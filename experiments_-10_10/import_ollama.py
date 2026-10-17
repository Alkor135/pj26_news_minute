r"""Регистрирует проверенный GGUF-экспорт как отдельную модель Ollama.

Проверяет манифест, SHA-256 и сигнатуру GGUF, создаёт Modelfile с системным
заданием обучения, вызывает ollama create и проверяет ответ через /api/chat.
Существующее имя модели не заменяет без явного --replace. Пробный запрос
проверяет JSON с одним целым score в диапазоне от -10 до +10.

Примеры запуска из корня проекта:
  .\.venv\Scripts\python.exe experiments_-10_10/import_ollama.py --export data/experiments_-10_10/RTX4070/export
  .\.venv\Scripts\python.exe experiments_-10_10/import_ollama.py --export data/experiments_-10_10/RTX3090Ti/export --name news-score-3090ti
  .\.venv\Scripts\python.exe experiments_-10_10/import_ollama.py --export data/experiments_-10_10/RTX4070/export --check-only
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib import error as urlerror
from urllib import request as urlrequest


DEFAULT_NAMES = {'RTX4070': 'news-score-4070', 'RTX3090Ti': 'news-score-3090ti'}


def sha256_file(path):
    """Вычисляет SHA-256 файла path потоковым чтением и возвращает hex-строку."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_export(folder):
    """Проверяет folder и целостность GGUF; возвращает манифест и путь к весам."""
    folder = Path(folder).resolve()
    source = folder / 'export_manifest.json'
    if not source.is_file():
        raise ValueError(f'Не найден манифест экспорта: {source}')
    manifest = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(manifest, dict):
        raise ValueError('Манифест экспорта должен быть объектом JSON')
    if manifest.get('profile') not in DEFAULT_NAMES:
        raise ValueError('Неизвестный профиль модели')
    if not isinstance(manifest.get('model'), str) or not manifest['model'].strip():
        raise ValueError('В манифесте отсутствует базовая модель')
    if not isinstance(manifest.get('revision'), str) or not re.fullmatch(r'[0-9a-f]{40}', manifest['revision']):
        raise ValueError('В манифесте отсутствует точная SHA-ревизия базы')
    prompt = manifest.get('system_prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('В манифесте отсутствует системное задание')
    filename = manifest.get('gguf_file')
    if not isinstance(filename, str) or Path(filename).name != filename or filename in ('.', '..'):
        raise ValueError('Имя GGUF должно обозначать файл внутри папки экспорта')
    gguf = folder / filename
    if not gguf.is_file() or gguf.stat().st_size <= 4:
        raise ValueError(f'Не найден непустой GGUF: {gguf}')
    with gguf.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise ValueError('У файла нет сигнатуры GGUF')
    digest = manifest.get('gguf_sha256')
    if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
        raise ValueError('В манифесте отсутствует SHA-256 файла GGUF')
    if sha256_file(gguf) != digest:
        raise ValueError('SHA-256 GGUF не совпадает с манифестом экспорта')
    return manifest, gguf


def build_modelfile(gguf, system_prompt):
    """Создаёт текст Modelfile для gguf с заданием system_prompt и нулевой температурой."""
    if '"""' in system_prompt:
        raise ValueError('Системное задание нельзя безопасно записать в Modelfile')
    return (f'FROM ./{Path(gguf).name}\n'
            'PARAMETER temperature 0\n'
            'PARAMETER num_predict 32\n'
            f'SYSTEM """{system_prompt}"""\n')


def parse_score(answer):
    """Возвращает целый score из ответа answer или отклоняет нарушение контракта."""
    try:
        data = json.loads(answer)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError('Ответ модели не является JSON') from error
    if not isinstance(data, dict) or set(data) != {'score'}:
        raise ValueError('Ответ должен содержать только поле score')
    score = data['score']
    if type(score) is not int or not -10 <= score <= 10:
        raise ValueError('score должен быть целым числом от -10 до +10')
    return score


def api_json(host, endpoint, payload=None, timeout=30):
    """Вызывает JSON API Ollama по host/endpoint и возвращает объект ответа."""
    url = host.rstrip('/') + endpoint
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = urlrequest.Request(url, data=body, headers={'Content-Type': 'application/json'})
    try:
        with urlrequest.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except (urlerror.URLError, TimeoutError) as exc:
        raise RuntimeError(f'Нет ответа Ollama {url}: {exc}') from exc


def existing_models(host):
    """Возвращает множество зарегистрированных имён моделей Ollama на host."""
    data = api_json(host, '/api/tags')
    if not isinstance(data, dict) or not isinstance(data.get('models'), list):
        raise RuntimeError('Ollama вернула неверный ответ /api/tags')
    return {str(item['name']) for item in data['models'] if isinstance(item, dict) and 'name' in item}


def smoke_test(host, name):
    """Отправляет тестовый запрос модели name и возвращает ответ и проверенный score."""
    message = json.dumps({'title': 'Банк России повысил ключевую ставку',
                          'instrument': 'MIX', 'horizon_minutes': 5}, ensure_ascii=False)
    response = api_json(host, '/api/chat', {
        'model': name, 'messages': [{'role': 'user', 'content': message}],
        'stream': False, 'format': 'json', 'options': {'temperature': 0, 'num_predict': 32}
    }, timeout=300)
    if not isinstance(response, dict) or not isinstance(response.get('message'), dict):
        raise RuntimeError('Ollama вернула неверную структуру /api/chat')
    answer = response['message'].get('content')
    return answer, parse_score(answer)


def import_model(args):
    """Проверяет экспорт, создаёт модель по args и возвращает её имя."""
    folder = args.export.resolve()
    manifest, gguf = validate_export(folder)
    name = args.name or DEFAULT_NAMES[manifest['profile']]
    if not re.fullmatch(r'[a-z0-9][a-z0-9._/-]*(?::[a-z0-9._-]+)?', name):
        raise ValueError('Некорректное имя модели Ollama')
    modelfile = build_modelfile(gguf, manifest['system_prompt'])
    if args.check_only:
        print(f'GGUF готов к импорту: {gguf}; имя: {name}', flush=True)
        return name
    names = existing_models(args.host)
    canonical = name if ':' in name else name + ':latest'
    if (name in names or canonical in names) and not args.replace:
        raise ValueError(f'Модель {name} уже существует; для осознанной замены задайте --replace')
    modelfile_path = folder / 'Modelfile'
    if modelfile_path.exists() and modelfile_path.read_text(encoding='utf-8') != modelfile:
        raise ValueError('Существующий Modelfile отличается от текущего экспорта')
    modelfile_path.write_text(modelfile, encoding='utf-8')
    environment = os.environ.copy()
    environment['OLLAMA_HOST'] = args.host
    subprocess.run([args.ollama, 'create', name, '-f', str(modelfile_path)],
                   check=True, env=environment)
    if name not in existing_models(args.host) and canonical not in existing_models(args.host):
        raise RuntimeError('После ollama create модель не найдена в /api/tags')
    report = {'ollama_name': name, 'profile': manifest['profile'],
              'gguf_sha256': manifest['gguf_sha256'], 'host': args.host}
    try:
        answer, score = smoke_test(args.host, name)
    except (ValueError, RuntimeError) as error:
        report['smoke_test'] = {'status': 'failed', 'error': str(error)}
        (folder / 'import_report.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        raise RuntimeError(f'Модель создана, но пробный ответ неверен: {error}') from error
    report['smoke_test'] = {'status': 'passed', 'score': score, 'answer': answer}
    (folder / 'import_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return name


def build_parser():
    """Создаёт парсер командной строки для импорта и проверки GGUF."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--export', required=True, type=Path, help='Папка с GGUF и export_manifest.json')
    parser.add_argument('--name', help='Имя новой модели; по умолчанию зависит от профиля GPU')
    parser.add_argument('--host', default='http://127.0.0.1:11434', help='Адрес локального Ollama API')
    parser.add_argument('--ollama', default='ollama', help='Команда или путь к ollama.exe')
    parser.add_argument('--replace', action='store_true', help='Явно разрешить обновление модели с тем же именем')
    parser.add_argument('--check-only', action='store_true', help='Проверить GGUF без Ollama и записи файлов')
    return parser


def main(argv=None):
    """Запускает импорт с argv; печатает результат и возвращает код завершения."""
    args = build_parser().parse_args(argv)
    try:
        name = import_model(args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Ошибка импорта: {error}', file=sys.stderr)
        return 1
    if not args.check_only:
        print(f'Модель Ollama создана и проверена: {name}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
