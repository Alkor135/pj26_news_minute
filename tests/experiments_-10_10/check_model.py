r"""Проверяет импортированную модель Ollama на сохранённой отложенной выборке.

Без параметров выбирает единственный импортированный профиль RTX4070/RTX3090Ti.
Сверяет исходную разметку по SHA-256, восстанавливает временное разделение
обучения и выбирает те же примеры, которые оценивались до экспорта.
В Ollama отправляет только заголовок, инструмент и горизонт. Считает MAE,
ошибки JSON, совпадение балла и знака; сравнивает с нулём и адаптером.
По умолчанию проверяет первые 32 сохранённых примера; --limit 0 — все.
Сентябрь допускает только при явной роли test в манифесте обучения.
Проверяет качество баллов/классов, а не доходность торговли.
Перезаписывает check_model_report.json в каталоге обучения; файловых логов нет.
Для задачи extremes вместо MAE считает precision/recall по трём классам
и матрицу ошибок. Профили PROFILE_extremes поддерживаются отдельно от score.

Запуск из корня проекта или кнопкой VS Code, при работающей Ollama:
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py --limit 0
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py --run data/experiments_-10_10/RTX3090Ti
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py --prepare-only
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py --run data/experiments_-10_10/RTX4070_extremes --limit 0
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_model.py --run data/experiments_-10_10/RTX4070_extremes_test_202609 --data data/experiments_-10_10/scores_test_202609.jsonl --limit 0
Параметры --name и --host выбирают модель и сервер; --data задаёт перенесённую
разметку с тем же SHA-256. --prepare-only проверяет данные без вызовов Ollama.
"""

import argparse
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments_-10_10'))
sys.path.insert(0, str(ROOT / 'experiments_-10_10' / 'source'))
import import_ollama as ollama
import training_common as training
from model_tasks import classification_metrics, parse_response, run_directories, score_class, task_name


def metrics(predictions):
    """Возвращает метрики списка predictions; None означает отсутствие корректного ответа."""
    valid = [p for p in predictions if p['predicted'] is not None]
    nonzero = [p for p in valid if p['actual'] != 0]
    count = len(predictions)
    return dict(sample_size=count, valid_answers=len(valid),
                invalid_fraction=(count - len(valid)) / count if count else None,
                mae=sum(abs(p['predicted'] - p['actual']) for p in valid) / len(valid) if valid else None,
                zero_baseline_mae_same_rows=sum(abs(p['actual']) for p in valid) / len(valid) if valid else None,
                zero_baseline_mae_all_rows=sum(abs(p['actual']) for p in predictions) / count if count else None,
                exact_accuracy=sum(p['predicted'] == p['actual'] for p in valid) / len(valid) if valid else None,
                direction_accuracy_nonzero_actual=sum(
                    (p['predicted'] > 0) - (p['predicted'] < 0) ==
                    (p['actual'] > 0) - (p['actual'] < 0) for p in nonzero
                ) / len(nonzero) if nonzero else None,
                direction_sample_size=len(nonzero))


def predict(host, name, row, system_prompt, task='score'):
    """Запрашивает модель name на host по row без метки; возвращает текст, значение task и задержку."""
    payload = dict(title=row['title'], instrument=row['instrument'], horizon_minutes=row['horizon'])
    started = time.perf_counter()
    response = ollama.api_json(host, '/api/chat', {
        'model': name, 'messages': [dict(role='system', content=system_prompt),
                                  dict(role='user', content=json.dumps(payload, ensure_ascii=False))],
        'stream': False, 'format': 'json', 'options': {'temperature': 0, 'num_predict': 32}
    }, timeout=300)
    if not isinstance(response, dict) or not isinstance(response.get('message'), dict):
        raise ValueError('Неверная структура ответа Ollama')
    answer = response['message'].get('content')
    elapsed = time.perf_counter() - started
    return answer, parse_response(answer, task), elapsed


def prepare_cases(run, data, limit):
    """Проверяет run, роль периода и SHA data; возвращает сохранённые отложенные строки и манифест."""
    manifest = json.loads((run / 'training_manifest.json').read_text(encoding='utf-8'))
    task_name(manifest)
    report = json.loads((run / 'data_report.json').read_text(encoding='utf-8'))
    if ollama.sha256_file(data) != report['data_sha256']:
        raise ValueError('Разметка отличается от обучения по SHA-256; требуется исходный файл')
    parameters = manifest['parameters']
    training.validate_evaluation_period(parameters)
    rows, _ = training.read_examples(data)
    _, valid, _ = training.split_examples(rows, parameters['train_start'],
                                         parameters['validation_start'], parameters['cutoff'])
    indexed = {(r['news_id'], r['instrument'], r['horizon']): r for r in valid}
    saved = json.loads((run / 'validation_predictions.json').read_text(encoding='utf-8'))
    selected, seen = [], set()
    for prediction in saved:
        key = prediction['news_id'], prediction['instrument'], prediction['horizon']
        if key in seen or key not in indexed or prediction['actual'] != indexed[key]['score']:
            raise ValueError('Сохранённые предсказания не соответствуют исходной валидации')
        seen.add(key)
        selected.append(dict(indexed[key], adapter_predicted=prediction['predicted']))
    if not selected:
        raise ValueError('Сохранённая валидация пуста')
    return selected[:limit] if limit else selected, manifest


def build_parser():
    """Возвращает параметры проверки модели; пути по умолчанию определяются от проекта."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, help='Каталог обучения; по умолчанию единственный импортированный профиль')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/experiments_-10_10/scores.jsonl', help='Исходная разметка')
    parser.add_argument('--name', help='Имя модели; по умолчанию из import_report.json')
    parser.add_argument('--host', help='Адрес Ollama; по умолчанию из import_report.json')
    parser.add_argument('--limit', type=int, default=32, help='Число примеров; 0 — вся сохранённая выборка')
    parser.add_argument('--prepare-only', action='store_true', help='Проверить данные без обращения к Ollama')
    return parser


def main(argv=None):
    """Проверяет данные и Ollama по argv; пишет отчёт и возвращает 0 либо 1 при ошибке."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    args = build_parser().parse_args(argv)
    try:
        if args.limit < 0:
            raise ValueError('--limit должен быть неотрицательным')
        if args.run is None:
            base = ROOT / 'data/experiments_-10_10'
            candidates = [path for path in run_directories(base)
                          if (path / 'export/import_report.json').is_file()]
            if len(candidates) != 1:
                raise ValueError('Нужен единственный импортированный профиль или явный --run ПУТЬ')
            run = candidates[0]
        else:
            run = args.run.resolve()
        rows, manifest = prepare_cases(run, args.data, args.limit)
        print(f'Проверка: {run.name}, примеров: {len(rows)}', flush=True)
        if args.prepare_only:
            return 0
        imported = json.loads((run / 'export/import_report.json').read_text(encoding='utf-8'))
        task = task_name(manifest)
        if task_name(imported) != task:
            raise ValueError('Задача импорта не совпадает с задачей обучения')
        name = args.name or imported['ollama_name']
        host = args.host or imported['host']
        names = ollama.existing_models(host)
        if name not in names and (name if ':' in name else name + ':latest') not in names:
            raise ValueError(f'В Ollama отсутствует модель {name}')
        predictions = []
        for number, row in enumerate(rows, 1):
            result = dict(news_id=row['news_id'], instrument=row['instrument'], horizon=row['horizon'],
                          actual=row['score'], adapter_predicted=row['adapter_predicted'], predicted=None)
            if task == 'extremes':
                result['actual_class'] = score_class(row['score'])
            try:
                raw, score, elapsed = predict(host, name, row, manifest['system_prompt'], task)
                result.update(raw=raw, predicted=score, seconds=elapsed)
            except ValueError as error:
                result['error'] = str(error)
            predictions.append(result)
            actual_text = f'{score_class(row["score"])} (балл {row["score"]})' if task == 'extremes' else str(row['score'])
            print(f'{number}/{len(rows)}: {row["instrument"]} {row["horizon"]} мин; '
                  f'факт={actual_text}, прогноз={result["predicted"]}', flush=True)
        metric_function = classification_metrics if task == 'extremes' else metrics
        summary = metric_function(predictions)
        adapter = metric_function([dict(actual=p['actual'], predicted=p['adapter_predicted']) for p in predictions])
        groups = {f'{i}_{h}': metric_function([p for p in predictions if p['instrument'] == i and p['horizon'] == h])
                  for i in ('MIX', 'RTS') for h in training.HORIZONS
                  if any(p['instrument'] == i and p['horizon'] == h for p in predictions)}
        report = dict(ollama_name=name, host=host, task=task,
                      evaluation_role=manifest['parameters'].get('evaluation_role', 'validation'),
                      model_identity_verified=False,
                      note='Проверяется модель под указанным именем; её веса могли быть заменены после импорта.',
                      data_sha256=ollama.sha256_file(args.data),
                      period={k: manifest['parameters'][k] for k in ('validation_start', 'cutoff')},
                      metrics=summary, adapter_metrics_same_sample=adapter,
                      by_instrument_horizon=groups, predictions=predictions)
        target = run / 'check_model_report.json'
        training.save_json(target, report)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f'Отчёт: {target}', flush=True)
        return 0 if summary['valid_answers'] == len(rows) else 1
    except (ValueError, KeyError, TypeError, RuntimeError, OSError) as error:
        print(f'Ошибка проверки: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
