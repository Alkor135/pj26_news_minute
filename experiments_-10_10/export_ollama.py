r"""Экспортирует обученный LoRA-адаптер в квантованную модель GGUF для Ollama.

Проверяет завершённое обучение и точную ревизию базы, объединяет адаптер с
неквантованной базой, сохраняет Safetensors, преобразует их средствами
llama.cpp в F16 GGUF и квантует до Q4_K_M (или выбранного формата).
Результат записывается в новую папку export вместе с манифестом и SHA-256.
Незавершённый результат обучения и уже существующий каталог экспорта не меняет.
Использует общее .venv с зависимостями из корневого requirements.txt.
Без --run выбирает единственный профиль RTX4070/RTX3090Ti с отметкой
завершения в data/experiments_-10_10 относительно корня проекта.
Если таких профилей нет или готовы оба, требует явный --run.
Путь llama.cpp по умолчанию: data/experiments_-10_10/llama.cpp.
Поддерживает score и extremes по манифесту; ищет также PROFILE_extremes.
Задача и формат ответа передаются в export_manifest.json без смены типа модели.

Примеры запуска из корня проекта:
  .\.venv\Scripts\python.exe experiments_-10_10/export_ollama.py
  .\.venv\Scripts\python.exe experiments_-10_10/export_ollama.py --run data/experiments_-10_10/RTX4070_extremes
  .\.venv\Scripts\python.exe experiments_-10_10/export_ollama.py --run data/experiments_-10_10/RTX4070 --llama-cpp-dir data/experiments_-10_10/llama.cpp
  .\.venv\Scripts\python.exe experiments_-10_10/export_ollama.py --run data/experiments_-10_10/RTX3090Ti --llama-cpp-dir data/experiments_-10_10/llama.cpp --device cuda:0
  .\.venv\Scripts\python.exe experiments_-10_10/export_ollama.py --run data/experiments_-10_10/RTX4070 --check-only
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from model_tasks import parse_response, run_directories, task_name
QUANTIZATIONS = ('Q4_K_M', 'Q5_K_M', 'Q8_0')


def sha256_file(path):
    """Вычисляет SHA-256 файла path потоковым чтением и возвращает hex-строку."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_training_run(run):
    """Проверяет завершённый run, задачу и совместимость адаптера; возвращает манифест."""
    run = Path(run)
    manifest_file = run / 'training_manifest.json'
    metrics_file = run / 'training_metrics.json'
    provenance_file = run / 'adapter_provenance.json'
    adapter = run / 'adapter'
    required = (manifest_file, metrics_file, run / 'score_metrics.json',
                run / 'validation_predictions.json', provenance_file,
                adapter / 'adapter_config.json',
                adapter / 'adapter_model.safetensors', adapter / 'tokenizer_config.json')
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise ValueError('Нет завершённого обучения или адаптера: ' + ', '.join(missing))
    manifest = json.loads(manifest_file.read_text(encoding='utf-8'))
    provenance = json.loads(provenance_file.read_text(encoding='utf-8'))
    adapter_config = json.loads((adapter / 'adapter_config.json').read_text(encoding='utf-8'))
    if any(not isinstance(item, dict) for item in (manifest, provenance, adapter_config)):
        raise ValueError('Манифест, отметка завершения и конфигурация адаптера должны быть объектами JSON')
    if task_name(manifest) != task_name(provenance):
        raise ValueError('Задача адаптера не совпадает с манифестом обучения')
    model = manifest.get('model')
    revision = manifest.get('revision')
    if not isinstance(model, str) or not model.strip():
        raise ValueError('В манифесте нет имени базовой модели')
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('В манифесте нет точной SHA-ревизии базовой модели')
    if adapter_config.get('base_model_name_or_path') != model:
        raise ValueError('Базовая модель адаптера не совпадает с манифестом обучения')
    if provenance.get('model') != model or provenance.get('revision') != revision:
        raise ValueError('База или её ревизия в отметке завершения не совпадают с манифестом')
    if sha256_file(adapter / 'adapter_model.safetensors') != provenance.get('adapter_sha256'):
        raise ValueError('SHA-256 адаптера не совпадает с отметкой завершения')
    if not isinstance(manifest.get('system_prompt'), str) or not manifest['system_prompt'].strip():
        raise ValueError('В манифесте отсутствует системное задание')
    if manifest.get('profile') not in ('RTX4070', 'RTX3090Ti'):
        raise ValueError('Неизвестный профиль обучения')
    return manifest


def resolve_llama_tools(args):
    """Находит конвертер и квантизатор из args; возвращает оба абсолютных пути."""
    directory = args.llama_cpp_dir.resolve() if args.llama_cpp_dir else None
    converter = args.converter or (directory / 'convert_hf_to_gguf.py' if directory else None)
    if args.quantizer:
        quantizer = args.quantizer
    elif directory:
        candidates = [directory / 'build' / 'bin' / 'Release' / 'llama-quantize.exe',
                      directory / 'build' / 'bin' / 'llama-quantize.exe',
                      directory / 'build' / 'bin' / 'llama-quantize']
        quantizer = next((path for path in candidates if path.is_file()), candidates[0])
    else:
        quantizer = None
    if not converter or not Path(converter).is_file():
        raise ValueError('Не найден convert_hf_to_gguf.py: задайте --llama-cpp-dir или --converter')
    if not quantizer or not Path(quantizer).is_file():
        raise ValueError('Не найден llama-quantize: задайте --llama-cpp-dir или --quantizer')
    return Path(converter).resolve(), Path(quantizer).resolve()


def check_gguf(path):
    """Проверяет непустой GGUF-файл path и его сигнатуру; возвращает размер."""
    path = Path(path)
    if not path.is_file() or path.stat().st_size <= 4:
        raise RuntimeError(f'GGUF не создан или пуст: {path}')
    with path.open('rb') as handle:
        if handle.read(4) != b'GGUF':
            raise RuntimeError(f'Неверная сигнатура GGUF: {path}')
    return path.stat().st_size


def smoke_merged_model(model, tokenizer, system_prompt, task='score'):
    """Проверяет ответ модели по контракту task и возвращает отчёт о формате."""
    import torch
    payload = json.dumps({'title': 'Банк России повысил ключевую ставку',
                          'instrument': 'MIX', 'horizon_minutes': 5}, ensure_ascii=False)
    messages = [{'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': payload}]
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
    tokens = tokens.to(model.device)
    model.eval()
    with torch.inference_mode():
        generated = model.generate(tokens, max_new_tokens=32, do_sample=False,
                                   pad_token_id=tokenizer.eos_token_id)
    answer = tokenizer.decode(generated[0, tokens.shape[-1]:], skip_special_tokens=True).strip()
    try:
        value = parse_response(answer, task)
    except ValueError as error:
        return {'status': 'failed', 'answer': answer, 'error': str(error)}
    return {'status': 'passed', 'answer': answer, 'class' if task == 'extremes' else 'score': value}


def merge_adapter(run, manifest, target, device):
    """Объединяет адаптер run с базой manifest; сохраняет target и возвращает проверку."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError('Установите зависимости из корневого requirements.txt') from error
    cache = ROOT / 'data' / 'experiments_-10_10' / 'hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна; используйте --device cpu')
    print(f'Загрузка {manifest["model"]}@{manifest["revision"]} на {device}', flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        manifest['model'], revision=manifest['revision'], cache_dir=str(cache),
        torch_dtype=torch.float16, device_map=device, low_cpu_mem_usage=True)
    wrapped = PeftModel.from_pretrained(base, str(Path(run) / 'adapter'), is_trainable=False)
    merged = wrapped.merge_and_unload(safe_merge=True)
    merged.config.use_cache = True
    merged.save_pretrained(target, safe_serialization=True, max_shard_size='4GB')
    tokenizer = AutoTokenizer.from_pretrained(Path(run) / 'adapter')
    tokenizer.save_pretrained(target)
    smoke = smoke_merged_model(merged, tokenizer, manifest['system_prompt'], task_name(manifest))
    if smoke['status'] != 'passed':
        print(f'Предупреждение: пробный ответ объединённой модели неверен: {smoke["error"]}',
              file=sys.stderr)
    del merged, wrapped, base
    if device.startswith('cuda'):
        torch.cuda.empty_cache()
    if not list(Path(target).glob('*.safetensors')):
        raise RuntimeError('Объединённые веса Safetensors не созданы')
    return smoke


def convert_and_quantize(merged, converter, quantizer, folder, quantization, converter_python):
    """Конвертирует merged средствами converter_python; сохраняет F16 и готовый GGUF."""
    f16 = folder / 'model-f16.gguf'
    result = folder / f'model-{quantization}.gguf'
    subprocess.run([converter_python, str(converter), str(merged), '--outfile', str(f16),
                    '--outtype', 'f16'], check=True)
    check_gguf(f16)
    subprocess.run([str(quantizer), str(f16), str(result), quantization], check=True)
    check_gguf(result)
    return result


def export_run(args):
    """Выбирает обучение из args или единственный готовый профиль; возвращает каталог экспорта."""
    if args.run is None:
        directory = ROOT / 'data' / 'experiments_-10_10'
        candidates = [path for path in run_directories(directory)
                      if (path / 'adapter_provenance.json').is_file()]
        if not candidates:
            raise ValueError('Не найдено завершённое обучение RTX4070/RTX3090Ti; '
                             'завершите обучение или задайте --run ПУТЬ')
        if len(candidates) > 1:
            raise ValueError('Найдены оба или несколько завершённых профилей; задайте --run ПУТЬ: '
                             + ', '.join(str(path) for path in candidates))
        run = candidates[0].resolve()
        print(f'Автоматически выбран результат обучения: {run}', flush=True)
    else:
        run = args.run.resolve()
    manifest = validate_training_run(run)
    output = (args.output or run / 'export').resolve()
    if args.check_only:
        print(f'Результат обучения готов к экспорту: {run}', flush=True)
        return None
    if output.exists():
        raise ValueError(f'Каталог экспорта уже существует: {output}; выберите новый --output')
    if output == run or output == run / 'adapter' or run / 'adapter' in output.parents:
        raise ValueError('Экспорт нельзя сохранять поверх результата обучения или внутри адаптера')
    converter, quantizer = resolve_llama_tools(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.export_', dir=output.parent) as temp:
        folder = Path(temp)
        merged = folder / 'merged'
        smoke = merge_adapter(run, manifest, merged, args.device)
        gguf = convert_and_quantize(merged, converter, quantizer, folder, args.quantization,
                                    args.converter_python)
        report = {
            'profile': manifest['profile'], 'model': manifest['model'],
            'task': task_name(manifest),
            'revision': manifest['revision'], 'system_prompt': manifest['system_prompt'],
            'training_manifest_sha256': sha256_file(run / 'training_manifest.json'),
            'adapter_sha256': sha256_file(run / 'adapter' / 'adapter_model.safetensors'),
            'adapter_provenance_sha256': sha256_file(run / 'adapter_provenance.json'),
            'quantization': args.quantization, 'gguf_file': gguf.name,
            'gguf_sha256': sha256_file(gguf), 'gguf_bytes': check_gguf(gguf),
            'f16_gguf_file': 'model-f16.gguf',
            'f16_gguf_sha256': sha256_file(folder / 'model-f16.gguf'),
            'f16_gguf_bytes': check_gguf(folder / 'model-f16.gguf'),
            'converter': str(converter), 'quantizer': str(quantizer),
            'merged_format': 'Safetensors FP16',
            'merged_smoke_test': smoke,
        }
        (folder / 'export_manifest.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        folder.rename(output)
    return output


def build_parser():
    """Возвращает парсер экспорта с автоматическим выбором обучения и стандартным llama.cpp."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path,
                        help='Папка обучения; по умолчанию единственный завершённый профиль RTX4070/RTX3090Ti')
    parser.add_argument('--output', type=Path, help='Новая папка результата; по умолчанию RUN/export')
    parser.add_argument('--llama-cpp-dir', type=Path,
                        default=ROOT / 'data' / 'experiments_-10_10' / 'llama.cpp',
                        help='Папка llama.cpp; по умолчанию data/experiments_-10_10/llama.cpp в проекте')
    parser.add_argument('--converter', type=Path, help='Путь к convert_hf_to_gguf.py вместо --llama-cpp-dir')
    parser.add_argument('--converter-python', default=sys.executable,
                        help='Python с зависимостями llama.cpp; по умолчанию текущий Python')
    parser.add_argument('--quantizer', type=Path, help='Путь к llama-quantize.exe вместо --llama-cpp-dir')
    parser.add_argument('--quantization', choices=QUANTIZATIONS, default='Q4_K_M')
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu',
                        help='Устройство объединения весов; по умолчанию CPU')
    parser.add_argument('--check-only', action='store_true', help='Проверить обучение без загрузки весов')
    return parser


def main(argv=None):
    """Запускает экспорт с argv; печатает результат и возвращает код завершения."""
    args = build_parser().parse_args(argv)
    try:
        result = export_run(args)
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f'Ошибка экспорта: {error}', file=sys.stderr)
        return 1
    if result:
        print(f'Готовый GGUF и манифест: {result}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
