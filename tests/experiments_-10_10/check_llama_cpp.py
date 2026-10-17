r"""Проверяет реальную конвертацию Qwen3 в GGUF и квантование Q4_K_M.

Создаёт маленькую Qwen3 со случайными весами и локальным токенизатором,
запускает convert_hf_to_gguf.py и llama-quantize, проверяет квантованные
тензоры и уменьшение размера. Интернет и GPU не нужны. Все временные файлы
находятся внутри tests и удаляются; файловые логи не создаются.

Запуск из корня после подготовки llama.cpp и сохранения токенизатора:
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_llama_cpp.py
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_llama_cpp.py --tokenizer-dir data/experiments_-10_10/RTX4070/adapter
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]


def run_checked(command):
    """Выполняет command без файлового лога; при ошибке возвращает диагностику исключением."""
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(f'Код завершения {result.returncode}:\n' + (result.stdout + result.stderr)[-6000:])


def check_tools(llama_dir, tokenizer_dir):
    """Проверяет инструменты llama_dir с локальным tokenizer_dir и возвращает отчёт."""
    import gguf
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    converter = llama_dir / 'convert_hf_to_gguf.py'
    quantizer = llama_dir / 'build' / 'bin' / 'Release' / 'llama-quantize.exe'
    for path in (converter, quantizer, tokenizer_dir / 'tokenizer_config.json'):
        if not path.is_file():
            raise ValueError(f'Не найден файл: {path}')
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(prefix='tmp_llama_cpp_', dir=ROOT / 'tests') as folder:
        work = Path(folder)
        model_dir = work / 'model'
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=len(tokenizer), hidden_size=256, intermediate_size=512,
            num_hidden_layers=1, num_attention_heads=8, num_key_value_heads=4,
            head_dim=32, max_position_embeddings=512, tie_word_embeddings=False,
            eos_token_id=tokenizer.eos_token_id)).half()
        model.save_pretrained(model_dir, safe_serialization=True)
        tokenizer.save_pretrained(model_dir)
        del model
        f16 = work / 'model-f16.gguf'
        quantized = work / 'model-Q4_K_M.gguf'
        print('Проверка: конвертация синтетической Qwen3 в F16 GGUF...', flush=True)
        run_checked([sys.executable, str(converter), str(model_dir), '--outtype', 'f16', '--outfile', str(f16)])
        print('Проверка: квантование GGUF в Q4_K_M...', flush=True)
        run_checked([str(quantizer), str(f16), str(quantized), 'Q4_K_M'])
        reader = gguf.GGUFReader(quantized)
        quantized_tensors = sum(tensor.tensor_type == gguf.GGMLQuantizationType.Q4_K for tensor in reader.tensors)
        tensor_count = len(reader.tensors)
        del reader
        gc.collect()
        if not quantized_tensors or quantized.stat().st_size >= f16.stat().st_size:
            raise RuntimeError('Не подтверждены тензоры Q4_K или уменьшение размера')
        return dict(result='Конвертация и квантование прошли',
                    tokenizer_dir=str(tokenizer_dir), tensors=tensor_count,
                    q4_k_tensors=quantized_tensors, f16_bytes=f16.stat().st_size,
                    q4_k_m_bytes=quantized.stat().st_size)


def main(argv=None):
    """Читает argv, запускает проверку и возвращает код завершения."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--llama-cpp-dir', type=Path, default=ROOT / 'data' / 'experiments_-10_10' / 'llama.cpp',
                        help='Папка исходников и квантизатора llama.cpp')
    parser.add_argument('--tokenizer-dir', type=Path, default=ROOT / 'data' / 'experiments_-10_10' / 'RTX4070' / 'adapter',
                        help='Локальный токенизатор Qwen3; новые веса не скачиваются')
    args = parser.parse_args(argv)
    try:
        print(json.dumps(check_tools(args.llama_cpp_dir.resolve(), args.tokenizer_dir.resolve()),
                         ensure_ascii=False, indent=2))
    except (ImportError, ValueError, RuntimeError, OSError) as error:
        print(f'Ошибка проверки llama.cpp: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
