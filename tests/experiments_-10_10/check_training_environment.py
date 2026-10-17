r"""Проверяет зависимости QLoRA, CUDA и один шаг обучения в текущем окружении.

Создаёт крошечную Qwen3 со случайными весами в tests, загружает её в NF4,
добавляет LoRA и выполняет один шаг AdamW через Trainer на GPU. Модели из интернета не
скачивает; временные файлы удаляет. Проверка не оценивает качество прогнозов
и размещение полной Qwen3 4B. Отчёт выводится в консоль, файловых логов нет.

Запуск из корня:
  .\.venv\Scripts\python.exe tests/experiments_-10_10/check_training_environment.py
"""

from importlib.metadata import version
import json
from pathlib import Path
import sys
import tempfile


def check_environment():
    """Проверяет синтетический шаг QLoRA на CUDA и возвращает версии и результат."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, BitsAndBytesConfig,
                              Qwen3Config, Qwen3ForCausalLM, Trainer, TrainingArguments,
                              default_data_collator)

    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch не видит GPU CUDA')
    torch.manual_seed(42)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    config = Qwen3Config(vocab_size=128, hidden_size=64, intermediate_size=128,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=4, head_dim=16, max_position_embeddings=64,
                         tie_word_embeddings=False, use_cache=False)
    with tempfile.TemporaryDirectory(prefix='tmp_qlora_', dir=Path(__file__).resolve().parents[1]) as folder:
        base_path = Path(folder) / 'base'
        Qwen3ForCausalLM(config).save_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, local_files_only=True, dtype=dtype,
            device_map={'': 0}, attn_implementation='sdpa',
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
        if not getattr(model, 'is_loaded_in_4bit', False):
            raise RuntimeError('Модель не загружена в 4-битном формате')
        model = prepare_model_for_kbit_training(
            model, gradient_checkpointing_kwargs={'use_reentrant': False})
        model = get_peft_model(model, LoraConfig(
            r=4, lora_alpha=8, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                            'gate_proj', 'up_proj', 'down_proj'],
            bias='none', task_type='CAUSAL_LM'))
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        tracked = next(parameter for name, parameter in model.named_parameters() if 'lora_B' in name)
        before = tracked.detach().clone()
        tokens = torch.randint(0, 128, (2, 16)).tolist()
        rows = [dict(input_ids=row, attention_mask=[1] * len(row), labels=row) for row in tokens]
        training_args = TrainingArguments(
            output_dir=str(Path(folder) / 'trainer'), max_steps=1,
            per_device_train_batch_size=2, learning_rate=1e-3,
            bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
            optim='adamw_torch', gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            save_strategy='no', logging_strategy='no', report_to=[], disable_tqdm=True,
            dataloader_num_workers=0, dataloader_pin_memory=False)
        print('Проверка: один шаг Trainer с QLoRA на GPU...', flush=True)
        trainer = Trainer(model=model, args=training_args, train_dataset=rows,
                          data_collator=default_data_collator)
        loss = trainer.train().training_loss
        if not torch.isfinite(torch.tensor(loss)):
            raise RuntimeError('Нечисловая ошибка обучения')
        torch.cuda.synchronize()
        if torch.equal(before, tracked.detach()):
            raise RuntimeError('Шаг оптимизатора не обновил LoRA-веса')
        model.save_pretrained(Path(folder) / 'adapter')
        if not (Path(folder) / 'adapter' / 'adapter_model.safetensors').is_file():
            raise RuntimeError('Адаптер не сохранён')
        return dict(python=sys.version.split()[0], gpu=torch.cuda.get_device_name(0),
                    cuda=torch.version.cuda, dtype=str(dtype), loss=loss,
                    trainable_parameters=sum(parameter.numel() for parameter in parameters),
                    packages={name: version(name) for name in (
                        'torch', 'transformers', 'peft', 'accelerate', 'bitsandbytes', 'safetensors')},
                    result='4-битная загрузка, обратное распространение, шаг AdamW и сохранение LoRA прошли')


def main():
    """Выводит отчёт проверки окружения и возвращает код завершения."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    try:
        print(json.dumps(check_environment(), ensure_ascii=False, indent=2))
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        print(f'Ошибка проверки окружения: {error}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
