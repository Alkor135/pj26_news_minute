# Вспомогательные модули эксперимента −10…+10

`training_common.py` содержит общую логику обучения для профилей RTX 4070
и RTX 3090 Ti. Модуль отдельно не запускается. Его импортируют
`../train_RTX4070.py` и `../train_RTX3090Ti.py`, добавляя эту папку в путь
поиска модулей. Она отделена от общей папки `source` в корне проекта.

Модуль читает CSV/JSONL с готовыми баллами, проверяет метки и обязательные поля,
разделяет историю по времени, исключает пересечения окон и перепечатки.
Затем загружает базовую модель Hugging Face в NF4, обучает LoRA, считает метрики
валидации и сохраняет адаптер с отметкой завершения и SHA-256 весов.
Файловых логов нет. Признаки модели — заголовок, инструмент и горизонт;
будущие ценовые метрики используются только как целевая разметка.

Команды из корня проекта в общем `.venv` с зависимостями из `requirements.txt`:

```powershell
.\.venv\Scripts\python.exe experiments_-10_10/train_RTX4070.py --dry-run
.\.venv\Scripts\python.exe experiments_-10_10/train_RTX4070.py
.\.venv\Scripts\python.exe experiments_-10_10/train_RTX3090Ti.py
.\.venv\Scripts\python.exe experiments_-10_10/train_RTX4070.py --help
```

По умолчанию `--data` указывает на `data/experiments_-10_10/scores.jsonl`,
а `--output` — на `data/experiments_-10_10/RTX4070` или `RTX3090Ti`.
Эти пути определяются от корня проекта независимо от рабочего каталога.
Основные параметры: `--model`, `--revision`, `--train-start`,
`--validation-start`, `--cutoff`, `--epochs`, `--batch-size`, `--accumulation`,
`--max-length`, `--rank`, `--learning-rate`, `--max-steps`, `--prediction-limit`
и `--seed`. `--dry-run` проверяет только данные без загрузки модели и GPU.

Результаты: `data_report.json`, `training_manifest.json`, `adapter/`,
`training_metrics.json`, `score_metrics.json`, `validation_predictions.json`
и `adapter_provenance.json`. Существующее завершённое обучение не перезаписывается.
Полное описание формата данных и этапов — в [README эксперимента](../README.md).
