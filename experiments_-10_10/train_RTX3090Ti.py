"""Дообучает Qwen2.5 7B методом QLoRA с настройками для RTX 3090 Ti 24 ГБ.

Читает готовые баллы, отделяет валидацию по времени, обучает один адаптер
для MIX/RTS и пяти горизонтов. База: Qwen/Qwen2.5-7B-Instruct;
batch=2, накопление=8, длина=512. Потребление памяти проверяется на реальном запуске.
Общая логика и параметры находятся в source/training_common.py; файловых логов нет.
Без --data использует data/experiments_-10_10/scores.jsonl из корня проекта.

Из корня проекта в .venv с зависимостями из корневого requirements.txt:
  python experiments_-10_10/train_RTX3090Ti.py
  python experiments_-10_10/train_RTX3090Ti.py --dry-run
  python experiments_-10_10/train_RTX3090Ti.py --data data/experiments_-10_10/scores.jsonl --epochs 1
  python experiments_-10_10/train_RTX3090Ti.py --help
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from training_common import main


if __name__ == '__main__':
    raise SystemExit(main('RTX3090Ti'))
