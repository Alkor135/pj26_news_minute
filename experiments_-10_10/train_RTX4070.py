"""Дообучает Qwen3 4B методом QLoRA с начальными настройками для RTX 4070 12 ГБ.

Читает готовые баллы, отделяет валидацию по времени, обучает один адаптер
для MIX/RTS и пяти горизонтов. База: Qwen/Qwen3-4B-Instruct-2507;
batch=1, накопление=16, длина=512. Потребление памяти проверяется на реальном запуске.
Общая логика и параметры находятся в source/training_common.py; файловых логов нет.
Без --data использует data/experiments_-10_10/scores.jsonl из корня проекта.
По умолчанию --task extremes: классы negative (-10..-8), none (-7..7),
positive (8..10); балансируется только обучение. Каталог RTX4070_extremes.
--task score сохраняет прежнюю балльную задачу и каталог RTX4070.
Финальный сентябрь задаётся --evaluation-role test и границами периода;
при этом нужен отдельный файл данных с сентябрём и новый --output.

Из корня проекта в .venv с зависимостями из корневого requirements.txt:
  python experiments_-10_10/train_RTX4070.py
  python experiments_-10_10/train_RTX4070.py --dry-run
  python experiments_-10_10/train_RTX4070.py --task score --output data/experiments_-10_10/RTX4070_score_retry
  python experiments_-10_10/train_RTX4070.py --data data/experiments_-10_10/scores.jsonl --epochs 1
  python experiments_-10_10/train_RTX4070.py --help
  python experiments_-10_10/train_RTX4070.py --evaluation-role test --validation-start 2026-09-01 --cutoff 2026-10-01 --data data/experiments_-10_10/scores_test_202609.jsonl --output data/experiments_-10_10/RTX4070_extremes_test_202609 --dry-run
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from training_common import main


if __name__ == '__main__':
    raise SystemExit(main('RTX4070'))
