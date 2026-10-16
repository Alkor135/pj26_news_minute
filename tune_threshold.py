r"""Исполняемая команда сравнения коэффициентов динамического порога.

Реализация в source/tune_threshold.py; читает готовую БД, не пересчитывает тики
или эмбеддинги. Сохраняет все значения, не выбирает лучший параметр автоматически.
Примеры:
  .\.venv\Scripts\python.exe tune_threshold.py
  .\.venv\Scripts\python.exe tune_threshold.py --multipliers 0.25 0.5 0.75 1
Все параметры: --help. Выход threshold_tuning рядом с БД; файловых логов нет.
"""

from source.tune_threshold import main

if __name__ == '__main__':
    main()
