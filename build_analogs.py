r"""Исполняемая команда расчёта пяти прошлых аналогов и динамических порогов.

Реализация находится в source/build_analogs.py, исходные данные не изменяются.
Примеры:
  .\.venv\Scripts\python.exe build_analogs.py
  .\.venv\Scripts\python.exe build_analogs.py --quantile 0.75 --min-similarity 0.5
  .\.venv\Scripts\python.exe build_analogs.py --threshold-multiplier 0.5
Параметры: --help. Выход: data/analog_analysis, прогресс в консоли, логов нет.
"""

from source.build_analogs import main

if __name__ == '__main__':
    main()
