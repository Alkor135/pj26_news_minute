r"""Исполняемая команда пробы фильтра новостей российского рынка.

Реализация в source/russian_market.py. Сначала месяц проверки и год истории классифицируются
без цен, затем include получает эмбеддинги, реакции и совместимые прошлые аналоги.
Примеры:
  .\.venv\Scripts\python.exe russian_market.py --stage classify
  .\.venv\Scripts\python.exe russian_market.py --month 2026-09 --history-years 1
  .\.venv\Scripts\python.exe russian_market.py --history-years 0
  .\.venv\Scripts\python.exe russian_market.py --threshold-multiplier 0.5
Параметры: --help. Годовой выход data/russian_market/2026-09_year1, логов нет.
"""

from source.russian_market import main

if __name__ == '__main__':
    main()
