r"""Открывает минутные свечи и исторические реакции на новости в finplot.

Реализация: source/view_news_finplot.py. Читает готовую БД реакций, показывает
события, пять прошлых аналогов, динамические пороги и полную RSS-ленту.
Примеры из корня:
  .\.venv\Scripts\python.exe view_news_finplot.py
  .\.venv\Scripts\python.exe view_news_finplot.py --instrument RTS --horizon 60
  .\.venv\Scripts\python.exe view_news_finplot.py --all-events --search санкц
  .\.venv\Scripts\python.exe view_news_finplot.py --selection-mode stable
  .\.venv\Scripts\python.exe view_news_finplot.py --selection-mode assessments
  .\.venv\Scripts\python.exe view_news_finplot.py --threshold-multiplier 0.5
  .\.venv\Scripts\python.exe view_news_finplot.py --database data\russian_market\2026-09\reactions\reactions.sqlite3 --analogs data\russian_market\2026-09\analogs\analogs.sqlite3 --selection-mode assessments
  .\.venv\Scripts\python.exe view_news_finplot.py --screenshot data\news_reactions\finplot.png
  .\.venv\Scripts\python.exe view_news_finplot.py --help
"""
from source.view_news_finplot import main

if __name__ == '__main__':
    main()
