r"""Запускает расчёт реакций MIX/RTS после получения новостей.

Реализация: source/build_reactions.py. Читает ZIP и выборку пилота, сохраняет
минутные свечи и все реакции в data/news_reactions без изменения источников.
Примеры из корня:
  .\.venv\Scripts\python.exe build_reactions.py
  .\.venv\Scripts\python.exe build_reactions.py --latency-seconds 5
  .\.venv\Scripts\python.exe build_reactions.py --all-news --month 2026-09
  .\.venv\Scripts\python.exe build_reactions.py --help
"""
from source.build_reactions import main

if __name__ == '__main__':
    main()
