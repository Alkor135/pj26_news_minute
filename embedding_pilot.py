r"""Запускает сравнение эмбеддингов новостей через локальный Ollama.

Реализация: source/embedding_pilot.py. Читает исходные БД без изменения, сохраняет
SQLite-кэш, аналоги и замеры в data/embedding_pilot.
Примеры из корня:
  .\.venv\Scripts\python.exe embedding_pilot.py
  .\.venv\Scripts\python.exe embedding_pilot.py --month 2026-09 --limit 2000 --batch-size 32
  .\.venv\Scripts\python.exe embedding_pilot.py --help
"""
from source.embedding_pilot import main

if __name__ == '__main__':
    main()
