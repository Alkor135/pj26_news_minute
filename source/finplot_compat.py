r"""Совместимый импорт finplot 1.9.7 при русской локали Windows.

Установленная библиотека проверяет locale.getdefaultlocale при импорте и вызывает
os._exit(1) для ru/be. Только на время импорта подставляется нейтральный ответ;
исходная функция восстанавливается в finally. Настройки ОС и файлы библиотеки
не меняются. Импорт выполняется в главном потоке до создания окна.

Используется просмотрщиком:
  .\.venv\Scripts\python.exe view_news_finplot.py
Проверка:
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_finplot_compat.py -v
Вспомогательный модуль source, самостоятельный запуск не требуется.
"""

import importlib
import locale


def neutral_import_locale(*args, **kwargs):
    """Возвращает нейтральную локаль только для импорта; args/kwargs игнорируются."""
    return 'en_US', 'UTF-8'


def load_finplot():
    """Возвращает модуль finplot, восстанавливая getdefaultlocale после импорта."""
    original = locale.getdefaultlocale
    try:
        locale.getdefaultlocale = neutral_import_locale
        return importlib.import_module('finplot')
    finally:
        locale.getdefaultlocale = original
