r"""Проверяет запуск finplot при русской локали и восстановление функции locale.

Изолированный дочерний Python необходим: исходный finplot вызывает os._exit(1),
поэтому обычный импорт завершил бы весь набор тестов. Графическое окно не создаётся.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

from pathlib import Path
import subprocess
import sys
import unittest


class FinplotCompatibilityTests(unittest.TestCase):
    """Ловит жёсткое завершение импорта и утечку временной функции локали."""

    def test_import_survives_russian_locale_and_restores_locale_function(self):
        """Проверяет реальный импорт finplot в дочернем процессе с русской локалью."""
        code = '''
import locale
def russian_locale(*args, **kwargs):
    """Возвращает тестовую русскую локаль, игнорируя аргументы интерфейса locale."""
    return ('ru_RU', 'cp1251')
locale.getdefaultlocale = russian_locale
from source.finplot_compat import load_finplot
module = load_finplot()
assert hasattr(module, 'candlestick_ochl')
assert locale.getdefaultlocale is russian_locale
print('ok')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).resolve().parents[1],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'ok')


if __name__ == '__main__':
    unittest.main()
