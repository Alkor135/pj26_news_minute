r"""Проверяет реальное построение finplot и переключение событий без рабочего стола.

Создаёт SQLite и RSS только во временной папке tests, запускает Qt offscreen,
проверяет свечи, маркеры соседних новостей и освобождение старого графика.
Проверяет категории/объяснения российского рынка и прогноз без будущих котировок.
Проверяет живое изменение k, пороги соседей и обратимый переход к прежнему отбору.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_finplot_smoke.py -v
"""

from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pandas as pd
from PyQt6 import QtWidgets

from source.build_reactions import create_store
from source.view_news_finplot import NewsWindow, configure_chart, fplt


class FinplotSmokeTests(unittest.TestCase):
    """Проверяет интеграцию реальных Qt, pandas и finplot на небольших данных."""

    def test_analog_mode_keeps_selected_query_without_future_quotes(self):
        """Показывает отобранную новость даже при no_entry и не применяет stable."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            database, analogs = root / 'reactions.sqlite3', root / 'analogs.sqlite3'
            record = dict(news_id='a', loaded_at='2026-09-08 12:00:00', title='Без будущих котировок',
                          instrument='MIX', horizon=30, status='no_entry', stable=False, points=None)
            relevance = dict(categories=['monetary'], groups=['monetary:inflation:RU:reported'], reason='Инфляция России', rule_version='test')
            with closing(create_store(database)) as c:
                c.execute('INSERT INTO reactions VALUES(?,?,?,?,?,?,?)', ('a', 'MIX', 30, 'no_entry', 0, None, json.dumps(record)))
                c.execute('INSERT INTO news VALUES(?,?,?,?,?)', ('a', record['loaded_at'], record['title'], '', json.dumps({'relevance': relevance})))
                c.commit()
            score = dict(selected=True, status='selected', forecast_direction='up', median_ratio=2.,
                         min_similarity=.8, actual_status='no_entry', actual_ratio=None,
                         neighbors=[dict(news_id=str(i), loaded_at=f'2026-09-0{i} 12:00:00', similarity=.8,
                                         points=8., return_frac=.008, threshold_frac=.01, threshold_points=10.) for i in range(1, 6)],
                         threshold=dict(status='ok', threshold_frac=.01, threshold_points=10.))
            with closing(sqlite3.connect(analogs)) as c:
                c.execute('CREATE TABLE scores(news_id TEXT,instrument TEXT,horizon INTEGER,model TEXT,payload TEXT)')
                c.execute('INSERT INTO scores VALUES(?,?,?,?,?)', ('a', 'MIX', 30, 'model', json.dumps(score)))
                c.execute('CREATE TABLE metadata(key TEXT,value TEXT)')
                c.execute('INSERT INTO metadata VALUES(?,?)', ('summary', json.dumps(dict(history_years=1,
                    query_start='2026-09-01', query_end_exclusive='2026-10-01'))))
                c.commit()
            window = NewsWindow(database, root, analog_database=analogs, selection_mode='all')
            try:
                app.processEvents()
                window.mode.setCurrentIndex(window.mode.findData('analogs'))
                self.assertEqual(window.table.rowCount(), 0)
                window.threshold_multiplier.setValue(.5)
                self.assertEqual(window.table.rowCount(), 1)
                self.assertEqual(window.neighbors_table.rowCount(), 5)
                self.assertEqual(window.neighbors_table.item(0, 4).text(), '5')
                window.threshold_multiplier.setValue(.4)
                self.assertEqual(window.current_record['forecast']['threshold_multiplier'], .4)
                self.assertEqual(window.neighbors_table.item(0, 4).text(), '4')
                self.assertEqual(window.current_record['status'], 'no_entry')
                self.assertIn('вверх', window.stats.text())
                self.assertIn('Инфляция России', window.stats.text())
                self.assertIn('Поиск аналогов: 1 год', window.notice.text())
                window.category.setCurrentIndex(window.category.findData('oil'))
                self.assertEqual(window.table.rowCount(), 0)
                window.category.setCurrentIndex(window.category.findData('monetary'))
                self.assertEqual(window.table.rowCount(), 1)
                window.threshold_multiplier.setValue(1.)
                self.assertEqual(window.table.rowCount(), 0)
                window.threshold_multiplier.setValue(.5)
                self.assertEqual(window.table.rowCount(), 1)
            finally:
                window.clear_chart()
                window.close()
                app.processEvents()

    def test_draw_news_markers_and_switch_instruments(self):
        """Ловит несовместимость временных индексов и утечку графиков при переключении."""
        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        configure_chart()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            database = root / 'reactions.sqlite3'
            with closing(create_store(database)) as c:
                for instrument in ('MIX', 'RTS'):
                    for i, ts in enumerate(pd.date_range('2026-09-01 11:00:00', periods=181, freq='min')):
                        price = 100 + i
                        c.execute('INSERT INTO minute_bars VALUES(?,?,?,?,?,?,?,?)',
                                  (instrument, str(ts), price, price + 2, price - 1, price + 1, 10, 3))
                    r = {'news_id': 'a', 'title': 'Контрольная новость', 'provider': 'prime',
                         'instrument': instrument, 'horizon': 30, 'loaded_at': '2026-09-01 12:00:00',
                         'entry_at': '2026-09-01 12:00:01', 'entry_price': 160,
                         'end_at': '2026-09-01 12:30:01', 'exit_at': '2026-09-01 12:30:00',
                         'exit_price': 190, 'points': 30, 'status': 'ok', 'stable': True}
                    c.execute('INSERT INTO reactions VALUES(?,?,?,?,?,?,?)',
                              ('a', instrument, 30, 'ok', 1, 30, json.dumps(r)))
                c.commit()
            with closing(sqlite3.connect(root / 'rss_news_2026_09.db')) as c:
                c.execute('CREATE TABLE news(loaded_at TEXT,date TEXT,title TEXT,provider TEXT)')
                c.execute('INSERT INTO news VALUES(?,?,?,?)',
                          ('2026-09-01 12:05:00', '2026-09-01 12:04:00', 'Соседняя новость', 'prime'))
                c.execute('INSERT INTO news VALUES(?,?,?,?)',
                          ('2026-09-01 09:05:00', '2026-09-01 09:04:00', 'Соседняя новость', 'prime'))
                c.commit()
            window = NewsWindow(database, root)
            try:
                window.show()
                window.select_event()
                app.processEvents()
                self.assertIsNotNone(window.canvas)
                self.assertEqual(len(window.canvas.axs), 2)
                lines = [item for item in window.canvas.axs[0].items if isinstance(item, fplt.FinLine)]
                self.assertAlmostEqual(lines[0].points[0][0], 60., places=3)
                self.assertIn('Соседняя новость', window.context.toPlainText())
                window.instrument.setCurrentText('RTS')
                window.select_event()
                app.processEvents()
                self.assertEqual(window.current_record['instrument'], 'RTS')
                self.assertEqual(len(fplt.windows), 1)
                self.assertFalse(window.grab().isNull())
            finally:
                window.clear_chart()
                window.close()
                app.processEvents()


if __name__ == '__main__':
    unittest.main()
