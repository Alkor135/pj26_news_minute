r"""Проверяет весь локальный результат расчёта аналогов на временные утечки.

Если data/analog_analysis/analogs.sqlite3 ещё не создана, тест пропускается.
Проверяет все пороги/соседей/решения и целостность SQLite, исходные данные не меняет.
Также проверяет готовые data/russian_market/*/analogs и совместимость групп событий.
Для годового режима проверяет собственную нижнюю границу каждого запроса и query_count.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_analog_output.py -v
"""

from contextlib import closing
import json
import unittest

from source.build_analogs import DEFAULT_OUTPUT, make_decision, readonly
from source.embedding_pilot import ROOT
from source.market_relevance import compatible_groups
from source.analog_selection import history_start


@unittest.skipUnless((DEFAULT_OUTPUT / 'analogs.sqlite3').is_file(), 'Сначала выполните build_analogs.py')
class AnalogOutputTests(unittest.TestCase):
    """Проверяет реальные сохранённые оценки, включая все причины отказа."""

    def test_all_thresholds_neighbors_and_decisions(self):
        """Ловит повторные/будущие даты и расхождение решения с сохранёнными аналогами."""
        paths = [DEFAULT_OUTPUT / 'analogs.sqlite3'] + list((ROOT / 'data' / 'russian_market').glob('*/analogs/analogs.sqlite3'))
        for path in paths:
            with self.subTest(database=str(path)):
                self.audit_database(path)

    def audit_database(self, path):
        """Проверяет все решения в read-only SQLite path, включая тематическую совместимость."""
        with closing(readonly(path)) as c:
            self.assertEqual(c.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
            config = json.loads(c.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
            news = {key: json.loads(payload) for key, payload in c.execute('SELECT news_id,payload FROM news')}
            for key, instrument, horizon, payload in c.execute('SELECT * FROM thresholds'):
                t = json.loads(payload)
                if t['status'] == 'ok':
                    self.assertLess(t['history_max_end'][:10], news[key]['loaded_at'][:10])
                    self.assertGreaterEqual(t['sample_count'], config['threshold_options']['min_samples'])
                    self.assertGreaterEqual(t['day_count'], config['threshold_options']['min_days'])
            count, selected = 0, 0
            for (payload,) in c.execute('SELECT payload FROM scores'):
                s = json.loads(payload)
                if config.get('query_start'):
                    self.assertGreaterEqual(s['loaded_at'], config['query_start'])
                if config.get('query_end_exclusive'):
                    self.assertLess(s['loaded_at'], config['query_end_exclusive'])
                days = [n['loaded_at'][:10] for n in s['neighbors']]
                self.assertEqual(len(days), len(set(days)))
                for n in s['neighbors']:
                    self.assertLess(n['loaded_at'][:10], s['loaded_at'][:10])
                    self.assertLess(n['end_at'], s['loaded_at'])
                    if config.get('history_years'):
                        self.assertGreaterEqual(n['loaded_at'], history_start(s['loaded_at'], config['history_years']))
                    self.assertLess(n['threshold_history_max_end'][:10], n['loaded_at'][:10])
                    if s.get('relevance'):
                        self.assertTrue(compatible_groups(s['relevance'], n.get('relevance')))
                decision = make_decision(s['neighbors'], s['loaded_at'], s['threshold'], config['min_similarity'], s.get('relevance'), config.get('history_years'))
                for field in ('selected', 'status', 'forecast_direction', 'median_ratio'):
                    self.assertEqual(s[field], decision[field])
                count += 1
                selected += s['selected']
            self.assertEqual(count, config.get('query_count', len(news)) * 2 * 5 * len(config['models']))
            self.assertEqual(selected, sum(g['selected'] for g in config['groups']))


if __name__ == '__main__':
    unittest.main()
