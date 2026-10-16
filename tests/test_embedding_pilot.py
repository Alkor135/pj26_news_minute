r"""Проверки корректности пилотного поиска аналогов новостей.

Проверяются чтение SQLite, дедупликация, векторы, сброс кэша при изменении
текста, временной фильтр, экранирование HTML и экспорт ручной разметки в CSV.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
Временные БД создаются только в папке tests, обращения к Ollama отсутствуют.
"""

import importlib
from contextlib import closing
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest


class PilotTests(unittest.TestCase):
    """Проверяет данные и поиск на небольших независимо заданных примерах."""

    def module(self):
        """Возвращает проверяемый модуль; отсутствие реализации даёт провал теста."""
        try:
            return importlib.import_module('source.embedding_pilot')
        except ModuleNotFoundError:
            self.fail('Модуль embedding_pilot ещё не реализован')

    def test_dedup_keeps_first_loading_and_counts_repeats(self):
        """Ловит выбор поздней загрузки и потерю дублей при нормализации пробелов."""
        m = self.module()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            db = Path(folder) / 'rss_news_2026_09.db'
            with closing(sqlite3.connect(db)) as c:
                c.execute('CREATE TABLE news(loaded_at TEXT,date TEXT,title TEXT,provider TEXT)')
                c.executemany('INSERT INTO news VALUES(?,?,?,?)', [
                    ('2026-09-02 12:00:00', '2026-09-02 11:59:00', 'ЦБ  повысил ставку', 'prime'),
                    ('2026-09-02 11:00:00', '2026-09-02 10:59:00', 'ЦБ повысил ставку', 'interfax'),
                    ('2026-10-01 11:00:00', '2026-09-02 10:59:00', 'Другая новость', 'prime'),
                    ('2026-09-03 11:00:00', '2026-09-03 10:59:00', '   ', 'prime'),
                ])
                c.commit()
            before = db.read_bytes()
            rows, stats = m.load_news(Path(folder), '2026-09')
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['loaded_at'], '2026-09-02 11:00:00')
            self.assertEqual(rows[0]['provider'], 'interfax')
            self.assertEqual(rows[0]['duplicate_count'], 2)
            self.assertEqual(stats['empty_titles'], 1)
            self.assertEqual(db.read_bytes(), before)

    def test_vector_normalization_and_invalid_values(self):
        """Ловит ненормированные, пустые, нулевые, бесконечные и разные векторы."""
        m = self.module()
        vector = m.normalize_vector([3, 4], expected_dim=2)
        self.assertAlmostEqual(vector[0], .6, places=6)
        self.assertAlmostEqual(vector[1], .8, places=6)
        for value in ([], [0, 0], [math.nan, 1], [math.inf, 1], [1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                m.normalize_vector(value, expected_dim=2)

    def test_neighbors_exclude_self_and_future(self):
        """Ловит самоаналог, нарушение временного зазора и обратную сортировку."""
        m = self.module()
        rows = [
            {'news_id': 'a', 'loaded_at': '2026-09-01 10:00:00'},
            {'news_id': 'b', 'loaded_at': '2026-09-01 09:00:00'},
            {'news_id': 'c', 'loaded_at': '2026-09-01 12:00:00'},
            {'news_id': 'd', 'loaded_at': '2026-09-01 14:00:00'},
            {'news_id': 'e', 'loaded_at': '2026-09-01 12:00:01'},
        ]
        vectors = [[.8, .6], [0, 1], [1, 0], [1, 0], [1, 0]]
        found = m.nearest(rows, vectors, 3, 5, past_only=True, embargo_minutes=120)
        self.assertEqual([x[0] for x in found], [2, 0, 1])
        self.assertAlmostEqual(found[1][1], .8)

    def test_cache_separates_model_versions(self):
        """Ловит смешивание одинаковых news_id при разных digest моделей."""
        m = self.module()
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with closing(m.open_cache(Path(folder) / 'pilot.db')) as c:
                m.save_vectors(c, 'digest-a', [('news-1', [1., 0.])])
                m.save_vectors(c, 'digest-b', [('news-1', [0., 1.])])
                self.assertEqual(list(m.cached_vectors(c, 'digest-a')['news-1']), [1., 0.])
                self.assertEqual(list(m.cached_vectors(c, 'digest-b')['news-1']), [0., 1.])
                self.assertEqual(m.cached_vectors(c, 'unknown'), {})

    def test_report_escapes_titles_and_keeps_review_fields(self):
        """Ловит исполнение разметки из заголовка и потерю строк ручной оценки."""
        import csv
        try:
            report = importlib.import_module('source.embedding_report')
        except ModuleNotFoundError:
            self.fail('Модуль embedding_report ещё не реализован')
        rows = [
            {'news_id': 'a', 'title': '<script>alert(1)</script>', 'loaded_at': '2026-09-01 10:00:00',
             'provider': 'prime', 'published_at': '', 'duplicate_count': 1},
            {'news_id': 'b', 'title': 'Аналог', 'loaded_at': '2026-09-01 09:00:00',
             'provider': 'prime', 'published_at': '', 'duplicate_count': 1},
        ]
        summary = {'month': '2026-09', 'sample_size': 2, 'seed': 42,
                   'corpus_id': 'test', 'query_ids': ['a'], 'models': {},
                   'source_stats': {'source_rows': 2, 'unique_titles': 2}, 'embargo_minutes': 120}
        neighbors = [{'model': 'test', 'mode': 'all', 'query_id': 'a', 'neighbor_id': 'b',
                      'rank': 1, 'similarity': .5}]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            report.write_report(Path(folder), rows, neighbors, summary)
            html = (Path(folder) / 'report.html').read_text(encoding='utf-8')
            self.assertNotIn('<script>alert(1)</script>', html)
            self.assertIn('&lt;script&gt;', html)
            with (Path(folder) / 'review.csv').open(encoding='utf-8-sig', newline='') as f:
                exported = list(csv.DictReader(f))
            self.assertEqual(exported[0]['neighbor_title'], 'Аналог')
            self.assertEqual(exported[0]['relevance_0_1_2'], '')

    def test_changed_input_invalidates_cached_vectors(self):
        """Ловит повторное использование вектора, если текст того же news_id изменился."""
        m = self.module()
        self.assertTrue(hasattr(m, 'save_news'), 'Нужна проверка изменения входного текста кэша')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with closing(m.open_cache(Path(folder) / 'pilot.db')) as c:
                m.save_news(c, [{'news_id': 'a', 'title': 'ЦБ повысил ставку'}], 'sample-1')
                m.save_vectors(c, 'digest', [('a', [1., 0.])])
                m.save_news(c, [{'news_id': 'a', 'title': 'ЦБ повысил ставку', 'provider': 'prime'}], 'sample-2')
                self.assertIn('a', m.cached_vectors(c, 'digest'))
                m.save_news(c, [{'news_id': 'a', 'title': 'цб повысил ставку'}], 'sample-3')
                self.assertEqual(m.cached_vectors(c, 'digest'), {})


if __name__ == '__main__':
    unittest.main()
