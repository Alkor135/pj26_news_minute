r"""Проверяет годовой поиск, идентификаторы событий и отдельный месяц проверки.

Синтетические SQLite размещаются только в tests. Будущие/старше года аналоги
исключены, пять дат не обязаны идти подряд, одинаковые тексты разных дней сохранены.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_year_history.py -v
"""

from contextlib import closing
from pathlib import Path
import sqlite3
import tempfile
import unittest
import json

from source.analog_selection import pick_neighbors


class YearHistoryTests(unittest.TestCase):
    """Ловит потерю повторных заголовков и смешивание истории с месяцем оценки."""

    def test_search_uses_calendar_year_and_arbitrary_distinct_dates(self):
        """Граничный случай ровно год допускается, более старый лучший косинус исключён."""
        days = ['2025-09-30', '2025-12-03', '2026-03-14', '2026-07-21', '2026-09-01']
        rows = [dict(news_id=str(i), loaded_at=d + ' 12:00:00', end_at=d + ' 12:30:01',
                     status='ok', threshold_status='ok', threshold_frac=.01, similarity=.9 - i / 100)
                for i, d in enumerate(days)]
        rows.append(dict(rows[0], news_id='too-old', loaded_at='2025-09-30 11:59:59', similarity=1.))
        found = pick_neighbors(rows, '2026-09-30 12:00:00', history_years=1)
        self.assertEqual({r['loaded_at'][:10] for r in found}, set(days))
        self.assertNotIn('too-old', [r['news_id'] for r in found])
        self.assertEqual(len(pick_neighbors(rows, '2026-10-01 12:00:00', history_years=1)), 4)

    def test_leap_day_year_start(self):
        """29 февраля предыдущего невисокосного года заменяется на 28 февраля."""
        from source.analog_selection import history_start
        self.assertEqual(history_start('2024-02-29 12:00:00', 1), '2023-02-28 12:00:00')

    def test_history_loader_keeps_same_title_on_different_days(self):
        """Одинаковые заголовки объединяются внутри даты, но не теряются за год."""
        from source.news_history import load_history
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            with closing(sqlite3.connect(root / 'rss_news_2026_09.db')) as c:
                c.execute('CREATE TABLE news(loaded_at TEXT,date TEXT,title TEXT,provider TEXT)')
                for day, hour in [('01', '12'), ('01', '13'), ('03', '12')]:
                    stamp = f'2026-09-{day} {hour}:00:00'
                    c.execute('INSERT INTO news VALUES(?,?,?,?)', (stamp, stamp, 'Одинаковый заголовок', 'test'))
                c.commit()
            rows, stats = load_history(root, '2026-09-01 00:00:00', '2026-10-01 00:00:00')
            self.assertEqual(len(rows), 2)
            self.assertNotEqual(rows[0]['news_id'], rows[1]['news_id'])
            self.assertEqual(rows[0]['title_key'], rows[1]['title_key'])
            self.assertEqual(rows[0]['duplicate_count'], 2)
            self.assertEqual(stats['source_rows'], 3)

    def test_query_period_is_separate_from_available_history(self):
        """Исторические новости становятся кандидатами, но не запросами сентября."""
        from source.build_analogs import query_indices
        news = [{'loaded_at': date} for date in ['2025-09-01 12:00:00', '2026-09-01 12:00:00', '2026-10-01 00:00:00']]
        self.assertEqual(query_indices(news, '2026-09-01', '2026-10-01'), [1])

    def test_vector_reuse_requires_exact_text_and_preserves_model_version(self):
        """Новые события переиспользуют вектор точного текста, но не другого регистра."""
        from source.news_history import seed_title_vectors
        from source.embedding_pilot import open_cache, save_vectors, cached_vectors
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            with closing(open_cache(Path(folder) / 'pilot.sqlite3')) as c:
                c.execute('INSERT INTO news VALUES(?,?)', ('old', json.dumps({'title': 'Текст'})))
                save_vectors(c, 'digest-a', [('old', [1., 0.])])
                save_vectors(c, 'digest-b', [('old', [0., 1.])])
                count = seed_title_vectors(c, [{'news_id': 'day-1', 'title': 'Текст'},
                                              {'news_id': 'day-2', 'title': 'Текст'},
                                              {'news_id': 'case', 'title': 'текст'}])
                self.assertEqual(count, 4)
                self.assertEqual(list(cached_vectors(c, 'digest-a')['day-1']), [1., 0.])
                self.assertEqual(list(cached_vectors(c, 'digest-b')['day-2']), [0., 1.])
                self.assertNotIn('case', cached_vectors(c, 'digest-a'))


if __name__ == '__main__':
    unittest.main()
