r"""Проверяет согласованность входов и независимость прогноза от исхода запроса.

Использует SQLite в памяти и синтетические векторы, без Ollama и реальных архивов.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_analog_pipeline.py -v
"""

from copy import deepcopy
from contextlib import closing
import json
import sqlite3
import unittest

from source.build_analogs import make_decision, model_manifest
from source.embedding_pilot import PREPARATION, save_vectors


class AnalogPipelineTests(unittest.TestCase):
    """Ловит подмену версии/текста и использование будущего результата при отборе."""

    def test_manifest_rejects_changed_text_missing_model_and_ignores_old_digest(self):
        """Требует актуальную версию manifest и точное совпадение текста эмбеддинга."""
        with closing(sqlite3.connect(':memory:')) as c:
            c.executescript('''CREATE TABLE news(news_id TEXT,payload TEXT);
                CREATE TABLE samples(corpus_id TEXT,news_id TEXT);
                CREATE TABLE models(digest TEXT,name TEXT);
                CREATE TABLE embeddings(news_id TEXT,digest TEXT,preparation TEXT,dimension INTEGER,vector BLOB);''')
            c.execute('INSERT INTO news VALUES(?,?)', ('a', json.dumps({'title': 'Текст'})))
            c.execute('INSERT INTO samples VALUES(?,?)', ('corpus', 'a'))
            c.executemany('INSERT INTO models VALUES(?,?)', [('old', 'model'), ('new', 'model')])
            save_vectors(c, 'new', [('a', [1., 0.])])
            manifest = dict(corpus_id='corpus', preparation=PREPARATION,
                            models={'model': {'current_run': {'digest': 'new'}}})
            news = [{'news_id': 'a', 'title': 'Текст'}]
            self.assertEqual(model_manifest(c, manifest, news, 'corpus', ['model']), [('new', 'model')])
            with self.assertRaisesRegex(ValueError, 'нет модели'):
                model_manifest(c, manifest, news, 'corpus', ['model', 'absent'])
            with self.assertRaisesRegex(ValueError, 'Тексты'):
                model_manifest(c, manifest, [{'news_id': 'a', 'title': 'текст'}], 'corpus', ['model'])

    def test_current_outcome_changes_do_not_change_decision(self):
        """Подмена исхода запроса на минус/нет данных не меняет отбор и направление."""
        past = [dict(news_id=str(i), loaded_at=f'2026-09-0{i} 12:00:00',
                     end_at=f'2026-09-0{i} 12:30:01', status='ok', points=10,
                     return_frac=.02, threshold_frac=.01, threshold_status='ok', similarity=.9)
                for i in range(1, 6)]
        query = dict(news_id='q', loaded_at='2026-09-08 12:00:00', end_at='2026-09-08 12:30:01',
                     status='ok', points=10, return_frac=.02, threshold_status='ok', threshold_frac=.01, similarity=1.)
        threshold = {'status': 'ok', 'threshold_frac': .01}
        decisions = []
        for status, points in [('ok', 10), ('ok', -10), ('no_entry', None)]:
            changed = deepcopy(query)
            changed.update(status=status, points=points, return_frac=None if points is None else points / 500.)
            decisions.append(make_decision(past + [changed], query['loaded_at'], threshold, .5))
        self.assertTrue(decisions[0]['selected'])
        self.assertEqual(decisions[0], decisions[1])
        self.assertEqual(decisions[0], decisions[2])


if __name__ == '__main__':
    unittest.main()
