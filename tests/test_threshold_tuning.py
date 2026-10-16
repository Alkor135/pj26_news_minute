r"""Проверяет изменение строгости порога без замены аналогов и подглядывания.

Синтетические оценки не требуют котировок или Ollama. Один и тот же набор соседей
переоценивается с k, а общий контроль результата сохраняет базовый порог k=1.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_threshold_tuning.py -v
"""

from copy import deepcopy
import unittest
from contextlib import closing
from pathlib import Path
import csv
import json
import sqlite3
import tempfile

from source.threshold_tuning import rescale_score


class ThresholdTuningTests(unittest.TestCase):
    """Проверяет масштабирование, неизменность соседей и независимость прогноза."""

    def score(self):
        """Возвращает пример пяти согласных, но недостаточно сильных аналогов."""
        return dict(selected=False, status='weak_median_move', forecast_direction=None,
            neighbors=[dict(news_id=str(i), loaded_at=f'2026-09-0{i} 12:00:00', similarity=.9,
                            points=8., return_frac=.008, threshold_frac=.01, threshold_points=10.) for i in range(1, 6)],
            threshold=dict(status='ok', threshold_frac=.01, threshold_points=10.),
            actual_status='ok', actual_direction='up', actual_ratio=.7, actual_points=7.)

    def test_multiplier_changes_threshold_but_not_neighbors(self):
        """k=0,5 допускает слабую медиану; пороги и отношения согласованно меняются."""
        original = self.score()
        changed = rescale_score(original, .5)
        self.assertFalse(rescale_score(original, 1.)['selected'])
        self.assertTrue(changed['selected'])
        self.assertEqual(changed['forecast_direction'], 'up')
        self.assertAlmostEqual(changed['median_ratio'], 1.6)
        self.assertAlmostEqual(changed['threshold']['threshold_points'], 5.)
        self.assertAlmostEqual(changed['actual_ratio'], 1.4)
        self.assertTrue(changed['strong_correct'])
        self.assertFalse(changed['base_strong_correct'])
        self.assertEqual([n['news_id'] for n in changed['neighbors']], [n['news_id'] for n in original['neighbors']])
        self.assertEqual(original['threshold']['threshold_points'], 10.)

    def test_repeated_scaling_always_uses_base_not_previous_value(self):
        """Переход 1→0,5→0,25→1 не накапливает множители."""
        original = self.score()
        half = rescale_score(original, .5)
        quarter = rescale_score(half, .25)
        self.assertAlmostEqual(quarter['threshold']['threshold_points'], 2.5)
        restored = rescale_score(quarter, 1.)
        self.assertAlmostEqual(restored['median_ratio'], .8)
        self.assertFalse(restored['selected'])
        self.assertAlmostEqual(restored['actual_ratio'], .7)

    def test_lower_threshold_does_not_relax_sign_or_query_history(self):
        """Малый k не отменяет пять совпадающих знаков и достаточную историю порога."""
        score = self.score()
        score['neighbors'][0]['points'] = -8.
        score['neighbors'][0]['return_frac'] = -.008
        self.assertEqual(rescale_score(score, .01)['status'], 'mixed_directions')
        score = self.score()
        score['threshold']['status'] = 'insufficient_pre_history'
        score['threshold']['threshold_frac'] = score['threshold']['threshold_points'] = None
        self.assertEqual(rescale_score(score, .01)['status'], 'query_threshold_unavailable')

    def test_current_outcome_does_not_change_selection_and_invalid_k_rejected(self):
        """Смена текущего исхода меняет только оценку; неположительный/NaN k запрещён."""
        first = rescale_score(self.score(), .5)
        changed = deepcopy(self.score())
        changed.update(actual_status='no_entry', actual_direction=None, actual_ratio=None, actual_points=None)
        second = rescale_score(changed, .5)
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(first['forecast_direction'], second['forecast_direction'])
        self.assertIsNone(second['strong_correct'])
        for value in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                rescale_score(self.score(), value)

    def test_grid_keeps_input_database_and_exports_all_trials_with_stable_columns(self):
        """Сравнение сохраняет все k, не меняет SQLite и создаёт полную схему пустого CSV."""
        from source.tune_threshold import run
        from source.build_analogs import digest_file
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            database = root / 'analogs.sqlite3'
            score = dict(self.score(), news_id='q', loaded_at='2026-09-08 12:00:00', model='model', instrument='MIX', horizon=30)
            with closing(sqlite3.connect(database)) as c:
                c.executescript('CREATE TABLE metadata(key TEXT,value TEXT); CREATE TABLE scores(payload TEXT); CREATE TABLE news(news_id TEXT,payload TEXT);')
                c.execute('INSERT INTO metadata VALUES(?,?)', ('summary', json.dumps({'min_similarity': .5})))
                c.execute('INSERT INTO scores VALUES(?)', (json.dumps(score),))
                c.execute('INSERT INTO news VALUES(?,?)', ('q', json.dumps({'title': 'Пример'})))
                c.commit()
            before = digest_file(database)
            totals = run(database, root / 'grid', [.5, 1., 2.])
            self.assertEqual([r['selected'] for r in totals], [1, 0, 0])
            self.assertEqual(digest_file(database), before)
            run(database, root / 'empty', [2.])
            with (root / 'empty' / 'selected.csv').open(encoding='utf-8-sig', newline='') as handle:
                reader = csv.DictReader(handle)
                self.assertIn('actual_status', reader.fieldnames)
                self.assertEqual(list(reader), [])


if __name__ == '__main__':
    unittest.main()
