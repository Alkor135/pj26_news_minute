r"""Проверяет согласованную шкалу и самостоятельную сборку файла разметки.

Создаёт временную SQLite только внутри tests, запускает build_rasmetka.py
с готовыми реакциями и проверяет совместимость вывода со скриптом дообучения.
Запуск из корня: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_rasmetka.py -v
"""

import json
import importlib.util
from contextlib import closing
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta

from source.build_reactions import create_store
from source.market_relevance import RULE_VERSION
from source.price_reactions import METRIC_VERSION

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'experiments_-10_10' / 'build_rasmetka.py'
SPEC = importlib.util.spec_from_file_location('build_rasmetka', SCRIPT)
build_rasmetka = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_rasmetka)


def reaction(**changes):
    """Возвращает полную искусственную реакцию; changes задаёт условия теста."""
    row = dict(news_id='news-1', instrument='MIX', horizon=5,
               loaded_at='2026-07-15 12:00:00', end_at='2026-07-15 12:05:01',
               title='Банк России повысил ключевую ставку', provider='test',
               status='ok', stable=False, points=100.0, return_pct=1.0,
               z_score=1.0, er=0.45, direction_fraction=0.7,
               mae_points=0.0, max_pullback_points=0.0, flags=[]) | changes
    row.setdefault('direction', 'up' if row['points'] and row['points'] > 0 else
                   'down' if row['points'] and row['points'] < 0 else 'flat')
    return row


class ScoreTests(unittest.TestCase):
    """Проверяет пороги, знак и отсутствие метки при неполных исходных данных."""

    def setUp(self):
        """Импортирует модуль после появления файла скрипта."""
        self.module = build_rasmetka

    def test_moderate_and_unstable_movement(self):
        """Умеренное ровное движение даёт ±5, сильное с откатом снижается."""
        self.assertEqual(self.module.score_reaction(reaction())[0], 5)
        self.assertEqual(self.module.score_reaction(reaction(points=-100.0))[0], -5)
        score, _, strength, quality = self.module.score_reaction(
            reaction(z_score=3.0, max_pullback_points=125.0))
        self.assertEqual(score, 4)
        self.assertEqual(strength, 1.0)
        self.assertEqual(quality, 0.4)

    def test_stable_extremes_and_cap_for_unstable(self):
        """Только исходный stable разрешает ±10, остальные ограничены ±9."""
        steady = reaction(z_score=2.0, stable=True)
        self.assertEqual(self.module.score_reaction(steady)[0], 10)
        self.assertEqual(self.module.score_reaction(steady | {'points': -100.0, 'direction': 'down'})[0], -10)
        almost = reaction(z_score=3.0, mae_points=0.0, max_pullback_points=0.0,
                          direction_fraction=0.69)
        self.assertEqual(self.module.score_reaction(almost)[0], 9)

    def test_zero_and_unlabeled_differ(self):
        """Малый z означает ноль; неполное окно или отсутствие sigma — пустую метку."""
        self.assertEqual(self.module.score_reaction(reaction(points=0.0, z_score=0.0))[0], 0)
        self.assertEqual(self.module.score_reaction(reaction(z_score=0.25))[0], 0)
        self.assertEqual(self.module.score_reaction(reaction(z_score=0.251))[0], 1)
        self.assertIsNone(self.module.score_reaction(reaction(status='gap', z_score=None))[0])
        self.assertIsNone(self.module.score_reaction(reaction(z_score=None))[0])
        self.assertIsNone(self.module.score_reaction(reaction(er=None))[0])

    def test_inconsistent_stable_is_error(self):
        """Повреждённый флаг stable не может присвоить крайний балл."""
        with self.assertRaises(ValueError):
            self.module.score_reaction(reaction(stable=True, z_score=1.0))
        with self.assertRaises(ValueError):
            self.module.score_reaction(reaction(stable=True, z_score=2.0, direction='flat'))


class BuildTests(unittest.TestCase):
    """Проверяет чтение SQLite и пригодность JSONL для обучения."""

    def setUp(self):
        """Создаёт временную БД реакций и каталог вывода внутри tests."""
        self.tmp = tempfile.TemporaryDirectory(prefix='tmp_rasmetka_', dir=ROOT / 'tests')
        self.addCleanup(self.tmp.cleanup)
        self.work = Path(self.tmp.name)
        self.database = self.work / 'reactions.sqlite3'
        with closing(create_store(self.database)) as connection:
            summary = dict(config=dict(metric_version=METRIC_VERSION,
                                       relevance_version=RULE_VERSION,
                                       horizons=[5, 15, 30, 60, 120],
                                       neutral_points={'MIX': 50.0, 'RTS': 20.0},
                                       latency_seconds=0.0),
                           news_count=2, reaction_count=3)
            connection.execute('INSERT INTO metadata VALUES (?,?)',
                               ('summary', json.dumps(summary, ensure_ascii=False)))
            for key, when, title in (
                ('news-1', '2026-07-15 12:00:00', 'Банк России повысил ключевую ставку'),
                ('news-2', '2026-09-15 12:00:00', 'Банк России понизил ключевую ставку')):
                connection.execute('INSERT INTO news VALUES (?,?,?,?,?)',
                                   (key, when, title, 'test', json.dumps({'news_id': key})))
            examples = [
                reaction(),
                reaction(instrument='RTS', status='gap', points=None, end_at=None, z_score=None),
                reaction(news_id='news-2', loaded_at='2026-09-15 12:00:00',
                         end_at='2026-09-15 12:05:01',
                         title='Банк России понизил ключевую ставку'),
            ]
            for row in examples:
                connection.execute('INSERT INTO reactions VALUES (?,?,?,?,?,?,?)',
                                   (row['news_id'], row['instrument'], row['horizon'],
                                    row['status'], int(row['stable']), row['points'],
                                    json.dumps(row, ensure_ascii=False)))
            connection.commit()

    def test_cli_exports_year_and_missing_label(self):
        """Сентябрь не попадает в год обучения, а неполный RTS остаётся с score=null."""
        output = self.work / 'scores.jsonl'
        result = subprocess.run([sys.executable, str(SCRIPT),
                                 '--reaction-db', str(self.database), '--output', str(output)],
                                text=True, capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 2)
        self.assertEqual({r['instrument']: r['score'] for r in rows}, {'MIX': 5, 'RTS': None})
        self.assertEqual({r['score_version'] for r in rows}, {'magnitude-quality-v1'})
        self.assertEqual(len({r['group_id'] for r in rows}), 1)
        report = json.loads((self.work / 'scores_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(report['scores']['5'], 1)
        self.assertEqual(report['missing_labels']['gap'], 1)

    def test_rejects_wrong_metric_version(self):
        """Не размечает БД с другой версией определения ценовой реакции."""
        with closing(sqlite3.connect(self.database)) as connection:
            summary = json.loads(connection.execute(
                "SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
            summary['config']['metric_version'] = 'старое правило'
            connection.execute('UPDATE metadata SET value=? WHERE key=?',
                               (json.dumps(summary, ensure_ascii=False), 'summary'))
            connection.commit()
        result = subprocess.run([sys.executable, str(SCRIPT),
                                 '--reaction-db', str(self.database),
                                 '--output', str(self.work / 'scores.jsonl')],
                                text=True, capture_output=True, encoding='utf-8')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.work / 'scores.jsonl').exists())


class RawSourceTests(unittest.TestCase):
    """Проверяет путь от исходных RSS и ZIP котировок до готового JSONL."""

    def test_cli_builds_reactions_from_source_files(self):
        """Разметчик сам фильтрует новость, считает реакцию и сохраняет десять строк."""
        with tempfile.TemporaryDirectory(prefix='tmp_rasmetka_raw_', dir=ROOT / 'tests') as folder:
            work = Path(folder)
            news, mix, rts = (work / name for name in ('news', 'mix', 'rts'))
            for path in (news, mix, rts):
                path.mkdir()
            with closing(sqlite3.connect(news / 'rss_news_202607.db')) as connection:
                connection.execute('CREATE TABLE news (loaded_at TEXT,date TEXT,title TEXT,provider TEXT)')
                connection.execute('INSERT INTO news VALUES (?,?,?,?)',
                                   ('2026-07-15 12:00:00', '2026-07-15',
                                    'Банк России повысил ключевую ставку', 'test'))
                connection.commit()
            origin = datetime(2026, 7, 15, 10, 59, 30)
            ticks = ['datetime,last,volume']
            for minute in range(71):
                stamp = origin + timedelta(minutes=minute)
                price = 100 + minute % 2 if minute < 61 else 100 + (minute - 61) * 5
                ticks.append(f'{stamp.isoformat(sep=" ")},{price},1')
            for path in (mix, rts):
                with zipfile.ZipFile(path / '20260715.zip', 'w') as archive:
                    archive.writestr('ticks.csv', '\n'.join(ticks) + '\n')
            output = work / 'scores.jsonl'
            result = subprocess.run([sys.executable, str(SCRIPT),
                                     '--start', '2026-07-15', '--end-exclusive', '2026-07-16',
                                     '--news-dir', str(news), '--mix-dir', str(mix),
                                     '--rts-dir', str(rts), '--output', str(output)],
                                    text=True, capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            rows = [json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows), 10)
            self.assertEqual(len({r['news_id'] for r in rows}), 1)
            self.assertTrue(any(r['status'] == 'ok' and r['score'] is not None for r in rows))
            self.assertTrue((work / 'reactions' / 'reactions.sqlite3').exists())


if __name__ == '__main__':
    unittest.main()
