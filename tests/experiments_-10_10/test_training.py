"""Проверяет защиту обучения от утечек, неверных меток и потери целевого ответа.

Создаёт временные JSONL только внутри tests; тяжёлые библиотеки и GPU не нужны.
Проверяет также выбор результата разметки по умолчанию для обоих профилей.
Прежний числовой режим запускается с явным --task score.
Общий модуль загружается из experiments_-10_10/source/training_common.py.
Запуск из корня: python -m unittest discover -s tests/experiments_-10_10 -v.
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = ROOT / 'experiments_-10_10'
MODULE = EXPERIMENT / 'source' / 'training_common.py'


def example(news_id, loaded_at, end_at, **changes):
    """Возвращает валидную строку разметки; changes переопределяет поля примера."""
    row = dict(news_id=news_id, title=f'Новость {news_id}', loaded_at=loaded_at,
               end_at=end_at, instrument='MIX', horizon=5, score=4,
               stable=False, score_version='magnitude-quality-v1')
    return row | changes


class TrainingTests(unittest.TestCase):
    """Проверяет реальные преобразования примеров и границы временных выборок."""

    def setUp(self):
        """Загружает проверяемый модуль и создаёт каталог данных внутри tests."""
        self.assertTrue(MODULE.exists(), 'Общий модуль обучения ещё не создан')
        spec = importlib.util.spec_from_file_location('training_common', MODULE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.tmp = tempfile.TemporaryDirectory(prefix='tmp_training_', dir=ROOT / 'tests')
        self.addCleanup(self.tmp.cleanup)

    def read(self, rows):
        """Записывает rows во временный JSONL и возвращает результат чтения модулем."""
        path = Path(self.tmp.name) / 'scores.jsonl'
        path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
        return self.module.read_examples(path)

    def test_missing_score_is_not_zero(self):
        """Отсутствующий ответ исключается, а настоящий ноль сохраняется."""
        rows, stats = self.read([
            example('a', '2025-10-01 10:00:00', '2025-10-01 10:05:01', score=None),
            example('b', '2025-10-01 11:00:00', '2025-10-01 11:05:01', score=0),
        ])
        self.assertEqual([r['score'] for r in rows], [0])
        self.assertEqual(stats['unlabeled_rows'], 1)

    def test_invalid_scores_and_unstable_extremes_rejected(self):
        """Запрещает дробные/булевы баллы, неверный диапазон и ±10 без stable."""
        for value in (True, 1.5, 11, float('nan'), 10, -10):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.read([example('a', '2025-10-01 10:00:00', '2025-10-01 10:05:01', score=value)])

    def test_conflicting_news_metadata_rejected(self):
        """Одинаковый news_id не должен обозначать разные моменты поступления."""
        with self.assertRaises(ValueError):
            self.read([
                example('a', '2025-10-01 10:00:00', '2025-10-01 10:05:01'),
                example('a', '2025-10-02 10:00:00', '2025-10-02 10:05:01', instrument='RTS'),
            ])

    def test_null_identifiers_and_non_object_rows_rejected(self):
        """Не превращает null в идентификатор 'None' и явно отвергает JSON-массив."""
        for changes in ({'news_id': None}, {'score_version': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read([example('a', '2025-10-01 10:00:00', '2025-10-01 10:05:01') | changes])
        with self.assertRaises(ValueError):
            self.read([['это не объект новости']])

    def test_purge_whole_news_and_keep_september_out(self):
        """Удаляет все горизонты новости с пересечением границы и весь сентябрь."""
        rows, _ = self.read([
            example('train', '2026-07-20 10:00:00', '2026-07-20 10:05:01'),
            example('cross', '2026-07-31 23:00:00', '2026-07-31 23:05:01'),
            example('cross', '2026-07-31 23:00:00', '2026-08-01 01:00:01', horizon=120),
            example('valid', '2026-08-15 10:00:00', '2026-08-15 10:05:01'),
            example('test', '2026-09-01 10:00:00', '2026-09-01 10:05:01'),
        ])
        train, valid, stats = self.module.split_examples(rows, '2025-09-01', '2026-08-01', '2026-09-01')
        self.assertEqual([r['news_id'] for r in train], ['train'])
        self.assertEqual([r['news_id'] for r in valid], ['valid'])
        self.assertEqual(stats['purged_window_rows'], 2)
        self.assertEqual(stats['outside_period_rows'], 1)

    def test_reprints_not_shared_between_train_and_validation(self):
        """Убирает из обучения перепечатки по нормализованному заголовку и group_id."""
        rows, _ = self.read([
            example('unique', '2026-07-01 10:00:00', '2026-07-01 10:05:01'),
            example('old', '2026-07-02 10:00:00', '2026-07-02 10:05:01', title='  СТАВКА   ЦБ '),
            example('new', '2026-08-02 10:00:00', '2026-08-02 10:05:01', title='ставка цб'),
            example('old2', '2026-07-03 10:00:00', '2026-07-03 10:05:01', group_id='event'),
            example('new2', '2026-08-03 10:00:00', '2026-08-03 10:05:01', group_id='event'),
        ])
        train, valid, stats = self.module.split_examples(rows, '2025-09-01', '2026-08-01', '2026-09-01')
        self.assertEqual([r['news_id'] for r in train], ['unique'])
        self.assertEqual(len(valid), 2)
        self.assertEqual(stats['purged_reprint_rows'], 2)

    def test_future_metrics_not_in_prompt(self):
        """Вход модели содержит только заголовок, инструмент и горизонт."""
        row = example('a', '2026-07-01 10:00:00', '2026-07-01 10:05:01',
                      z_score=9876, return_pct=1234)
        prompt = json.dumps(self.module.make_messages(row), ensure_ascii=False)
        self.assertNotIn('9876', prompt)
        self.assertNotIn('1234', prompt)
        self.assertNotIn('stable', prompt)
        self.assertNotIn('score_version', prompt)

    def test_response_only_mask_and_no_silent_truncation(self):
        """Маскирует вход; слишком длинный пример отклоняет вместо обрезки ответа."""
        row = example('a', '2026-07-01 10:00:00', '2026-07-01 10:05:01')
        encoded = self.module.encode_example(row, CharacterTokenizer(), 4096)
        supervised = [v for v in encoded['labels'] if v != -100]
        self.assertTrue(supervised)
        self.assertIn('"score":4', ''.join(chr(v) for v in supervised))
        self.assertEqual(encoded['labels'][0], -100)
        with self.assertRaises(ValueError):
            self.module.encode_example(row, CharacterTokenizer(), 8)

    def test_cli_dry_run_both_profiles(self):
        """Оба запуска проверяют данные без GPU и записывают разные профили моделей."""
        self.read([
            example('train', '2026-07-01 10:00:00', '2026-07-01 10:05:01'),
            example('valid', '2026-08-01 10:00:00', '2026-08-01 10:05:01'),
        ])
        for gpu in ('RTX4070', 'RTX3090Ti'):
            output = Path(self.tmp.name) / gpu
            result = subprocess.run([
                sys.executable, str(EXPERIMENT / f'train_{gpu}.py'),
                '--data', str(Path(self.tmp.name) / 'scores.jsonl'),
                '--output', str(output), '--dry-run',
                '--task', 'score',
            ], capture_output=True, text=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'data_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['profile'], gpu)
            self.assertEqual(report['train']['rows'], 1)
            self.assertEqual(report['validation']['rows'], 1)
            self.assertFalse((output / 'adapter').exists())

    def test_default_data_for_both_profiles(self):
        """Без --data оба профиля выбирают результат разметки; явный путь имеет приоритет."""
        for gpu in ('RTX4070', 'RTX3090Ti'):
            with self.subTest(gpu=gpu):
                parser = self.module.build_parser(gpu)
                self.assertEqual(parser.parse_args([]).data,
                                 ROOT / 'data' / 'experiments_-10_10' / 'scores.jsonl')
                custom = Path(self.tmp.name) / 'other.jsonl'
                self.assertEqual(parser.parse_args(['--data', str(custom)]).data, custom)


class CharacterTokenizer:
    """Минимальный детерминированный токенизатор для проверки маски целевого ответа."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        """Преобразует messages в коды символов; при запросе добавляет начало ответа."""
        text = ''.join(f"<{m['role']}>{m['content']}<end>" for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(c) for c in text] if tokenize else text


if __name__ == '__main__':
    unittest.main()
