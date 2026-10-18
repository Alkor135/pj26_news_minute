r"""Проверяет трёхклассовый контракт, балансировку и оценку редких сигналов.

Использует только синтетические строки в памяти, без GPU и Ollama.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests/experiments_-10_10 -p test_extremes.py -v
"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments_-10_10/source'))


class ExtremesTests(unittest.TestCase):
    """Проверяет результат преобразований и защиту от ложного улучшения метрик."""

    def setUp(self):
        """Загружает общий контракт задач и данные без тяжёлых библиотек."""
        import model_tasks
        self.tasks = model_tasks

    def test_class_boundaries(self):
        """Границы ±8 включаются в крайние классы, ±7 остаются в none."""
        self.assertEqual([self.tasks.score_class(v) for v in (-10, -8, -7, 0, 7, 8, 10)],
                         ['negative', 'negative', 'none', 'none', 'none', 'positive', 'positive'])

    def test_balancing_retains_all_rows_and_does_not_mutate_validation(self):
        """Балансировка воспроизводимо дополняет редкие классы без изменения исходных строк."""
        rows = [dict(news_id=str(i), score=v) for i, v in enumerate([-8, 9, 0, 1, -2])]
        balanced, report = self.tasks.balance_training(rows, seed=42)
        self.assertEqual(report['after'], {'negative': 3, 'none': 3, 'positive': 3})
        self.assertEqual(len(rows), 5)
        self.assertEqual({r['news_id'] for r in balanced}, {r['news_id'] for r in rows})
        self.assertEqual(balanced, self.tasks.balance_training(rows, seed=42)[0])

    def test_missing_class_rejected(self):
        """Отсутствие одного класса нельзя исправить повторением других примеров."""
        with self.assertRaisesRegex(ValueError, 'класс'):
            self.tasks.balance_training([{'score': 0}, {'score': 8}], seed=42)

    def test_strict_response_contract(self):
        """Ответы с баллом, лишними полями и неизвестными классами отклоняются."""
        self.assertEqual(self.tasks.parse_response('{"class":"negative"}', 'extremes'), 'negative')
        for text in ('{"score":-10}', '{"class":"positive","score":9}', '{"class":"other"}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.tasks.parse_response(text, 'extremes')

    def test_signal_metrics_include_false_positives_and_invalid_answers(self):
        """Ложные сигналы портят precision, пропуски и неверный JSON уменьшают recall."""
        report = self.tasks.classification_metrics([
            dict(actual=8, predicted='positive'), dict(actual=0, predicted='positive'),
            dict(actual=-9, predicted=None), dict(actual=-8, predicted='none'),
        ])
        self.assertEqual(report['classes']['positive']['precision'], 0.5)
        self.assertEqual(report['classes']['positive']['recall'], 1)
        self.assertEqual(report['classes']['negative']['recall'], 0)
        self.assertIsNone(report['classes']['negative']['precision'])
        self.assertEqual(report['invalid_fraction'], 0.25)
        self.assertEqual(report['confusion']['negative']['invalid'], 1)

    def test_default_training_task(self):
        """Оба профиля по умолчанию обучают классификацию; старый режим доступен явно."""
        import training_common
        for profile in ('RTX4070', 'RTX3090Ti'):
            parser = training_common.build_parser(profile)
            self.assertEqual(parser.parse_args([]).task, 'extremes')
            self.assertEqual(parser.parse_args(['--task', 'score']).task, 'score')

    def test_training_encodes_class_and_masks_input(self):
        """При обучении extremes целевой ответ содержит класс, а не исходный балл."""
        import training_common
        row = dict(title='Новость', instrument='MIX', horizon=5, score=9, task='extremes', news_id='a')
        result = training_common.encode_example(row, CharacterTokenizer(), 4096)
        response = ''.join(chr(v) for v in result['labels'] if v != -100)
        self.assertIn('{"class":"positive"}', response)
        self.assertNotIn('score', response)
        self.assertEqual(result['labels'][0], -100)
        user = training_common.make_messages(row)[1]['content']
        self.assertEqual(json.loads(user), dict(title='Новость', instrument='MIX', horizon_minutes=5))

    def test_dry_run_balances_train_but_not_validation(self):
        """Реальный CLI разделяет данные до балансировки и сохраняет исходные баллы."""
        with tempfile.TemporaryDirectory(prefix='tmp_extremes_', dir=ROOT / 'tests') as temp:
            folder = Path(temp)
            rows = []
            for month in ('07', '08'):
                for i, score in enumerate((-8, 9, 0, 1)):
                    rows.append(dict(news_id=f'{month}_{i}', title=f'Новость {month}_{i}',
                                     loaded_at=f'2026-{month}-01 10:00:00',
                                     end_at=f'2026-{month}-01 10:05:01', instrument='MIX',
                                     horizon=5, score=score, stable=False, score_version='test'))
            data = folder / 'scores.jsonl'
            data.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            output = folder / 'result'
            result = subprocess.run([sys.executable, str(ROOT / 'experiments_-10_10/train_RTX4070.py'),
                                     '--data', str(data), '--output', str(output), '--dry-run'],
                                    capture_output=True, encoding='utf-8')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            report = json.loads((output / 'data_report.json').read_text(encoding='utf-8'))
            self.assertEqual(report['task'], 'extremes')
            self.assertEqual(report['train_before_balance']['rows'], 4)
            self.assertEqual(report['train']['rows'], 6)
            self.assertEqual(report['validation']['rows'], 4)
            self.assertEqual(report['validation']['classes'], {'negative': 1, 'positive': 1, 'none': 2})
            self.assertEqual(report['parameters']['prediction_limit'], 0)
            self.assertEqual(data.read_text(encoding='utf-8').count('"score"'), 8)

    def test_extreme_export_import_contract(self):
        """Проверка импорта выбирает отдельное имя и не принимает балл вместо класса."""
        spec = importlib.util.spec_from_file_location('extreme_import', ROOT / 'experiments_-10_10/import_ollama.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        import hashlib
        with tempfile.TemporaryDirectory(prefix='tmp_extremes_', dir=ROOT / 'tests') as temp:
            folder = Path(temp)
            gguf = folder / 'model.gguf'
            gguf.write_bytes(b'GGUFexample')
            manifest = dict(profile='RTX4070', task='extremes', model='Qwen/Example', revision='a' * 40,
                            system_prompt=self.tasks.EXTREME_PROMPT, gguf_file=gguf.name,
                            gguf_sha256=hashlib.sha256(gguf.read_bytes()).hexdigest())
            (folder / 'export_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            args = module.build_parser().parse_args(['--export', str(folder), '--check-only'])
            self.assertEqual(module.import_model(args), 'news-extremes-4070')
            self.assertIn('class', module.build_modelfile(gguf, manifest['system_prompt']))


class CharacterTokenizer:
    """Кодирует символы для проверки маски и целевого ответа без загрузки модели."""

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        """Возвращает коды текста messages с началом ответа при add_generation_prompt."""
        text = ''.join(f'<{m["role"]}>{m["content"]}<end>' for m in messages)
        if add_generation_prompt:
            text += '<assistant>'
        return [ord(char) for char in text]


if __name__ == '__main__':
    unittest.main()
