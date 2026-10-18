r"""Проверяет метрики оценки Ollama и исключение утечки целевого балла.

Внешний Ollama не запускается. Временные данные создаются только в tests.
Проверяет допуск финального сентября по явной роли test, без допуска в validation.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests/experiments_-10_10 -p test_check_model.py -v
"""

import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


def load_check():
    """Возвращает модуль проверочного скрипта без выполнения его main."""
    spec = importlib.util.spec_from_file_location('check_model', Path(__file__).with_name('check_model.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModelCheckTests(unittest.TestCase):
    """Проверяет честное сравнение метрик и содержимое запросов модели."""

    def test_invalid_answers_do_not_improve_baseline_comparison(self):
        """MAE модели и нуля сравниваются на одинаковых успешно оценённых строках."""
        module = load_check()
        metrics = module.metrics([
            {'actual': 10, 'predicted': None},
            {'actual': -2, 'predicted': -1},
            {'actual': 0, 'predicted': 0},
        ])
        self.assertEqual(metrics['mae'], 0.5)
        self.assertEqual(metrics['zero_baseline_mae_same_rows'], 1)
        self.assertEqual(metrics['zero_baseline_mae_all_rows'], 4)
        self.assertAlmostEqual(metrics['invalid_fraction'], 1 / 3)
        self.assertEqual(metrics['direction_accuracy_nonzero_actual'], 1)

    def test_no_valid_answers_have_no_mae(self):
        """Полностью неверные ответы не превращаются в нулевую ошибку."""
        module = load_check()
        result = module.metrics([{'actual': 1, 'predicted': None}])
        self.assertIsNone(result['mae'])
        self.assertEqual(result['invalid_fraction'], 1)

    def test_prepare_september_requires_explicit_test_role(self):
        """Сохранённый сентябрьский тест восстанавливается без использования его в обучении."""
        module = load_check()
        with tempfile.TemporaryDirectory(prefix='tmp_model_check_', dir=ROOT / 'tests') as temp:
            run = Path(temp)
            rows = [dict(news_id=month, title=f'Новость {month}', instrument='MIX', horizon=5,
                         score=8, stable=False, score_version='test',
                         loaded_at=f'2026-{month}-15 10:00:00', end_at=f'2026-{month}-15 10:05:01')
                    for month in ('08', '09')]
            data = run / 'scores.jsonl'
            data.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
            parameters = dict(train_start='2025-09-01', validation_start='2026-09-01',
                              cutoff='2026-10-01', evaluation_role='test')
            manifest = dict(task='extremes', parameters=parameters)
            (run / 'training_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            (run / 'data_report.json').write_text(json.dumps({'data_sha256': hashlib.sha256(data.read_bytes()).hexdigest()}), encoding='utf-8')
            (run / 'validation_predictions.json').write_text(json.dumps([
                dict(news_id='09', instrument='MIX', horizon=5, actual=8, predicted='positive')
            ]), encoding='utf-8')
            selected, _ = module.prepare_cases(run, data, 0)
            self.assertEqual([r['news_id'] for r in selected], ['09'])
            manifest['parameters']['evaluation_role'] = 'validation'
            (run / 'training_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'test'):
                module.prepare_cases(run, data, 0)

    def test_request_excludes_label_and_disallows_extra_fields(self):
        """В запрос не попадает actual; ответ с лишними полями отклоняется."""
        module = load_check()
        row = {'news_id': 'x', 'title': 'Новость', 'instrument': 'RTS',
               'horizon': 15, 'score': 10, 'loaded_at': '2026-08-10'}
        with patch.object(module.ollama, 'api_json', return_value={
                'message': {'content': '{"score":10,"extra":true}'}}) as api:
            with self.assertRaises(ValueError):
                module.predict('http://localhost:11434', 'example', row, 'Инструкция')
        payload = api.call_args.args[2]
        import json
        self.assertEqual(json.loads(payload['messages'][1]['content']),
                         {'title': 'Новость', 'instrument': 'RTS', 'horizon_minutes': 15})


if __name__ == '__main__':
    unittest.main()
