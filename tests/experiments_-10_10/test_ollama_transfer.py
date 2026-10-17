r"""Проверяет защиту экспорта и импорта модели без загрузки больших весов.

Создаёт временные учебные артефакты только внутри tests.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests/experiments_-10_10 -v
"""

import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / 'experiments_-10_10'


def load_script(name):
    """Загружает скрипт name из каталога эксперимента и возвращает модуль."""
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OllamaTransferTests(unittest.TestCase):
    """Проверяет совместимость артефактов и контракт ответа при переносе."""

    def setUp(self):
        """Создаёт изолированные каталоги примеров и загружает проверяемые скрипты."""
        self.tmp = tempfile.TemporaryDirectory(prefix='tmp_ollama_', dir=ROOT / 'tests')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.export = load_script('export_ollama')
        self.importer = load_script('import_ollama')

    def training_run(self, adapter_base='Qwen/Qwen2.5-7B-Instruct'):
        """Создаёт минимальный результат обучения и возвращает его каталог."""
        run = self.root / 'RTX3090Ti'
        adapter = run / 'adapter'
        adapter.mkdir(parents=True)
        (run / 'training_manifest.json').write_text(json.dumps({
            'profile': 'RTX3090Ti', 'model': 'Qwen/Qwen2.5-7B-Instruct',
            'revision': 'a' * 40, 'system_prompt': 'Верни только JSON.'
        }), encoding='utf-8')
        (run / 'training_metrics.json').write_text('{}', encoding='utf-8')
        (run / 'score_metrics.json').write_text('{}', encoding='utf-8')
        (run / 'validation_predictions.json').write_text('[]', encoding='utf-8')
        (adapter / 'adapter_config.json').write_text(json.dumps({
            'base_model_name_or_path': adapter_base
        }), encoding='utf-8')
        (adapter / 'adapter_model.safetensors').write_bytes(b'weights')
        (adapter / 'tokenizer_config.json').write_text('{}', encoding='utf-8')
        (run / 'adapter_provenance.json').write_text(json.dumps({
            'model': 'Qwen/Qwen2.5-7B-Instruct', 'revision': 'a' * 40,
            'adapter_sha256': hashlib.sha256(b'weights').hexdigest()
        }), encoding='utf-8')
        return run

    def test_export_rejects_adapter_for_another_base(self):
        """Не допускает объединения адаптера с моделью другого семейства."""
        run = self.training_run(adapter_base='Qwen/Another-Model')
        with self.assertRaisesRegex(ValueError, '(?i)базов'):
            self.export.validate_training_run(run)

    def test_export_requires_finished_adapter(self):
        """Один манифест без обученных весов не считается результатом обучения."""
        run = self.training_run()
        (run / 'adapter' / 'adapter_model.safetensors').unlink()
        with self.assertRaisesRegex(ValueError, 'адаптер'):
            self.export.validate_training_run(run)

    def test_export_requires_completion_marker(self):
        """Адаптер без отметки окончания всего обучения не экспортируется."""
        run = self.training_run()
        (run / 'adapter_provenance.json').unlink()
        with self.assertRaisesRegex(ValueError, 'завершён'):
            self.export.validate_training_run(run)

    def test_export_rejects_swapped_adapter_of_same_model(self):
        """Адаптер другого запуска с тем же именем базы обнаруживается по SHA."""
        run = self.training_run()
        (run / 'adapter' / 'adapter_model.safetensors').write_bytes(b'other')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.export.validate_training_run(run)

    def test_export_rejects_other_base_revision(self):
        """Отметка завершения другой SHA-ревизии базы отклоняется."""
        run = self.training_run()
        path = run / 'adapter_provenance.json'
        marker = json.loads(path.read_text(encoding='utf-8'))
        marker['revision'] = 'b' * 40
        path.write_text(json.dumps(marker), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'ревизи'):
            self.export.validate_training_run(run)

    def test_check_only_works_after_export(self):
        """Проверка исходного обучения доступна и после создания папки export."""
        run = self.training_run()
        (run / 'export').mkdir()
        args = SimpleNamespace(run=run, output=None, check_only=True)
        self.assertIsNone(self.export.export_run(args))

    def test_import_rejects_changed_gguf(self):
        """Импорт отвергает файл GGUF, изменившийся после экспорта."""
        folder = self.root / 'export'
        folder.mkdir()
        gguf = folder / 'model-Q4_K_M.gguf'
        gguf.write_bytes(b'GGUFwrong')
        (folder / 'export_manifest.json').write_text(json.dumps({
            'profile': 'RTX3090Ti', 'model': 'Qwen/Qwen2.5-7B-Instruct',
            'revision': 'a' * 40, 'system_prompt': 'Верни только JSON.',
            'gguf_file': gguf.name, 'gguf_sha256': hashlib.sha256(b'GGUForiginal').hexdigest(),
            'quantization': 'Q4_K_M'
        }), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'SHA-256'):
            self.importer.validate_export(folder)

    def test_modelfile_uses_exported_weights_and_training_prompt(self):
        """Запуск Ollama сохраняет системное задание и выбирает готовый GGUF."""
        content = self.importer.build_modelfile(
            self.root / 'model-Q4_K_M.gguf', 'Верни только JSON.')
        self.assertIn('FROM ./model-Q4_K_M.gguf', content)
        self.assertIn('SYSTEM """Верни только JSON."""', content)
        self.assertIn('PARAMETER temperature 0', content)

    def test_smoke_rejects_invalid_score(self):
        """Проверка ответа исключает bool, дроби, лишние поля и выход за шкалу."""
        for answer in ('{"score":true}', '{"score":1.5}', '{"score":11}',
                       '{"score":0,"extra":1}', 'не JSON'):
            with self.subTest(answer=answer), self.assertRaises(ValueError):
                self.importer.parse_score(answer)
        self.assertEqual(self.importer.parse_score('{"score":-10}'), -10)

    def test_export_publishes_complete_folder_only_after_conversion(self):
        """Готовый каталог появляется после конверсии с проверенным GGUF и манифестом."""
        run = self.training_run()
        converter = self.root / 'convert_hf_to_gguf.py'
        quantizer = self.root / 'llama-quantize.exe'
        converter.write_text('# пример', encoding='utf-8')
        quantizer.write_bytes(b'exe')
        target = run / 'export'
        args = SimpleNamespace(run=run, output=None, check_only=False,
                               llama_cpp_dir=None, converter=converter, quantizer=quantizer,
                               device='cpu', quantization='Q4_K_M', converter_python='python')

        def fake_merge(source, manifest, merged, device):
            """Имитирует только тяжёлую загрузку весов, создавая выходной файл."""
            merged.mkdir()
            (merged / 'model.safetensors').write_bytes(b'merged')
            return {'status': 'passed', 'score': -3, 'answer': '{"score":-3}'}

        def fake_conversion(merged, convert_tool, quant_tool, folder, quantization, converter_python):
            """Имитирует только внешнюю конвертацию, создавая сигнатуру GGUF."""
            self.assertFalse(target.exists())
            (folder / 'model-f16.gguf').write_bytes(b'GGUFfull')
            result = folder / 'model-Q4_K_M.gguf'
            result.write_bytes(b'GGUFexample')
            return result

        with patch.object(self.export, 'merge_adapter', side_effect=fake_merge), \
             patch.object(self.export, 'convert_and_quantize', side_effect=fake_conversion):
            self.assertEqual(self.export.export_run(args), target)
        self.assertTrue((target / 'model-Q4_K_M.gguf').is_file())
        self.assertTrue((target / 'model-f16.gguf').is_file())
        manifest = json.loads((target / 'export_manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['gguf_sha256'], hashlib.sha256(b'GGUFexample').hexdigest())
        self.assertEqual(manifest['f16_gguf_sha256'], hashlib.sha256(b'GGUFfull').hexdigest())
        self.assertEqual(manifest['revision'], 'a' * 40)
        with self.assertRaisesRegex(ValueError, 'существует'):
            self.export.export_run(args)

    def test_import_refuses_duplicate_before_writing_modelfile(self):
        """Существующее имя без --replace не меняет ни модель, ни Modelfile."""
        folder = self.root / 'export'
        folder.mkdir()
        gguf = folder / 'model-Q4_K_M.gguf'
        gguf.write_bytes(b'GGUFexample')
        (folder / 'export_manifest.json').write_text(json.dumps({
            'profile': 'RTX4070', 'model': 'Qwen/Qwen3-4B-Instruct-2507',
            'revision': 'a' * 40, 'system_prompt': 'Верни только JSON.',
            'gguf_file': gguf.name, 'gguf_sha256': hashlib.sha256(b'GGUFexample').hexdigest()
        }), encoding='utf-8')
        args = SimpleNamespace(export=folder, name=None, check_only=False,
                               host='http://127.0.0.1:11434', replace=False, ollama='ollama')
        with patch.object(self.importer, 'existing_models', return_value={'news-score-4070:latest'}):
            with self.assertRaisesRegex(ValueError, 'уже существует'):
                self.importer.import_model(args)
        self.assertFalse((folder / 'Modelfile').exists())

    def test_import_creates_model_and_checks_reply(self):
        """Успешный импорт связывает GGUF с Ollama и сохраняет проверенный балл."""
        folder = self.root / 'export'
        folder.mkdir()
        gguf = folder / 'model-Q4_K_M.gguf'
        gguf.write_bytes(b'GGUFexample')
        (folder / 'export_manifest.json').write_text(json.dumps({
            'profile': 'RTX4070', 'model': 'Qwen/Qwen3-4B-Instruct-2507',
            'revision': 'a' * 40, 'system_prompt': 'Верни только JSON.',
            'gguf_file': gguf.name, 'gguf_sha256': hashlib.sha256(b'GGUFexample').hexdigest()
        }), encoding='utf-8')
        args = SimpleNamespace(export=folder, name=None, check_only=False,
                               host='http://127.0.0.1:11434', replace=False, ollama='ollama')
        installed = False

        def fake_api(host, endpoint, payload=None, timeout=30):
            """Заменяет только внешний Ollama API и возвращает состояние модели."""
            if endpoint == '/api/tags':
                return {'models': [{'name': 'news-score-4070:latest'}] if installed else []}
            self.assertEqual(endpoint, '/api/chat')
            self.assertEqual(payload['model'], 'news-score-4070')
            self.assertFalse(payload['stream'])
            return {'message': {'content': '{"score":-3}'}}

        def fake_cli(command, check, env):
            """Заменяет только внешнюю CLI-команду создания модели."""
            nonlocal installed
            self.assertEqual(command[:3], ['ollama', 'create', 'news-score-4070'])
            self.assertTrue((folder / 'Modelfile').is_file())
            installed = True

        with patch.object(self.importer, 'api_json', side_effect=fake_api), \
             patch.object(self.importer.subprocess, 'run', side_effect=fake_cli):
            self.assertEqual(self.importer.import_model(args), 'news-score-4070')
        report = json.loads((folder / 'import_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['smoke_test']['score'], -3)
        self.assertEqual(report['smoke_test']['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
