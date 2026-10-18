r"""Проверяет порядок этапов оркестратора RTX 4070 без запуска обучения.

Все подмены ограничены внешними процессами. Временные файлы только в tests.
Проверяет реальное удаление только выбранной разметки и её сводки во временной папке.
Последний этап 8 формирует HTML из готового отчёта без обращения к модели.
Режим final-test использует полный год обучения и сентябрь 2026 только для итоговой проверки.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests/experiments_-10_10 -p test_orchestrator.py -v
"""

import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]


class OrchestratorTests(unittest.TestCase):
    """Проверяет сквозную передачу путей, режим просмотра и остановку при ошибке."""

    def setUp(self):
        """Загружает оркестратор и создаёт временный каталог результата."""
        path = ROOT / 'experiments_-10_10/orchestrator_RTX4070.py'
        spec = importlib.util.spec_from_file_location('orchestrator', path)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory(prefix='tmp_orchestrator_', dir=ROOT / 'tests')
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'run'

    def test_order_and_paths(self):
        """Каждый этап получает один и тот же каталог; классификация выбирается явно."""
        args = self.module.build_parser().parse_args(['--run', str(self.run)])
        commands = self.module.build_commands(args)
        self.assertEqual([stage for stage, _, _ in commands], [3, 4, 5, 6, 7, 8])
        self.assertIn('extremes', commands[1][2])
        self.assertIn(str(self.run), commands[1][2])
        self.assertIn(str(self.run), commands[2][2])
        self.assertIn(str(self.run / 'export'), commands[3][2])
        self.assertEqual(commands[4][2][-2:], ['--limit', '0'])
        self.assertIn('build_html_report.py', commands[5][2][1])
        self.assertIn(str(self.run), commands[5][2])

    def test_final_test_has_separate_paths_and_september_boundaries(self):
        """Финальный профиль переносит август в обучение и не меняет старые каталоги."""
        args = self.module.build_parser().parse_args(['--final-test'])
        commands = self.module.build_commands(args)
        label = commands[0][2]
        train = commands[1][2]
        self.assertEqual(label[label.index('--end-exclusive') + 1], '2026-10-01')
        self.assertEqual(train[train.index('--validation-start') + 1], '2026-09-01')
        self.assertEqual(train[train.index('--cutoff') + 1], '2026-10-01')
        self.assertEqual(train[train.index('--evaluation-role') + 1], 'test')
        self.assertIn('scores_test_202609.jsonl', label[label.index('--output') + 1])
        self.assertTrue(train[train.index('--output') + 1].endswith('RTX4070_extremes_test_202609'))
        self.assertIn('news-extremes-4070-test-202609', commands[3][2])

    def test_final_test_preserves_explicit_output(self):
        """Явные пути и имя имеют приоритет над отдельными каталогами финального профиля."""
        args = self.module.build_parser().parse_args(['--final-test', '--run', str(self.run), '--name', 'custom-test'])
        commands = self.module.build_commands(args)
        self.assertIn(str(self.run), commands[1][2])
        self.assertIn('custom-test', commands[3][2])

    def test_dry_run_does_not_start_processes(self):
        """Просмотр команд работает без данных, CUDA и Ollama."""
        with patch.object(self.module.subprocess, 'run') as execute:
            self.assertEqual(self.module.main(['--dry-run', '--run', str(self.run)]), 0)
        execute.assert_not_called()

    def test_only_html_stage_does_not_run_model(self):
        """Запуск с этапа 8 вызывает только создание отчёта."""
        args = self.module.build_parser().parse_args(['--from-stage', '8', '--run', str(self.run)])
        commands = self.module.build_commands(args)
        self.assertEqual(len(commands), 1)
        self.assertEqual(commands[0][0], 8)
        self.assertEqual(Path(commands[0][2][1]).name, 'build_html_report.py')

    def test_failure_stops_following_stages(self):
        """Ошибка импорта не допускает запуска проверки модели."""
        with patch.object(self.module.subprocess, 'run', side_effect=subprocess.CalledProcessError(2, ['python'])) as execute:
            self.assertEqual(self.module.main(['--from-stage', '6', '--run', str(self.run)]), 2)
        self.assertEqual(execute.call_count, 1)

    def test_training_directory_is_not_overwritten(self):
        """Уже начатое обучение отклоняется до дорогого пересчёта разметки."""
        self.run.mkdir()
        (self.run / 'training_manifest.json').write_text('{}', encoding='utf-8')
        with patch.object(self.module.subprocess, 'run') as execute:
            self.assertEqual(self.module.main(['--run', str(self.run)]), 1)
        execute.assert_not_called()

    def test_delete_labels_before_label_stage(self):
        """Флаг удаляет JSONL и его сводку до запуска разметки, сохраняя соседние файлы."""
        folder = Path(self.temp.name)
        data = folder / 'custom.jsonl'
        summary = folder / 'custom_summary.json'
        other = folder / 'other.jsonl'
        for path in (data, summary, other):
            path.write_text('данные', encoding='utf-8')

        def check_deleted(*args, **kwargs):
            """Проверяет состояние файлов в момент старта дочернего процесса."""
            self.assertFalse(data.exists())
            self.assertFalse(summary.exists())
            self.assertTrue(other.exists())

        with patch.object(self.module.subprocess, 'run', side_effect=check_deleted) as execute:
            self.assertEqual(self.module.main(['--delete-labels', '--data', str(data), '--to-stage', '3']), 0)
        self.assertEqual(execute.call_count, 1)

    def test_dry_run_delete_keeps_files(self):
        """Просмотр удаления сохраняет оба реальных файла и не вызывает процессы."""
        data = Path(self.temp.name) / 'custom.jsonl'
        summary = data.with_name('custom_summary.json')
        for path in (data, summary):
            path.write_text('данные', encoding='utf-8')
        with patch.object(self.module.subprocess, 'run') as execute:
            self.assertEqual(self.module.main(['--delete-labels', '--dry-run', '--data', str(data)]), 0)
        self.assertTrue(data.exists())
        self.assertTrue(summary.exists())
        execute.assert_not_called()

    def test_delete_without_label_stage_is_rejected(self):
        """Разметку нельзя удалить, если её восстановление не входит в выбранные этапы."""
        data = Path(self.temp.name) / 'keep.jsonl'
        data.write_text('данные', encoding='utf-8')
        with patch.object(self.module.subprocess, 'run') as execute:
            self.assertEqual(self.module.main(['--delete-labels', '--from-stage', '4', '--data', str(data)]), 1)
        self.assertTrue(data.exists())
        execute.assert_not_called()

    def test_delete_missing_labels_is_allowed(self):
        """Первый запуск с отсутствующей разметкой не считается ошибкой удаления."""
        with patch.object(self.module.subprocess, 'run'):
            self.assertEqual(self.module.main(['--delete-labels', '--to-stage', '3', '--data',
                                               str(Path(self.temp.name) / 'missing.jsonl')]), 0)


if __name__ == '__main__':
    unittest.main()
