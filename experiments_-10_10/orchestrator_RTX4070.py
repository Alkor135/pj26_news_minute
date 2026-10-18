r"""Последовательно запускает эксперимент трёх классов на RTX 4070.

Этапы: 3 — build_rasmetka.py (загрузка RSS, фильтр, баллы), 4 — обучение
train_RTX4070.py с --task extremes, 5 — export_ollama.py, 6 — import_ollama.py,
7 — check_model.py на всей сохранённой валидации, 8 — build_html_report.py
по готовым ответам. Этап 4.1 (подготовка llama.cpp)
выполняется заранее по README. Каждый процесс запускается текущим Python
из корня проекта. При первой ошибке следующие этапы не выполняются.
Без параметров запускает этапы 3–8; этап 3 пересоздаёт файл разметки.
Готовые обучение и экспорт не перезаписывает; замену модели Ollama разрешает
только --replace. Прогресс выводится в консоль, файловых логов нет.
--delete-labels перед этапом 3 удаляет только выбранный JSONL и его
файл STEM_summary.json. Без этапа 3 этот флаг запрещён. Отсутствующие файлы
пропускаются; папки не удаляются. При --dry-run файлы сохраняются.
--final-test обучает на сентябре 2025 — августе 2026 и проверяет только
после обучения на сентябре 2026. Использует отдельные JSONL, каталог и имя
с суффиксом test_202609/test-202609. По умолчанию остаётся августовская валидация.

Запуск в подготовленном .venv из корня проекта или кнопкой VS Code:
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --dry-run
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --delete-labels
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --delete-labels --dry-run
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --from-stage 4
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --from-stage 4 --to-stage 4
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --from-stage 5
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --from-stage 7
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --from-stage 8
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --final-test --dry-run
  .\.venv\Scripts\python.exe experiments_-10_10/orchestrator_RTX4070.py --final-test
--data меняет путь JSONL, --run — единый каталог нового обучения,
--reaction-db — готовую базу реакций для этапа 3, --name — имя новой модели.
--dry-run только печатает команды и ничего не запускает.
"""

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'experiments_-10_10'


def build_parser():
    """Возвращает параметры диапазона этапов, общих путей и безопасного просмотра."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--from-stage', type=int, choices=range(3, 9), default=3,
                        help='Первый этап; 3 — разметка, 4 — обучение, 5 — экспорт, 6 — импорт, 7 — проверка, 8 — HTML')
    parser.add_argument('--to-stage', type=int, choices=range(3, 9), default=8, help='Последний этап включительно')
    parser.add_argument('--final-test', action='store_true', help='Полный год обучения до 01.09.2026; финальный тест за сентябрь 2026')
    parser.add_argument('--data', type=Path, help='JSONL; по умолчанию scores.jsonl либо scores_test_202609.jsonl')
    parser.add_argument('--run', type=Path,
                        help='Каталог нового обучения, экспорта и отчётов')
    parser.add_argument('--reaction-db', type=Path, help='Готовая база реакций вместо чтения тиков на этапе 3')
    parser.add_argument('--name', help='Имя модели; по умолчанию news-extremes-4070 либо news-extremes-4070-test-202609')
    parser.add_argument('--replace', action='store_true', help='Разрешить замену существующего имени модели при импорте')
    parser.add_argument('--delete-labels', action='store_true',
                        help='Перед этапом 3 удалить выбранный JSONL разметки и его сводку')
    parser.add_argument('--dry-run', action='store_true', help='Показать команды без выполнения')
    return parser


def resolve_profile(args):
    """Возвращает копию args с путями и периодом профиля; явные пути/имя сохраняет."""
    resolved = argparse.Namespace(**vars(args))
    base = ROOT / 'data/experiments_-10_10'
    resolved.data = args.data or base / ('scores_test_202609.jsonl' if args.final_test else 'scores.jsonl')
    resolved.run = args.run or base / ('RTX4070_extremes_test_202609' if args.final_test else 'RTX4070_extremes')
    resolved.name = args.name or ('news-extremes-4070-test-202609' if args.final_test else 'news-extremes-4070')
    resolved.train_start = '2025-09-01'
    resolved.validation_start = '2026-09-01' if args.final_test else '2026-08-01'
    resolved.cutoff = '2026-10-01' if args.final_test else '2026-09-01'
    resolved.evaluation_role = 'test' if args.final_test else 'validation'
    return resolved


def build_commands(args):
    """Возвращает список (номер, название, команда) для периода и этапов args."""
    args = resolve_profile(args)
    if args.from_stage > args.to_stage:
        raise ValueError('--from-stage не должен превышать --to-stage')
    if args.delete_labels and args.from_stage != 3:
        raise ValueError('--delete-labels разрешён только при запуске этапа 3 (разметка)')
    run = args.run.resolve()
    data = args.data.resolve()
    label = [sys.executable, str(SCRIPTS / 'build_rasmetka.py'), '--output', str(data),
             '--start', args.train_start, '--end-exclusive', args.cutoff]
    if args.reaction_db:
        label.extend(['--reaction-db', str(args.reaction_db.resolve())])
    train = [sys.executable, str(SCRIPTS / 'train_RTX4070.py'), '--task', 'extremes',
             '--data', str(data), '--output', str(run), '--train-start', args.train_start,
             '--validation-start', args.validation_start, '--cutoff', args.cutoff,
             '--evaluation-role', args.evaluation_role]
    export = [sys.executable, str(SCRIPTS / 'export_ollama.py'), '--run', str(run)]
    importer = [sys.executable, str(SCRIPTS / 'import_ollama.py'), '--export', str(run / 'export'),
                '--name', args.name]
    if args.replace:
        importer.append('--replace')
    check = [sys.executable, str(ROOT / 'tests/experiments_-10_10/check_model.py'),
             '--run', str(run), '--data', str(data), '--name', args.name, '--limit', '0']
    html = [sys.executable, str(SCRIPTS / 'build_html_report.py'), '--run', str(run), '--data', str(data)]
    commands = [(3, 'Разметка', label), (4, 'Обучение трёх классов', train),
                (5, 'Объединение и экспорт GGUF', export), (6, 'Импорт в Ollama', importer),
                (7, 'Проверка модели', check), (8, 'HTML-отчёт', html)]
    return [entry for entry in commands if args.from_stage <= entry[0] <= args.to_stage]


def delete_label_files(data, dry_run=False):
    """Удаляет JSONL data и его сводку; при dry_run только печатает пути, возвращает None."""
    data = data.resolve()
    if data.suffix.lower() != '.jsonl':
        raise ValueError('Для удаления разметки --data должен указывать на файл .jsonl')
    paths = (data, data.with_name(data.stem + '_summary.json'))
    for path in paths:
        if path.exists() and not path.is_file():
            raise ValueError(f'Вместо файла разметки обнаружена папка: {path}')
    for path in paths:
        if dry_run:
            print(f'Будет удалён перед разметкой: {path}', flush=True)
        elif path.exists():
            path.unlink()
            print(f'Удалён файл разметки: {path}', flush=True)
        else:
            print(f'Файл отсутствует, удаление не требуется: {path}', flush=True)


def main(argv=None):
    """Проверяет этапы argv, при запросе удаляет разметку и выполняет цепочку; возвращает код."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    args = resolve_profile(build_parser().parse_args(argv))
    try:
        commands = build_commands(args)
        stages = {stage for stage, _, _ in commands}
        run = args.run.resolve()
        if not args.dry_run and 4 in stages and run.exists():
            if any(path.name != 'data_report.json' for path in run.iterdir()):
                raise ValueError('Каталог уже содержит обучение; задайте новый --run '
                                 'или продолжите с --from-stage 5')
        if not args.dry_run and 5 in stages and (run / 'export').exists():
            raise ValueError('Экспорт уже существует; продолжите с --from-stage 6')
        if args.delete_labels:
            delete_label_files(args.data, args.dry_run)
        for stage, name, command in commands:
            print(f'\nЭтап {stage}: {name}\n{subprocess.list2cmdline(command)}', flush=True)
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
        print('Команды показаны; выполнение не запускалось.' if args.dry_run
              else 'Выбранные этапы успешно завершены.', flush=True)
        return 0
    except subprocess.CalledProcessError as error:
        print(f'Оркестратор остановлен: этап завершился с кодом {error.returncode}. '
              'Следующие этапы не запускались.', file=sys.stderr)
        return error.returncode if error.returncode > 0 else 1
    except (ValueError, OSError) as error:
        print(f'Ошибка оркестратора: {error}', file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Оркестратор остановлен пользователем.', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
