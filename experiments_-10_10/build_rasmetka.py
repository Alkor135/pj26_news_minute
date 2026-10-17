r"""Этап 3: размечает реакции MIX и RTS на новости баллами от −10 до +10.

По умолчанию читает RSS за сентябрь 2025 — август 2026, применяет текущий
тематический фильтр проекта, вызывает общий расчёт ценовых реакций и сохраняет
JSONL для независимых скриптов дообучения (этап 4). При --reaction-db повторно
использует готовую БД реакций, проверяя версию расчёта и параметры. Исходные
RSS SQLite, архивы котировок и готовая БД реакций открываются только для чтения.
Неполные окна сохраняются со score=null; файловых логов скрипт не создаёт.

Примеры запуска из корня проекта:
  .\.venv\Scripts\python.exe experiments_-10_10/build_rasmetka.py
  .\.venv\Scripts\python.exe experiments_-10_10/build_rasmetka.py --reaction-db data\russian_market\2026-09_year1\reactions\reactions.sqlite3
  .\.venv\Scripts\python.exe experiments_-10_10/build_rasmetka.py --start 2025-09-01 --end-exclusive 2026-09-01
  .\.venv\Scripts\python.exe experiments_-10_10/build_rasmetka.py --help
Параметры и формат результата описаны в соседнем README.md.
"""

import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import sys
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from source.build_reactions import build as build_reactions
from source.market_relevance import RULE_VERSION, classify_title
from source.news_history import load_history
from source.price_data import DEFAULT_TICK_DIRS
from source.price_reactions import HORIZONS, METRIC_VERSION

DEFAULT_OUTPUT = ROOT / 'data' / 'experiments_-10_10' / 'scores.jsonl'
SCORE_VERSION = 'magnitude-quality-v1'
NEUTRAL_Z = 0.25
NEUTRAL_POINTS = {'MIX': 50.0, 'RTS': 20.0}


def finite_number(value):
    """Возвращает конечное вещественное число из value либо None при пропуске/ошибке."""
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def score_reaction(row):
    """Возвращает (балл или None, причина, сила, качество) по согласованной шкале.

    row содержит результат measure_reaction. ±10 выдаются лишь при исходном
    stable и выполнении всех порогов; слабые/неустойчивые реакции остаются в
    диапазоне ±1…9. Неполные и невалидные метрики не заменяются нулём.
    """
    if row.get('status') != 'ok':
        return None, str(row.get('status') or 'unknown_status'), None, None
    if not isinstance(row.get('stable'), bool):
        raise ValueError('stable должен быть логическим значением')
    points = finite_number(row.get('points'))
    z = finite_number(row.get('z_score'))
    if points is None or z is None or z < 0:
        return None, 'missing_or_invalid_magnitude', None, None
    magnitude = abs(points)
    if magnitude == 0 and z > 0:
        return None, 'inconsistent_zero_movement', None, None
    strength = min(1.0, z / 2.0)
    if row['stable'] and (magnitude == 0 or z < 2 or
                          row.get('direction') != ('up' if points > 0 else 'down')):
        raise ValueError('stable противоречит направлению, итоговому движению или z_score')
    if not row['stable'] and z <= NEUTRAL_Z:
        return 0, 'neutral', strength, None
    values = [finite_number(row.get(key)) for key in
              ('er', 'direction_fraction', 'mae_points', 'max_pullback_points')]
    if (any(value is None for value in values) or not 0 <= values[0] <= 1
            or not 0 <= values[1] <= 1 or values[2] < 0 or values[3] < 0):
        if row['stable']:
            raise ValueError('stable имеет неполные или недопустимые метрики качества')
        return None, 'missing_or_invalid_quality', strength, None
    er, direction_fraction, mae, pullback = values
    quality = min(1.0, er / 0.45, direction_fraction / 0.70,
                  1.0 if mae == 0 else magnitude / (2 * mae),
                  1.0 if pullback == 0 else magnitude / (2 * pullback))
    if row['stable']:
        if (er < 0.45 or direction_fraction < 0.70 or
                mae > 0.5 * magnitude or pullback > 0.5 * magnitude):
            raise ValueError('stable противоречит метрикам качества')
        return 10 if points > 0 else -10, 'stable', strength, quality
    if magnitude == 0:
        return None, 'inconsistent_zero_movement', strength, quality
    unsigned = min(9, max(1, math.floor(10 * strength * quality + 0.5)))
    return unsigned if points > 0 else -unsigned, 'intermediate', strength, quality


def group_for_title(title):
    """Возвращает стабильный идентификатор точных перепечаток title после нормализации."""
    normalized = ' '.join(unicodedata.normalize('NFKC', title).casefold().split())
    return 'title-sha256:' + hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def check_reaction_config(config):
    """Проверяет совместимость параметров готовой БД с версией этой разметки."""
    if config.get('metric_version') != METRIC_VERSION:
        raise ValueError('Версия ценовых метрик в БД отличается от текущей')
    if config.get('relevance_version') not in (None, RULE_VERSION):
        raise ValueError('Версия фильтра новостей в БД отличается от текущей')
    if tuple(config.get('horizons', ())) != HORIZONS:
        raise ValueError('Горизонты в БД отличаются от 5/15/30/60/120 минут')
    if config.get('neutral_points') != NEUTRAL_POINTS or config.get('latency_seconds') != 0.0:
        raise ValueError('Параметры stable/входа в БД отличаются от параметров разметки')


def read_reactions(database, start, end_exclusive):
    """Читает в [start,end_exclusive) полные и неполные реакции из database только на чтение.

    Повторно проверяет тематический фильтр по заголовку, целостность паспорта
    расчёта и согласованность payload с индексными полями SQLite.
    """
    uri = Path(database).resolve().as_uri() + '?mode=ro'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        metadata = connection.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
        if metadata is None:
            raise ValueError('В БД реакций нет паспорта расчёта')
        check_reaction_config(json.loads(metadata[0])['config'])
        cursor = connection.execute('''
            SELECT n.news_id,n.loaded_at,n.title,n.provider,
                   r.instrument,r.horizon,r.status,r.stable,r.points,r.payload
            FROM news AS n JOIN reactions AS r USING(news_id)
            WHERE n.loaded_at >= ? AND n.loaded_at < ?
            ORDER BY n.loaded_at,n.news_id,r.instrument,r.horizon
        ''', (start, end_exclusive))
        for news_id, loaded_at, title, provider, instrument, horizon, status, stable, points, payload in cursor:
            if classify_title(title)['decision'] != 'include':
                continue
            if instrument not in NEUTRAL_POINTS or horizon not in HORIZONS:
                raise ValueError(f'{news_id}: неизвестный инструмент или горизонт')
            reaction = json.loads(payload)
            if (reaction.get('news_id') != news_id or reaction.get('loaded_at') != loaded_at or
                    reaction.get('title') != title or reaction.get('instrument') != instrument or
                    reaction.get('horizon') != horizon or reaction.get('status') != status or
                    reaction.get('stable') != bool(stable) or reaction.get('points') != points):
                raise ValueError(f'{news_id}: payload реакции не совпадает с индексом SQLite')
            yield reaction


def prepare_news(news_dir, start, end_exclusive):
    """Загружает RSS и возвращает новости с решением include и статистику отбора."""
    rows, source_stats = load_history(Path(news_dir), start, end_exclusive)
    included = [row for row in rows if classify_title(row['title'])['decision'] == 'include']
    if not included:
        raise ValueError('За период нет новостей с решением include')
    return included, dict(source=source_stats, unique_news=len(rows), included_news=len(included),
                          excluded_or_uncertain=len(rows) - len(included))


def calculate_reactions(news_dir, tick_dirs, output, start, end_exclusive):
    """Рассчитывает реакции через общие модули и возвращает путь БД и статистику новостей."""
    rows, selection = prepare_news(news_dir, start, end_exclusive)
    config = dict(metric_version=METRIC_VERSION, relevance_version=RULE_VERSION,
                  horizons=HORIZONS, neutral_points=NEUTRAL_POINTS, latency_seconds=0.0,
                  max_gap_seconds=120.0, max_entry_wait_seconds=60.0,
                  max_exit_age_seconds=60.0, time_field='loaded_at', timezone='Europe/Moscow',
                  loaded_at_timezone_confirmed=False,
                  tick_timezone_evidence='Загрузчик quote_download/FINAM_quote_downloader: mstime=on',
                  stable_thresholds=dict(z=2.0, er=0.45, direction_fraction=0.70,
                                         mae_fraction=0.5, pullback_fraction=0.5))
    folder = output.parent / 'reactions'
    summary = build_reactions(rows, tick_dirs, folder, config)
    return folder / 'reactions.sqlite3', selection | {'reaction_summary': summary}


def label_row(reaction):
    """Возвращает строку JSONL для обучения вместе с метриками и причиной метки."""
    score, reason, strength, quality = score_reaction(reaction)
    keys = ('news_id', 'title', 'loaded_at', 'end_at', 'instrument', 'horizon',
            'status', 'stable', 'provider', 'entry_at', 'entry_price', 'exit_at',
            'exit_price', 'points', 'return_pct', 'direction', 'mfe_points',
            'mae_points', 'max_pullback_points', 'max_up_points', 'max_down_points',
            'er', 'direction_fraction', 'sigma_pre', 'z_score', 'flags')
    result = {key: reaction.get(key) for key in keys}
    result.update(score=score, score_version=SCORE_VERSION,
                  group_id=group_for_title(reaction['title']), label_reason=reason,
                  strength=strength, quality=quality,
                  metric_version=METRIC_VERSION, relevance_version=RULE_VERSION)
    return result


def write_output(reactions, output, source_info):
    """Записывает размеченный JSONL и сводку через временные файлы, возвращает сводку.

    Все реакции обрабатываются до замены готовых файлов; неполные строки
    сохраняются с score=null. Файлы .build имеют постоянные имена.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_name(output.name + '.build')
    summary_path = output.with_name(output.stem + '_summary.json')
    report_pending = summary_path.with_name(summary_path.name + '.build')
    scores, missing, tasks, statuses = Counter(), Counter(), Counter(), Counter()
    count, news = 0, set()
    try:
        with pending.open('w', encoding='utf-8', newline='\n') as stream:
            for reaction in reactions:
                row = label_row(reaction)
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                count += 1
                news.add(row['news_id'])
                tasks[f"{row['instrument']}_{row['horizon']}"] += 1
                statuses[row['status']] += 1
                if row['score'] is None:
                    missing[row['label_reason']] += 1
                else:
                    scores[str(row['score'])] += 1
        if count == 0:
            raise ValueError('За заданный период нет реакций для разметки')
        with pending.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        summary = dict(stage=3, score_version=SCORE_VERSION, metric_version=METRIC_VERSION,
                       relevance_version=RULE_VERSION, neutral_z=NEUTRAL_Z,
                       reactions=count, news=len(news), scores=dict(sorted(scores.items())),
                       missing_labels=dict(sorted(missing.items())),
                       statuses=dict(sorted(statuses.items())), tasks=dict(sorted(tasks.items())),
                       jsonl_sha256=digest, source=source_info)
        report_pending.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + '\n',
                                  encoding='utf-8')
        pending.replace(output)
        report_pending.replace(summary_path)
        return summary
    except Exception:
        pending.unlink(missing_ok=True)
        report_pending.unlink(missing_ok=True)
        raise


def build_parser():
    """Создаёт параметры независимого третьего этапа разметки."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', default='2025-09-01', help='Начало RSS-периода, включительно, МСК')
    parser.add_argument('--end-exclusive', default='2026-09-01', help='Конец RSS-периода, исключительно, МСК')
    parser.add_argument('--news-dir', type=Path, default=Path(r'C:\Users\Alkor\gd\db_rss'), help='Исходные SQLite RSS')
    parser.add_argument('--mix-dir', type=Path, default=DEFAULT_TICK_DIRS['MIX'], help='ZIP котировок MIX')
    parser.add_argument('--rts-dir', type=Path, default=DEFAULT_TICK_DIRS['RTS'], help='ZIP котировок RTS')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Выходной JSONL')
    parser.add_argument('--reaction-db', type=Path, help='Готовая SQLite реакций вместо повторного чтения тиков')
    return parser


def main(argv=None):
    """Выполняет этап 3 по CLI-аргументам argv и возвращает код завершения."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        start, end = datetime.fromisoformat(args.start), datetime.fromisoformat(args.end_exclusive)
        if start.tzinfo or end.tzinfo or not start < end:
            raise ValueError('Нужны возрастающие локальные даты МСК без смещения')
        begin, finish = start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')
        if args.reaction_db is None:
            database, selection = calculate_reactions(
                args.news_dir, {'MIX': args.mix_dir, 'RTS': args.rts_dir}, args.output, begin, finish)
        else:
            database, selection = args.reaction_db, {'mode': 'existing_reactions'}
        source_info = dict(reaction_db=str(database.resolve()), period_start=begin,
                           period_end_exclusive=finish, **selection)
        summary = write_output(read_reactions(database, begin, finish), args.output, source_info)
        print(json.dumps(dict(output=str(args.output.resolve()), summary=summary),
                         ensure_ascii=False, default=str), flush=True)
        return 0
    except (ValueError, OSError, KeyError, sqlite3.Error) as error:
        parser.exit(1, f'Ошибка разметки: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
