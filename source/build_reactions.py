r"""Создаёт таблицу реакций MIX/RTS на новости и минутный кэш для finplot.

Берёт текущую выборку пилота с готовыми эмбеддингами, читает дневные ZIP, считает
реакции 5/15/30/60/120 минут по loaded_at. Сохраняет все события, включая неполные,
в отдельные SQLite/CSV. Исходные архивы и БД эмбеддингов не изменяются.
Готовая SQLite заменяется только после успешного полного расчёта.

Примеры из корня проекта:
  .\.venv\Scripts\python.exe build_reactions.py
  .\.venv\Scripts\python.exe build_reactions.py --latency-seconds 5
  .\.venv\Scripts\python.exe build_reactions.py --mix-dir C:\data_quote\data_finam_MIX_tick_zip --rts-dir C:\data_quote\data_finam_RTS_tick_zip
  .\.venv\Scripts\python.exe build_reactions.py --all-news --month 2026-09
Все параметры: --help. Прогресс в консоли, файловые логи не создаются.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import pandas as pd

from source.embedding_pilot import load_news
from source.embedding_report import write_csv
from source.price_data import DEFAULT_TICK_DIRS, minute_bars, read_ticks
from source.price_reactions import HORIZONS, METRIC_VERSION, measure_reaction

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / 'data' / 'news_reactions'


def selected_news(pilot_dir):
    """Читает ровно текущую выборку из pilot_dir; возвращает новости и corpus_id."""
    summary = json.loads((pilot_dir / 'summary.json').read_text(encoding='utf-8'))
    with closing(sqlite3.connect((pilot_dir / 'pilot.sqlite3').resolve().as_uri() + '?mode=ro', uri=True)) as c:
        rows = [json.loads(row[0]) for row in c.execute(
            'SELECT n.payload FROM news n JOIN samples s USING(news_id) WHERE s.corpus_id=?',
            (summary['corpus_id'],))]
    if len(rows) != summary['sample_size']:
        raise ValueError('Размер сохранённой выборки не соответствует summary.json')
    return sorted(rows, key=lambda r: (r['loaded_at'], r['news_id'])), summary['corpus_id']


def create_store(path):
    """Создаёт выходную SQLite path; возвращает соединение с таблицами данных и метрик."""
    c = sqlite3.connect(path)
    c.executescript('''
        CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE news(news_id TEXT PRIMARY KEY,loaded_at TEXT,title TEXT,provider TEXT,payload TEXT);
        CREATE TABLE archives(instrument TEXT,day TEXT,status TEXT,payload TEXT,PRIMARY KEY(instrument,day));
        CREATE TABLE minute_bars(instrument TEXT,datetime TEXT,open REAL,high REAL,low REAL,close REAL,
                                 volume REAL,tick_count INTEGER,PRIMARY KEY(instrument,datetime));
        CREATE TABLE reactions(news_id TEXT,instrument TEXT,horizon INTEGER,status TEXT,
                               stable INTEGER,points REAL,payload TEXT,PRIMARY KEY(news_id,instrument,horizon));
        CREATE INDEX reaction_selection ON reactions(instrument,horizon,status,stable);
    ''')
    return c


def build(rows, tick_dirs, output, config):
    """Рассчитывает rows по tick_dirs с config, пишет SQLite/CSV в output; возвращает сводку.

    Временная SQLite имеет постоянное имя reactions.build.sqlite3. Существующий
    готовый файл не меняется при ошибке расчёта; все неполные реакции сохраняются.
    """
    output.mkdir(parents=True, exist_ok=True)
    pending = output / 'reactions.build.sqlite3'
    if pending.exists():
        pending.unlink()
    c = create_store(pending)
    all_reactions = []
    coverage = []
    groups = {}
    for row in rows:
        groups.setdefault(row['loaded_at'][:10], []).append(row)
    try:
        c.executemany('INSERT INTO news VALUES(?,?,?,?,?)',
                      [(r['news_id'], r['loaded_at'], r['title'], r['provider'], json.dumps(r, ensure_ascii=False))
                       for r in rows])
        for instrument, folder in tick_dirs.items():
            for day, events in sorted(groups.items()):
                archive = folder / (day.replace('-', '') + '.zip')
                ticks = bars = None
                status, audit = 'missing_archive', {'path': str(archive)}
                if archive.exists():
                    try:
                        ticks, audit = read_ticks(archive)
                        bars = minute_bars(ticks)
                        status = 'ok'
                    except (ValueError, OSError, zipfile.BadZipFile, pd.errors.ParserError) as error:
                        status, audit = 'archive_error', {'path': str(archive), 'error': str(error)}
                c.execute('INSERT INTO archives VALUES(?,?,?,?)',
                          (instrument, day, status, json.dumps(audit, ensure_ascii=False)))
                coverage.append({'instrument': instrument, 'day': day, 'status': status, **audit})
                if bars is not None:
                    c.executemany('INSERT INTO minute_bars VALUES(?,?,?,?,?,?,?,?)',
                                  [(instrument, str(ts), float(b.open), float(b.high), float(b.low),
                                    float(b.close), float(b.volume), int(b.tick_count)) for ts, b in bars.iterrows()])
                for event in events:
                    for horizon in HORIZONS:
                        if ticks is None:
                            result = {'status': status, 'stable': False, 'points': None,
                                      'direction': None, 'flags': [status, 'continuous_contract_unverified']}
                        else:
                            result = measure_reaction(
                                ticks, bars, event['loaded_at'], horizon,
                                neutral_points=config['neutral_points'][instrument],
                                latency_seconds=config['latency_seconds'],
                                max_entry_wait_seconds=config['max_entry_wait_seconds'],
                                max_exit_age_seconds=config['max_exit_age_seconds'],
                                max_gap_seconds=config['max_gap_seconds'])
                        result.update(news_id=event['news_id'], instrument=instrument, horizon=horizon,
                                      loaded_at=event['loaded_at'], title=event['title'], provider=event['provider'])
                        c.execute('INSERT INTO reactions VALUES(?,?,?,?,?,?,?)',
                                  (event['news_id'], instrument, horizon, result['status'],
                                   int(result['stable']), result['points'], json.dumps(result, ensure_ascii=False)))
                        all_reactions.append(result)
                c.commit()
                print(f'{instrument} {day}: {status}, новостей {len(events)}, '
                      f'тиков {len(ticks) if ticks is not None else 0}', flush=True)
        counts = Counter(r['status'] for r in all_reactions)
        stable_counts = Counter(f'{r["instrument"]}_{r["horizon"]}' for r in all_reactions if r['stable'])
        manifest_hash = hashlib.sha256(json.dumps(coverage, sort_keys=True).encode()).hexdigest()
        summary = {'config': config, 'news_count': len(rows), 'reaction_count': len(all_reactions),
                   'status_counts': dict(counts), 'stable_counts': dict(stable_counts),
                   'source_manifest_sha256': manifest_hash, 'created_at': pd.Timestamp.now(tz='Europe/Moscow').isoformat(),
                   'archive_status_counts': dict(Counter(r['status'] for r in coverage))}
        c.execute('INSERT INTO metadata VALUES(?,?)', ('summary', json.dumps(summary, ensure_ascii=False)))
        c.commit()
        if c.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('SQLite не прошла integrity_check')
    finally:
        c.close()
    pending.replace(output / 'reactions.sqlite3')
    export_results(output, rows, all_reactions, coverage, summary)
    return summary


def export_results(output, rows, reactions, coverage, summary):
    """Экспортирует reactions и coverage в CSV/JSON, включая широкую таблицу по rows."""
    columns = ['news_id', 'instrument', 'horizon', 'loaded_at', 'title', 'provider', 'status',
               'entry_at', 'entry_price', 'end_at', 'exit_at', 'exit_price', 'points', 'return_pct',
               'direction', 'mfe_points', 'mae_points', 'max_pullback_points', 'max_up_points',
               'max_down_points', 'er', 'direction_fraction', 'sigma_pre', 'z_score', 'stable', 'flags']
    exported = [{**r, 'flags': '|'.join(r['flags'])} for r in reactions]
    write_csv(output / 'reactions_long.csv', exported, columns)
    indexed = {(r['news_id'], r['instrument'], r['horizon']): r for r in reactions}
    wide = []
    for event in rows:
        for instrument in ('MIX', 'RTS'):
            item = {key: event[key] for key in ('news_id', 'loaded_at', 'title', 'provider')}
            item['instrument'] = instrument
            for h in HORIZONS:
                r = indexed[(event['news_id'], instrument, h)]
                for key in ('points', 'return_pct', 'direction', 'status', 'stable', 'er',
                            'mfe_points', 'mae_points', 'max_pullback_points', 'z_score'):
                    item[f'{key}_{h}'] = r.get(key)
            wide.append(item)
    write_csv(output / 'news_reactions.csv', wide, list(wide[0]) if wide else ['news_id'])
    write_csv(output / 'coverage.csv', coverage, ['instrument', 'day', 'status', 'rows', 'first', 'last', 'sha256', 'path', 'error'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    """Разбирает параметры CLI и запускает сборку; печатает итоговые счётчики."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pilot-dir', type=Path, default=ROOT / 'data' / 'embedding_pilot', help='Текущий пилот эмбеддингов')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Выходная папка')
    p.add_argument('--mix-dir', type=Path, default=DEFAULT_TICK_DIRS['MIX'], help='Архивы MIX')
    p.add_argument('--rts-dir', type=Path, default=DEFAULT_TICK_DIRS['RTS'], help='Архивы RTS')
    p.add_argument('--latency-seconds', type=float, default=0., help='Задержка после получения новости')
    p.add_argument('--max-gap-seconds', type=float, default=120., help='Максимальный пробел между сделками')
    p.add_argument('--max-entry-wait-seconds', type=float, default=60., help='Максимальное ожидание первой сделки')
    p.add_argument('--max-exit-age-seconds', type=float, default=60., help='Максимальная давность цены выхода')
    p.add_argument('--all-news', action='store_true', help='Весь месяц новостей вместо выборки эмбеддингов')
    p.add_argument('--month', default='2026-09', help='Месяц для --all-news')
    p.add_argument('--news-dir', type=Path, default=Path(r'C:\Users\Alkor\gd\db_rss'), help='Исходные новости для --all-news')
    args = p.parse_args()
    if min(args.latency_seconds, args.max_gap_seconds, args.max_entry_wait_seconds, args.max_exit_age_seconds) < 0:
        p.error('Временные параметры должны быть неотрицательными')
    if args.all_news:
        rows, _ = load_news(args.news_dir, args.month)
        corpus_id = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    else:
        rows, corpus_id = selected_news(args.pilot_dir)
    if not rows:
        p.error('Нет новостей для расчёта')
    config = {'metric_version': METRIC_VERSION, 'corpus_id': corpus_id, 'all_news': args.all_news,
              'pilot_db': str((args.pilot_dir / 'pilot.sqlite3').resolve()), 'horizons': HORIZONS,
              'latency_seconds': args.latency_seconds, 'max_gap_seconds': args.max_gap_seconds,
              'max_entry_wait_seconds': args.max_entry_wait_seconds, 'max_exit_age_seconds': args.max_exit_age_seconds,
              'neutral_points': {'MIX': 50., 'RTS': 20.}, 'time_field': 'loaded_at',
              'timezone': 'Europe/Moscow', 'loaded_at_timezone_confirmed': False,
              'tick_timezone_evidence': 'Загрузчик quote_download/FINAM_quote_downloader: mstime=on',
              'stable_thresholds': {'z': 2., 'er': .45, 'direction_fraction': .7, 'mae_fraction': .5, 'pullback_fraction': .5}}
    summary = build(rows, {'MIX': args.mix_dir, 'RTS': args.rts_dir}, args.output, config)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
