r"""Строит оценки новостей по пяти прошлым дням и динамическому фону.

Читает реакции и готовые эмбеддинги, дополняет фон предыдущими архивами Finam.
Порог каждой новости использует только предшествующие дни и завершённые минуты.
Ранжирование аналогов не использует знак/силу исходов; текущий исход добавляется
после решения. SQLite хранит все отказы, пять соседей и происхождение порогов.
Кэш фона проверяется по SHA256 архива и параметрам расчёта. Логов нет.
Если новости содержат relevance, аналоги ограничиваются общими category:event:object:stage.
--history-years задаёт скользящую историю, --query-start/--query-end отделяют период оценки.
--threshold-multiplier задаёт k для T=k*T_base, не меняя набор пяти аналогов.

Запуск из корня:
  .\.venv\Scripts\python.exe build_analogs.py
  .\.venv\Scripts\python.exe build_analogs.py --quantile 0.75 --min-similarity 0.5
  .\.venv\Scripts\python.exe build_analogs.py --threshold-multiplier 0.5
  .\.venv\Scripts\python.exe build_analogs.py --history-years 1 --query-start 2026-09-01 --query-end 2026-10-01
Все параметры: build_analogs.py --help. Результаты: data/analog_analysis.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from source.analog_selection import assess_neighbors, pick_neighbors
from source.build_reactions import DEFAULT_OUTPUT as REACTIONS
from source.dynamic_threshold import BackgroundIndex, dynamic_threshold
from source.embedding_pilot import PREPARATION, ROOT, cached_vectors
from source.embedding_report import write_csv
from source.price_data import DEFAULT_TICK_DIRS, minute_bars, read_ticks
from source.price_reactions import HORIZONS, METRIC_VERSION, measure_reaction, pre_volatility
from source.market_relevance import compatible_groups
from source.threshold_tuning import rescale_score, validate_multiplier

DEFAULT_OUTPUT = ROOT / 'data' / 'analog_analysis'


def digest_file(path):
    """Возвращает SHA256 содержимого path, читая файл блоками по одному мегабайту."""
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def readonly(path):
    """Открывает SQLite path только для чтения; соединение закрывает вызывающий."""
    return sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)


def collect_background(folder, instrument, first_day, last_day, output, config, step, lookback):
    """Возвращает фон instrument из folder с кэшем output и правилами config.

    Берёт lookback доступных архивов до first_day и все до last_day. Фон по сетке
    step минут использует те же тиковые границы и ограничения пропусков, что новости.
    Неудачные архивы прерывают расчёт, чтобы ошибка не выглядела отсутствием данных.
    """
    paths = sorted(p for p in folder.glob('????????.zip') if p.stem.isdigit())
    previous = [p for p in paths if p.stem < first_day.replace('-', '')][-lookback:]
    paths = previous + [p for p in paths if first_day.replace('-', '') <= p.stem <= last_day.replace('-', '')]
    settings = {k: config[k] for k in ('latency_seconds', 'max_entry_wait_seconds', 'max_exit_age_seconds', 'max_gap_seconds')}
    cache_key = hashlib.sha256(json.dumps([METRIC_VERSION, settings, step, list(HORIZONS)], sort_keys=True).encode()).hexdigest()
    with closing(sqlite3.connect(output / 'background.sqlite3')) as cache:
        cache.execute('CREATE TABLE IF NOT EXISTS archives(instrument TEXT,day TEXT,fingerprint TEXT,payload TEXT,PRIMARY KEY(instrument,day))')
        rows = []
        for number, path in enumerate(paths, 1):
            fingerprint = cache_key + ':' + digest_file(path)
            old = cache.execute('SELECT fingerprint,payload FROM archives WHERE instrument=? AND day=?', (instrument, path.stem)).fetchone()
            if old and old[0] == fingerprint:
                found = json.loads(old[1])
            else:
                ticks, _ = read_ticks(path)
                bars = minute_bars(ticks)
                found = []
                for anchor in pd.date_range(bars.index[0].ceil(f'{step}min'), bars.index[-1], freq=f'{step}min'):
                    if pre_volatility(bars, anchor) is None:
                        continue
                    for horizon in HORIZONS:
                        r = measure_reaction(ticks, bars, str(anchor), horizon, **settings)
                        if r['status'] == 'ok' and r['sigma_pre']:
                            found.append(dict(day=anchor.strftime('%Y-%m-%d'), start_at=str(anchor),
                                              end_at=r['end_at'], horizon=horizon,
                                              return_frac=r['return_pct'] / 100, sigma_pre=r['sigma_pre']))
                cache.execute('INSERT OR REPLACE INTO archives VALUES(?,?,?,?)',
                              (instrument, path.stem, fingerprint, json.dumps(found)))
                cache.commit()
            rows.extend(found)
            if number % 5 == 0 or number == len(paths):
                print(f'Фон {instrument}: {number}/{len(paths)} архивов, {len(rows)} интервалов', flush=True)
    return rows


def make_decision(candidates, loaded_at, threshold, min_similarity, query_relevance=None, history_years=None):
    """Возвращает прогноз по candidates/threshold без исхода запроса.

    query_relevance при наличии ограничивает кандидатов совместимыми группами;
    отсутствие классификации сохраняет прежний режим без тематического фильтра.
    history_years — календарное окно поиска, None означает всю доступную историю.
    """
    if query_relevance is not None:
        candidates = [r for r in candidates if compatible_groups(query_relevance, r.get('relevance', {}))]
    neighbors = pick_neighbors(candidates, loaded_at, history_years=history_years)
    result = assess_neighbors(neighbors, min_similarity)
    if threshold['status'] != 'ok':
        result.update(selected=False, status='query_threshold_unavailable', forecast_direction=None)
    return {**result, 'neighbors': neighbors, 'threshold': threshold}


def query_indices(news, start=None, end=None):
    """Возвращает индексы запросов news за [start,end); None снимает соответствующую границу."""
    if start and end and start >= end:
        raise ValueError('Начало периода оценки должно предшествовать концу')
    return [i for i, n in enumerate(news) if (not start or n['loaded_at'] >= start) and (not end or n['loaded_at'] < end)]


def model_manifest(pilot, manifest, news, corpus_id, requested):
    """Проверяет тексты/состав/подготовку и возвращает единственный digest каждого requested.

    pilot — соединение кэша, manifest — summary.json пилота; news — новости реакций.
    Исторические версии из models не подменяют digest текущего manifest.
    Несовпадение входов прерывает расчёт до чтения архивов.
    """
    if manifest['corpus_id'] != corpus_id or manifest['preparation'] != PREPARATION:
        raise ValueError('Выборка/подготовка эмбеддингов не соответствует реакциям')
    texts = {key: json.loads(payload)['title'] for key, payload in pilot.execute('SELECT news_id,payload FROM news')}
    sample = {r[0] for r in pilot.execute('SELECT news_id FROM samples WHERE corpus_id=?', (corpus_id,))}
    if sample != {n['news_id'] for n in news} or any(texts.get(n['news_id']) != n['title'] for n in news):
        raise ValueError('Тексты или состав новостей изменились: пересоберите реакции')
    versions = []
    for model in dict.fromkeys(requested):
        if model not in manifest['models']:
            raise ValueError(f'В текущем пилоте нет модели {model}')
        digest = manifest['models'][model]['current_run']['digest']
        if pilot.execute('SELECT name FROM models WHERE digest=?', (digest,)).fetchone() != (model,):
            raise ValueError(f'Метаданные версии модели не совпали: {model}')
        vectors = cached_vectors(pilot, digest)
        if any(n['news_id'] not in vectors for n in news):
            raise ValueError(f'Не хватает эмбеддингов {model}')
        versions.append((digest, model))
    return versions


def build(args):
    """Рассчитывает фон, пороги и оценки по CLI args, атомарно публикует SQLite и CSV."""
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    reaction_sha = digest_file(args.reaction_db)
    with closing(readonly(args.reaction_db)) as source:
        news = [json.loads(r[0]) for r in source.execute('SELECT payload FROM news ORDER BY loaded_at,news_id')]
        reactions = {(r['news_id'], r['instrument'], r['horizon']): r for (payload,) in source.execute('SELECT payload FROM reactions') for r in [json.loads(payload)]}
        config = json.loads(source.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])['config']
        bars_by_inst = {}
        for instrument in DEFAULT_TICK_DIRS:
            bars = pd.read_sql_query('SELECT datetime,close FROM minute_bars WHERE instrument=? ORDER BY datetime', source, params=(instrument,))
            bars.index = pd.DatetimeIndex(pd.to_datetime(bars.pop('datetime')))
            bars_by_inst[instrument] = bars
    options = {k: getattr(args, k) for k in ('quantile', 'lookback_days', 'time_band_minutes', 'min_samples', 'min_days', 'volatility_fraction')}
    manifest = json.loads((args.pilot_dir / 'summary.json').read_text(encoding='utf-8'))
    indices = query_indices(news, getattr(args, 'query_start', None), getattr(args, 'query_end', None))
    if not indices:
        raise ValueError('В периоде оценки нет новостей')
    history_years = getattr(args, 'history_years', None)
    threshold_multiplier = validate_multiplier(getattr(args, 'threshold_multiplier', 1.))
    with closing(readonly(args.pilot_dir / 'pilot.sqlite3')) as pilot:
        versions = model_manifest(pilot, manifest, news, config['corpus_id'], args.models)
    backgrounds = {}
    for instrument, folder in [('MIX', args.mix_dir), ('RTS', args.rts_dir)]:
        backgrounds[instrument] = BackgroundIndex(collect_background(folder, instrument, news[0]['loaded_at'][:10],
            news[-1]['loaded_at'][:10], output, config, args.background_step, args.lookback_days))
    thresholds = {}
    for instrument, bars in bars_by_inst.items():
        for n in news:
            when = pd.Timestamp(n['loaded_at'])
            sigma = pre_volatility(bars, when)
            i = bars.index.searchsorted(when.floor('min'), side='left') - 1
            reference = float(bars['close'].iloc[i]) if i >= 0 else None
            for horizon in HORIZONS:
                thresholds[n['news_id'], instrument, horizon] = dynamic_threshold(
                    backgrounds[instrument], n['loaded_at'], sigma, reference, horizon, **options)
    print('Динамические пороги рассчитаны', flush=True)
    pending = output / 'analogs.build.sqlite3'
    if pending.exists():
        pending.unlink()
    summary = {'reaction_sha256': reaction_sha, 'preparation': PREPARATION, 'corpus_id': config['corpus_id'],
               'threshold_options': options, 'background_step': args.background_step,
               'min_similarity': args.min_similarity, 'news_count': len(news), 'models': {}, 'groups': []}
    summary['relevance_version'] = manifest.get('relevance_version')
    summary.update(query_count=len(indices), query_start=getattr(args, 'query_start', None),
                   query_end_exclusive=getattr(args, 'query_end', None), history_years=history_years,
                   threshold_multiplier=threshold_multiplier)
    if summary['relevance_version'] and any(not n.get('relevance') or n['relevance']['rule_version'] != summary['relevance_version']
                                           or n['relevance']['decision'] != 'include' for n in news):
        raise ValueError('Отсутствует или не совпадает классификация российского рынка')
    csv_rows = []
    with closing(sqlite3.connect(pending)) as store, closing(readonly(args.pilot_dir / 'pilot.sqlite3')) as pilot:
        store.executescript('''CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT);
            CREATE TABLE news(news_id TEXT PRIMARY KEY,payload TEXT);
            CREATE TABLE thresholds(news_id TEXT,instrument TEXT,horizon INTEGER,payload TEXT,PRIMARY KEY(news_id,instrument,horizon));
            CREATE TABLE scores(news_id TEXT,instrument TEXT,horizon INTEGER,model_digest TEXT,model TEXT,status TEXT,selected INTEGER,payload TEXT,
                PRIMARY KEY(news_id,instrument,horizon,model_digest));
            CREATE INDEX selection ON scores(instrument,horizon,model,selected);''')
        store.executemany('INSERT INTO news VALUES(?,?)', [(n['news_id'], json.dumps(n, ensure_ascii=False)) for n in news])
        store.executemany('INSERT INTO thresholds VALUES(?,?,?,?)', [(*key, json.dumps(value)) for key, value in thresholds.items()])
        for digest, model in versions:
            vectors = cached_vectors(pilot, digest)
            matrix = np.asarray([vectors[n['news_id']] for n in news], dtype=np.float32)
            similarity = np.clip(matrix[indices] @ matrix.T, -1, 1)
            summary['models'][model] = digest
            for instrument in DEFAULT_TICK_DIRS:
                for horizon in HORIZONS:
                    counts, selected, valid, correct, strong = Counter(), 0, 0, 0, 0
                    candidates = []
                    for n in news:
                        r = reactions[n['news_id'], instrument, horizon]
                        t = thresholds[n['news_id'], instrument, horizon]
                        candidates.append(dict(news_id=n['news_id'], loaded_at=n['loaded_at'],
                            end_at=r.get('end_at'), status=r['status'], points=r.get('points'),
                            return_frac=(r['return_pct'] / 100 if r.get('return_pct') is not None else None),
                            threshold_status=t['status'], threshold_frac=t['threshold_frac'],
                            threshold_points=t['threshold_points'], threshold_history_max_end=t['history_max_end'],
                            relevance=n.get('relevance')))
                    for query_row, qi in enumerate(indices):
                        n = news[qi]
                        past = [dict(r, similarity=float(similarity[query_row, i])) for i, r in enumerate(candidates)]
                        score = make_decision(past, n['loaded_at'], thresholds[n['news_id'], instrument, horizon], args.min_similarity, n.get('relevance'), history_years)
                        actual = reactions[n['news_id'], instrument, horizon]
                        score.update(news_id=n['news_id'], loaded_at=n['loaded_at'], instrument=instrument,
                                     horizon=horizon, model=model, actual_status=actual['status'], actual_points=actual.get('points'),
                                     actual_direction=None, actual_ratio=None, direction_correct=None, strong_correct=None)
                        score['relevance'] = n.get('relevance')
                        if actual['status'] == 'ok' and score['threshold']['status'] == 'ok':
                            score['actual_direction'] = 'up' if actual['points'] > 0 else 'down' if actual['points'] < 0 else 'flat'
                            score['actual_ratio'] = abs(actual['return_pct'] / 100) / score['threshold']['threshold_frac']
                        score = rescale_score(score, threshold_multiplier, args.min_similarity)
                        if score['selected'] and score['direction_correct'] is not None:
                            valid += 1
                            correct += score['direction_correct']
                            strong += score['strong_correct']
                        counts[score['status']] += 1
                        selected += score['selected']
                        store.execute('INSERT INTO scores VALUES(?,?,?,?,?,?,?,?)', (n['news_id'], instrument, horizon, digest, model,
                            score['status'], int(score['selected']), json.dumps(score, ensure_ascii=False)))
                        relevance = score.get('relevance') or {}
                        csv_rows.append({k: v for k, v in score.items() if k not in ('neighbors', 'threshold', 'relevance')} | {
                            'title': n['title'], 'threshold_points': score['threshold']['threshold_points'],
                            'relevance_categories': '|'.join(relevance.get('categories', [])),
                            'relevance_groups': '|'.join(relevance.get('groups', [])), 'relevance_reason': relevance.get('reason', ''),
                            'neighbor_ids': '|'.join(r['news_id'] for r in score['neighbors'])})
                    summary['groups'].append(dict(model=model, instrument=instrument, horizon=horizon, statuses=dict(counts),
                        selected=selected, actual_valid=valid, direction_correct=correct, strong_correct=strong))
                    print(f'{model} {instrument} {horizon}: отобрано {selected}, исход известен {valid}, знак совпал {correct}', flush=True)
                    store.commit()
        if not summary['models']:
            raise ValueError('Нет запрошенных моделей в кэше')
        store.execute('INSERT INTO metadata VALUES(?,?)', ('summary', json.dumps(summary, ensure_ascii=False)))
        store.commit()
    pending.replace(output / 'analogs.sqlite3')
    write_csv(output / 'scores.csv', csv_rows, list(csv_rows[0]))
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return summary


def main():
    """Разбирает аргументы, проверяет параметры и запускает полный расчёт аналогов."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--reaction-db', type=Path, default=REACTIONS / 'reactions.sqlite3')
    p.add_argument('--pilot-dir', type=Path, default=ROOT / 'data' / 'embedding_pilot')
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    p.add_argument('--mix-dir', type=Path, default=DEFAULT_TICK_DIRS['MIX'])
    p.add_argument('--rts-dir', type=Path, default=DEFAULT_TICK_DIRS['RTS'])
    p.add_argument('--models', nargs='+', default=['qwen3-embedding:0.6b', 'bge-m3:latest'])
    p.add_argument('--quantile', type=float, default=.75)
    p.add_argument('--lookback-days', type=int, default=20)
    p.add_argument('--time-band-minutes', type=int, default=60)
    p.add_argument('--min-samples', type=int, default=30)
    p.add_argument('--min-days', type=int, default=5)
    p.add_argument('--volatility-fraction', type=float, default=.5)
    p.add_argument('--background-step', type=int, default=15)
    p.add_argument('--min-similarity', type=float, default=.5)
    p.add_argument('--threshold-multiplier', type=float, default=1., help='Положительный k для динамического порога T=k*T_base')
    p.add_argument('--history-years', type=int, help='Скользящая история в календарных годах, по умолчанию без ограничения')
    p.add_argument('--query-start', help='Начало периода оценки включительно, YYYY-MM-DD')
    p.add_argument('--query-end', help='Конец периода оценки исключительно, YYYY-MM-DD')
    args = p.parse_args()
    if args.history_years is not None and args.history_years < 1:
        p.error('Число лет должно быть положительным')
    if args.background_step < 1 or not -1 <= args.min_similarity <= 1:
        p.error('Шаг должен быть положительным, косинус от -1 до 1')
    dynamic_threshold([], '2026-01-01', None, None, 30, **{k: getattr(args, k) for k in
        ('quantile', 'lookback_days', 'time_band_minutes', 'min_samples', 'min_days', 'volatility_fraction')})
    build(args)


if __name__ == '__main__':
    main()
