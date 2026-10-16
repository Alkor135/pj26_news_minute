r"""Проверяет новости выбранного месяца с поиском аналогов за предыдущий год.

Сначала классифицирует все уникальные заголовки без котировок и сохраняет решения
и слепую выборку для ручной оценки. Векторизует только include, рассчитывает реакции
и пять прошлых аналогов внутри совместимых групп. Старые результаты сохраняются.
Неоднозначные заголовки не дополняются догадками LLM; ручная оценка не подменяется
автоматическими правилами. Файловых логов нет; кэши допускают повторный запуск.
По умолчанию загружает год истории плюс месяц проверки; для каждого запроса
отдельно ограничивает аналоги календарным годом до loaded_at. Повторы объединяет
внутри даты, сохраняя одинаковые заголовки разных дней. --history-years 0 оставляет
прежнюю месячную пробу. Годовые результаты сохраняет в отдельной папке.
--threshold-multiplier меняет строгость готового динамического порога, по умолчанию 1.

Примеры из корня:
  .\.venv\Scripts\python.exe russian_market.py --stage classify
  .\.venv\Scripts\python.exe russian_market.py --month 2026-09 --history-years 1
  .\.venv\Scripts\python.exe russian_market.py --threshold-multiplier 0.5
  .\.venv\Scripts\python.exe russian_market.py --month 2026-09 --history-years 0
  .\.venv\Scripts\python.exe russian_market.py --models qwen3-embedding:0.6b
Параметры: --help. Годовой выход data/russian_market/2026-09_year1.
"""

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import csv
import json
from pathlib import Path
import random
import shutil
from types import SimpleNamespace
from datetime import datetime

from source.build_analogs import build as build_analogs
from source.build_reactions import build as build_reactions
from source.embedding_pilot import ROOT, PREPARATION, DEFAULT_MODELS, load_news, open_cache, save_news, compute_model
from source.embedding_report import write_csv
from source.market_relevance import RULE_VERSION, classify_title
from source.price_data import DEFAULT_TICK_DIRS
from source.price_reactions import HORIZONS, METRIC_VERSION
from source.news_history import load_history, seed_title_vectors
from source.analog_selection import history_start
from source.threshold_tuning import validate_multiplier


def classify_month(news_dir, month, output, history_years=0):
    """Классифицирует month и history_years до него из news_dir; пишет отчёт в output.

    Ручная выборка стратифицирована по решению, без ценовых результатов, seed=42.
    Самооценка правил не считается независимой ручной валидацией. Заполненные поля
    human_* для неизменного текста сохраняются, включая выпавшие из новой выборки.
    Возвращает все include, включая исторические. history_years=0 сохраняет
    прежнюю месячную дедупликацию; при положительном значении используется дата+текст.
    """
    query_start, query_end = month_bounds(month)
    if history_years:
        rows, stats = load_history(news_dir, history_start(query_start, history_years), query_end)
    else:
        rows, stats = load_news(news_dir, month)
    decisions, included = [], []
    for row in rows:
        relevance = classify_title(row['title'])
        item = dict(row, relevance=relevance)
        if relevance['decision'] == 'include':
            included.append(item)
        decisions.append({k: row[k] for k in ('news_id', 'loaded_at', 'title', 'provider')} | {
            'decision': relevance['decision'], 'categories': '|'.join(relevance['categories']),
            'groups': '|'.join(relevance['groups']), 'reason': relevance['reason'], 'rule_version': RULE_VERSION})
    output.mkdir(parents=True, exist_ok=True)
    columns = ['news_id', 'loaded_at', 'title', 'provider', 'decision', 'categories', 'groups', 'reason', 'rule_version']
    write_csv(output / 'classification.csv', decisions, columns)
    previous_review = {}
    review_path = output / 'review_blind.csv'
    if review_path.exists():
        with review_path.open(encoding='utf-8-sig', newline='') as handle:
            previous_review = {(r['news_id'], r['title']): r for r in csv.DictReader(handle)}
    review, rng = [], random.Random(42)
    for status in ('include', 'uncertain', 'exclude'):
        group = [r for r in decisions if r['decision'] == status]
        review.extend(dict(r, human_decision='', human_category='', human_comment='') for r in rng.sample(group, min(60, len(group))))
    present = {r['news_id'] for r in review}
    by_id = {r['news_id']: r for r in decisions}
    for (key, title), old in previous_review.items():
        annotations = {k: old.get(k, '') for k in ('human_decision', 'human_category', 'human_comment')}
        if any(annotations.values()) and key in by_id and by_id[key]['title'] == title and key not in present:
            review.append(dict(by_id[key], **annotations))
            present.add(key)
    for row in review:
        old = previous_review.get((row['news_id'], row['title']), {})
        row.update({k: old.get(k, row.get(k, '')) for k in ('human_decision', 'human_category', 'human_comment')})
    write_csv(review_path, review, columns + ['human_decision', 'human_category', 'human_comment'])
    summary = dict(month=month, rule_version=RULE_VERSION, source_stats=stats, unique_count=len(rows),
                   history_years=history_years, query_start=query_start, query_end_exclusive=query_end,
                   decisions=dict(Counter(r['decision'] for r in decisions)),
                   categories=dict(Counter(category for r in included for category in r['relevance']['categories'])),
                   review_status='Ожидает независимой ручной разметки; цены при классификации не используются')
    (output / 'classification_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return included


def month_bounds(month):
    """Возвращает начало month и следующего месяца как строки МСК с нулевым временем."""
    start = datetime.strptime(month, '%Y-%m')
    end = datetime(start.year + (start.month == 12), start.month % 12 + 1, 1)
    return start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')


def run(args):
    """Выполняет цикл по args, отделяя историю кандидатов от месяца проверяемых запросов."""
    suffix = f'_year{args.history_years}' if args.history_years else ''
    output = args.output or ROOT / 'data' / 'russian_market' / (args.month + suffix)
    rows = classify_month(args.news_dir, args.month, output, args.history_years)
    query_start, query_end = month_bounds(args.month)
    if args.stage == 'classify':
        return
    if len(rows) < 6:
        raise ValueError('Недостаточно включённых заголовков для пробы')
    pilot_dir = output / 'pilot'
    pilot_dir.mkdir(exist_ok=True)
    previous = ROOT / 'data' / 'russian_market' / args.month / 'pilot' / 'pilot.sqlite3'
    if not previous.exists():
        previous = ROOT / 'data' / 'embedding_pilot' / 'pilot.sqlite3'
    if not (pilot_dir / 'pilot.sqlite3').exists() and previous.exists():
        shutil.copy2(previous, pilot_dir / 'pilot.sqlite3')
    corpus_id = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    manifest = dict(month=args.month, sample_size=len(rows), corpus_id=corpus_id, preparation=PREPARATION,
                    relevance_version=RULE_VERSION, history_years=args.history_years,
                    query_start=query_start, query_end_exclusive=query_end, models={})
    with closing(open_cache(pilot_dir / 'pilot.sqlite3')) as cache:
        print(f'Переиспользовано векторов по точному тексту: {seed_title_vectors(cache, rows)}', flush=True)
        save_news(cache, rows, corpus_id)
        for model in dict.fromkeys(args.models):
            _, metrics = compute_model(cache, rows, model, args.ollama_url, args.batch_size, corpus_id)
            manifest['models'][model] = metrics
    (pilot_dir / 'summary.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    config = dict(metric_version=METRIC_VERSION, corpus_id=corpus_id, all_news=False,
                  pilot_db=str(pilot_dir / 'pilot.sqlite3'), horizons=HORIZONS, latency_seconds=0.,
                  max_gap_seconds=120., max_entry_wait_seconds=60., max_exit_age_seconds=60.,
                  neutral_points={'MIX': 50., 'RTS': 20.}, time_field='loaded_at', timezone='Europe/Moscow',
                  loaded_at_timezone_confirmed=False, relevance_version=RULE_VERSION)
    build_reactions(rows, DEFAULT_TICK_DIRS, output / 'reactions', config)
    analog_dir = output / 'analogs'
    analog_dir.mkdir(exist_ok=True)
    old_background = ROOT / 'data' / 'analog_analysis' / 'background.sqlite3'
    if not (analog_dir / 'background.sqlite3').exists() and old_background.exists():
        shutil.copy2(old_background, analog_dir / 'background.sqlite3')
    summary = build_analogs(SimpleNamespace(output=analog_dir, reaction_db=output / 'reactions' / 'reactions.sqlite3',
        pilot_dir=pilot_dir, mix_dir=DEFAULT_TICK_DIRS['MIX'], rts_dir=DEFAULT_TICK_DIRS['RTS'], models=args.models,
        quantile=.75, lookback_days=20, time_band_minutes=60, min_samples=30, min_days=5,
        volatility_fraction=.5, background_step=15, min_similarity=.5,
        history_years=args.history_years or None, query_start=query_start, query_end=query_end,
        threshold_multiplier=args.threshold_multiplier))
    print('Готово: ' + str(analog_dir / 'analogs.sqlite3'), flush=True)
    print('Отборов: ' + str(sum(g['selected'] for g in summary['groups'])), flush=True)


def main():
    """Разбирает параметры CLI и запускает независимую пробу фильтра RTS/MIX."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--month', default='2026-09', help='Месяц loaded_at, YYYY-MM')
    parser.add_argument('--history-years', type=int, default=1, help='Календарных лет истории до запроса, 0 — только месячная проба')
    parser.add_argument('--news-dir', type=Path, default=Path(r'C:\Users\Alkor\gd\db_rss'), help='Исходные БД RSS')
    parser.add_argument('--output', type=Path, help='Папка результатов, по умолчанию data/russian_market/месяц')
    parser.add_argument('--stage', choices=['classify', 'all'], default='all', help='Только классификация или полный расчёт')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Точные теги установленных моделей эмбеддингов')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Локальный сервер Ollama')
    parser.add_argument('--batch-size', type=int, default=32, help='Заголовков в пакете эмбеддингов')
    parser.add_argument('--threshold-multiplier', type=float, default=1., help='Коэффициент k динамического порога')
    args = parser.parse_args()
    if args.batch_size < 1 or args.history_years < 0:
        parser.error('Размер пакета должен быть положительным, число лет неотрицательным')
    try:
        validate_multiplier(args.threshold_multiplier)
    except ValueError as error:
        parser.error(str(error))
    run(args)


if __name__ == '__main__':
    main()
