r"""Сравнивает коэффициенты динамического порога по готовой БД без пересчёта котировок.

Переоценивает одни и те же пять аналогов для каждого k, сохраняет все испробованные
значения и отборы. Автоматически лучший параметр не выбирает. Сентябрьские результаты
после подбора являются исследовательскими, а не независимым тестом параметра.
Считает достижения эффективного порога и общего базового порога k=1 отдельно.

Примеры из корня:
  .\.venv\Scripts\python.exe tune_threshold.py
  .\.venv\Scripts\python.exe tune_threshold.py --multipliers 0.1 0.25 0.5 0.75 1 1.5
  .\.venv\Scripts\python.exe tune_threshold.py --database data\russian_market\2026-09_year1\analogs\analogs.sqlite3
Выход по умолчанию: threshold_tuning рядом с исходной БД. Файловых логов нет.
"""

import argparse
from contextlib import closing
import json
from pathlib import Path

from source.build_analogs import digest_file, readonly
from source.embedding_pilot import ROOT
from source.embedding_report import write_csv
from source.threshold_tuning import rescale_score, validate_multiplier

DETAIL_COLUMNS = ['threshold_multiplier', 'news_id', 'loaded_at', 'model', 'instrument', 'horizon',
                  'forecast_direction', 'actual_status', 'actual_points', 'median_ratio', 'actual_ratio',
                  'actual_ratio_base', 'direction_correct', 'strong_correct', 'base_strong_correct', 'title', 'neighbor_ids']


def summarize(scores, multiplier, model='Все', instrument='Все', horizon=0):
    """Сводит переоценённые scores для multiplier и указанных меток группы в одну строку."""
    selected = [s for s in scores if s['selected']]
    valid = [s for s in selected if s.get('direction_correct') is not None]
    return dict(threshold_multiplier=multiplier, model=model, instrument=instrument, horizon=horizon,
                assessments=len(scores), selected=len(selected), unique_news=len({s['news_id'] for s in selected}),
                actual_valid=len(valid), direction_correct=sum(s['direction_correct'] for s in valid),
                effective_target_hits=sum(s['strong_correct'] for s in valid),
                base_target_hits=sum(s['base_strong_correct'] for s in valid))


def run(database, output, multipliers):
    """Читает database только для чтения, пишет сравнение multipliers в output, возвращает общую сводку."""
    multipliers = list(dict.fromkeys(validate_multiplier(k) for k in multipliers))
    if not multipliers:
        raise ValueError('Укажите хотя бы один коэффициент порога')
    with closing(readonly(database)) as c:
        metadata = json.loads(c.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
        scores = [json.loads(p) for (p,) in c.execute('SELECT payload FROM scores')]
        titles = {key: json.loads(p)['title'] for key, p in c.execute('SELECT news_id,payload FROM news')}
    groups = sorted({(s['model'], s['instrument'], s['horizon']) for s in scores})
    summary, details = [], []
    for k in multipliers:
        scaled = [rescale_score(s, k, metadata['min_similarity']) for s in scores]
        summary.append(summarize(scaled, k))
        for model, instrument, horizon in groups:
            subset = [s for s in scaled if (s['model'], s['instrument'], s['horizon']) == (model, instrument, horizon)]
            summary.append(summarize(subset, k, model, instrument, horizon))
        for s in scaled:
            if s['selected']:
                details.append({key: s.get(key) for key in ('threshold_multiplier', 'news_id', 'loaded_at', 'model',
                    'instrument', 'horizon', 'forecast_direction', 'actual_status', 'actual_points', 'median_ratio', 'actual_ratio',
                    'actual_ratio_base', 'direction_correct', 'strong_correct', 'base_strong_correct')} | {
                    'title': titles[s['news_id']], 'neighbor_ids': '|'.join(n['news_id'] for n in s['neighbors'])})
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'summary.csv', summary, list(summary[0]))
    write_csv(output / 'selected.csv', details, DETAIL_COLUMNS)
    totals = [r for r in summary if r['model'] == 'Все']
    (output / 'summary.json').write_text(json.dumps(dict(source_database=str(database.resolve()),
        source_sha256=digest_file(database), multipliers=multipliers, source_config=metadata,
        parameter_selection='Все значения сохранены; автоматический выбор отсутствует', totals=totals),
        ensure_ascii=False, indent=2), encoding='utf-8')
    for row in totals:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return totals


def main():
    """Разбирает CLI и строит таблицу исследовательского сравнения коэффициентов порога."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--database', type=Path, default=ROOT / 'data/russian_market/2026-09_year1/analogs/analogs.sqlite3', help='Готовые оценки аналогов')
    p.add_argument('--output', type=Path, help='Папка сравнения, по умолчанию threshold_tuning рядом с БД')
    p.add_argument('--multipliers', type=float, nargs='+', default=[.1, .25, .5, .75, 1., 1.25, 1.5], help='Положительные коэффициенты k')
    args = p.parse_args()
    if not args.database.is_file():
        p.error('БД оценок не найдена. Сначала выполните russian_market.py или build_analogs.py')
    try:
        run(args.database, args.output or args.database.parent / 'threshold_tuning', args.multipliers)
    except ValueError as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
