r"""Экспорт пилота эмбеддингов в HTML, CSV и JSON.

Создаёт сравнение моделей рядом и отдельные списки аналогов за весь месяц и
только из прошлого. Все заголовки экранируются. CSV содержит пустые поля для
ручной разметки релевантности; исходные БД и эмбеддинги модуль не изменяет.

Используется командой:
  .\.venv\Scripts\python.exe embedding_pilot.py --month 2026-09 --limit 2000
Вспомогательный модуль source. Самостоятельный запуск не требуется. Параметры write_report:
выходная папка, записи новостей, найденные аналоги, метаданные эксперимента.
"""

import csv
from html import escape
import json


def write_csv(path, records, columns):
    """Записывает records в path с колонками columns, UTF-8 BOM и заголовком."""
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)


def write_report(output, rows, neighbors, summary):
    """Создаёт отчёты в output по rows, neighbors и summary; возвращает None.

    rows — новости с идентификаторами, neighbors — ранги и косинусы по моделям,
    summary — выборка, параметры и замеры. Результаты с постоянными именами
    перезаписываются; пользовательскую разметку следует сохранять отдельной копией.
    """
    output.mkdir(parents=True, exist_ok=True)
    by_id = {row['news_id']: row for row in rows}
    review = []
    for hit in neighbors:
        query, neighbor = by_id[hit['query_id']], by_id[hit['neighbor_id']]
        review.append({**hit, 'similarity': round(hit['similarity'], 6),
                       'query_title': query['title'], 'query_loaded_at': query['loaded_at'],
                       'neighbor_title': neighbor['title'], 'neighbor_loaded_at': neighbor['loaded_at'],
                       'neighbor_provider': neighbor['provider'],
                       'relevance_0_1_2': '', 'same_event_yes_no': '', 'action_match_yes_no': '', 'notes': ''})
    columns = ['model', 'mode', 'query_id', 'query_title', 'query_loaded_at', 'rank',
               'neighbor_id', 'neighbor_title', 'neighbor_loaded_at', 'neighbor_provider',
               'similarity', 'relevance_0_1_2', 'same_event_yes_no', 'action_match_yes_no', 'notes']
    write_csv(output / 'review.csv', review, columns)
    write_csv(output / 'news.csv', rows,
              ['news_id', 'loaded_at', 'published_at', 'title', 'provider', 'duplicate_count'])
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    groups = {}
    for hit in neighbors:
        groups.setdefault((hit['query_id'], hit['mode'], hit['model']), []).append(hit)
    models = list(summary['models']) or list(dict.fromkeys(hit['model'] for hit in neighbors))
    timing = []
    for model in models:
        entry = summary['models'].get(model, {})
        measured = entry.get('benchmark') or entry.get('current_run', {})
        rate = measured.get('texts_per_second')
        speed = f'{rate:.1f}' if rate is not None else 'нет замера'
        timing.append(f'<tr><td>{escape(model)}</td><td>{measured.get("generated_count", 0)}</td>'
                      f'<td>{measured.get("warmup_seconds", 0):.2f}</td>'
                      f'<td>{measured.get("embedding_seconds", 0):.2f}</td><td>{speed}</td></tr>')
    sections = []
    for number, query_id in enumerate(summary['query_ids'], 1):
        query = by_id[query_id]
        panels = []
        for mode, caption in [('past', 'Только прошлые новости'), ('all', 'Весь месяц: исследование сходства')]:
            cards = []
            for model in models:
                items = []
                for hit in groups.get((query_id, mode, model), []):
                    row = by_id[hit['neighbor_id']]
                    items.append(f'<li><div class="score">#{hit["rank"]} · {hit["similarity"]:.3f}</div>'
                                 f'<div>{escape(row["title"])}</div><small>{escape(row["loaded_at"])}'
                                 f' · {escape(str(row["provider"]))} · записей: {row["duplicate_count"]}</small></li>')
                content = ''.join(items) or '<li>Нет доступных прошлых аналогов в этой выборке.</li>'
                cards.append(f'<section class="model"><h4>{escape(model)}</h4><ol>{content}</ol></section>')
            panels.append(f'<h3>{caption}</h3><div class="grid">{"".join(cards)}</div>')
        sections.append(f'<details {"open" if number <= 2 else ""}><summary><span class="number">{number:02}</span>'
                        f'{escape(query["title"])}</summary><p class="meta">Получено: {escape(query["loaded_at"])}'
                        f' · {escape(str(query["provider"]))} · записей: {query["duplicate_count"]}</p>'
                        f'{"".join(panels)}</details>')
    document = '''<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Сравнение эмбеддингов новостей</title><style>
*{box-sizing:border-box}body{margin:0;background:#f3f5f8;color:#17233a;font:16px/1.55 "Segoe UI",sans-serif}
main{max-width:1280px;margin:40px auto;padding:0 24px}h1{font-size:36px;letter-spacing:-1px;margin:8px 0}
h2{margin-top:32px}h3{font-size:17px;margin:24px 0 10px}h4{margin:0;padding:14px 18px;background:#e9eff8}
.eyebrow{color:#3163ba;font-size:13px;font-weight:700;letter-spacing:2px}.lead{max-width:940px;color:#53627a}
.info{background:#fff;border-left:4px solid #3163ba;padding:18px 22px;margin:24px 0}.meta,small{color:#64748b}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px 15px;text-align:left;border-bottom:1px solid #e2e8f0}
th{font-size:13px;color:#53627a}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}
details{background:#fff;border:1px solid #dce3ed;border-radius:9px;margin:14px 0;padding:18px 22px}
summary{cursor:pointer;font-weight:650;font-size:18px}.number{color:#3163ba;margin-right:14px;font-size:14px}
.model{border:1px solid #e2e8f0;border-radius:7px;overflow:hidden}ol{margin:0;padding:0;list-style:none}
li{padding:13px 18px;border-top:1px solid #e9eef5}.score{font-size:12px;font-weight:700;color:#3163ba;margin-bottom:5px}
small{display:block;font-size:12px;margin-top:6px}a{color:#235bb8}.footer{font-size:13px;color:#64748b;margin:35px 0}
@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:0 12px}h1{font-size:28px}td,th{padding:8px;font-size:12px}}
</style></head><body><main>'''
    document += f'''<div class="eyebrow">PJ26 · ПИЛОТ ЭМБЕДДИНГОВ</div>
<h1>Похожие новости — рядом</h1>
<p class="lead">{summary['sample_size']} уникальных заголовков · {escape(summary['month'])} ·
{len(summary['query_ids'])} запросов · seed {summary['seed']}. Раскройте заголовок для сравнения моделей.</p>
<div class="info"><strong>Что проверяем:</strong> смысловое сходство заголовков, скорость и пригодность аналогов.
Косинусное сходство не является вероятностью роста. Цены и торговая доходность здесь ещё не рассчитаны.
В режиме прошлого используется loaded_at и зазор {summary['embargo_minutes']} минут; это только временной
фильтр. Смысловые перепечатки одного события пока могут попасть в выдачу.</div>
<h2>Измерения на этом компьютере</h2><p class="meta">Прогрев отдельно. Скорость — новые заголовки / время
HTTP-запросов после прогрева, без чтения SQLite. Кэш не считается ускорением модели.
Порядок моделей фиксирован, повторных прогонов нет.</p>
<table><thead><tr><th>Модель</th><th>Новых текстов</th><th>Прогрев, с</th><th>Расчёт, с</th><th>Текстов/с</th></tr></thead>
<tbody>{''.join(timing)}</tbody></table>
<h2>Проверка аналогов</h2><p>Для разметки скачайте <a href="review.csv">review.csv</a> и сохраните копию:
релевантность 0 — нет, 1 — только тема, 2 — аналогичное событие; отдельно укажите перепечатку
и совпадение действия. Сравнивайте качество по этим оценкам, а не по величине косинуса между моделями.</p>
<p><a href="news.csv">Выборка новостей</a> · <a href="summary.json">Метаданные и замеры</a></p>
{''.join(sections)}<p class="footer">Выборка случайная из уникальных заголовков месяца; запросы частично тематические.
Повторы объединены по нормализованному заголовку внутри месяца. Вне выборки могут существовать более подходящие аналоги.
Отпечаток выборки: {escape(summary['corpus_id'])}</p></main></body></html>'''
    (output / 'report.html').write_text(document, encoding='utf-8')
