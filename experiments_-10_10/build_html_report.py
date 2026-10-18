r"""Создаёт автономный HTML-отчёт из готовой проверки модели трёх классов.

Читает RUN/check_model_report.json, строит выводы, метрики, матрицу ошибок,
сравнение адаптера и Ollama, таблицу примеров с поиском и фильтрами.
Заголовки добавляет из JSONL только после совпадения SHA-256 с отчётом.
Добавляет вывод о готовности к реальной торговле и для ложных сигналов
направление, цены и условный результат в пунктах без издержек. Пункты берёт
из разметки, а не из балла; короткий сигнал меняет знак движения цены.
Показывает роль evaluation_role: валидация либо отдельный финальный тест.
Окружение обучения, GPU, Ollama и интернет не требуются; модель не запускается.
По умолчанию RUN=data/experiments_-10_10/RTX4070_extremes от корня проекта.
Перезаписывает RUN/check_model_report.html; исходные JSON/разметку не меняет.
Файловых логов нет. Поддерживает task=extremes, не прежний балльный режим.

Запуск отдельно, в том числе кнопкой VS Code:
  .\.venv\Scripts\python.exe experiments_-10_10/build_html_report.py
  .\.venv\Scripts\python.exe experiments_-10_10/build_html_report.py --run data/experiments_-10_10/RTX3090Ti_extremes
  .\.venv\Scripts\python.exe experiments_-10_10/build_html_report.py --output data/experiments_-10_10/report.html
Параметр --data выбирает исходный JSONL для заголовков. При его отсутствии
или несовпадении SHA-256 отчёт строится без заголовков с пояснением.
Готовый HTML открыть в браузере; сервер и внешние библиотеки не используются.
"""

import argparse
import hashlib
from html import escape
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent / 'source'))
from model_tasks import CLASSES, classification_metrics, score_class

LABELS = {'negative': 'Сильная отрицательная', 'none': 'Без экстремума',
          'positive': 'Сильная положительная', 'invalid': 'Неверный ответ'}


def percent(value):
    """Возвращает процент с одним десятичным знаком либо тире для отсутствующего значения."""
    return '—' if value is None else f'{100 * value:.1f}%'.replace('.', ',')


def load_news(data, expected_hash):
    """Читает тексты и котировки из data при совпадении expected_hash; возвращает словарь и пояснение."""
    if not data.is_file():
        return {}, 'Разметка не найдена: заголовки и пункты недоступны. Метрики взяты из готовой проверки.'
    with data.open('rb') as handle:
        actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if actual_hash != expected_hash:
        return {}, 'SHA-256 разметки не совпадает с отчётом: заголовки и пункты не добавлены во избежание подмены примеров.'
    news = {}
    with data.open(encoding='utf-8-sig') as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                key = row['news_id'], row['instrument'], int(row['horizon'])
                fields = ('title', 'loaded_at', 'points', 'entry_price', 'exit_price',
                          'entry_at', 'exit_at', 'flags')
                news[key] = {field: row.get(field) for field in fields}
    return news, ''


def finite_quote(value):
    """Возвращает конечное число котировки value либо None; неверные значения отклоняет."""
    if value is None or value == '':
        return None
    if isinstance(value, bool):
        raise ValueError('Котировка не может быть логическим значением')
    number = float(value)
    if not math.isfinite(number):
        raise ValueError('Котировка должна быть конечным числом')
    return number


def price_outcome(prediction, quote):
    """Возвращает движение цены, пункты сигнала и совпадение направления prediction по quote.

positive трактуется как условная покупка, negative — как условная продажа.
None не заменяется нулём. Если сохранены обе цены и points, проверяет их
согласованность. Комиссии, исполнение и размер позиции не моделирует.
"""
    entry = finite_quote(quote.get('entry_price'))
    exit_price = finite_quote(quote.get('exit_price'))
    movement = finite_quote(quote.get('points'))
    if entry is not None and exit_price is not None:
        difference = exit_price - entry
        if movement is not None and not math.isclose(movement, difference, abs_tol=1e-6, rel_tol=1e-9):
            raise ValueError('Пункты разметки не совпадают с разностью exit_price - entry_price')
        movement = difference
    signal = prediction['predicted']
    signed_points = None
    direction = 'unknown'
    if signal not in ('negative', 'positive'):
        direction = 'no-signal'
    elif movement is not None:
        signed_points = movement if signal == 'positive' else -movement
        direction = 'correct' if signed_points > 0 else 'wrong' if signed_points < 0 else 'flat'
    return dict(movement=movement, signed_points=signed_points, direction=direction,
                entry_price=entry, exit_price=exit_price)


def format_points(value, signed=False):
    """Возвращает число value с пробелами и запятой; signed добавляет плюс, None даёт тире."""
    if value is None:
        return '—'
    text = f'{abs(value):,.2f}'.rstrip('0').rstrip('.').replace(',', ' ').replace('.', ',')
    return ('−' if value < 0 else '+' if signed and value > 0 else '') + text


def direction_label(value):
    """Возвращает русский ответ о совпадении направления по коду value."""
    return {'correct': 'Да', 'wrong': 'Нет', 'flat': 'Нет движения',
            'unknown': 'Нет котировок', 'no-signal': 'Нет сигнала'}[value]


def class_badge(value):
    """Возвращает экранированный HTML значка класса value."""
    value = value if value in CLASSES else 'invalid'
    return f'<span class="badge {value}">{LABELS[value]}</span>'


def table(headers, rows, css=''):
    """Возвращает HTML таблицы из экранированных headers и заранее подготовленных ячеек rows."""
    head = ''.join(f'<th>{escape(str(h))}</th>' for h in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{cell}</td>' for cell in row) + '</tr>' for row in rows)
    return f'<div class="table-scroll"><table class="{css}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def render_report(report, news, warning):
    """Возвращает HTML проверки report с текстами/котировками news, выводом о торговле и warning."""
    if report.get('task') != 'extremes' or not isinstance(report.get('predictions'), list):
        raise ValueError('Нужен check_model_report.json задачи extremes с массивом predictions')
    predictions = report['predictions']
    if not predictions:
        raise ValueError('В отчёте нет проверенных примеров')
    seen = set()
    for p in predictions:
        score_class(p['actual'])
        key = p['news_id'], p['instrument'], p['horizon']
        if key in seen or p['instrument'] not in ('MIX', 'RTS') or p['horizon'] not in (5, 15, 30, 60, 120):
            raise ValueError('Повтор или неверный инструмент/горизонт в предсказаниях')
        seen.add(key)
    metrics = classification_metrics(predictions)
    classes = metrics['classes']
    support = classes['negative']['support'] + classes['positive']['support']
    hits = classes['negative']['true_positives'] + classes['positive']['true_positives']
    false = classes['negative']['false_positives'] + classes['positive']['false_positives']
    headline = f'Обнаружено {hits} из {support} экстремальных реакций'
    if support == 0:
        conclusion = 'В этой выборке нет фактических экстремумов: полноту их обнаружения оценить нельзя.'
    elif hits == 0:
        conclusion = f'Модель пропустила все {support} экстремумов. Правильных экстремальных сигналов нет.'
    else:
        conclusion = f'Обнаружено {hits} экстремумов с правильным направлением; пропущено или перепутано {support - hits}.'
    conclusion += f' Выдано {metrics["signal_count"]} экстремальных сигналов, из них ложных — {false}.'
    model = escape(str(report.get('ollama_name', 'Не указана')))
    period = report.get('period', {})
    role = 'Финальный тест' if report.get('evaluation_role') == 'test' else 'Валидация'
    dates = escape(f'{period.get("validation_start", "—")} → {period.get("cutoff", "—")} (конец не включён)')
    class_table = table(['Фактический класс', 'В выборке', 'Сигналов', 'Верных', 'Ложных', 'Пропущено', 'Точность', 'Полнота', 'F1'], [
        [class_badge(c), str(classes[c]['support']), str(classes[c]['predicted_count']),
         str(classes[c]['true_positives']), str(classes[c]['false_positives']),
         str(classes[c]['false_negatives']), percent(classes[c]['precision']),
         percent(classes[c]['recall']), percent(classes[c]['f1'])] for c in CLASSES])
    confusion = table(['Факт ↓ / прогноз →', *[LABELS[c] for c in CLASSES], LABELS['invalid']], [
        [class_badge(a), *[f'<span class="{"diagonal" if a == p else "off-diagonal"}">{metrics["confusion"][a][p]}</span>'
                           for p in (*CLASSES, 'invalid')]] for a in CLASSES])
    comparison_rows = []
    if all('adapter_predicted' in p for p in predictions):
        adapter = classification_metrics([dict(actual=p['actual'], predicted=p['adapter_predicted']) for p in predictions])
        comparison_rows.append(('До экспорта · адаптер', adapter))
    comparison_rows.append(('После импорта · Ollama', metrics))
    comparison = table(['Модель', 'Верных отрицательных', 'Верных положительных', 'Ложных сигналов', 'Неверный JSON', 'Общая точность'], [
        [name, str(m['classes']['negative']['true_positives']), str(m['classes']['positive']['true_positives']),
         str(sum(m['classes'][c]['false_positives'] for c in ('negative', 'positive'))),
         str(m['sample_size'] - m['valid_answers']), percent(m['accuracy'])] for name, m in comparison_rows])
    grouped = []
    for instrument in ('MIX', 'RTS'):
        for horizon in (5, 15, 30, 60, 120):
            subset = [p for p in predictions if p['instrument'] == instrument and p['horizon'] == horizon]
            if not subset:
                continue
            m = classification_metrics(subset)
            for c in ('negative', 'positive'):
                group = m['classes'][c]
                grouped.append([instrument, str(horizon), class_badge(c), str(group['support']),
                                str(group['predicted_count']), percent(group['precision']), percent(group['recall'])])
    groups = table(['Инструмент', 'Минут', 'Класс', 'В выборке', 'Сигналов', 'Точность', 'Полнота'], grouped)
    outcomes = []
    for p in predictions:
        info = news.get((p['news_id'], p['instrument'], p['horizon']), {})
        outcome = price_outcome(p, info)
        if p['predicted'] in ('negative', 'positive') and p['predicted'] != score_class(p['actual']):
            outcomes.append((p, info, outcome))
    measured = sum(o['signed_points'] is not None for _, _, o in outcomes)
    right = sum(o['direction'] == 'correct' for _, _, o in outcomes)
    wrong = sum(o['direction'] == 'wrong' for _, _, o in outcomes)
    flat = sum(o['direction'] == 'flat' for _, _, o in outcomes)
    false_rows = []
    for p, info, o in outcomes:
        title = escape(str(info.get('title') or p['news_id']))
        times = escape(f'{info.get("entry_at") or "—"} → {info.get("exit_at") or "—"}')
        false_rows.append([f'{title}<small>{times}</small>', escape(p['instrument']), str(p['horizon']),
                           class_badge(p['predicted']), str(p['actual']), class_badge(score_class(p['actual'])),
                           f'{format_points(o["entry_price"])} → {format_points(o["exit_price"])}',
                           format_points(o['movement'], True), direction_label(o['direction']),
                           format_points(o['signed_points'], True)])
    false_table = table(['Новость / вход → выход', 'Инструмент', 'Минут', 'Прогноз', 'Балл', 'Факт',
                         'Цена входа → выхода', 'Движение цены, пт', 'Направление угадано', 'По сигналу, пт'], false_rows)
    point_rows = []
    for instrument in ('MIX', 'RTS'):
        for horizon in (5, 15, 30, 60, 120):
            subset = [o for p, _, o in outcomes if p['instrument'] == instrument and p['horizon'] == horizon]
            known = [o for o in subset if o['signed_points'] is not None]
            if not subset:
                continue
            total = math.fsum(o['signed_points'] for o in known) if known else None
            point_rows.append([instrument, str(horizon), str(len(subset)), str(len(known)),
                               str(sum(o['direction'] == 'correct' for o in known)),
                               format_points(total, True), format_points(total / len(known) if known else None, True)])
    points_summary = table(['Инструмент', 'Минут', 'Ложных сигналов', 'С котировками', 'Направление угадано',
                            'Сумма по сигналам, пт', 'Среднее по сигналу, пт'], point_rows)
    source_flags = {flag for _, info, _ in outcomes for flag in (info.get('flags') or [])}
    flags_note = ''
    if source_flags & {'continuous_contract_unverified', 'synthetic_fractions'}:
        flags_note = ('В разметке отмечены непроверенная склейка контрактов и/или синтетические доли. '
                      'Исполнимость этих цен на конкретном контракте требует отдельной проверки.')
    example_rows = []
    for index, p in enumerate(predictions, 1):
        actual = score_class(p['actual'])
        predicted = p['predicted'] if p['predicted'] in CLASSES else 'invalid'
        kinds = []
        if actual != predicted:
            kinds.append('errors')
        if predicted in ('negative', 'positive'):
            kinds.extend(['signals', 'correct-signals' if predicted == actual else 'false-signals'])
            if predicted != actual:
                kinds.append('false-negative-signal' if predicted == 'negative' else 'false-positive-signal')
        if actual in ('negative', 'positive') and actual != predicted:
            kinds.append('missed')
        if predicted == 'invalid':
            kinds.append('invalid')
        info = news.get((p['news_id'], p['instrument'], p['horizon']), {})
        o = price_outcome(p, info)
        if o['direction'] in ('correct', 'wrong', 'flat'):
            kinds.append('direction-' + o['direction'])
        title = escape(str(info.get('title') or 'Заголовок недоступен'))
        news_id = escape(str(p['news_id']))
        loaded = escape(str(info.get('loaded_at', '—')))
        error = escape(str(p.get('error', '')))
        cells = [str(index), f'{title}<small>{loaded}<br><span class="news-id">{news_id}</span></small>',
                 escape(p['instrument']), str(p['horizon']), str(p['actual']), class_badge(actual),
                 class_badge(predicted) + (f'<small>{error}</small>' if error else ''),
                 format_points(o['movement'], True), direction_label(o['direction']),
                 format_points(o['signed_points'], True)]
        example_rows.append(f'<tr data-kind="{" ".join(kinds)}" data-instrument="{escape(p["instrument"])}" '
                            f'data-horizon="{p["horizon"]}">' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr>')
    examples = ''.join(example_rows)
    warning_html = f'<p class="notice">{escape(warning)}</p>' if warning else ''
    identity = escape(str(report.get('note', 'Проверяется результат модели под сохранённым именем.')))
    data_hash = escape(str(report.get('data_sha256', 'Не указан')))
    return PAGE.replace('@@BODY@@', f'''
<header><p class="eyebrow">ЭКСПЕРИМЕНТ · НОВОСТИ → MIX / RTS</p><h1>Проверка экстремальных реакций</h1>
<p class="subtitle">{model} · {role} · {dates}</p></header>
<section class="trade-verdict"><h2>Можно использовать для торговли?</h2>
<p><b>Пока нет. Готовность к реальной торговле не подтверждена.</b> В текущем отчёте нет полноценной
проверки торговой стратегии с издержками и исполнением. Это не утверждение об убыточности:
ошибка экстремального класса может сопровождаться верным направлением и положительным условным результатом.</p>
<p>Для решения о реальном запуске нужны фиксированные правила сделок, проверка конкретных контрактов,
комиссии, спред, проскальзывание, задержка модели, совместный учёт позиций и проверка на неиспользованном периоде.</p></section>
<section class="verdict"><span class="eyebrow">ГЛАВНЫЙ РЕЗУЛЬТАТ</span><h2>{headline}</h2><p>{conclusion}</p></section>
<div class="cards"><article><strong>{len(predictions)}</strong><span>проверенных сочетаний</span></article>
<article><strong>{support}</strong><span>фактических экстремумов</span></article>
<article><strong>{hits}</strong><span>правильных сигналов</span></article>
<article><strong>{false}</strong><span>ложных сигналов</span></article></div>
<p class="caption">Это сочетания «новость — инструмент — горизонт», не независимые новости.
Уникальных идентификаторов новостей: {len({p['news_id'] for p in predictions})}.</p>
<section><h2>Ложные экстремальные сигналы: направление и пункты</h2>
<p>«Ложный» здесь означает ошибку класса ±8…±10. Направление проверено отдельно по изменению цены,
а не по знаку балла: при балле 0 цена тоже могла измениться.</p>
<p><b>Направление угадано: {right} из {measured}</b> сигналов с котировками; ошибочно — {wrong},
без движения — {flat}, без котировок — {len(outcomes) - measured}.</p>{false_table}
<h3>Итоги условных сигналов по инструментам и горизонтам</h3>{points_summary}
<p class="notice">Движение цены = цена выхода − цена входа. По сигналу: для positive — это движение,
для negative — движение с обратным знаком. Плюс означает движение в пользу условной позиции.
Пункты показаны в единицах котировки исходной разметки, не в рублях и не в баллах −10…+10.</p>
<p>Цены входа и выхода взяты из разметки; фактическое исполнение заявки не моделируется. Комиссии,
спред, проскальзывание и время ответа модели не учтены. Суммы — арифметика отдельных примеров,
не доходность портфеля: новости и горизонты могут пересекаться. MIX и RTS между собой не складываются.
{flags_note}</p></section>
<section><h2>Качество каждого класса</h2><p>negative: −10…−8 · none: −7…+7 · positive: +8…+10.
«Без экстремума» не означает отсутствие движения.</p>{class_table}
<div class="definitions"><p><b>Точность:</b> сколько выданных сигналов этого класса подтвердилось.</p>
<p><b>Полнота:</b> какую долю фактических случаев этого класса модель обнаружила.</p>
<p><b>Пропущено:</b> фактические случаи класса, для которых выдан другой класс или неверный ответ.</p>
<p><b>F1:</b> сводный показатель точности и полноты. Тире означает, что показатель не определён,
например нет выданных сигналов. Это не 100%.</p></div></section>
<section><h2>Матрица ошибок</h2><p>Строки — факт, столбцы — прогноз. Совпадения лежат на диагонали.</p>{confusion}</section>
<section><h2>До и после экспорта</h2><p>Сравнение на одинаковых примерах текущего отчёта.</p>{comparison}
<p class="notice">Общая точность Ollama — {percent(metrics['accuracy'])}. Постоянный ответ «без экстремума» —
{percent(metrics['no_signal_baseline_accuracy'])}. Доля обычных случаев высока, поэтому общая точность
не заменяет проверку сильных реакций. Неверных ответов: {len(predictions) - metrics['valid_answers']}.</p></section>
<section><h2>Инструменты и горизонты</h2>{groups}</section>
<section><h2>Каждый пример</h2>{warning_html}<div class="filters">
<input id="search" type="search" placeholder="Найти заголовок или идентификатор" aria-label="Поиск">
<select id="kind" aria-label="Тип примера"><option value="">Все примеры</option><option value="errors">Все ошибки</option>
<option value="signals">Все экстремальные сигналы</option><option value="correct-signals">Верные сигналы</option>
<option value="false-signals">Ложные сигналы</option><option value="missed">Пропущенные экстремумы</option>
<option value="false-negative-signal">Ложные: сильная отрицательная</option>
<option value="false-positive-signal">Ложные: сильная положительная</option>
<option value="direction-correct">Сигналы: направление угадано</option>
<option value="direction-wrong">Сигналы: направление ошибочно</option>
<option value="invalid">Неверные ответы</option></select>
<select id="instrument" aria-label="Инструмент"><option value="">MIX и RTS</option><option>MIX</option><option>RTS</option></select>
<select id="horizon" aria-label="Горизонт"><option value="">Все горизонты</option>
{''.join(f'<option value="{h}">{h} минут</option>' for h in (5, 15, 30, 60, 120))}</select></div>
<p id="visible-count" class="caption"></p><div class="table-scroll examples"><table id="examples"><thead><tr>
<th>№</th><th>Новость / дата получения</th><th>Инструмент</th><th>Минут</th><th>Балл</th><th>Факт</th><th>Прогноз</th>
<th>Движение цены, пт</th><th>Направление угадано</th><th>По сигналу, пт</th>
</tr></thead><tbody>{examples}</tbody></table></div></section>
<footer><p>Отчёт построен из сохранённой проверки, без повторного обучения и запросов к модели.
Метрики пересчитаны по сохранённым ответам. Это оценка классов реакции, не доходности торговли.</p>
<details><summary>Воспроизводимость</summary><p>{identity}</p><p>SHA-256 разметки: <code>{data_hash}</code></p>
<p>Контрольные суммы весов Ollama этим отчётом не проверяются. Заголовки, даты и идентификаторы — локальные данные:
учитывайте это при передаче HTML другим людям.</p></details></footer>''')


PAGE = r'''<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Результаты проверки модели — MIX / RTS</title><style>
:root{font-family:Segoe UI,Arial,sans-serif;color:#1d293d;background:#f3f5f8;line-height:1.55}
*{box-sizing:border-box}body{margin:0}main{max-width:1380px;margin:auto;padding:40px 28px}
header{padding:0 0 24px}.eyebrow{font-size:11px;font-weight:700;letter-spacing:1.8px;color:#526783}
h1{font-size:34px;line-height:1.2;margin:8px 0}h2{font-size:23px;line-height:1.35;margin:0 0 12px}
.subtitle,.caption,small{color:#627089}.subtitle{margin:12px 0}.caption{font-size:13px}
section{background:white;border:1px solid #e1e6ef;border-radius:14px;padding:26px;margin:22px 0}
.verdict{background:#172d47;color:white;border:0}.verdict .eyebrow{color:#a7c4e8}.verdict h2{margin-top:12px;font-size:29px}
.trade-verdict{border-left:5px solid #a45928;background:#fff9ef}
.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:16px}.cards article{background:white;border:1px solid #e1e6ef;border-radius:12px;padding:20px}
.cards strong{display:block;font-size:32px;font-weight:650}.cards span{font-size:13px;color:#627089}
.table-scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;text-align:left}
th{background:#eef2f7;font-weight:600;color:#42516b;white-space:nowrap}td,th{padding:12px 10px;border-bottom:1px solid #e8edf3;vertical-align:top}
tbody tr:hover{background:#f8faff}.badge{display:inline-block;padding:4px 9px;border-radius:6px;font-size:12px;white-space:nowrap}
.negative{background:#ffedef;color:#a02838}.positive{background:#e5f6ee;color:#1c6b4c}.none{background:#edf1f7;color:#506077}.invalid{background:#fff1d8;color:#8a5c12}
.diagonal{font-weight:700;background:#e5f0fa;display:block;padding:5px;border-radius:5px}.off-diagonal{display:block;padding:5px}
.definitions{display:grid;grid-template-columns:1fr 1fr;gap:0 22px;margin-top:16px;font-size:13px;color:#526079}
.notice{background:#f1f5fb;border-left:3px solid #7795b7;padding:14px 16px;font-size:14px}
.filters{display:flex;gap:10px;flex-wrap:wrap;margin:18px 0}input,select{font:inherit;font-size:13px;padding:10px;border:1px solid #ccd5e2;border-radius:7px;background:white;color:#1d293d}input{flex:1;min-width:240px}
.examples{max-height:700px}.examples th{position:sticky;top:0;z-index:1}.examples td:nth-child(2){min-width:300px;max-width:540px}
small{display:block;font-size:11px;margin-top:6px}.news-id{word-break:break-all}footer{font-size:13px;color:#627089;padding:0 6px 24px}code{word-break:break-all}
[hidden]{display:none!important}@media(max-width:760px){main{padding:20px 12px}h1{font-size:27px}section{padding:18px}.cards{grid-template-columns:1fr 1fr}.definitions{grid-template-columns:1fr}.filters>*{width:100%}}
@media print{main{padding:0}section{break-inside:avoid}.filters{display:none}.examples{max-height:none;overflow:visible}body{background:white}}
</style></head><body><main>@@BODY@@</main><script>
const rows=Array.from(document.querySelectorAll('#examples tbody tr'));
function filterRows(){
 const query=document.getElementById('search').value.trim().toLocaleLowerCase('ru');
 const kind=document.getElementById('kind').value;
 const instrument=document.getElementById('instrument').value;
 const horizon=document.getElementById('horizon').value;
 let visible=0;
 for(const row of rows){
  const show=(!query||row.textContent.toLocaleLowerCase('ru').includes(query))&&
    (!kind||row.dataset.kind.split(' ').includes(kind))&&
    (!instrument||row.dataset.instrument===instrument)&&(!horizon||row.dataset.horizon===horizon);
  row.hidden=!show;if(show)visible++;
 }
 document.getElementById('visible-count').textContent=`Показано ${visible} из ${rows.length} примеров`;
}
for(const id of ['search','kind','instrument','horizon'])document.getElementById(id).addEventListener('input',filterRows);
filterRows();
</script></body></html>'''


def build_parser():
    """Возвращает параметры источника проверки, необязательной разметки и выходного HTML."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--run', type=Path, default=ROOT / 'data/experiments_-10_10/RTX4070_extremes', help='Каталог готовой проверки')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/experiments_-10_10/scores.jsonl', help='JSONL для заголовков')
    parser.add_argument('--output', type=Path, help='HTML; по умолчанию RUN/check_model_report.html')
    return parser


def main(argv=None):
    """Создаёт только HTML по параметрам argv; возвращает 0 либо 1 при неверных данных."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    args = build_parser().parse_args(argv)
    try:
        source = args.run.resolve() / 'check_model_report.json'
        target = (args.output or args.run / 'check_model_report.html').resolve()
        if target.suffix.lower() != '.html':
            raise ValueError('--output должен иметь расширение .html')
        report = json.loads(source.read_text(encoding='utf-8'))
        if not isinstance(report, dict):
            raise ValueError('Отчёт проверки должен быть объектом JSON')
        news, warning = load_news(args.data, report.get('data_sha256'))
        html = render_report(report, news, warning)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(html, encoding='utf-8')
        print(f'HTML-отчёт создан: {target}', flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'Ошибка создания HTML-отчёта: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
