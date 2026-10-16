r"""Отбирает пять текстовых аналогов из разных прошлых дней и оценивает согласие.

Сначала ранжируются допустимые прошлые события по сходству, выбирается максимум
одна новость на дату. Затем проверяются все пять исходов без замены неудобного
соседа. Результат текущей новости в интерфейс оценки не передаётся.
history_years ограничивает возраст аналога календарными годами; None — вся история.

Использование:
  .\.venv\Scripts\python.exe build_analogs.py --min-similarity 0.5
  .\.venv\Scripts\python.exe russian_market.py --history-years 1
  .\.venv\Scripts\python.exe view_news_finplot.py --selection-mode analogs
Модуль source отдельно не запускается. Все оценки/отказы сохраняет build_analogs.
"""

import math
import statistics
from datetime import datetime


def history_start(loaded_at, years):
    """Возвращает loaded_at минус years календарных лет; 29 февраля заменяет на 28."""
    if not isinstance(years, int) or years < 1:
        raise ValueError('Число лет должно быть положительным целым')
    stamp = datetime.strptime(loaded_at, '%Y-%m-%d %H:%M:%S')
    try:
        start = stamp.replace(year=stamp.year - years)
    except ValueError:
        start = stamp.replace(year=stamp.year - years, day=28)
    return start.strftime('%Y-%m-%d %H:%M:%S')


def pick_neighbors(candidates, query_loaded_at, count=5, history_years=None):
    """Возвращает count ближайших candidates, максимум по одному в каждом прошлом дне.

    Требует status=ok, положительный порог, завершённое до query_loaded_at окно.
    Знак, размер движения, ER/MAE/stable в ранжировании не используются.
    history_years — календарное окно от loaded_at включительно; None снимает предел.
    """
    query_day = query_loaded_at[:10]
    cutoff = history_start(query_loaded_at, history_years) if history_years is not None else ''
    eligible = [r for r in candidates if r['loaded_at'][:10] < query_day and
                r['loaded_at'] >= cutoff and
                r.get('end_at') and r['end_at'] < query_loaded_at and r.get('status') == 'ok'
                and r.get('threshold_status') == 'ok' and r.get('threshold_frac', 0) > 0]
    days, found = set(), []
    for row in sorted(eligible, key=lambda r: (-r['similarity'], r['loaded_at'], r['news_id'])):
        day = row['loaded_at'][:10]
        if day not in days:
            found.append(row)
            days.add(day)
        if len(found) == count:
            break
    return found


def assess_neighbors(neighbors, min_similarity=.5):
    """Оценивает ровно пять neighbors; возвращает решение, направление и причины отказа.

    min_similarity — исследовательский минимум косинуса. Нужны пять совпадающих
    ненулевых знаков и медиана abs(return_frac)/threshold_frac >=1. Данные текущего
    результата функция не принимает. Прямолинейность/просадки не фильтруются.
    """
    result = {'selected': False, 'status': 'insufficient_neighbors', 'forecast_direction': None,
              'consensus_direction': None, 'median_ratio': None, 'median_return_frac': None,
              'min_similarity': min((r['similarity'] for r in neighbors), default=None),
              'neighbor_count': len(neighbors)}
    if len(neighbors) != 5:
        return result
    result['median_ratio'] = statistics.median(abs(r['return_frac']) / r['threshold_frac'] for r in neighbors)
    result['median_return_frac'] = statistics.median(r['return_frac'] for r in neighbors)
    if result['min_similarity'] < min_similarity:
        result['status'] = 'low_similarity'
        return result
    signs = [int(math.copysign(1, r['points'])) if r['points'] else 0 for r in neighbors]
    if 0 in signs or len(set(signs)) != 1:
        result['status'] = 'mixed_directions'
        return result
    result['consensus_direction'] = 'up' if signs[0] > 0 else 'down'
    if result['median_ratio'] < 1:
        result['status'] = 'weak_median_move'
        return result
    result.update(selected=True, status='selected', forecast_direction=result['consensus_direction'])
    return result
