r"""Меняет строгость динамического порога, сохраняя пять выбранных аналогов.

Эффективный порог T=k*T_base, где T_base рассчитан из прошлой истории. Переоценка
не выбирает новых соседей и не использует исход запроса для принятия решения.
При повторном изменении k масштаб всегда берётся от сохранённой базы, без накопления.
Оценка результата показана и по изменяемому T, и по постоянному базовому T_base.

Использование:
  .\.venv\Scripts\python.exe tune_threshold.py --multipliers 0.25 0.5 0.75 1
  .\.venv\Scripts\python.exe view_news_finplot.py --threshold-multiplier 0.5
Модуль source отдельно не запускается, исходные оценки/БД не изменяет.
"""

from copy import deepcopy
import math

from source.analog_selection import assess_neighbors


def validate_multiplier(value):
    """Возвращает конечный положительный float value или возбуждает ValueError."""
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError('Коэффициент порога k должен быть конечным положительным числом')
    return value


def rescale_score(score, multiplier, min_similarity=.5):
    """Возвращает копию score с эффективным порогом multiplier*T_base.

    score — сохранённая оценка с неизменными neighbors и threshold; min_similarity
    берётся из параметров исходного расчёта. Поддерживает прежние оценки с k=1.
    Вход не изменяет. direction_correct/strong_correct относятся только к выбранным
    прогнозам с известным исходом; base_strong_correct использует постоянный T_base.
    """
    k = validate_multiplier(multiplier)
    previous = validate_multiplier(score.get('threshold_multiplier', 1.))
    result = deepcopy(score)
    for item in result['neighbors'] + [result['threshold']]:
        for key in ('threshold_frac', 'threshold_points'):
            base = item.get('base_' + key, item.get(key) / previous if item.get(key) is not None else None)
            item['base_' + key] = base
            item[key] = base * k if base is not None else None
    actual_base = score.get('actual_ratio_base', score.get('actual_ratio') * previous if score.get('actual_ratio') is not None else None)
    result.update(assess_neighbors(result['neighbors'], min_similarity))
    if result['threshold']['status'] != 'ok':
        result.update(selected=False, status='query_threshold_unavailable', forecast_direction=None)
    result.update(threshold_multiplier=k, actual_ratio_base=actual_base,
                  actual_ratio=actual_base / k if actual_base is not None else None,
                  median_ratio_base=result['median_ratio'] * k if result['median_ratio'] is not None else None,
                  direction_correct=None, strong_correct=None, base_strong_correct=None)
    if result['selected'] and result.get('actual_status') == 'ok' and actual_base is not None:
        correct = result.get('actual_direction') == result['forecast_direction']
        result.update(direction_correct=correct, strong_correct=correct and result['actual_ratio'] >= 1,
                      base_strong_correct=correct and actual_base >= 1)
    return result
