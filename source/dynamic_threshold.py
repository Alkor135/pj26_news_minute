r"""Вычисляет динамический порог движения только по прошлым котировкам.

Выбирает предшествующие торговые дни, близкие часы и предсобытийную волатильность.
Квантиль модулей фоновых доходностей, нормированных на собственную sigma,
масштабируется на sigma текущей новости. При нехватке данных фиксированной
подстановки нет. Эпизоды текущего дня и будущие интервалы исключены.

Использование из корня:
  .\.venv\Scripts\python.exe build_analogs.py --quantile 0.75 --lookback-days 20
  .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dynamic_analogs.py -v
Модуль source отдельно не запускается; параметры функций описаны ниже.
"""

import math

import numpy as np
import pandas as pd


class BackgroundIndex:
    """Хранит фон одного инструмента в массивах по горизонтам для повторного поиска."""

    def __init__(self, rows):
        """Индексирует rows с day/start_at/end_at/horizon/return_frac/sigma_pre."""
        groups = {}
        for row in rows:
            if row.get('sigma_pre') and math.isfinite(row['sigma_pre']) and row['sigma_pre'] > 0:
                groups.setdefault(row['horizon'], []).append(row)
        self.groups = {}
        for horizon, items in groups.items():
            days = np.array([r['day'] for r in items])
            self.groups[horizon] = {
                'day': days, 'all_days': np.unique(days),
                'minute': np.array([int(r['start_at'][11:13]) * 60 + int(r['start_at'][14:16]) for r in items]),
                'weekend': np.array([pd.Timestamp(str(day)).dayofweek >= 5 for day in days]),
                'end': np.array([r['end_at'] for r in items]),
                'sigma': np.array([r['sigma_pre'] for r in items], dtype=float),
                'return': np.array([r['return_frac'] for r in items], dtype=float),
            }


def dynamic_threshold(background, loaded_at, sigma_pre, reference_price, horizon, *,
                      quantile=.75, lookback_days=20, time_band_minutes=60,
                      min_samples=30, min_days=5, volatility_fraction=.5):
    """Возвращает порог/покрытие для loaded_at без использования будущего.

    background — список фоновых интервалов или BackgroundIndex одного инструмента;
    sigma_pre и reference_price известны до новости. horizon — минуты; quantile —
    квантиль; lookback_days — предыдущие торговые дни; time_band_minutes — полоса
    часов; min_samples/min_days — минимальная история; volatility_fraction — доля
    ближайших по log(sigma) интервалов. При отказе threshold_frac/points равны None.
    """
    if (not 0 < quantile < 1 or not 0 < volatility_fraction <= 1 or
            min(lookback_days, min_samples, min_days) < 1 or time_band_minutes < 0):
        raise ValueError('Некорректные настройки динамического порога')
    result = {'status': 'insufficient_pre_history', 'threshold_frac': None, 'threshold_points': None,
              'sample_count': 0, 'day_count': 0, 'history_max_end': None, 'history_first_day': None,
              'quantile': quantile, 'candidate_count': 0}
    if (sigma_pre is None or reference_price is None or not math.isfinite(sigma_pre) or
            not math.isfinite(reference_price) or sigma_pre <= 0 or reference_price <= 0):
        return result
    index = background if isinstance(background, BackgroundIndex) else BackgroundIndex(background)
    data = index.groups.get(horizon)
    result['status'] = 'insufficient_background'
    if data is None:
        return result
    when = pd.Timestamp(loaded_at)
    day, stamp = when.strftime('%Y-%m-%d'), when.strftime('%Y-%m-%d %H:%M:%S')
    previous_days = data['all_days'][data['all_days'] < day][-lookback_days:]
    mask = (np.isin(data['day'], previous_days) & (data['end'] < stamp) &
            (data['weekend'] == (when.dayofweek >= 5)) &
            (np.abs(data['minute'] - (when.hour * 60 + when.minute)) <= time_band_minutes))
    positions = np.flatnonzero(mask)
    result['candidate_count'] = len(positions)
    if len(positions) < min_samples:
        return result
    distances = np.abs(np.log(data['sigma'][positions] / sigma_pre))
    keep = max(min_samples, math.ceil(len(positions) * volatility_fraction))
    positions = positions[np.argsort(distances, kind='stable')[:keep]]
    days = np.unique(data['day'][positions])
    result.update(sample_count=len(positions), day_count=len(days),
                  history_max_end=str(max(data['end'][positions])), history_first_day=str(min(days)))
    if len(days) < min_days:
        return result
    scaled = np.abs(data['return'][positions]) / data['sigma'][positions] * sigma_pre
    threshold = float(np.quantile(scaled, quantile, method='linear'))
    if not math.isfinite(threshold) or threshold <= 0:
        result['status'] = 'zero_threshold'
        return result
    result.update(status='ok', threshold_frac=threshold, threshold_points=float(reference_price * threshold))
    return result
