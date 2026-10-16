r"""Ретроспективные метрики движения цены после получения новости.

Вход — первая сделка следующей секунды после loaded_at+latency, поскольку доли
секунды Finam искусственные. Выход — последняя сделка строго до секунды окончания
горизонта. Неполные окна не получают направление/доходность. Будущие цены
используются только для разметки; волатильность берётся из завершённых прошлых минут.

Использование:
  .\.venv\Scripts\python.exe build_reactions.py --latency-seconds 0
  .\.venv\Scripts\python.exe view_news_finplot.py --instrument RTS --horizon 60
Параметры measure_reaction описаны в функции; модуль source отдельно не запускается.
"""

import math

import numpy as np
import pandas as pd

HORIZONS = (5, 15, 30, 60, 120)
METRIC_VERSION = 'loaded-at-whole-second-fresh-prehistory-v2'


def pre_volatility(bars, available_at):
    """Возвращает sigma 60 доходностей до available_at или None при нехватке истории.

    bars — минутные свечи. Требуются 61 подряд идущая завершённая минута;
    последняя свеча должна непосредственно предшествовать минуте новости;
    открытая в момент новости свеча полностью исключается.
    """
    cutoff = bars.index.searchsorted(available_at.floor('min'), side='left')
    history = bars.iloc[max(0, cutoff - 61):cutoff]
    if (len(history) != 61 or
            history.index[-1] != available_at.floor('min') - pd.Timedelta(minutes=1) or
            not (np.diff(history.index.as_unit('ns').asi8) == 60_000_000_000).all()):
        return None
    closes = history['close'].to_numpy(dtype=float)
    sigma = float(np.std(np.diff(closes) / closes[:-1], ddof=1))
    return sigma if math.isfinite(sigma) and sigma > 0 else None


def measure_reaction(ticks, bars, loaded_at, horizon, neutral_points=0., latency_seconds=0.,
                     max_entry_wait_seconds=60., max_exit_age_seconds=60., max_gap_seconds=120.):
    """Возвращает статус и метрики реакции по ticks/bars после loaded_at.

    horizon — астрономические минуты после входа; neutral_points — нейтральная
    зона в пунктах. latency_seconds — задержка обработки; max_entry_wait_seconds,
    max_exit_age_seconds и max_gap_seconds ограничивают отсутствие сделок.
    Все времена локальные МСК без смещения. Неполный результат содержит None
    вместо движения; метка stable вычисляется по будущему и не является сигналом.
    """
    if horizon <= 0 or min(neutral_points, latency_seconds, max_entry_wait_seconds,
                            max_exit_age_seconds, max_gap_seconds) < 0:
        raise ValueError('Горизонт должен быть положительным, остальные параметры — неотрицательными')
    available = pd.Timestamp(loaded_at)
    decision = available + pd.Timedelta(seconds=latency_seconds)
    result = {'horizon': horizon, 'loaded_at': str(available), 'status': 'no_entry',
              'entry_at': None, 'entry_price': None, 'exit_at': None, 'end_at': None,
              'exit_price': None, 'points': None, 'return_pct': None, 'direction': None,
              'mfe_points': None, 'mae_points': None, 'max_pullback_points': None,
              'max_up_points': None, 'max_down_points': None, 'er': None,
              'direction_fraction': None, 'sigma_pre': None, 'z_score': None, 'stable': False,
              'flags': ['continuous_contract_unverified', 'synthetic_fractions']}
    entry_i = ticks.index.searchsorted(decision.floor('s') + pd.Timedelta(seconds=1), side='left')
    if entry_i == len(ticks):
        return result
    entry = ticks.index[entry_i].floor('s')
    result.update(entry_at=str(entry), entry_price=float(ticks['last'].iloc[entry_i]))
    if (entry - decision).total_seconds() > max_entry_wait_seconds:
        result['status'] = 'entry_wait'
        return result
    end = entry + pd.Timedelta(minutes=horizon)
    result['end_at'] = str(end)
    if entry.date() != available.date() or end.date() != entry.date():
        result['status'] = 'cross_day'
        return result
    exit_i = ticks.index.searchsorted(end, side='left') - 1
    exit_at = ticks.index[exit_i].floor('s')
    if exit_i <= entry_i or (end - exit_at).total_seconds() > max_exit_age_seconds:
        result['status'] = 'endpoint_missing'
        return result
    seconds = ticks.index[entry_i:exit_i + 1].as_unit('ns').asi8 // 1_000_000_000
    if len(seconds) > 1 and (np.diff(seconds) > max_gap_seconds).any():
        result['status'] = 'gap'
        return result
    prices = ticks['last'].iloc[entry_i:exit_i + 1].to_numpy(dtype=float)
    p0, p1 = prices[0], prices[-1]
    points = float(p1 - p0)
    sign = float(np.sign(points))
    signed = sign * (prices - p0)
    known_bars = bars[(bars.index >= entry.floor('min')) &
                      (bars.index + pd.Timedelta(minutes=1) <= end)]
    path = np.r_[p0, known_bars['close'].to_numpy(dtype=float), p1]
    distance = float(np.abs(np.diff(path)).sum())
    er = abs(points) / distance if distance else None
    fraction = float((sign * (known_bars['close'] - p0) > 0).mean()) if sign and len(known_bars) else None
    sigma = pre_volatility(bars, available)
    z = abs(points / p0) / (sigma * math.sqrt(horizon)) if sigma else None
    mae = float(max(0., -signed.min())) if sign else None
    pullback = float((np.maximum.accumulate(signed) - signed).max()) if sign else None
    result.update(status='ok', exit_at=str(exit_at), exit_price=float(p1), points=points,
                  return_pct=float(points / p0 * 100),
                  direction='flat' if abs(points) <= neutral_points else ('up' if points > 0 else 'down'),
                  mfe_points=float(signed.max()) if sign else None, mae_points=mae,
                  max_pullback_points=pullback, max_up_points=float(max(0., prices.max() - p0)),
                  max_down_points=float(max(0., p0 - prices.min())), er=er,
                  direction_fraction=fraction, sigma_pre=sigma, z_score=z)
    if sigma is None:
        result['flags'].append('insufficient_pre_history')
    result['stable'] = bool(result['direction'] != 'flat' and z is not None and z >= 2 and
                            er is not None and er >= .45 and fraction is not None and fraction >= .7
                            and mae <= .5 * abs(points) and pullback <= .5 * abs(points))
    return result
