r"""Чтение дневных ZIP тиков Finam и построение минутных свечей.

Архивы открываются только для чтения. Проверяются формат, цены, объёмы, дата и
порядок строк; сделки с одинаковым временем не удаляются. Пустые минуты не
заполняются. Дробные секунды источника искусственные и не задают точное время.

Модуль используется командами:
  .\.venv\Scripts\python.exe build_reactions.py
  .\.venv\Scripts\python.exe view_news_finplot.py --instrument MIX --horizon 30
Вспомогательный модуль source. Самостоятельный запуск не нужен. read_ticks(path) возвращает сделки и паспорт
архива; minute_bars(ticks) возвращает OHLCV с числом сделок за минуту.
"""

import hashlib
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

DEFAULT_TICK_DIRS = {
    'MIX': Path(r'C:\data_quote\data_finam_MIX_tick_zip'),
    'RTS': Path(r'C:\data_quote\data_finam_RTS_tick_zip'),
}


def read_ticks(path):
    """Читает ZIP path; возвращает проверенную таблицу сделок и метаданные источника.

    В ZIP ожидается один CSV datetime,last,volume за дату YYYYMMDD имени архива.
    При некорректных данных возникает ValueError, исходные строки не исправляются.
    """
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if not n.endswith('/')]
        if len(names) != 1 or not names[0].lower().endswith('.csv'):
            raise ValueError(f'{path.name}: ожидается один CSV')
        with archive.open(names[0]) as stream:
            ticks = pd.read_csv(stream)
    if list(ticks.columns) != ['datetime', 'last', 'volume']:
        raise ValueError(f'{path.name}: неверные колонки')
    if ticks.empty:
        raise ValueError(f'{path.name}: пустые тики')
    stamps = pd.DatetimeIndex(pd.to_datetime(ticks.pop('datetime'), format='ISO8601')).as_unit('ns')
    if stamps.tz is not None or stamps.hasnans or not stamps.is_monotonic_increasing:
        raise ValueError(f'{path.name}: нарушено время или порядок строк')
    expected_day = pd.Timestamp(path.stem).date()
    if any(d != expected_day for d in stamps.date):
        raise ValueError(f'{path.name}: сделки другой даты')
    for col in ('last', 'volume'):
        ticks[col] = pd.to_numeric(ticks[col], errors='raise')
        if not np.isfinite(ticks[col]).all() or (ticks[col] <= 0).any():
            raise ValueError(f'{path.name}: неверные {col}')
    ticks.index = stamps
    ticks.index.name = 'datetime'
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    metadata = {'path': str(path.resolve()), 'sha256': digest,
                'rows': len(ticks), 'first': str(stamps[0]), 'last': str(stamps[-1]),
                'timezone': 'Europe/Moscow', 'synthetic_fractions': True,
                'continuous_contract_unverified': True}
    return ticks, metadata


def minute_bars(ticks):
    """Агрегирует ticks в OHLCV по [минута, следующая минута), не заполняя пробелы."""
    bars = ticks['last'].resample('min', closed='left', label='left').ohlc()
    bars['volume'] = ticks['volume'].resample('min').sum()
    bars['tick_count'] = ticks['last'].resample('min').size()
    return bars.dropna(subset=['open'])
