r"""Проверки минутных свечей и ценовых реакций после получения новости.

Контрольные цены и время заданы вручную: исключаются искусственные доли секунды,
будущие свечи в нормировке и неполные интервалы. Ollama и исходные ZIP не нужны.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import importlib
import unittest

import pandas as pd


class ReactionTests(unittest.TestCase):
    """Проверяет временные границы, свечи и метрики на искусственных сделках."""

    def modules(self):
        """Возвращает модули данных/метрик; отсутствие реализации даёт провал теста."""
        try:
            return importlib.import_module('source.price_data'), importlib.import_module('source.price_reactions')
        except ModuleNotFoundError:
            self.fail('Модули расчёта реакций ещё не реализованы')

    def ticks(self, times, prices):
        """Создаёт сделки с times и prices и единичным объёмом для ручных примеров."""
        return pd.DataFrame({'last': prices, 'volume': [1] * len(prices)},
                            index=pd.DatetimeIndex(pd.to_datetime(times)).as_unit('ns'))

    def test_entry_excludes_entire_synthetic_second_and_exit_excludes_endpoint_second(self):
        """Ловит вход по условным миллисекундам и включение секунды после конца окна."""
        data, calc = self.modules()
        ticks = self.ticks(['2026-09-01 12:00:00.001', '2026-09-01 12:00:00.999',
                            '2026-09-01 12:00:01.000', '2026-09-01 12:00:59.000',
                            '2026-09-01 12:01:00.000', '2026-09-01 12:01:01.000'],
                           [900, 999, 100, 120, 130, 9999])
        result = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 12:00:00',
                                       1, neutral_points=1)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['entry_at'], '2026-09-01 12:00:01')
        self.assertEqual(result['points'], 30)
        self.assertEqual(result['max_up_points'], 30)
        self.assertEqual(result['exit_at'], '2026-09-01 12:01:00')

    def test_minute_candles_do_not_fill_missing_minutes(self):
        """Ловит неправильный порядок OHLC, объём и искусственное заполнение пустых минут."""
        data, _ = self.modules()
        ticks = self.ticks(['2026-09-01 12:00:01', '2026-09-01 12:00:20',
                            '2026-09-01 12:00:59', '2026-09-01 12:02:01'], [100, 90, 110, 120])
        bars = data.minute_bars(ticks)
        self.assertEqual(len(bars), 2)
        self.assertEqual(list(bars.iloc[0][['open', 'high', 'low', 'close', 'volume']]), [100, 110, 90, 110, 3])
        self.assertNotIn(pd.Timestamp('2026-09-01 12:01:00'), bars.index)

    def test_gap_and_late_entry_are_not_valid_reactions(self):
        """Ловит перенос входа через отсутствие торгов и расчёт через длинный пробел."""
        data, calc = self.modules()
        ticks = self.ticks(['2026-09-01 12:00:01', '2026-09-01 12:04:00',
                            '2026-09-01 12:05:00', '2026-09-01 12:06:00'], [100, 120, 130, 140])
        r = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 12:00:00', 5)
        self.assertEqual(r['status'], 'gap')
        self.assertIsNone(r['points'])
        r = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 11:00:00', 5)
        self.assertEqual(r['status'], 'entry_wait')

    def test_excursions_use_ticks_and_drawdown_is_not_mae(self):
        """Ловит смешение отката от максимума и отклонения от цены входа."""
        data, calc = self.modules()
        ticks = self.ticks(['2026-09-01 12:00:01', '2026-09-01 12:00:10',
                            '2026-09-01 12:00:20', '2026-09-01 12:00:30',
                            '2026-09-01 12:01:00'], [100, 95, 140, 110, 130])
        r = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 12:00:00', 1, neutral_points=1)
        self.assertEqual(r['mfe_points'], 40)
        self.assertEqual(r['mae_points'], 5)
        self.assertEqual(r['max_pullback_points'], 30)
        self.assertEqual(r['direction'], 'up')

    def test_missing_endpoint_and_crossing_day_remain_incomplete(self):
        """Ловит выдачу движения за час при наличии всего минуты котировок."""
        data, calc = self.modules()
        ticks = self.ticks(['2026-09-01 23:49:01', '2026-09-01 23:49:30'], [100, 110])
        r = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 23:49:00', 5)
        self.assertEqual(r['status'], 'endpoint_missing')
        self.assertIsNone(r['direction'])
        r = calc.measure_reaction(ticks, data.minute_bars(ticks), '2026-09-01 23:49:00', 30)
        self.assertEqual(r['status'], 'cross_day')

    def test_pre_volatility_uses_only_completed_bars_before_news(self):
        """Ловит утечку будущего роста в нормировку силы движения."""
        _, calc = self.modules()
        index = pd.date_range('2026-09-01 10:59:00', periods=63, freq='min')
        bars = pd.DataFrame({'close': [100 + i % 2 for i in range(63)]}, index=index)
        before = calc.pre_volatility(bars, pd.Timestamp('2026-09-01 12:00:30'))
        bars.loc[bars.index >= pd.Timestamp('2026-09-01 12:00:00'), 'close'] = 9999
        after = calc.pre_volatility(bars, pd.Timestamp('2026-09-01 12:00:30'))
        self.assertGreater(before, 0)
        self.assertEqual(before, after)

    def test_pre_volatility_rejects_stale_history_after_break(self):
        """Ловит использование старого непрерывного участка после перерыва торгов."""
        _, calc = self.modules()
        index = pd.date_range('2026-09-01 10:59:00', periods=61, freq='min')
        bars = pd.DataFrame({'close': [100 + i % 2 for i in range(61)]}, index=index)
        self.assertIsNone(calc.pre_volatility(bars, pd.Timestamp('2026-09-01 13:00:00')))


if __name__ == '__main__':
    unittest.main()
