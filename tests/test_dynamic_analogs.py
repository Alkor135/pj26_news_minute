r"""Проверки динамического порога и пяти аналогов без будущих данных.

Синтетические примеры ловят подсматривание, повторные дни, замену неудачного
соседа и влияние результата текущей новости на отбор.
Даты могут быть непоследовательными, с промежутками в несколько месяцев.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -v
"""

import importlib
import unittest


class DynamicAnalogTests(unittest.TestCase):
    """Проверяет временные условия, квантиль и выбор пяти независимых дат."""

    def module(self, name):
        """Возвращает модуль name или проваливает тест, если реализация отсутствует."""
        try:
            return importlib.import_module('source.' + name)
        except ModuleNotFoundError:
            self.fail('Отсутствует реализация ' + name)

    def background(self):
        """Возвращает четыре вручную заданных прошлых фоновых интервала."""
        return [dict(day=f'2026-09-0{i}', start_at=f'2026-09-0{i} 12:00:01',
                     end_at=f'2026-09-0{i} 12:30:01', horizon=30,
                     return_frac=move, sigma_pre=.01) for i, move in enumerate([.01, .02, .03, .04], 1)]

    def candidates(self):
        """Возвращает семь соседей, включая один повтор дня и несовпадающий шестой день."""
        rows = []
        for i in range(1, 7):
            rows.append(dict(news_id=str(i), loaded_at=f'2026-09-0{i} 12:00:00',
                             end_at=f'2026-09-0{i} 12:30:01', status='ok',
                             similarity=1 - i * .05, points=10., return_frac=.02,
                             threshold_frac=.01, threshold_status='ok'))
        rows.insert(1, {**rows[0], 'news_id': 'same-day', 'similarity': .94})
        return rows

    def test_threshold_is_dynamic_and_excludes_query_day_and_future(self):
        """Ловит включение текущего дня/будущего и фиксированный денежный порог."""
        m = self.module('dynamic_threshold')
        bg = self.background()
        bg += [dict(day='2026-09-07', start_at='2026-09-07 11:00:00', end_at='2026-09-07 11:30:00',
                    horizon=30, return_frac=9., sigma_pre=.01)]
        kwargs = dict(horizon=30, quantile=.75, min_samples=4, min_days=4, volatility_fraction=1.)
        a = m.dynamic_threshold(bg, '2026-09-07 12:00:00', .01, 100., **kwargs)
        b = m.dynamic_threshold(bg, '2026-09-07 12:00:00', .02, 100., **kwargs)
        self.assertEqual(a['status'], 'ok')
        self.assertAlmostEqual(a['threshold_frac'], .0325)
        self.assertAlmostEqual(a['threshold_points'], 3.25)
        self.assertAlmostEqual(b['threshold_frac'], .065)
        self.assertEqual(a['sample_count'], 4)
        self.assertLess(a['history_max_end'], '2026-09-07 00:00:00')

    def test_threshold_refuses_zero_or_insufficient_background(self):
        """Ловит тихую подстановку фиксированного порога при нехватке/нулевом фоне."""
        m = self.module('dynamic_threshold')
        r = m.dynamic_threshold(self.background(), '2026-09-07 12:00:00', .01, 100., 30)
        self.assertEqual(r['status'], 'insufficient_background')
        self.assertIsNone(r['threshold_frac'])
        bg = [{**r, 'return_frac': 0.} for r in self.background()]
        r = m.dynamic_threshold(bg, '2026-09-07 12:00:00', .01, 100., 30,
                                min_samples=4, min_days=4, volatility_fraction=1.)
        self.assertEqual(r['status'], 'zero_threshold')

    def test_neighbors_have_five_distinct_past_dates(self):
        """Ловит повтор даты, включение дня запроса и незавершённое окно реакции."""
        m = self.module('analog_selection')
        items = self.candidates()
        items += [{**items[0], 'news_id': 'same-query-day', 'loaded_at': '2026-09-08 09:00:00', 'similarity': 1.}]
        items += [{**items[0], 'news_id': 'future-end', 'end_at': '2026-09-09 12:00:00', 'similarity': 1.}]
        found = m.pick_neighbors(items, '2026-09-08 12:00:00')
        self.assertEqual([r['news_id'] for r in found], ['1', '2', '3', '4', '5'])

    def test_disagreeing_neighbor_is_not_replaced_by_sixth(self):
        """Ловит поиск пяти удобных исходов вместо проверки пяти ближайших."""
        m = self.module('analog_selection')
        items = self.candidates()
        items[0]['points'], items[0]['return_frac'] = -10., -.02
        result = m.assess_neighbors(m.pick_neighbors(items, '2026-09-08 12:00:00'), min_similarity=.5)
        self.assertFalse(result['selected'])
        self.assertEqual(result['status'], 'mixed_directions')

    def test_dates_can_be_far_apart_and_need_not_be_consecutive(self):
        """Принимает пять произвольных прошлых дат с большими промежутками между ними."""
        m = self.module('analog_selection')
        template = self.candidates()[0]
        days = ['2025-10-03', '2026-01-08', '2026-04-14', '2026-07-21', '2026-09-01']
        rows = [dict(template, news_id=str(i), loaded_at=day + ' 12:00:00', end_at=day + ' 12:30:01')
                for i, day in enumerate(days)]
        found = m.pick_neighbors(rows, '2026-09-08 12:00:00')
        self.assertEqual({r['loaded_at'][:10] for r in found}, set(days))

    def test_median_magnitude_allows_one_weak_neighbor_and_uses_no_query_outcome(self):
        """Ловит требование силы всех пяти вместо медианы и фильтрацию по устойчивости."""
        m = self.module('analog_selection')
        items = self.candidates()
        items[0].update(return_frac=.001, stable=False, er=.01, mae_points=999)
        found = m.pick_neighbors(items, '2026-09-08 12:00:00')
        result = m.assess_neighbors(found, min_similarity=.5)
        self.assertTrue(result['selected'])
        self.assertEqual(result['forecast_direction'], 'up')
        self.assertAlmostEqual(result['median_ratio'], 2.)


if __name__ == '__main__':
    unittest.main()
