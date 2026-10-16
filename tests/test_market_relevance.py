r"""Проверяет фильтр российского рынка без цен, Ollama и будущих данных.

Примеры отделяют экономический объект/событие от совпадений подстрок.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_market_relevance.py -v
"""

import unittest
from unittest.mock import patch
import csv
from pathlib import Path
import tempfile

from source.market_relevance import classify_title, compatible_groups


class MarketRelevanceTests(unittest.TestCase):
    """Проверяет включения, отказы, неопределённость и сопоставимость аналогов."""

    def test_core_events_are_included_with_explanation(self):
        """Выделяет российские решения и международные нефтяные события."""
        cases = [
            ('Банк России снизил ключевую ставку до 15%', 'monetary'),
            ('ЕС утвердил новые санкции против российских банков', 'geopolitics'),
            ('Россия и США проведут переговоры по Украине', 'geopolitics'),
            ('ОПЕК+ решила увеличить добычу нефти', 'oil'),
            ('Россия продлила обязательную продажу валютной выручки', 'fx'),
            ('Сбербанк утвердил дивиденды за 2025 год', 'corporate'),
        ]
        for title, category in cases:
            with self.subTest(title=title):
                result = classify_title(title)
                self.assertEqual(result['decision'], 'include')
                self.assertIn(category, result['categories'])
                self.assertTrue(result['reason'])
                self.assertTrue(result['groups'])

    def test_foreign_equities_gas_and_substring_noise_do_not_pass(self):
        """Отбрасывает прежние ложные аналоги и совпадения газ внутри газета."""
        for title in ['StoneX повысил целевую цену Securitize после решения SEC',
                      'Газета сообщила о выборах во Франции',
                      'Запасы природного газа в США выросли на 50 млрд кубических футов',
                      'ЦБ Турции снизил ключевую ставку',
                      'СМИ: Венгрия заблокировала переговоры о членстве Украины в ЕС',
                      'Нефть дешевеет после роста: США и Иран обсуждают Ормуз',
                      'FTSE 100 вырос на фоне новостей о нефти и Ормузе',
                      'Газпром запустил школьный конкурс рисунков']:
            with self.subTest(title=title):
                self.assertNotEqual(classify_title(title)['decision'], 'include')

    def test_ambiguous_title_is_not_invented_into_event(self):
        """Одного российского имени или слова нефть недостаточно для включения."""
        for title in ['Путин выступил с заявлением', 'Нефть: что будет дальше', 'Сбербанк сообщил новости']:
            self.assertEqual(classify_title(title)['decision'], 'uncertain')

    def test_analogs_need_same_object_and_event_type(self):
        """Не смешивает разные компании и противоположные решения по ставке."""
        cut = classify_title('Банк России снизил ключевую ставку')
        cut2 = classify_title('ЦБ РФ понизил ключевую ставку до 17%')
        hike = classify_title('Банк России повысил ключевую ставку')
        sber = classify_title('Сбербанк утвердил дивиденды')
        gazprom = classify_title('Газпром утвердил дивиденды')
        self.assertTrue(compatible_groups(cut, cut2))
        self.assertFalse(compatible_groups(cut, hike))
        self.assertFalse(compatible_groups(sber, gazprom))
        expected = classify_title('Эксперт ожидает, что Банк России снизит ключевую ставку')
        self.assertFalse(compatible_groups(cut, expected))

    def test_pipeline_filters_incompatible_candidates_before_ranking(self):
        """Чужая компания с лучшим косинусом не занимает место тематического аналога."""
        from source.build_analogs import make_decision
        relevance = classify_title('Сбербанк утвердил дивиденды')
        foreign = classify_title('Газпром утвердил дивиденды')
        rows = [dict(news_id=str(i), loaded_at=f'2026-09-0{i} 12:00:00',
                     end_at=f'2026-09-0{i} 12:30:01', status='ok', similarity=.8,
                     points=10, return_frac=.02, threshold_frac=.01, threshold_status='ok', relevance=relevance)
                for i in range(1, 6)]
        rows.append(dict(rows[0], news_id='wrong', similarity=.99, points=-10, relevance=foreign))
        decision = make_decision(rows, '2026-09-08 12:00:00', {'status': 'ok'}, .5, relevance)
        self.assertTrue(decision['selected'])
        self.assertNotIn('wrong', [r['news_id'] for r in decision['neighbors']])

    def test_restoration_is_not_disruption_and_intention_is_not_approval(self):
        """Различает восстановление/нарушение поставок и планы/утверждение дивидендов."""
        restored = classify_title('Трамп заявил о восстановлении поставок нефти через Ормузский пролив')
        disrupted = classify_title('Поставки нефти через Ормузский пролив прекратились из-за блокады')
        self.assertTrue(restored['groups'])
        self.assertTrue(any(':supply_restoration:' in group for group in restored['groups']))
        self.assertTrue(any(':supply_disruption:' in group for group in disrupted['groups']))
        self.assertFalse(compatible_groups(restored, disrupted))
        planned = classify_title('Сбербанк рассчитывает выплатить рекордные дивиденды по итогам 2026 года')
        approved = classify_title('Сбербанк утвердил дивиденды')
        self.assertEqual(planned['stage'], 'expectation')
        self.assertFalse(compatible_groups(planned, approved))
        self.assertEqual(classify_title('Сбер: дивиденды за 2026 год')['decision'], 'uncertain')

    def test_classification_preserves_human_review_on_rerun(self):
        """Повторный расчёт не стирает ручную оценку неизменного заголовка."""
        from source.russian_market import classify_month
        from source.embedding_report import write_csv
        news = [dict(news_id='a', loaded_at='2026-09-01 12:00:00', title='Сбербанк утвердил дивиденды', provider='test')]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            output = Path(folder)
            with patch('source.russian_market.load_news', return_value=(news, {})):
                classify_month(output, '2026-09', output)
                path = output / 'review_blind.csv'
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    rows = list(csv.DictReader(handle))
                rows[0]['human_decision'] = 'include'
                rows[0]['human_comment'] = 'Проверено вручную'
                write_csv(path, rows, list(rows[0]))
                classify_month(output, '2026-09', output)
                with path.open(encoding='utf-8-sig', newline='') as handle:
                    reviewed = list(csv.DictReader(handle))[0]
                self.assertEqual(reviewed['human_comment'], 'Проверено вручную')


if __name__ == '__main__':
    unittest.main()
