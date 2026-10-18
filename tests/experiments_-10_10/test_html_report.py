r"""Проверяет HTML-отчёт по готовым результатам без Ollama и повторных расчётов.

Проверяет числа, экранирование новостей и создание только HTML во временной папке tests.
Проверяет направление по ценам, пункты условного сигнала и отдельный вывод о готовности к торговле.
Запуск: .\.venv\Scripts\python.exe -m unittest discover -s tests/experiments_-10_10 -p test_html_report.py -v
"""

import importlib.util
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class HtmlReportTests(unittest.TestCase):
    """Проверяет достоверность и безопасное отображение сохранённых результатов."""

    def setUp(self):
        """Загружает модуль отчёта и создаёт синтетическую проверку трёх классов."""
        spec = importlib.util.spec_from_file_location('html_report', ROOT / 'experiments_-10_10/build_html_report.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.report = dict(task='extremes', ollama_name='example', period={'validation_start': '2026-08-01', 'cutoff': '2026-09-01'},
                           predictions=[dict(news_id='a', instrument='MIX', horizon=5, actual=-8,
                                             predicted='none', adapter_predicted='none'),
                                        dict(news_id='b', instrument='RTS', horizon=15, actual=0,
                                             predicted='positive', adapter_predicted='none')])

    def test_html_explains_missed_extremes_and_false_signals(self):
        """Отчёт показывает пропущенные экстремумы, ложные сигналы и сравнение с адаптером."""
        result = self.module.render_report(self.report, {}, '')
        self.assertIn('0 из 1', result)
        self.assertIn('До экспорта', result)
        self.assertIn('Ложные сигналы', result)
        self.assertIn('filterRows', result)
        self.assertIn('negative', result)

    def test_news_html_is_escaped(self):
        """Текст новости не превращается в исполняемый HTML."""
        news = {('a', 'MIX', 5): {'title': '<img src=x onerror=alert(1)>', 'loaded_at': '2026-08-01'}}
        result = self.module.render_report(self.report, news, '')
        self.assertNotIn('<img src=x', result)
        self.assertIn('&lt;img', result)

    def test_main_writes_html_without_modifying_inputs(self):
        """Отдельный запуск читает JSON и создаёт HTML без наличия остальных этапов."""
        with tempfile.TemporaryDirectory(prefix='tmp_html_', dir=ROOT / 'tests') as temp:
            run = Path(temp)
            source = run / 'check_model_report.json'
            source.write_text(json.dumps(self.report), encoding='utf-8')
            before = source.read_bytes()
            self.assertEqual(self.module.main(['--run', str(run), '--data', str(run / 'missing.jsonl')]), 0)
            self.assertTrue((run / 'check_model_report.html').is_file())
            self.assertEqual(source.read_bytes(), before)

    def test_bad_data_hash_does_not_attach_wrong_titles(self):
        """Отчёт не подмешивает новости из изменённой разметки."""
        with tempfile.TemporaryDirectory(prefix='tmp_html_', dir=ROOT / 'tests') as temp:
            data = Path(temp) / 'scores.jsonl'
            data.write_text('{}', encoding='utf-8')
            news, warning = self.module.load_news(data, '0' * 64)
            self.assertEqual(news, {})
            self.assertIn('SHA-256', warning)

    def test_short_signal_points_reverse_price_move(self):
        """Падение на 175 пунктов соответствует +175 по отрицательному сигналу."""
        result = self.module.price_outcome({'predicted': 'negative'},
                                           {'entry_price': 1000, 'exit_price': 825, 'points': -175})
        self.assertEqual(result['movement'], -175)
        self.assertEqual(result['signed_points'], 175)
        self.assertEqual(result['direction'], 'correct')

    def test_loaded_quotes_are_preserved_after_hash_check(self):
        """После проверки SHA-256 сохраняются цены, время и флаги выбранной разметки."""
        with tempfile.TemporaryDirectory(prefix='tmp_html_', dir=ROOT / 'tests') as temp:
            data = Path(temp) / 'scores.jsonl'
            row = dict(news_id='a', instrument='MIX', horizon=5, title='Новость',
                       points=-175, entry_price=1000, exit_price=825, entry_at='2026-08-01 10:00:01',
                       exit_at='2026-08-01 10:05:00', flags=['continuous_contract_unverified'])
            data.write_text(json.dumps(row), encoding='utf-8')
            news, warning = self.module.load_news(data, hashlib.sha256(data.read_bytes()).hexdigest())
            self.assertEqual(warning, '')
            self.assertEqual(news[('a', 'MIX', 5)]['points'], -175)
            self.assertEqual(news[('a', 'MIX', 5)]['exit_price'], 825)
            self.assertIn('continuous_contract_unverified', news[('a', 'MIX', 5)]['flags'])

    def test_neutral_score_does_not_hide_price_direction(self):
        """Направление проверяется по котировкам даже при фактическом балле ноль."""
        result = self.module.price_outcome({'predicted': 'positive', 'actual': 0}, {'points': -20})
        self.assertEqual(result['signed_points'], -20)
        self.assertEqual(result['direction'], 'wrong')

    def test_missing_or_inconsistent_quotes_not_presented_as_profit(self):
        """Без котировок результат неизвестен; противоречивые цены не принимаются."""
        result = self.module.price_outcome({'predicted': 'positive'}, {})
        self.assertIsNone(result['signed_points'])
        self.assertEqual(result['direction'], 'unknown')
        with self.assertRaises(ValueError):
            self.module.price_outcome({'predicted': 'positive'},
                                       {'entry_price': 100, 'exit_price': 120, 'points': -20})

    def test_false_signal_table_has_direction_and_points(self):
        """Ошибка экстремального класса показывается отдельно от верного направления."""
        news = {('b', 'RTS', 15): {'title': 'Рост', 'points': 20, 'entry_price': 100, 'exit_price': 120}}
        result = self.module.render_report(self.report, news, '')
        self.assertIn('Можно использовать для торговли?', result)
        self.assertIn('Готовность к реальной торговле не подтверждена', result)
        self.assertIn('Направление угадано', result)
        self.assertIn('По сигналу, пт', result)
        self.assertIn('+20', result)
        self.assertIn('Направление угадано: 1 из 1', result)

    @unittest.skipUnless(shutil.which('node'), 'Для проверки JavaScript нужен Node.js')
    def test_javascript_filters_without_browser(self):
        """Исполняет встроенные фильтры на минимальном DOM без браузера, сервера и сети."""
        html = self.module.render_report(self.report, {}, '')
        script = html.rsplit('<script>', 1)[1].split('</script>', 1)[0]
        probe = r'''
// Проверяет фильтры отчёта на двух синтетических строках без браузера.
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const script=JSON.parse(fs.readFileSync(0,'utf8'));
const rows=[
 {dataset:{kind:'errors missed',instrument:'MIX',horizon:'5'},textContent:'Новость А'},
 {dataset:{kind:'errors signals false-signals',instrument:'RTS',horizon:'15'},textContent:'Новость Б'}
];
const fields={};
// Имитирует только поля фильтров и регистрацию обработчиков событий.
for(const id of ['search','kind','instrument','horizon','visible-count']){
 fields[id]={value:'',textContent:'',addEventListener(){/* Ввод проверяется прямым вызовом фильтра. */}};
}
// Предоставляет данным отчёта минимальный интерфейс поиска элементов.
const context={document:{querySelectorAll(){return rows;},getElementById(id){return fields[id];}}};
vm.runInNewContext(script,context);
assert.equal(rows.filter(r=>!r.hidden).length,2);
fields.kind.value='false-signals';context.filterRows();
assert.equal(rows.filter(r=>!r.hidden).length,1);assert.equal(rows[1].hidden,false);
fields.instrument.value='MIX';context.filterRows();assert.equal(rows.every(r=>r.hidden),true);
fields.kind.value='missed';context.filterRows();assert.equal(rows[0].hidden,false);
fields.search.value='НОВОСТЬ Б';context.filterRows();assert.equal(rows.every(r=>r.hidden),true);
'''
        result = subprocess.run([shutil.which('node'), '-e', probe], input=json.dumps(script),
                                capture_output=True, encoding='utf-8')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
