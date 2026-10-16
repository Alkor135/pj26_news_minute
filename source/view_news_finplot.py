r"""Показывает минутные свечи MIX/RTS и исторические реакции на новости в finplot.

Читает готовую reactions.sqlite3 без изменения, позволяет выбрать инструмент,
горизонт, новость, режим пяти прошлых аналогов или фильтр устойчивости. Показывает время загрузки, вход, конец
окна, минутные маркеры соседних новостей полной RSS-ленты и метрики движения.
Контекст содержит исходные загрузки, включая повторы, без дедупликации заголовков.
Для БД российского рынка показывает категорию, основания фильтра и совместимые аналоги.
Для годовой БД показывает длину истории и отдельный период проверки.
Поле «Порог × k» меняет строгость без пересчёта котировок и без замены аналогов.
Цвет и метка устойчивости вычислены по будущим ценам, это не торговый сигнал.

Примеры запуска из корня проекта после build_reactions.py:
  .\.venv\Scripts\python.exe view_news_finplot.py
  .\.venv\Scripts\python.exe view_news_finplot.py --instrument RTS --horizon 60
  .\.venv\Scripts\python.exe view_news_finplot.py --all-events --search санкц
  .\.venv\Scripts\python.exe view_news_finplot.py --selection-mode analogs
  .\.venv\Scripts\python.exe view_news_finplot.py --threshold-multiplier 0.5
  .\.venv\Scripts\python.exe view_news_finplot.py --screenshot data\news_reactions\finplot.png
Параметры: --help. PNG можно также сохранить кнопкой окна. Файловых логов нет.
"""

import argparse
from contextlib import closing
from functools import partial
import json
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo

import pandas as pd
from PyQt6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from source.build_reactions import DEFAULT_OUTPUT
from source.price_reactions import HORIZONS
from source.finplot_compat import load_finplot
from source.build_analogs import DEFAULT_OUTPUT as ANALOGS, digest_file, readonly
from source.threshold_tuning import rescale_score, validate_multiplier

fplt = load_finplot()

STATUS_NAMES = {'ok': 'Полный', 'gap': 'Пробел тиков', 'entry_wait': 'Поздний вход',
                'no_entry': 'Нет входа', 'endpoint_missing': 'Нет конца окна',
                'cross_day': 'Другая дата', 'missing_archive': 'Нет архива', 'archive_error': 'Ошибка архива'}
SCORE_NAMES = {'selected': 'Отобрана', 'low_similarity': 'Слабое сходство',
               'mixed_directions': 'Разные направления', 'weak_median_move': 'Слабое движение аналогов',
               'insufficient_neighbors': 'Мало прошлых дней', 'query_threshold_unavailable': 'Нет порога запроса'}
CATEGORY_NAMES = {'geopolitics': 'Геополитика', 'monetary': 'Ставка/инфляция', 'oil': 'Нефть',
                  'fx': 'Рубль/валюта', 'corporate': 'Компании'}


def configure_chart():
    """Задаёт светлые цвета finplot, московскую шкалу и отключает восстановление окон."""
    fplt.background, fplt.foreground = '#ffffff', '#334155'
    fplt.candle_bull_color = fplt.candle_bull_body_color = '#13856f'
    fplt.candle_bear_color = fplt.candle_bear_body_color = '#d35255'
    fplt.volume_bull_color = fplt.volume_bull_body_color = '#acd6cb'
    fplt.volume_bear_color = fplt.volume_bear_body_color = '#ebbec1'
    fplt.cross_hair_color = '#637188'
    fplt.display_timezone = ZoneInfo('Europe/Moscow')
    fplt.key_esc_close = False
    fplt.autoviewrestore(False)
    pg.setConfigOptions(background=fplt.background, foreground=fplt.foreground)


def moscow(value):
    """Преобразует value в московский Timestamp наносекундной точности для finplot.

    Явные ns нужны для примитивов finplot 1.9.7: pandas 3 по умолчанию создаёт
    Timestamp в микросекундах, которые библиотека ошибочно трактует как наносекунды.
    """
    return pd.Timestamp(value).tz_localize('Europe/Moscow').as_unit('ns')


def metric_text(value, digits=2):
    """Форматирует числовое value с digits знаками; отсутствующее значение заменяет тире."""
    return '—' if value is None else f'{value:.{digits}f}'


def read_context_news(folder, month):
    """Читает из folder все исходные загрузки за month, включая повторные заголовки.

    Возвращает отсортированный список loaded_at/title/provider/date и происхождения.
    В отличие от выборки исследования, записи контекста не объединяются за месяц.
    """
    start = pd.Timestamp(month + '-01')
    end = start + pd.offsets.MonthBegin(1)
    rows = []
    for path in sorted(folder.glob('rss_news_*.db')):
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as c:
            for rowid, loaded, published, title, provider in c.execute(
                    'SELECT rowid,loaded_at,date,title,provider FROM news WHERE loaded_at>=? AND loaded_at<?',
                    (str(start), str(end))):
                rows.append({'loaded_at': loaded, 'published_at': published, 'title': title or '',
                             'provider': provider or '', 'origin': path.name, 'rowid': rowid})
    return sorted(rows, key=lambda r: (r['loaded_at'], r['origin'], r['rowid']))


class NewsCanvas(pg.GraphicsLayoutWidget):
    """Встраивает в окно две связанные панели finplot: свечи и объём."""

    def __init__(self, owner, bars, record, context):
        """Строит график bars, отметки record и соседние новости context для owner."""
        super().__init__(parent=owner)
        self.title, self.show_maximized = 'Новости и минутные свечи', False
        self.axes = list(fplt.create_plot_widget(self, rows=2, init_zoom_periods=len(bars)))
        price, volume = self.axes
        for i, axis in enumerate(self.axes):
            self.addItem(axis, row=i, col=0)
            axis.set_visible(xgrid=True, ygrid=True, xaxis=True)
            axis.getAxis('right').setWidth(85)
        self.ci.layout.setRowStretchFactor(0, 5)
        self.ci.layout.setRowStretchFactor(1, 1)
        price.setLabel('right', 'пункты')
        volume.setLabel('right', 'контракты')
        fplt.candlestick_ochl(bars[['open', 'close', 'high', 'low']], ax=price)
        fplt.volume_ocv(bars[['open', 'close', 'volume']], ax=volume)
        low, high = float(bars.low.min()), float(bars.high.max())
        span = max(high - low, 1.)
        positions = []
        for item in context:
            minute = moscow(item['loaded_at']).floor('min')
            if minute in bars.index:
                positions.append(minute)
        if positions:
            indexes = pd.DatetimeIndex(sorted(set(positions)))
            values = bars.loc[indexes, 'high'] + span * .025
            fplt.plot(values.rename('rss'), style='^', color='#9ca8b9', width=.6,
                      legend='Другие новости минуты', ax=price)
        when = moscow(record['loaded_at'])
        if bars.index[0] <= when <= bars.index[-1] + pd.Timedelta(minutes=1):
            fplt.add_line((when, low - span * .02), (when, high + span * .04),
                          color='#2563eb', width=2, ax=price)
            fplt.add_text((when, high + span * .05), 'Получена новость', color='#2563eb', ax=price)
        if record.get('entry_at'):
            entry = moscow(record['entry_at'])
            if bars.index[0] <= entry <= bars.index[-1]:
                fplt.add_text((entry, record['entry_price']), 'Вход →', color='#7c3aed', ax=price, anchor=(1, 1))
        if record['status'] == 'ok':
            entry, end = moscow(record['entry_at']), moscow(record['end_at'])
            color = '#14866e' if record['points'] >= 0 else '#c44652'
            fplt.add_line((entry, record['entry_price']), (moscow(record['exit_at']), record['exit_price']),
                          color=color, width=2, ax=price)
            fplt.add_vertical_band(entry, end, color=color + '18', ax=price)
            fplt.add_line((end, low), (end, high), color=color, style='--', ax=price)
            fplt.add_text((end, record['exit_price']), f'{record["points"]:+g} п. / {record["horizon"]} мин',
                          color=color, ax=price)
        fplt.refresh()

    @property
    def axs(self):
        """Предоставляет finplot список осей встроенного графика."""
        return self.axes


class NewsWindow(QtWidgets.QMainWindow):
    """Окно выбора новостей, исторических реакций и контекста полной ленты."""

    def __init__(self, database, news_dir, instrument='MIX', horizon=30, stable_only=True, search='',
                 analog_database=None, selection_mode=None, threshold_multiplier=None):
        """Открывает database и analog_database только для чтения; news_dir даёт RSS-контекст.

        instrument/horizon задают начальные фильтры. Без analog_database сохранён
        прежний режим stable_only; selection_mode выбирает аналоги/оценки/устойчивые/все.
        threshold_multiplier переопределяет k из БД, не меняя её; None использует сохранённый k.
        """
        super().__init__()
        self.database, self.news_dir = database, news_dir
        self.analog_database = analog_database
        self.canvas = None
        self.context_cache = {}
        self.records = []
        self.current_record = None
        self.setWindowTitle('PJ26 · Новости и пять прошлых аналогов · finplot')
        self.resize(1500, 980)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel('Инструмент'))
        self.instrument = QtWidgets.QComboBox()
        self.instrument.addItems(['MIX', 'RTS'])
        self.instrument.setCurrentText(instrument)
        toolbar.addWidget(self.instrument)
        toolbar.addWidget(QtWidgets.QLabel('Горизонт, мин'))
        self.horizon = QtWidgets.QComboBox()
        self.horizon.addItems([str(h) for h in HORIZONS])
        self.horizon.setCurrentText(str(horizon))
        toolbar.addWidget(self.horizon)
        self.stable_only = QtWidgets.QCheckBox('Только устойчивые по будущим ценам')
        self.stable_only.setChecked(stable_only)
        toolbar.addWidget(self.stable_only)
        self.mode = QtWidgets.QComboBox()
        for label, key in [('Пять прошлых аналогов', 'analogs'), ('Все оценки аналогов', 'assessments'),
                           ('Устойчивые по будущему', 'stable'), ('Все события', 'all')]:
            self.mode.addItem(label, key)
        initial_mode = selection_mode or ('analogs' if analog_database else ('stable' if stable_only else 'all'))
        self.mode.setCurrentIndex(self.mode.findData(initial_mode))
        if analog_database is None:
            self.mode.model().item(0).setEnabled(False)
            self.mode.model().item(1).setEnabled(False)
            self.mode.setToolTip('Для режима аналогов выполните build_analogs.py и откройте окно заново')
        toolbar.addWidget(self.mode)
        self.stable_only.hide()
        self.model = QtWidgets.QComboBox()
        scope_text = ''
        summary = {}
        if analog_database:
            with closing(readonly(analog_database)) as c:
                names = [r[0] for r in c.execute('SELECT DISTINCT model FROM scores ORDER BY model DESC')]
                if c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='metadata'").fetchone():
                    saved = c.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()
                    summary = json.loads(saved[0]) if saved else {}
                    if summary.get('history_years'):
                        scope_text = (f' Поиск аналогов: {summary["history_years"]} год до каждой новости. '
                                      f'Период проверки: {summary.get("query_start", "—")} — '
                                      f'{summary.get("query_end_exclusive", "—")} (конец не включён).')
            self.model.addItems(names)
        toolbar.addWidget(self.model)
        self.category = QtWidgets.QComboBox()
        self.category.addItem('Все категории', '')
        for key, label in CATEGORY_NAMES.items():
            self.category.addItem(label, key)
        toolbar.addWidget(self.category)
        toolbar.addWidget(QtWidgets.QLabel('Порог ×'))
        self.threshold_multiplier = QtWidgets.QDoubleSpinBox()
        self.threshold_multiplier.setDecimals(2)
        self.threshold_multiplier.setRange(.01, 100.)
        self.threshold_multiplier.setSingleStep(.05)
        initial_multiplier = validate_multiplier(threshold_multiplier if threshold_multiplier is not None else summary.get('threshold_multiplier', 1.))
        self.threshold_multiplier.setMaximum(max(100., initial_multiplier))
        self.threshold_multiplier.setValue(initial_multiplier)
        if self.threshold_multiplier.value() != initial_multiplier:
            raise ValueError('В finplot коэффициент k должен быть не меньше 0,01 и иметь не более двух знаков после запятой')
        self.threshold_multiplier.setToolTip('T = k × динамический базовый порог. k<1 мягче; k=1 — прежний вариант. Соседи неизменны.')
        self.threshold_multiplier.setEnabled(analog_database is not None)
        self.min_similarity = summary.get('min_similarity', .5)
        toolbar.addWidget(self.threshold_multiplier)
        self.search = QtWidgets.QLineEdit(search)
        self.search.setPlaceholderText('Поиск в заголовках')
        toolbar.addWidget(self.search, 1)
        button = QtWidgets.QPushButton('Сохранить PNG')
        button.clicked.connect(self.save_default_screenshot)
        toolbar.addWidget(button)
        layout.addLayout(toolbar)
        self.notice = QtWidgets.QLabel('Исторический отбор, а не торговый сигнал. Совпадение по времени не доказывает влияние новости. '
                                      'МСК: часовой пояс loaded_at принят предположительно; смена контракта в склейке не проверена.')
        self.notice.setWordWrap(True)
        if scope_text:
            self.notice.setText(self.notice.text() + scope_text)
        self.notice.setStyleSheet('padding:8px; background:#fff4d7; color:#695018')
        layout.addWidget(self.notice)
        self.headline = QtWidgets.QLabel()
        self.headline.setWordWrap(True)
        self.headline.setStyleSheet('font-size:16px; font-weight:600; padding:5px')
        layout.addWidget(self.headline)
        self.stats = QtWidgets.QLabel()
        self.stats.setWordWrap(True)
        layout.addWidget(self.stats)
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(['МСК / loaded_at', 'Новость', 'Пункты', 'ER', 'Z', 'Статус'])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().hide()
        self.table.setColumnWidth(0, 135)
        self.table.setColumnWidth(1, 310)
        splitter.addWidget(self.table)
        right = QtWidgets.QWidget()
        self.right_layout = QtWidgets.QVBoxLayout(right)
        self.chart_holder = QtWidgets.QWidget()
        self.chart_layout = QtWidgets.QVBoxLayout(self.chart_holder)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.addWidget(self.chart_holder, 5)
        tabs = QtWidgets.QTabWidget()
        analog_panel = QtWidgets.QWidget()
        analog_layout = QtWidgets.QVBoxLayout(analog_panel)
        back = QtWidgets.QPushButton('Вернуться к выбранной новости (двойной щелчок по аналогу открывает его график)')
        back.clicked.connect(self.select_event)
        analog_layout.addWidget(back)
        self.neighbors_table = QtWidgets.QTableWidget(0, 6)
        self.neighbors_table.setHorizontalHeaderLabels(['Дата МСК', 'Заголовок', 'Косинус', 'Пункты', 'Порог, п.', '|R| / T'])
        self.neighbors_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.neighbors_table.setColumnWidth(0, 150)
        self.neighbors_table.setColumnWidth(1, 300)
        self.neighbors_table.verticalHeader().setDefaultSectionSize(24)
        self.neighbors_table.cellDoubleClicked.connect(self.show_neighbor)
        analog_layout.addWidget(self.neighbors_table)
        tabs.addTab(analog_panel, 'Пять прошлых аналогов')
        self.context = QtWidgets.QPlainTextEdit()
        self.context.setReadOnly(True)
        tabs.addTab(self.context, 'Полная RSS-лента (будущее — только контекст)')
        tabs.setMinimumHeight(270)
        tabs.setMaximumHeight(290)
        self.right_layout.addWidget(tabs, 2)
        splitter.addWidget(right)
        splitter.setSizes([590, 900])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self.instrument.currentTextChanged.connect(self.reload_events)
        self.horizon.currentTextChanged.connect(self.reload_events)
        self.stable_only.toggled.connect(self.reload_events)
        self.mode.currentIndexChanged.connect(self.reload_events)
        self.model.currentTextChanged.connect(self.reload_events)
        self.category.currentIndexChanged.connect(self.reload_events)
        self.threshold_multiplier.valueChanged.connect(self.reload_events)
        self.search.textChanged.connect(self.reload_events)
        self.table.itemSelectionChanged.connect(self.select_event)
        self.reload_events()

    def connection(self):
        """Возвращает read-only соединение SQLite текущего окна; вызывающий закрывает его."""
        return sqlite3.connect(self.database.resolve().as_uri() + '?mode=ro', uri=True)

    def clear_chart(self):
        """Удаляет прежний график и его ссылки finplot перед переключением новости."""
        if self.canvas is not None:
            old = self.canvas
            self.chart_layout.removeWidget(old)
            if old in fplt.windows:
                fplt.windows.remove(old)
            fplt.master_data.pop(old, None)
            old.close()
            old.deleteLater()
            self.canvas = None

    def reload_events(self, *unused):
        """Применяет фильтры и обновляет таблицу; unused принимает параметры сигналов Qt."""
        previous_row = self.table.currentRow()
        previous_id = self.records[previous_row]['news_id'] if 0 <= previous_row < len(self.records) else None
        with closing(self.connection()) as c:
            records = [json.loads(r[0]) for r in c.execute(
                'SELECT payload FROM reactions WHERE instrument=? AND horizon=? ORDER BY stable DESC, '
                'json_extract(payload,\'$.z_score\') DESC, abs(points) DESC',
                (self.instrument.currentText(), int(self.horizon.currentText())))]
            relevance = {key: json.loads(payload).get('relevance') for key, payload in c.execute('SELECT news_id,payload FROM news')}
        for r in records:
            r['relevance'] = relevance.get(r['news_id'])
        needle = self.search.text().casefold()
        mode = self.mode.currentData()
        self.threshold_multiplier.setEnabled(self.analog_database is not None and mode in ('analogs', 'assessments'))
        scores = {}
        if mode in ('analogs', 'assessments') and self.analog_database:
            with closing(readonly(self.analog_database)) as c:
                scores = {key: rescale_score(json.loads(payload), self.threshold_multiplier.value(), self.min_similarity) for key, payload in c.execute(
                    'SELECT news_id,payload FROM scores WHERE instrument=? AND horizon=? AND model=?',
                    (self.instrument.currentText(), int(self.horizon.currentText()), self.model.currentText()))}
            for r in records:
                r['forecast'] = scores.get(r['news_id'])
            records.sort(key=lambda r: (r['loaded_at'], r['news_id']))
        self.records = [r for r in records if (mode != 'stable' or r['stable'])
                        and (mode not in ('analogs', 'assessments') or r['news_id'] in scores)
                        and (mode != 'analogs' or scores[r['news_id']]['selected'])
                        and (not self.category.currentData() or self.category.currentData() in (r.get('relevance') or {}).get('categories', []))
                        and needle in r['title'].casefold()]
        self.table.setHorizontalHeaderLabels(['МСК / loaded_at', 'Новость', 'Пункты',
            'Прогноз' if scores else 'ER', 'Медиана |R|/T' if scores else 'Z', 'Отбор' if scores else 'Статус'])
        self.table.blockSignals(True)
        self.table.clearContents()
        self.table.setRowCount(len(self.records))
        for i, r in enumerate(self.records):
            values = [r['loaded_at'][5:], r['title'], metric_text(r['points'], 0),
                      metric_text(r.get('er')), metric_text(r.get('z_score')), STATUS_NAMES.get(r['status'], r['status'])]
            if r.get('forecast'):
                s = r['forecast']
                values[3:] = [{'up': 'Вверх', 'down': 'Вниз'}.get(s['forecast_direction'], '—'),
                              metric_text(s['median_ratio']), SCORE_NAMES.get(s['status'], s['status'])]
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(r['title'])
                self.table.setItem(i, j, item)
        self.table.blockSignals(False)
        self.statusBar().showMessage(f'Найдено {len(self.records)} записей · k={self.threshold_multiplier.value():g} · Пустые минуты не заполняются · '
                                    'CSV вариантов порога: tune_threshold.py' if mode in ('analogs', 'assessments') else
                                    f'Найдено {len(self.records)} записей · Исторические метрики устойчивости')
        if self.records:
            self.table.selectRow(next((i for i, r in enumerate(self.records) if r['news_id'] == previous_id), 0))
        else:
            self.current_record = None
            self.clear_chart()
            self.headline.setText('По текущим фильтрам событий нет. Измените режим, модель или горизонт.')
            self.stats.clear()
            self.context.clear()
            self.neighbors_table.setRowCount(0)

    def select_event(self):
        """Показывает выбранную строку, её минутные свечи, метрики и полный новостной контекст."""
        index = self.table.currentRow()
        if not 0 <= index < len(self.records):
            return
        r = self.records[index]
        self.neighbor_records = r.get('forecast', {}).get('neighbors', []) if r.get('forecast') else []
        self.neighbors_table.setRowCount(len(self.neighbor_records))
        with closing(self.connection()) as c:
            titles = dict(c.execute('SELECT news_id,title FROM news'))
        for i, n in enumerate(self.neighbor_records):
            values = [n['loaded_at'], titles.get(n['news_id'], ''), metric_text(n['similarity'], 3),
                      metric_text(n['points'], 0), metric_text(n['threshold_points'], 0),
                      metric_text(abs(n['return_frac']) / n['threshold_frac'])]
            for j, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setToolTip(value)
                self.neighbors_table.setItem(i, j, item)
        self.display_record(r)

    def show_neighbor(self, row, column):
        """Открывает график аналога row; column — неиспользуемая колонка сигнала Qt."""
        if not 0 <= row < len(self.neighbor_records):
            return
        n = self.neighbor_records[row]
        with closing(self.connection()) as c:
            result = c.execute('SELECT payload FROM reactions WHERE news_id=? AND instrument=? AND horizon=?',
                (n['news_id'], self.instrument.currentText(), int(self.horizon.currentText()))).fetchone()
        if result:
            self.display_record(json.loads(result[0]))

    def display_record(self, r):
        """Показывает реакцию r и свечи; прогноз прикреплён только к текущему запросу."""
        self.current_record = r
        when = pd.Timestamp(r['loaded_at'])
        left = when - pd.Timedelta(minutes=60)
        right = when + pd.Timedelta(minutes=max(125, r['horizon'] + 5))
        with closing(self.connection()) as c:
            bars = pd.read_sql_query('SELECT datetime,open,high,low,close,volume,tick_count FROM minute_bars '
                                     'WHERE instrument=? AND datetime>=? AND datetime<=? ORDER BY datetime', c,
                                     params=(r['instrument'], str(left.floor('min')), str(right)))
        self.headline.setText(f'{r["instrument"]} · {r["loaded_at"]} МСК · {r["title"]}')
        self.stats.setText(f'Статус: {STATUS_NAMES.get(r["status"], r["status"])} | '
                           f'{r["horizon"]} мин: {metric_text(r["points"], 0)} п. '
                           f'({metric_text(r.get("return_pct"))}%) | ER {metric_text(r.get("er"))} | '
                           f'Z {metric_text(r.get("z_score"))} | MFE {metric_text(r.get("mfe_points"), 0)} | '
                           f'MAE {metric_text(r.get("mae_points"), 0)} | Откат {metric_text(r.get("max_pullback_points"), 0)} п.\n'
                           f'Вход: {r.get("entry_at") or "—"} · Конец горизонта: {r.get("end_at") or "—"} · '
                           f'Устойчивое по будущему: {"да" if r["stable"] else "нет"}')
        if r.get('forecast'):
            s = r['forecast']
            self.stats.setText(self.stats.text() + '\n' +
                f'Отбор: {SCORE_NAMES.get(s["status"], s["status"])} · Прогноз: '
                f'{ {"up": "вверх", "down": "вниз"}.get(s["forecast_direction"], "—")} · '
                f'Медиана |R|/T аналогов: {metric_text(s["median_ratio"])} · '
                f'Мин. косинус: {metric_text(s["min_similarity"], 3)} · '
                f'Порог запроса (k={s["threshold_multiplier"]:g}): {metric_text(s["threshold"]["threshold_points"], 0)} п. · '
                f'Фактическое |R|/T: {metric_text(s["actual_ratio"])}')
        relevance = r.get('relevance') or (r.get('forecast') or {}).get('relevance')
        if relevance:
            categories = ', '.join(CATEGORY_NAMES.get(k, k) for k in relevance['categories'])
            self.stats.setText(self.stats.text() + f'\nСвязь с RTS/MIX: {categories} · {relevance["reason"]}')
            self.stats.setToolTip('Группы сравнения: ' + ', '.join(relevance['groups']) + '\n' + relevance['rule_version'])
        month = r['loaded_at'][:7]
        if month not in self.context_cache:
            try:
                self.context_cache[month] = read_context_news(self.news_dir, month)
            except (OSError, sqlite3.Error):
                self.context_cache[month] = []
        context = [n for n in self.context_cache[month] if str(left) <= n['loaded_at'] <= str(right)]
        self.context.setPlainText('\n'.join(f'{n["loaded_at"][11:]} · {n["provider"]} · {n["title"]}' for n in context)
                                  or 'Полная RSS-лента в выбранном окне недоступна.')
        self.clear_chart()
        if not bars.empty:
            bars.index = pd.DatetimeIndex(pd.to_datetime(bars.pop('datetime'))).tz_localize('Europe/Moscow').as_unit('ns')
            self.canvas = NewsCanvas(self, bars, r, context)
            self.chart_layout.addWidget(self.canvas)
        else:
            self.statusBar().showMessage('В выбранном временном окне нет минутных котировок.')

    def save_default_screenshot(self):
        """Сохраняет изображение текущего окна в finplot.png рядом с базой реакций."""
        self.save_screenshot(self.database.parent / 'finplot.png')

    def save_screenshot(self, path):
        """Сохраняет собственное окно Qt в PNG path и сообщает путь в строке состояния."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.grab().save(str(path), 'PNG'):
            raise OSError(f'Не удалось сохранить {path}')
        self.statusBar().showMessage(f'PNG сохранён: {path.resolve()}')


def main():
    """Разбирает CLI, открывает окно finplot и запускает цикл событий Qt."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--database', type=Path, default=DEFAULT_OUTPUT / 'reactions.sqlite3', help='Готовая БД реакций')
    p.add_argument('--news-dir', type=Path, default=Path(r'C:\Users\Alkor\gd\db_rss'), help='Полная RSS-лента для контекста')
    p.add_argument('--instrument', choices=['MIX', 'RTS'], default='MIX', help='Инструмент при открытии')
    p.add_argument('--horizon', type=int, choices=HORIZONS, default=30, help='Горизонт при открытии')
    p.add_argument('--all-events', action='store_true', help='Показать также нейтральные и неполные события')
    p.add_argument('--search', default='', help='Фильтр текста заголовка')
    p.add_argument('--analogs', type=Path, default=ANALOGS / 'analogs.sqlite3', help='БД пяти прошлых аналогов')
    p.add_argument('--selection-mode', choices=['analogs', 'assessments', 'stable', 'all'], default='analogs', help='Режим отбора')
    p.add_argument('--screenshot', type=Path, help='Сохранить PNG после открытия окна')
    p.add_argument('--threshold-multiplier', type=float, help='Начальный коэффициент k, по умолчанию из БД или 1')
    args = p.parse_args()
    if args.threshold_multiplier is not None:
        try:
            validate_multiplier(args.threshold_multiplier)
        except ValueError as error:
            p.error(str(error))
    if not args.database.is_file():
        p.error('БД реакций отсутствует. Сначала выполните build_reactions.py')
    mode = 'all' if args.all_events else args.selection_mode
    requires_analogs = mode in ('analogs', 'assessments')
    analog_database = args.analogs if args.analogs.is_file() else None
    if requires_analogs and analog_database is None:
        p.error('БД аналогов отсутствует. Сначала выполните build_analogs.py')
    if analog_database:
        with closing(readonly(analog_database)) as c:
            metadata = json.loads(c.execute("SELECT value FROM metadata WHERE key='summary'").fetchone()[0])
        if metadata['reaction_sha256'] != digest_file(args.database):
            if requires_analogs:
                p.error('БД реакций изменилась. Повторите build_analogs.py')
            analog_database = None
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    configure_chart()
    try:
        window = NewsWindow(args.database, args.news_dir, args.instrument, args.horizon,
                            not args.all_events, args.search, analog_database, mode, args.threshold_multiplier)
    except ValueError as error:
        p.error(str(error))
    window.show()
    if args.screenshot:
        QtCore.QTimer.singleShot(1800, partial(window.save_screenshot, args.screenshot))
    app.exec()


if __name__ == '__main__':
    main()
