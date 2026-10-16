# Вспомогательные модули

В корне оставлены шесть исполняемых Python-скриптов. Вычисления, экспорт и интерфейс
находятся в этом пакете; напрямую файлы source не запускаются.

| Модуль | Назначение и параметры |
|---|---|
| `embedding_pilot.py` | Выборка по loaded_at, дедупликация, API Ollama, SQLite-кэш, косинусный поиск и main; CLI передаётся корневой командой. |
| `embedding_report.py` | `write_report(output,rows,neighbors,summary)` создаёт HTML/CSV/JSON; `write_csv` сохраняет UTF-8 BOM. |
| `price_data.py` | `read_ticks(path)` проверяет ZIP и возвращает сделки/паспорт; `minute_bars(ticks)` строит OHLCV без пустых минут. |
| `price_reactions.py` | `measure_reaction(ticks,bars,loaded_at,horizon,...)` считает движение/экстремумы/устойчивость с ограничениями ожидания и пробелов; `pre_volatility` берёт только завершённые прошлые минуты. |
| `build_reactions.py` | `build(rows,tick_dirs,output,config)` создаёт отдельную БД реакций/свечей, широкий/длинный CSV и покрытие. |
| `view_news_finplot.py` | `NewsWindow(database,news_dir,...)`: выбор события/инструмента/горизонта, finplot, метрики и соседние новости; чтение готовой БД. |
| `finplot_compat.py` | `load_finplot()` временно подставляет нейтральный getdefaultlocale при импорте finplot 1.9.7, затем восстанавливает функцию. Устраняет os._exit при русской локали без изменения ОС/библиотеки. |
| `__init__.py` | Пакет source, без действий при импорте. |
| `dynamic_threshold.py` | `BackgroundIndex(rows)` индексирует фон; `dynamic_threshold(...,quantile=.75,lookback_days=20,time_band_minutes=60,min_samples=30,min_days=5,volatility_fraction=.5)` оценивает порог только по прошлым дням, часам и волатильности. |
| `analog_selection.py` | `pick_neighbors(candidates,query_loaded_at,count=5)` выбирает разные прошлые даты до оценки исходов; `assess_neighbors(neighbors,min_similarity=.5)` требует пять совпадающих знаков и медиану силы >=1. |
| `build_analogs.py` | `build(args)` читает готовые векторы/реакции, строит кэш фона с SHA256, собственные пороги аналогов и все оценки SQLite/CSV. `model_manifest` проверяет точный текст, состав и версии. `make_decision` не принимает фактический исход запроса. |

Примеры из корня проекта:

```powershell
.\.venv\Scripts\python.exe embedding_pilot.py --month 2026-09 --limit 2000
.\.venv\Scripts\python.exe build_reactions.py --latency-seconds 0
.\.venv\Scripts\python.exe build_analogs.py --quantile 0.75 --min-similarity 0.5
.\.venv\Scripts\python.exe view_news_finplot.py --instrument MIX --horizon 30
.\.venv\Scripts\python.exe view_news_finplot.py --selection-mode assessments
.\.venv\Scripts\python.exe view_news_finplot.py --selection-mode stable
.\.venv\Scripts\python.exe view_news_finplot.py --screenshot data\news_reactions\finplot.png
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Полные параметры и определения метрик — в корневом README. Файловых логов нет.
Исходные БД/ZIP открываются только для чтения.

Времена графика явно локализованы в МСК и преобразованы в ns для корректного
положения примитивов finplot 1.9.7 при pandas 3. Прогнозная доступность задаётся
loaded_at, весь его секундный интервал исключается из входа из-за условных долей.
Контекст графика читает все исходные загрузки, включая повторные заголовки,
через `read_context_news(folder,month)`, без месячной дедупликации. Предсобытийная
волатильность требует актуальной последней свечи непосредственно перед новостью.

В новом режиме `NewsWindow(...,analog_database=...,selection_mode='analogs')`
показывает отбор по пяти аналогам; `assessments` показывает и отказы. Вкладка
аналогов содержит даты, косинусы, пункты, собственные пороги и |R|/T; двойной
щелчок открывает график аналога, кнопка возвращает запрос. Полная RSS-лента
на отдельной вкладке. Режимы `stable` и `all` сохраняют прежний просмотр.
Текущий фактический исход не ограничивает список прогнозов. `--analogs` задаёт БД
аналогов, по умолчанию data/analog_analysis/analogs.sqlite3; SHA256 реакций
проверяется перед запуском GUI. Остальные параметры команды build_analogs:
`--reaction-db`, `--pilot-dir`, `--output`, `--mix-dir`, `--rts-dir`, `--models`,
`--lookback-days`, `--time-band-minutes`, `--min-samples`, `--min-days`,
`--volatility-fraction`, `--background-step`. Все значения по умолчанию: `--help`.

## Проба фильтра российского рынка

`market_relevance.py`: `classify_title(title)` возвращает include/uncertain/exclude,
причины и группы category:event:object:stage; `compatible_groups(left,right)`
проверяет их пересечение. Версия правил ru-market-rules-v3. Цены и LLM не используются.
`russian_market.py`: `classify_month(news_dir,month,output)` классифицирует всю
ленту месяца, `run(args)` строит отдельные пилот/реакции/аналоги. Корневая команда:

```powershell
.\.venv\Scripts\python.exe russian_market.py --stage classify --history-years 0
.\.venv\Scripts\python.exe russian_market.py --month 2026-09 --history-years 0
```

Месячный режим: `--history-years 0`, `--month 2026-09`, `--stage classify|all`, `--news-dir`, `--output`
(по умолчанию data/russian_market/<месяц>), `--models` (обе), `--ollama-url`
(http://localhost:11434), `--batch-size 32`. В classify котировки/Ollama не читаются.
Старые результаты сохраняются; в новый кэш копируются пригодные векторы и фон.
classification.csv содержит все решения, review_blind.csv — выборку без цен и
поля human_*. Заполненные ручные поля неизменных текстов сохраняются при повторе.
classification_summary.json — сводка, папки pilot/reactions/analogs — расчёты.

`make_decision(...,query_relevance=None,history_years=None)` сохраняет прежний режим; при наличии
классификации ограничивает кандидатов общими группами до ранжирования. CSV оценок
получает relevance_categories/groups/reason, SQLite — полный relevance.
Пять дат произвольные, не обязательно подряд, ограничения возраста текстового
аналога нет. 20 дней относятся только к калибровке ценового фона.
GUI содержит список категорий, объяснение фильтра и подсказку с группой/версией.
Корпоративный словарь — исследовательский список, не исторический состав индекса;
веса не применяются. Неопределённые события и стадии остаются вне основного набора.

## Годовая история аналогов

По умолчанию `russian_market.py --history-years 1 --month 2026-09` загружает
историю с 01.09.2025 до 01.10.2026 исключительно, а оценки строит только для
сентября 2026. Для каждого запроса допускается кандидат не раньше loaded_at
минус один календарный год. Текущий день исключён. Разные даты не обязаны идти
подряд. Предыдущая месячная проба воспроизводится с `--history-years 0`.

Годовой выход: data/russian_market/2026-09_year1, старый месячный сохранён.
Новый модуль `news_history.py`: `load_history(folder,start,end)` читает RSS один раз
за интервал [start,end), объединяет одинаковые заголовки только внутри даты,
возвращает события и покрытие по месяцам. news_id = дата+текст, title_key = текст.
`seed_title_vectors(cache,rows)` переносит векторы точного текста под новые ID,
сохраняя digest/preparation. Разный регистр не смешивается. Это переиспользование
вычислений; повторные события разных дней остаются отдельными строками.

`classify_month(...,history_years=0)` поддерживает прежние вызовы; годовой CLI
передаёт 1. `month_bounds(month)` задаёт отдельный месяц проверки.
`history_start(loaded_at,years)` учитывает 29 февраля. `pick_neighbors` и
`make_decision` принимают history_years=None для прежнего неограниченного поиска.
`build_analogs.py`: новые --history-years, --query-start (включительно), --query-end
(исключительно). `query_indices(news,start,end)` выбирает только запросы периода;
исторические события остаются в news/thresholds и используются как кандидаты.
Матрица сходства содержит запросы*история вместо полного квадрата истории.
summary хранит query_count, query_start, query_end_exclusive и history_years.

Покрытие исходной БД сохраняется в classification_summary.json. Если начало БД
позже запрошенного начала года, история не дополняется вымышленными записями.
20 дней для динамического ценового фона остаются отдельным параметром.

## Настраиваемая строгость динамического порога

`threshold_tuning.py`: `validate_multiplier(value)` проверяет конечный k>0;
`rescale_score(score,multiplier,min_similarity=.5)` возвращает копию оценки с
T=k*T_base для запроса и его пяти аналогов. Соседи/даты/косинусы не меняются.
Повторное изменение использует base_threshold_frac/points и actual_ratio_base,
не накапливая множители. Результат содержит effective strong_correct и общий
base_strong_correct по T_base (k=1); текущий исход влияет только на эти оценки.

`tune_threshold.py`: `run(database,output,multipliers)` читает SQLite только для
чтения, сохраняет summary.csv по всем k и группам, selected.csv всех отборов и
summary.json с исходными параметрами/SHA256. `summarize` формирует строку группы.
Пустой selected.csv сохраняет полную схему. Лучшее значение не выбирается.

```powershell
.\.venv\Scripts\python.exe tune_threshold.py --multipliers 0.1 0.25 0.5 0.75 1 1.25 1.5
.\.venv\Scripts\python.exe view_news_finplot.py --database data\russian_market\2026-09_year1\reactions\reactions.sqlite3 --analogs data\russian_market\2026-09_year1\analogs\analogs.sqlite3 --threshold-multiplier 0.5
```

У tune_threshold параметры --database (по умолчанию годовые оценки сентября),
--output (threshold_tuning рядом с БД), --multipliers (список положительных k).
Для построения БД: build_analogs.py/russian_market.py --threshold-multiplier 0.5;
по умолчанию 1. В thresholds SQLite сохраняются базовые калибровки, в scores —
эффективные и базовые пороги. summary содержит threshold_multiplier.

`NewsWindow(...,threshold_multiplier=None)` берёт k из БД, None совместим со
старым форматом (1). CLI --threshold-multiplier задаёт начальное значение. В окне
поле «Порог ×» меняет отбор сразу, сохраняя выбранную новость, если она остаётся
в списке. Два десятичных знака, минимум 0.01; несовместимое CLI-значение даёт ошибку
вместо округления. Поле доступно в режимах аналогов/оценок. GUI не изменяет SQLite;
для CSV вариантов применяется tune_threshold.py. Устойчивость legacy-режима
и прочие ограничения не меняются. Сравнение параметров на сентябре — исследование;
для проверки выбранного k нужен последующий период, не использованный при подборе.
