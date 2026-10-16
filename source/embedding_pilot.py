r"""Сравнение локальных эмбеддингов новостей через Ollama.

Читает SQLite только для чтения, объединяет одинаковые заголовки в выбранном
месяце по loaded_at, выбирает воспроизводимую выборку и сохраняет float32-векторы
в отдельную SQLite. Рассчитывает аналоги за весь месяц и только из прошлого,
создаёт HTML/CSV и замеры скорости. Прерванный расчёт возобновляется из кэша.
Ценовые реакции и торговые сигналы этот скрипт пока не рассчитывает.

Примеры запуска из корня проекта:
  .\.venv\Scripts\python.exe embedding_pilot.py
  .\.venv\Scripts\python.exe embedding_pilot.py --month 2026-09 --limit 2000 --batch-size 32
  .\.venv\Scripts\python.exe embedding_pilot.py --models bge-m3:latest --limit 1000
  .\.venv\Scripts\python.exe embedding_pilot.py --news-dir C:\Users\Alkor\gd\db_rss --output data\embedding_pilot
Параметры: --help. Файловые логи не создаются; прогресс выводится в консоль.
"""

import argparse
from array import array
from datetime import datetime, timedelta
import hashlib
import heapq
import json
import math
from pathlib import Path
import random
import sqlite3
import struct
import time
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, ProxyHandler, build_opener

PREPARATION = 'title-nfkc-whitespace-v1-symmetric-no-prefix'
DEFAULT_MODELS = ['qwen3-embedding:0.6b', 'bge-m3:latest']
ROOT = Path(__file__).resolve().parents[1]


def normalize_title(title):
    """Возвращает NFKC-заголовок без лишних пробелов; title — исходная строка."""
    return ' '.join(unicodedata.normalize('NFKC', title).split())


def load_news(folder, month):
    """Читает БД из folder за month по loaded_at; возвращает записи и счётчики.

    Одинаковые заголовки без учёта регистра объединяются только внутри месяца;
    сохраняются первая загрузка, источник, происхождение и число повторов.
    """
    start = datetime.strptime(month, '%Y-%m')
    end = (start.replace(day=28) + timedelta(days=4)).replace(day=1)
    records = {}
    stats = {'source_rows': 0, 'empty_titles': 0, 'invalid_timestamps': 0, 'files': 0}
    for path in sorted(folder.glob('rss_news_*.db')):
        c = sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)
        try:
            cursor = c.execute(
                'SELECT rowid,loaded_at,date,title,provider FROM news WHERE loaded_at>=? AND loaded_at<?',
                (start.strftime('%Y-%m-%d %H:%M:%S'), end.strftime('%Y-%m-%d %H:%M:%S')),
            )
            stats['files'] += 1
            for rowid, loaded, published, title, provider in cursor:
                stats['source_rows'] += 1
                try:
                    datetime.strptime(loaded, '%Y-%m-%d %H:%M:%S')
                except (ValueError, TypeError):
                    stats['invalid_timestamps'] += 1
                    continue
                text = normalize_title(title or '')
                if not text:
                    stats['empty_titles'] += 1
                    continue
                key = hashlib.sha256(text.casefold().encode('utf-8')).hexdigest()
                origin = {'file': path.name, 'rowid': rowid, 'provider': provider}
                if key not in records:
                    records[key] = {
                        'news_id': key, 'loaded_at': loaded, 'published_at': published,
                        'title': text, 'provider': provider, 'duplicate_count': 0, 'origins': [],
                    }
                item = records[key]
                item['duplicate_count'] += 1
                item['origins'].append(origin)
                if loaded < item['loaded_at']:
                    item.update(loaded_at=loaded, published_at=published, title=text, provider=provider)
        finally:
            c.close()
    stats['unique_titles'] = len(records)
    return sorted(records.values(), key=lambda item: (item['loaded_at'], item['news_id'])), stats


def normalize_vector(values, expected_dim=None):
    """Проверяет размерность/конечность values и возвращает единичный float32-вектор."""
    if not values or (expected_dim is not None and len(values) != expected_dim):
        raise ValueError('Пустой вектор или неверная размерность')
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in values):
        raise ValueError('Вектор содержит нечисловые или неконечные значения')
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError('Нулевая или неконечная норма вектора')
    return array('f', (x / norm for x in values))


def open_cache(path):
    """Открывает отдельный SQLite-кэш path и создаёт таблицы; возвращает соединение."""
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(path)
    c.executescript('''
        CREATE TABLE IF NOT EXISTS news(news_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS samples(corpus_id TEXT,news_id TEXT,PRIMARY KEY(corpus_id,news_id));
        CREATE TABLE IF NOT EXISTS models(digest TEXT PRIMARY KEY,name TEXT,metadata TEXT);
        CREATE TABLE IF NOT EXISTS embeddings(
            news_id TEXT,digest TEXT,preparation TEXT,dimension INTEGER,vector BLOB,
            PRIMARY KEY(news_id,digest,preparation));
        CREATE TABLE IF NOT EXISTS benchmarks(
            corpus_id TEXT,digest TEXT,preparation TEXT,metrics TEXT,
            PRIMARY KEY(corpus_id,digest,preparation));
    ''')
    return c


def save_vectors(c, digest, records):
    """Записывает пары (news_id, vector) для digest в c и фиксирует пакет на диске."""
    c.executemany('INSERT OR REPLACE INTO embeddings VALUES(?,?,?,?,?)', [
        (key, digest, PREPARATION, len(v), struct.pack(f'<{len(v)}f', *v)) for key, v in records
    ])
    c.commit()


def save_news(c, rows, corpus_id):
    """Сохраняет rows и состав corpus_id в c; изменение текста сбрасывает его векторы.

    Идентификатор новости не зависит от регистра, но вход модели зависит;
    изменение источника или времени без изменения текста не сбрасывает вектор.
    """
    with c:
        for row in rows:
            old = c.execute('SELECT payload FROM news WHERE news_id=?', (row['news_id'],)).fetchone()
            if old and json.loads(old[0])['title'] != row['title']:
                c.execute('DELETE FROM embeddings WHERE news_id=?', (row['news_id'],))
            c.execute('INSERT OR REPLACE INTO news VALUES(?,?)',
                      (row['news_id'], json.dumps(row, ensure_ascii=False)))
            c.execute('INSERT OR IGNORE INTO samples VALUES(?,?)', (corpus_id, row['news_id']))


def cached_vectors(c, digest):
    """Возвращает словарь news_id→float32-вектор из c для digest и текущей подготовки."""
    result = {}
    for key, dim, blob in c.execute(
        'SELECT news_id,dimension,vector FROM embeddings WHERE digest=? AND preparation=?',
        (digest, PREPARATION),
    ):
        values = struct.unpack(f'<{dim}f', blob)
        result[key] = normalize_vector(values, expected_dim=dim)
    return result


def nearest(rows, vectors, query_index, top_k, past_only=False, embargo_minutes=120):
    """Возвращает top_k пар (индекс, косинус) для query_index в rows/vectors.

    Исключает сам запрос; past_only оставляет loaded_at не позже запроса минус
    embargo_minutes. Векторы должны быть единичными, одного пространства и размера.
    """
    cutoff = datetime.strptime(rows[query_index]['loaded_at'], '%Y-%m-%d %H:%M:%S')
    cutoff = (cutoff - timedelta(minutes=embargo_minutes)).strftime('%Y-%m-%d %H:%M:%S')
    query = vectors[query_index]
    scores = (
        (i, max(-1., min(1., math.sumprod(query, vector))))
        for i, (row, vector) in enumerate(zip(rows, vectors, strict=True))
        if row['news_id'] != rows[query_index]['news_id']
        and (not past_only or row['loaded_at'] <= cutoff)
    )
    return heapq.nlargest(top_k, scores, key=lambda pair: (pair[1], -pair[0]))


def api_json(base_url, endpoint, payload=None, timeout=120):
    """Выполняет локальный запрос Ollama; возвращает JSON либо понятную ошибку.

    base_url — адрес сервера, endpoint — путь API, payload — тело POST или None
    для GET, timeout — предел ожидания в секундах. HTTP-прокси не используется.
    """
    parsed = urlparse(base_url)
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost', '127.0.0.1', '::1'):
        raise ValueError('Для этого пилота разрешён только локальный Ollama по HTTP')
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    req = Request(base_url.rstrip('/') + endpoint, data=body, headers={'Content-Type': 'application/json'})
    try:
        with build_opener(ProxyHandler({})).open(req, timeout=timeout) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f'Ollama HTTP {exc.code}: {exc.read(1500).decode("utf-8", errors="replace")}') from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f'Нет ответа Ollama: {exc}') from exc


def embed_batch(base_url, model, titles, dimension):
    """Получает проверенные векторы titles от model; возвращает векторы и метаданные API."""
    response = api_json(base_url, '/api/embed', {
        'model': model, 'input': titles, 'truncate': False, 'keep_alive': '5m',
    })
    values = response.get('embeddings', [])
    if len(values) != len(titles):
        raise ValueError(f'Ollama вернул {len(values)} векторов вместо {len(titles)}')
    return [normalize_vector(v, dimension) for v in values], response


def compute_model(c, rows, model, base_url, batch_size, corpus_id):
    """Вычисляет отсутствующие векторы model для rows, сохраняет кэш и замеры.

    c — выходная SQLite, base_url — Ollama, batch_size — размер пакета,
    corpus_id — отпечаток выборки. Возвращает векторы по порядку rows и метрики.
    """
    tags = api_json(base_url, '/api/tags')['models']
    tag = next((x for x in tags if x['name'] == model), None)
    if tag is None:
        raise ValueError(f'Модель {model} не установлена. Автоматическая загрузка отключена.')
    digest = tag['digest']
    dimension = tag.get('details', {}).get('embedding_length')
    if not dimension:
        raise ValueError(f'В метаданных {model} отсутствует размерность embedding_length')
    c.execute('INSERT OR REPLACE INTO models VALUES(?,?,?)',
              (digest, model, json.dumps(tag, ensure_ascii=False)))
    c.commit()
    cache = cached_vectors(c, digest)
    pending = [r for r in rows if r['news_id'] not in cache]
    metrics = {'model': model, 'digest': digest, 'dimension': dimension,
               'cached_count': len(rows) - len(pending), 'generated_count': len(pending),
               'warmup_seconds': 0., 'embedding_seconds': 0., 'api_load_seconds': 0.,
               'prompt_tokens': 0, 'texts_per_second': None, 'batch_size': batch_size}
    if pending:
        print(f'{model}: прогрев, затем {len(pending)} новых заголовков', flush=True)
        start = time.perf_counter()
        embed_batch(base_url, model, [pending[0]['title']], dimension)
        metrics['warmup_seconds'] = time.perf_counter() - start
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset:offset + batch_size]
            start = time.perf_counter()
            vectors, response = embed_batch(base_url, model, [r['title'] for r in batch], dimension)
            metrics['embedding_seconds'] += time.perf_counter() - start
            metrics['api_load_seconds'] += response.get('load_duration', 0) / 1e9
            metrics['prompt_tokens'] += response.get('prompt_eval_count', 0)
            save_vectors(c, digest, [(r['news_id'], v) for r, v in zip(batch, vectors, strict=True)])
            cache.update((r['news_id'], v) for r, v in zip(batch, vectors, strict=True))
            print(f'{model}: {min(offset + batch_size, len(pending))}/{len(pending)}', flush=True)
        metrics['texts_per_second'] = len(pending) / metrics['embedding_seconds']
        c.execute('INSERT OR REPLACE INTO benchmarks VALUES(?,?,?,?)',
                  (corpus_id, digest, PREPARATION, json.dumps(metrics)))
        c.commit()
    stored = c.execute('SELECT metrics FROM benchmarks WHERE corpus_id=? AND digest=? AND preparation=?',
                       (corpus_id, digest, PREPARATION)).fetchone()
    ordered = [cache[r['news_id']] for r in rows]
    if any(len(v) != dimension for v in ordered):
        raise ValueError('Размерность кэша не совпадает с метаданными модели')
    return ordered, {'current_run': metrics, 'benchmark': json.loads(stored[0]) if stored else None}


def select_queries(rows, count, seed):
    """Возвращает до count индексов запросов: тематические примеры и случайное дополнение."""
    topics = ('ставк', 'санкци', 'переговор', 'нефт', 'инфляц', 'рубл', 'дивиденд', 'трамп')
    rng = random.Random(seed)
    chosen = []
    for word in topics:
        matches = [i for i, r in enumerate(rows) if word in r['title'].casefold() and i not in chosen]
        chosen.extend(rng.sample(matches, min(2, len(matches), max(0, count - len(chosen)))))
    remainder = [i for i in range(len(rows)) if i not in chosen]
    chosen.extend(rng.sample(remainder, min(len(remainder), count - len(chosen))))
    return sorted(chosen)


def main():
    """Разбирает CLI, выполняет пилот и сохраняет SQLite, JSON, CSV и HTML; возвращает None."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser._optionals.title = 'Параметры'
    parser.add_argument('--news-dir', type=Path, default=Path(r'C:\Users\Alkor\gd\db_rss'), help='Папка исходных БД')
    parser.add_argument('--output', type=Path, default=ROOT / 'data' / 'embedding_pilot', help='Папка кэша и отчётов')
    parser.add_argument('--month', default='2026-09', help='Месяц loaded_at, YYYY-MM; по умолчанию 2026-09')
    parser.add_argument('--limit', type=int, default=2000, help='Размер выборки; по умолчанию 2000')
    parser.add_argument('--seed', type=int, default=42, help='Начальное значение генератора; по умолчанию 42')
    parser.add_argument('--batch-size', type=int, default=32, help='Размер пакета API; по умолчанию 32')
    parser.add_argument('--queries', type=int, default=30, help='Число запросов для проверки; по умолчанию 30')
    parser.add_argument('--top-k', type=int, default=5, help='Число аналогов; по умолчанию 5')
    parser.add_argument('--embargo-minutes', type=int, default=120, help='Минимальный возраст прошлого аналога, минут')
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS, help='Точные теги установленных моделей')
    parser.add_argument('--ollama-url', default='http://localhost:11434', help='Локальный HTTP-адрес Ollama')
    args = parser.parse_args()
    if min(args.limit, args.batch_size, args.queries, args.top_k) < 1 or args.embargo_minutes < 0:
        parser.error('Размеры должны быть положительными, временной зазор — неотрицательным')
    if not args.news_dir.is_dir():
        parser.error('Папка новостей отсутствует')
    rows, stats = load_news(args.news_dir, args.month)
    if len(rows) < 2:
        parser.error('Для сравнения нужны хотя бы два уникальных заголовка')
    rows = sorted(random.Random(args.seed).sample(rows, min(args.limit, len(rows))),
                  key=lambda r: (r['loaded_at'], r['news_id']))
    corpus_id = hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=True)
    c = open_cache(args.output / 'pilot.sqlite3')
    try:
        save_news(c, rows, corpus_id)
        print(f'Выборка: {len(rows)} из {stats["unique_titles"]} уникальных заголовков', flush=True)
        summary = {'month': args.month, 'seed': args.seed, 'sample_size': len(rows), 'corpus_id': corpus_id,
                   'preparation': PREPARATION, 'source': str(args.news_dir.resolve()), 'source_stats': stats,
                   'time_field': 'loaded_at', 'timezone_assumption': 'Europe/Moscow, не подтверждено',
                   'embargo_minutes': args.embargo_minutes, 'top_k': args.top_k, 'models': {},
                   'created_at': datetime.now().astimezone().isoformat()}
        query_indices = select_queries(rows, min(args.queries, len(rows)), args.seed)
        neighbors = []
        for model in dict.fromkeys(args.models):
            vectors, metrics = compute_model(c, rows, model, args.ollama_url, args.batch_size, corpus_id)
            summary['models'][model] = metrics
            for mode in ('all', 'past'):
                for qi in query_indices:
                    hits = nearest(rows, vectors, qi, args.top_k, mode == 'past', args.embargo_minutes)
                    for rank, (ni, score) in enumerate(hits, 1):
                        neighbors.append({'model': model, 'mode': mode, 'query_id': rows[qi]['news_id'],
                                          'neighbor_id': rows[ni]['news_id'], 'rank': rank, 'similarity': score})
        summary['query_ids'] = [rows[i]['news_id'] for i in query_indices]
        from source.embedding_report import write_report
        write_report(args.output, rows, neighbors, summary)
        print(f'Отчёт: {(args.output / "report.html").resolve()}', flush=True)
    finally:
        c.close()


if __name__ == '__main__':
    main()
