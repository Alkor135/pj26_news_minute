r"""Читает многомесячную RSS-историю, сохраняя одинаковые заголовки разных дней.

Исходные SQLite открывает только для чтения. Внутри календарной даты объединяет
загрузки одинакового нормализованного текста, сохраняя первое loaded_at.
news_id зависит от даты и текста, title_key — только от текста для переиспользования
эмбеддинга. Это дедупликация загрузок, не доказательство независимости событий.

Использование: .\.venv\Scripts\python.exe russian_market.py --history-years 1
Проверки: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_year_history.py -v
"""

from collections import Counter
from contextlib import closing
from datetime import datetime
import hashlib
import json
import sqlite3

from source.embedding_pilot import normalize_title


def load_history(folder, start, end):
    """Возвращает события и покрытие из folder за [start,end), времена локальные МСК.

    Дата плюс casefold-текст задают идентификатор; первая загрузка выбирается
    независимо от порядка файлов/провайдеров. Пустые/невалидные строки учитываются.
    """
    records, by_month = {}, Counter()
    stats = dict(source_rows=0, empty_titles=0, invalid_timestamps=0, files=0,
                 requested_start=start, requested_end_exclusive=end, identity='day-title-v1')
    for path in sorted(folder.glob('rss_news_*.db')):
        with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as c:
            stats['files'] += 1
            for rowid, loaded, published, title, provider in c.execute(
                    'SELECT rowid,loaded_at,date,title,provider FROM news WHERE loaded_at>=? AND loaded_at<?', (start, end)):
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
                by_month[loaded[:7]] += 1
                title_key = hashlib.sha256(text.casefold().encode()).hexdigest()
                key = hashlib.sha256((loaded[:10] + ':' + title_key).encode()).hexdigest()
                item = records.setdefault(key, dict(news_id=key, title_key=title_key, loaded_at=loaded,
                    published_at=published, title=text, provider=provider, duplicate_count=0, origins=[]))
                item['duplicate_count'] += 1
                item['origins'].append(dict(file=path.name, rowid=rowid, provider=provider))
                if loaded < item['loaded_at']:
                    item.update(loaded_at=loaded, published_at=published, title=text, provider=provider)
    rows = sorted(records.values(), key=lambda r: (r['loaded_at'], r['news_id']))
    stats.update(unique_titles=len(rows), source_rows_by_month=dict(sorted(by_month.items())),
                 first_loaded_at=rows[0]['loaded_at'] if rows else None,
                 last_loaded_at=rows[-1]['loaded_at'] if rows else None)
    return rows, stats


def seed_title_vectors(cache, rows):
    """Копирует в cache готовые векторы точного текста под новыми news_id из rows.

    Использует только совпадающий текст, digest и preparation, включая версии
    исходного кэша. Возвращает число добавленных записей; исходные файлы не меняет.
    """
    by_title = {}
    for key, payload in cache.execute('SELECT news_id,payload FROM news'):
        by_title.setdefault(json.loads(payload)['title'], []).append(key)
    before = cache.total_changes
    for row in rows:
        for previous in by_title.get(row['title'], []):
            cache.execute('INSERT OR IGNORE INTO embeddings SELECT ?,digest,preparation,dimension,vector '
                          'FROM embeddings WHERE news_id=?', (row['news_id'], previous))
    cache.commit()
    return cache.total_changes - before
