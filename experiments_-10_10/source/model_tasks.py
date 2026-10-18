"""Общий контракт балльной задачи и трёх классов экстремальной реакции.

Преобразует исходные баллы в классы, балансирует только переданное обучение,
строго проверяет JSON и рассчитывает precision/recall без удаления ошибок.
Модуль отдельно не запускается. Примеры использования через скрипты проекта:
  python experiments_-10_10/train_RTX4070.py --task extremes --dry-run
  python experiments_-10_10/train_RTX4070.py --task score --dry-run
Будущие метрики используются только для целевого класса; файловых логов нет.
"""

import json
import random


CLASSES = ('negative', 'none', 'positive')
TASKS = ('score', 'extremes')
EXTREME_PROMPT = (
    'По заголовку новости определи класс последующей реакции указанного фьючерса '
    'на указанном горизонте. Верни только JSON с полем class: '
    'negative для сильной отрицательной реакции (баллы -10..-8), '
    'positive для сильной положительной реакции (баллы 8..10), '
    'none при отсутствии экстремальной реакции (баллы -7..7). '
    'none не означает отсутствие движения. Не добавляй другие поля.'
)


def score_class(score):
    """Возвращает класс целого score в диапазоне −10…10; неверные баллы отклоняет."""
    if type(score) is not int or not -10 <= score <= 10:
        raise ValueError('Для класса нужен целый балл -10..10')
    return 'negative' if score <= -8 else 'positive' if score >= 8 else 'none'


def task_name(manifest):
    """Возвращает проверенную задачу манифеста; старые манифесты относятся к score."""
    task = manifest.get('task', 'score')
    if task not in TASKS:
        raise ValueError(f'Неизвестная задача модели: {task}')
    return task


def run_directories(base):
    """Возвращает стандартные каталоги обоих профилей и обоих режимов внутри base."""
    return [base / (profile + suffix) for profile in ('RTX4070', 'RTX3090Ti')
            for suffix in ('', '_extremes')]


def balance_training(rows, seed):
    """Дополняет rows до равных классов с seed; возвращает новый список и счётчики.

Сохраняет каждый исходный пример, копии выбирает с возвращением. Исходный
список не меняет; валидацию вызывать через эту функцию запрещено.
"""
    groups = {c: [row for row in rows if score_class(row['score']) == c] for c in CLASSES}
    if any(not group for group in groups.values()):
        raise ValueError('В обучении должны присутствовать все три класса')
    target = max(map(len, groups.values()))
    rng = random.Random(seed)
    balanced = list(rows)
    for group in groups.values():
        balanced.extend(rng.choices(group, k=target - len(group)))
    rng.shuffle(balanced)
    return balanced, dict(method='oversample-to-largest', seed=seed,
                          before={c: len(groups[c]) for c in CLASSES},
                          after={c: target for c in CLASSES})


def parse_response(text, task='score'):
    """Возвращает балл или класс из text по task; лишние поля и неверные типы отклоняет."""
    if task not in TASKS:
        raise ValueError(f'Неизвестная задача: {task}')
    try:
        data = json.loads(text)
    except (TypeError, ValueError) as error:
        raise ValueError('Ответ модели не является JSON') from error
    key = 'class' if task == 'extremes' else 'score'
    if not isinstance(data, dict) or set(data) != {key}:
        raise ValueError(f'Ответ должен содержать только поле {key}')
    value = data[key]
    if task == 'extremes':
        if not isinstance(value, str) or value not in CLASSES:
            raise ValueError('class должен быть negative, none или positive')
    elif type(value) is not int or not -10 <= value <= 10:
        raise ValueError('score должен быть целым числом от -10 до +10')
    return value


def classification_metrics(predictions):
    """Возвращает метрики predictions с actual-баллом и predicted-классом либо None.

Неверные ответы входят в знаменатель recall и accuracy и отдельный столбец
матрицы. Precision без сигналов равен None; это не успешная точность.
"""
    confusion = {c: {p: 0 for p in (*CLASSES, 'invalid')} for c in CLASSES}
    for row in predictions:
        actual = score_class(row['actual'])
        predicted = row['predicted'] if row['predicted'] in CLASSES else 'invalid'
        confusion[actual][predicted] += 1
    count = len(predictions)
    per_class = {}
    for c in CLASSES:
        tp = confusion[c][c]
        support = sum(confusion[c].values())
        signals = sum(confusion[a][c] for a in CLASSES)
        precision = tp / signals if signals else None
        recall = tp / support if support else None
        per_class[c] = dict(support=support, predicted_count=signals, true_positives=tp,
                            false_positives=signals - tp, false_negatives=support - tp,
                            precision=precision, recall=recall,
                            f1=2 * tp / (signals + support) if signals + support else None)
    invalid = sum(confusion[c]['invalid'] for c in CLASSES)
    return dict(sample_size=count, valid_answers=count - invalid,
                invalid_fraction=invalid / count if count else None,
                accuracy=sum(confusion[c][c] for c in CLASSES) / count if count else None,
                no_signal_baseline_accuracy=sum(confusion['none'].values()) / count if count else None,
                signal_count=sum(per_class[c]['predicted_count'] for c in ('negative', 'positive')),
                classes=per_class, confusion=confusion)
