"""Общая подготовка данных и QLoRA-обучение классов реакции либо балла −10…+10.

Читает готовую разметку CSV/JSONL, проверяет метки, делит историю по времени,
исключает пересекающиеся окна и перепечатки. На вход модели передаются только
заголовок, инструмент и горизонт; ошибка обучения считается по JSON-ответу.
Без --data читает data/experiments_-10_10/scores.jsonl относительно корня проекта.
Сохраняет адаптер, токенизатор, отчёты и точную ревизию базовой модели.
После всех оценок записывает SHA-256 адаптера и ревизию базы в отметку
завершения, которую проверяет последующий экспорт в Ollama.
Файловые логи не создаются; ход работы выводится в консоль.
Обучение запускается в общем .venv проекта с зависимостями из корневого requirements.txt.
По умолчанию task=extremes: баллы ±8…±10 преобразуются в negative/positive,
остальные — в none. Балансируется только обучение после временного разделения.
Результаты сохраняются в PROFILE_extremes; --task score сохраняет прежнюю задачу.

Запуск из корня через профиль:
  python experiments_-10_10/train_RTX4070.py --dry-run
  python experiments_-10_10/train_RTX3090Ti.py --epochs 1
  python experiments_-10_10/train_RTX4070.py --task score --dry-run
Параметры: python experiments_-10_10/train_RTX4070.py --help.
Этот модуль отдельно не запускается. Формат разметки описан в соседнем README.md.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_tasks import (EXTREME_PROMPT, TASKS, balance_training, classification_metrics,
                         parse_response, score_class)
MSK = timezone(timedelta(hours=3))
HORIZONS = (5, 15, 30, 60, 120)
PROFILES = {
    'RTX4070': dict(model='Qwen/Qwen3-4B-Instruct-2507', batch_size=1, accumulation=16),
    'RTX3090Ti': dict(model='Qwen/Qwen2.5-7B-Instruct', batch_size=2, accumulation=8),
}
SYSTEM_PROMPT = (
    'Оцени последующую реакцию указанного фьючерса на указанном горизонте '
    'по заголовку новости. Верни только JSON с целым полем score от -10 до 10. '
    'Знак означает направление, модуль — величину и качество движения; '
    'крайние оценки соответствуют устойчивой реакции.'
)


def parse_time(value):
    """Возвращает datetime МСК из ISO-строки; время без зоны трактует как МСК."""
    parsed = datetime.fromisoformat(str(value))
    return parsed.replace(tzinfo=MSK) if parsed.tzinfo is None else parsed.astimezone(MSK)


def integer(value, name):
    """Проверяет конечное целое value и возвращает int; name используется в ошибке."""
    if isinstance(value, bool):
        raise ValueError(f'{name}: логическое значение не является целым баллом')
    number = float(value)
    if not math.isfinite(number) or number != int(number):
        raise ValueError(f'{name}: требуется конечное целое число')
    return int(number)


def normalize_text(text):
    """Возвращает текст без различий регистра и повторных пробелов для поиска копий."""
    return ' '.join(text.casefold().split())


def read_examples(path):
    """Читает CSV/JSONL path и возвращает (валидные строки, счётчики).

    Строки без score пропускаются. Неверная структура, пустые идентификаторы,
    неверная метка, конфликт news_id или повтор news_id/instrument/horizon
    вызывает ValueError. Исходный файл не изменяется.
    """
    path = Path(path)
    if path.suffix.lower() not in ('.csv', '.jsonl'):
        raise ValueError('Поддерживаются только CSV и JSONL')
    result, identities, keys, versions = [], {}, set(), set()
    stats = dict(source_rows=0, unlabeled_rows=0)
    with path.open(encoding='utf-8-sig', newline='') as handle:
        records = csv.DictReader(handle) if path.suffix.lower() == '.csv' else (
            json.loads(line) for line in handle if line.strip())
        for position, raw in enumerate(records, 1):
            stats['source_rows'] += 1
            try:
                if not isinstance(raw, dict):
                    raise ValueError('Ожидается объект с полями новости')
                if raw.get('score') in (None, ''):
                    stats['unlabeled_rows'] += 1
                    continue
                score = integer(raw['score'], 'score')
                horizon = integer(raw['horizon'], 'horizon')
                if not -10 <= score <= 10 or horizon not in HORIZONS:
                    raise ValueError('Баллы должны быть −10…10, горизонты — 5,15,30,60,120')
                if raw.get('instrument') not in ('MIX', 'RTS'):
                    raise ValueError('Неизвестный инструмент')
                stable_text = str(raw['stable']).lower()
                if stable_text not in ('true', 'false', '1', '0'):
                    raise ValueError('stable должен быть true/false или 1/0')
                stable = stable_text in ('true', '1')
                if stable != (abs(score) == 10):
                    raise ValueError('Крайние метки ±10 должны точно соответствовать stable')
                if raw.get('status', 'ok') != 'ok':
                    raise ValueError('Неполное окно не может иметь числовой балл')
                title, news_id = str(raw['title']).strip(), str(raw['news_id']).strip()
                score_version = str(raw['score_version']).strip()
                if (not title or not news_id or not score_version
                        or any(raw[key] is None for key in ('title', 'news_id', 'score_version'))):
                    raise ValueError('Нужны непустые title, news_id, score_version')
                loaded, end = parse_time(raw['loaded_at']), parse_time(raw['end_at'])
                if end < loaded + timedelta(minutes=horizon):
                    raise ValueError('end_at должен обозначать конец полного горизонта')
                identity = (loaded, title)
                if news_id in identities and identities[news_id] != identity:
                    raise ValueError('Противоречивые время/заголовок для одного news_id')
                identities[news_id] = identity
                key = (news_id, raw['instrument'], horizon)
                if key in keys:
                    raise ValueError('Повтор news_id/instrument/horizon')
                keys.add(key)
                versions.add(score_version)
                result.append(dict(news_id=news_id, title=title, loaded_at=loaded,
                                   end_at=end, instrument=raw['instrument'], horizon=horizon,
                                   score=score, stable=stable, score_version=score_version,
                                   group_id=str(raw.get('group_id') or '').strip()))
            except (KeyError, ValueError, TypeError, OverflowError) as error:
                raise ValueError(f'Строка данных {position}: {error}') from error
    if len(versions) > 1:
        raise ValueError('В одном обучении нельзя смешивать разные score_version')
    stats['score_version'] = next(iter(versions), None)
    return sorted(result, key=lambda r: (r['loaded_at'], r['news_id'], r['instrument'], r['horizon'])), stats


def group_keys(row):
    """Возвращает ключи новости, точного нормализованного текста и внешней группы копий."""
    keys = {('news', row['news_id']), ('text', normalize_text(row['title']))}
    if row['group_id']:
        keys.add(('group', row['group_id']))
    return keys


def split_examples(rows, start, validation_start, cutoff):
    """Возвращает train, validation и статистику с временными границами [start,cutoff).

    Все размеченные горизонты новости удаляются из соответствующей выборки,
    если хотя бы один ответ доступен лишь на/после её верхней границы. Перепечатки
    из валидации удаляются из обучения по идентификатору, тексту или group_id.
    """
    start, boundary, cutoff = map(parse_time, (start, validation_start, cutoff))
    if not start < boundary < cutoff:
        raise ValueError('Требуется train-start < validation-start < cutoff')
    ends = {}
    for row in rows:
        ends[row['news_id']] = max(ends.get(row['news_id'], row['end_at']), row['end_at'])
    train, valid = [], []
    stats = dict(outside_period_rows=0, purged_window_rows=0, purged_reprint_rows=0)
    for row in rows:
        if not start <= row['loaded_at'] < cutoff:
            stats['outside_period_rows'] += 1
            continue
        is_train = row['loaded_at'] < boundary
        if ends[row['news_id']] >= (boundary if is_train else cutoff):
            stats['purged_window_rows'] += 1
            continue
        (train if is_train else valid).append(row)
    validation_keys = set().union(*(group_keys(r) for r in valid)) if valid else set()
    clean_train = [r for r in train if not group_keys(r) & validation_keys]
    stats['purged_reprint_rows'] = len(train) - len(clean_train)
    if not clean_train or not valid:
        raise ValueError('После разделения нужны непустые обучение и валидация')
    return clean_train, valid, stats


def make_messages(row):
    """Возвращает чат по row с заданием его task; во входе только заголовок, инструмент и горизонт."""
    payload = dict(title=row['title'], instrument=row['instrument'], horizon_minutes=row['horizon'])
    prompt = EXTREME_PROMPT if row.get('task') == 'extremes' else SYSTEM_PROMPT
    return [dict(role='system', content=prompt),
            dict(role='user', content=json.dumps(payload, ensure_ascii=False))]


def encode_example(row, tokenizer, max_length):
    """Возвращает токены ответа по row.task с маской входа; отвергает превышение max_length."""
    messages = make_messages(row)
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    target = {'class': score_class(row['score'])} if row.get('task') == 'extremes' else {'score': row['score']}
    answer = json.dumps(target, separators=(',', ':'))
    full = tokenizer.apply_chat_template(messages + [dict(role='assistant', content=answer)],
                                         tokenize=True, add_generation_prompt=False)
    if full[:len(prefix)] != prefix or len(full) <= len(prefix):
        raise ValueError('Шаблон токенизатора несовместим с маской целевого ответа')
    if len(full) > max_length:
        raise ValueError(f"Новость {row['news_id']}: {len(full)} токенов > --max-length {max_length}")
    return dict(input_ids=full, attention_mask=[1] * len(full),
                labels=[-100] * len(prefix) + full[len(prefix):])


class TokenizedExamples:
    """Хранит строки и токенизирует их по запросу, не дублируя все токены в памяти."""

    def __init__(self, rows, tokenizer, max_length):
        """Сохраняет rows/tokenizer и проверяет длины всех примеров до начала обучения."""
        self.rows, self.tokenizer, self.max_length = rows, tokenizer, max_length
        for row in rows:
            encode_example(row, tokenizer, max_length)

    def __len__(self):
        """Возвращает число обучающих сочетаний новости, инструмента и горизонта."""
        return len(self.rows)

    def __getitem__(self, index):
        """Возвращает токенизированный пример с индексом index и маской ответа."""
        return encode_example(self.rows[index], self.tokenizer, self.max_length)


def summarize(rows):
    """Возвращает размеры, даты, распределения исходных баллов и трёх классов для rows."""
    return dict(rows=len(rows), news=len({r['news_id'] for r in rows}),
                first_loaded_at=rows[0]['loaded_at'].isoformat(),
                last_loaded_at=rows[-1]['loaded_at'].isoformat(),
                scores=dict(sorted(Counter(str(r['score']) for r in rows).items())),
                classes=dict(Counter(score_class(r['score']) for r in rows)),
                tasks=dict(Counter(f"{r['instrument']}_{r['horizon']}" for r in rows)),
                without_group_id=sum(not r['group_id'] for r in rows))


def save_json(path, payload):
    """Перезаписывает JSON-отчёт path данными payload; даты/пути записывает строками."""
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def evaluate_predictions(model, tokenizer, rows, limit, seed):
    """Генерирует ответы на воспроизводимой подвыборке rows и возвращает метрики/предсказания.

    limit=0 означает всю валидацию. Для extremes считает precision/recall классов;
    для score — MAE и нулевой ориентир. Сохраняет исходный actual-балл в обоих режимах.
    """
    import torch

    selected = random.Random(seed).sample(rows, min(limit, len(rows))) if limit else list(rows)
    task = rows[0].get('task', 'score')
    predictions, errors, zero_errors = [], [], []
    model.eval()
    model.config.use_cache = True
    for row in selected:
        ids = tokenizer.apply_chat_template(make_messages(row), tokenize=True, add_generation_prompt=True)
        inputs = torch.tensor([ids], device=model.device)
        with torch.inference_mode():
            generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                       do_sample=False, max_new_tokens=32,
                                       pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True).strip()
        predicted = None
        try:
            predicted = parse_response(text, task)
        except ValueError:
            pass
        predictions.append(dict(news_id=row['news_id'], instrument=row['instrument'],
                                horizon=row['horizon'], actual=row['score'], predicted=predicted, raw=text))
        if task == 'extremes':
            predictions[-1]['actual_class'] = score_class(row['score'])
        if task == 'score' and predicted is not None:
            errors.append(abs(predicted - row['score']))
            zero_errors.append(abs(row['score']))
    if task == 'extremes':
        return classification_metrics(predictions), predictions
    return dict(sample_size=len(selected), valid_answers=len(errors),
                invalid_fraction=1 - len(errors) / len(selected),
                mae=sum(errors) / len(errors) if errors else None,
                zero_baseline_mae_same_rows=sum(zero_errors) / len(zero_errors) if zero_errors else None,
                zero_baseline_mae_all_rows=sum(abs(r['score']) for r in selected) / len(selected)), predictions


def train_adapter(args, train_rows, validation_rows):
    """Обучает QLoRA по args, сохраняет адаптер и отчёты; требует CUDA и зависимости.

    train_rows уже преобразованы и сбалансированы в main, validation_rows сохранены
    без балансировки. База из Hugging Face заморожена; обучаются только LoRA-матрицы.
    """
    cache = ROOT / 'data' / 'experiments_-10_10' / 'hf_cache'
    os.environ.setdefault('HF_HOME', str(cache))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    try:
        import torch
        from huggingface_hub import model_info
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                                  DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    except ImportError as error:
        raise RuntimeError(f'В окружении {sys.executable} отсутствуют зависимости обучения. '
                           'Установите корневой requirements.txt по разделу установки в '
                           'experiments_-10_10/README.md и выберите .venv в VS Code.') from error
    if not torch.cuda.is_available():
        raise RuntimeError('Для QLoRA нужна NVIDIA CUDA; текущий PyTorch не видит GPU')
    set_seed(args.seed)
    revision = model_info(args.model, revision=args.revision).sha
    gpu = torch.cuda.get_device_name(0)
    print(f'GPU: {gpu}; модель: {args.model}; ревизия: {revision}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision, cache_dir=str(cache))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    train_data = TokenizedExamples(train_rows, tokenizer, args.max_length)
    valid_data = TokenizedExamples(validation_rows, tokenizer, args.max_length)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                     bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=revision, cache_dir=str(cache), quantization_config=quantization,
        torch_dtype=dtype, device_map={'': 0}, attn_implementation='sdpa')
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                            gradient_checkpointing_kwargs={'use_reentrant': False})
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none', task_type='CAUSAL_LM'))
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(args.output / 'trainer'), num_train_epochs=args.epochs,
        max_steps=args.max_steps, learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.accumulation, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        bf16=dtype == torch.bfloat16, fp16=dtype == torch.float16,
        optim='adamw_torch', warmup_ratio=0.03, lr_scheduler_type='cosine',
        eval_strategy='epoch', save_strategy='no', logging_strategy='steps', logging_steps=10,
        report_to=[], seed=args.seed, data_seed=args.seed, dataloader_num_workers=0,
        remove_unused_columns=False, prediction_loss_only=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_data,
                      eval_dataset=valid_data,
                      data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    manifest = dict(profile=args.profile, model=args.model, revision=revision, gpu=gpu,
                    parameters=vars(args), task=args.task,
                    system_prompt=EXTREME_PROMPT if args.task == 'extremes' else SYSTEM_PROMPT,
                    packages={name: version(name) for name in
                              ('torch', 'transformers', 'peft', 'bitsandbytes', 'accelerate')},
                    validation_policy='chronological-purge-reprints-v1',
                    loss='response-only causal cross-entropy',
                    checkpoint_policy='final adapter; no intermediate checkpoints')
    save_json(args.output / 'training_manifest.json', manifest)
    trained = trainer.train()
    validation_metrics = trainer.evaluate()
    adapter_dir = args.output / 'adapter'
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    save_json(args.output / 'training_metrics.json', dict(train=trained.metrics, validation=validation_metrics))
    numeric, predictions = evaluate_predictions(model, tokenizer, validation_rows, args.prediction_limit, args.seed)
    save_json(args.output / 'score_metrics.json', numeric)
    if args.task == 'extremes':
        save_json(args.output / 'classification_metrics.json', numeric)
    save_json(args.output / 'validation_predictions.json', predictions)
    with (adapter_dir / 'adapter_model.safetensors').open('rb') as handle:
        adapter_digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    save_json(args.output / 'adapter_provenance.json', dict(
        model=args.model, revision=revision, task=args.task, adapter_sha256=adapter_digest))
    print(f'Адаптер сохранён: {adapter_dir}', flush=True)


def build_parser(profile):
    """Возвращает argparse-парсер с параметрами и начальными настройками profile."""
    defaults = PROFILES[profile]
    parser = argparse.ArgumentParser(description=f'QLoRA-обучение оценки новостей: профиль {profile}')
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'data' / 'experiments_-10_10' / 'scores.jsonl',
                        help='Готовая разметка CSV/JSONL; по умолчанию data/experiments_-10_10/scores.jsonl')
    parser.add_argument('--task', choices=TASKS, default='extremes', help='Задача: extremes (по умолчанию) либо прежний score')
    parser.add_argument('--balance', choices=('oversample', 'none'), default='oversample',
                        help='Балансировка классов extremes только в обучении; по умолчанию oversample')
    parser.add_argument('--output', type=Path,
                        help='Каталог: data/experiments_-10_10/PROFILE_extremes либо PROFILE для score')
    parser.add_argument('--model', default=defaults['model'], help='Идентификатор модели Hugging Face')
    parser.add_argument('--revision', default='main', help='Ревизия базы; разрешённый SHA сохраняется в отчёте')
    parser.add_argument('--train-start', default='2025-09-01', help='Начало истории, включительно, МСК')
    parser.add_argument('--validation-start', default='2026-08-01', help='Начало отложенной валидации, МСК')
    parser.add_argument('--cutoff', default='2026-09-01', help='Конец доступной истории, исключительно, МСК')
    parser.add_argument('--epochs', type=float, default=1, help='Число проходов по обучению')
    parser.add_argument('--max-steps', type=int, default=-1, help='Ограничение шагов; положительное заменяет epochs')
    parser.add_argument('--batch-size', type=int, default=defaults['batch_size'], help='Примеры на один шаг GPU')
    parser.add_argument('--accumulation', type=int, default=defaults['accumulation'], help='Накопление градиентов')
    parser.add_argument('--max-length', type=int, default=512, help='Максимум токенов входа и ответа; превышение — ошибка')
    parser.add_argument('--rank', type=int, default=16, help='Ранг LoRA')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Начальная скорость обучения')
    parser.add_argument('--prediction-limit', type=int,
                        help='Лимит генерации; по умолчанию вся валидация extremes, 256 для score; 0 — все')
    parser.add_argument('--seed', type=int, default=42, help='Начальное состояние генераторов случайных чисел')
    parser.add_argument('--dry-run', action='store_true', help='Только проверка данных, без модели и GPU')
    return parser


def main(profile, argv=None):
    """Делит данные по времени, задаёт классы и балансирует только обучение; запускает профиль argv."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8')
    parser = build_parser(profile)
    args = parser.parse_args(argv)
    args.profile = profile
    args.prediction_limit = args.prediction_limit if args.prediction_limit is not None else (0 if args.task == 'extremes' else 256)
    args.output = args.output or ROOT / 'data' / 'experiments_-10_10' / (profile + ('_extremes' if args.task == 'extremes' else ''))
    try:
        if (any(getattr(args, name) <= 0 for name in ('batch_size', 'accumulation', 'max_length', 'rank'))
                or not math.isfinite(args.epochs) or args.epochs <= 0
                or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
                or args.prediction_limit < 0 or args.max_steps == 0 or args.max_steps < -1):
            raise ValueError('Параметры размеров/скорости/эпох должны быть положительными; проверьте лимиты')
        rows, read_stats = read_examples(args.data)
        train, valid, split_stats = split_examples(rows, args.train_start, args.validation_start, args.cutoff)
        train = [dict(r, task=args.task) for r in train]
        valid = [dict(r, task=args.task) for r in valid]
        original_train = summarize(train)
        balance = dict(method='none')
        if args.task == 'extremes':
            if set(score_class(r['score']) for r in train) != {'negative', 'none', 'positive'}:
                raise ValueError('В обучении должны присутствовать все три класса')
            if args.balance == 'oversample':
                train, balance = balance_training(train, args.seed)
        args.output = args.output.resolve()
        if args.output.exists():
            allowed = {'data_report.json'}
            if any(p.name not in allowed for p in args.output.iterdir()):
                raise ValueError('Каталог результатов уже содержит обучение; задайте другой --output')
        args.output.mkdir(parents=True, exist_ok=True)
        with args.data.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        report = dict(profile=profile, task=args.task, model=args.model, data_sha256=digest, parameters=vars(args),
                      train_before_balance=original_train, balance=balance,
                      read=read_stats, split=split_stats, train=summarize(train), validation=summarize(valid),
                      warning='Без group_id обнаруживаются только точные нормализованные перепечатки; '
                              'близкие переформулировки требуют предварительной группировки.')
        save_json(args.output / 'data_report.json', report)
        print(f'Обучение: {len(train)} примеров; валидация: {len(valid)}; модель: {args.model}', flush=True)
        if args.dry_run:
            print(f'Проверка данных завершена: {args.output / "data_report.json"}', flush=True)
        else:
            train_adapter(args, train, valid)
        return 0
    except (ValueError, OSError, RuntimeError) as error:
        print(f'Ошибка: {error}', file=sys.stderr)
        return 1
