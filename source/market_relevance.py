r"""Объяснимый фильтр заголовков для исследования RTS/MIX, без ценовых признаков.

Требует предмет и событие: политика России, рубль, нефть или ограниченный список
российских компаний. Это исследовательский словарь, не исторический состав индекса
и не доказанная оценка влияния. Неоднозначные заголовки получают uncertain.
Газ и иностранная макростатистика вне основного набора. LLM не вызывается.

Использование: .\.venv\Scripts\python.exe russian_market.py --stage classify
Проверки: .\.venv\Scripts\python.exe -m unittest discover -s tests -p test_market_relevance.py -v
"""

import re
import unicodedata

RULE_VERSION = 'ru-market-rules-v3'
COMPANIES = {
    'SBER': r'\bсбер(?:банк|а|ом|у)?\b|\bsber\b',
    'GAZP': r'\bгазпром(?:а|ом|у)?\b|\bgazp\b',
    'LKOH': r'\bлукойл\w*|\blkoh\b',
    'ROSN': r'\bроснефт\w*|\brosn\b',
    'NVTK': r'\bноват[эе]к\w*|\bnvtk\b',
    'GMKN': r'\bнорникел\w*|\bнорильск\w* никел\w*|\bgmkn\b',
    'TATN': r'\bтатнефт\w*|\btatn\b',
    'VTBR': r'\bвтб\b|\bvtbr\b',
}


def matches(pattern, text):
    """Возвращает наличие регулярного pattern в нормализованном text."""
    return re.search(pattern, text) is not None


def classify_title(title):
    """Возвращает решение/причины/группы по одному title, не используя цены или дату.

    groups задают category:event:object:stage для поиска сопоставимых аналогов. Список
    компаний фиксирован как исследовательская область, веса/членство не утверждаются.
    Одно сообщение может иметь несколько каналов; итоговая причина объясняет все.
    """
    text = ' '.join(unicodedata.normalize('NFKC', title or '').casefold().replace('ё', 'е').split())
    result = dict(decision='exclude', categories=[], groups=[], reasons=[], reason='',
                  rule_version=RULE_VERSION, entities=[], membership_verified=False)
    stage = ('expectation' if matches(r'ожида\w*|может|могут|прогноз\w*|счита\w*|допуска\w*|планиру\w*|рассчитыва\w*|намерен\w*|предлага\w*', text) else
             'scheduled' if matches(r'проведут|пройдут|состоитс\w*|запланирован\w*', text) else
             'reported' if matches(r'утверди\w*|решил\w*|одобри\w*|ввел\w*|введ\w*|продли\w*|снизи\w*|понизи\w*|повыс\w*|сохрани\w*|состави\w*|ускори\w*|замедли\w*|вырос\w*|сократи\w*|достиг\w*|увеличи\w*|прекрати\w*', text) else
             'commentary' if matches(r'заяви\w*|сообщи\w*|предупреди\w*|сказа\w*|подтверди\w*|отмети\w*|обсуди\w*|призна\w*', text) else 'unspecified')
    result['stage'] = stage
    if matches(r'^(?:цены на )?нефть (?:дешев\w*|дорож\w*|снизи\w*|вырос\w*|раст\w*|упа\w*|подешев\w*|подорож\w*)', text):
        result.update(decision='uncertain', reason='Обзор уже произошедшего движения нефти; отдельное новое событие не установлено',
                      reasons=['Обзор уже произошедшего движения нефти; отдельное новое событие не установлено'])
        return result
    if matches(r'^(?:индекс\w* )?(?:ftse|dax|cac\s*40|nikkei|nasdaq|s&p\s*500)\b', text):
        result.update(reason='Обзор иностранного фондового индекса вне основного набора',
                      reasons=['Обзор иностранного фондового индекса вне основного набора'])
        return result
    companies = [name for name, pattern in COMPANIES.items() if matches(pattern, text)]
    result['entities'] = companies
    ru = matches(r'\bросси\w*|\bрф\b|\bроссийск\w*', text)
    geopolitical = ru or matches(r'\bукраин\w*|\bкиев\w*|\bкремл\w*', text)
    cbr = matches(r'банк\w* россии|\bцб\s*(?:рф|россии)\b|\bнабиуллин\w*', text)
    oil = matches(r'\bнефт\w*|\bbrent\b|\burals\b|\bопек\w*', text)
    gas = matches(r'\bгаз(?:а|у|ом|е|овый|овые|ового)?\b|\bспг\b|\bhenry hub\b', text)
    events = []
    if geopolitical:
        for event, pattern in [('sanctions', r'санкци\w*|эмбарго|ценов\w* потол\w*'),
                               ('ceasefire', r'прекращен\w* огня|перемири\w*'),
                               ('negotiations', r'переговор\w*|мирн\w* (?:план|соглашен)\w*'),
                               ('escalation', r'эскалаци\w*|удар\w* по (?:нефт\w*|нпз|порт\w*)')]:
            if matches(pattern, text):
                if event == 'negotiations' and not ru and matches(r'членств\w*|вступлен\w*|евроинтеграци\w*', text):
                    continue
                events.append(('geopolitics', event, 'RU', 'Российский геополитический/санкционный канал'))
    if cbr and matches(r'ставк\w*|денежно.кредитн\w*', text):
        action = ('guidance' if stage == 'expectation' or matches(r'\bне (?:снизи\w*|повыс\w*|понизи\w*)', text) else
                  'cut' if matches(r'снизи\w*|понизи\w*|снизил\w*', text) else
                  'hike' if matches(r'повыс\w*|повыш\w*', text) else
                  'hold' if matches(r'сохрани\w*|остави\w*|без изменен\w*', text) else 'guidance')
        events.append(('monetary', 'rate_' + action, 'RU', 'Ставка или сигнал Банка России'))
    if ru and matches(r'инфляци\w*|индекс потребительск\w* цен', text) and matches(r'росстат|состави\w*|ускори\w*|замедли\w*|достиг\w*|вырос\w*|снизи\w*', text):
        events.append(('monetary', 'inflation', 'RU', 'Показатель инфляции России'))
    if (ru or matches(r'\bминфин\w*|\bрубл\w*', text)) and matches(
            r'валютн\w* выручк\w*|продаж\w* валют\w*|покупк\w* валют\w*|валютн\w* операци\w*|валютн\w* ограничен\w*|контрол\w* капитал\w*', text):
        events.append(('fx', 'regulation_operations', 'RU', 'Валютные операции или регулирование России'))
    if oil:
        if matches(r'\bопек\w*', text) and matches(r'добыч\w*|квот\w*|постав\w*|сокращ\w*', text):
            events.append(('oil', 'opec_supply', 'GLOBAL_OIL', 'Решение/сигнал ОПЕК по предложению нефти'))
        if matches(r'\bнефт\w*|\burals\b', text) and ru and matches(r'экспорт\w*|эмбарго|санкци\w*|потол\w*|нпз|налог\w*|ндпи', text):
            events.append(('oil', 'russian_exports', 'RU_OIL', 'Российский нефтяной экспорт/ограничения/налоги'))
        restoration = matches(r'восстановлен\w*|возобновлен\w*|разблокир\w*|возобнови\w*|восстанови\w*', text)
        if restoration and matches(r'постав\w*|транзит\w*|пролив\w*|добыч\w*', text):
            events.append(('oil', 'supply_restoration', 'GLOBAL_OIL', 'Восстановление нефтяных поставок'))
        if not restoration and matches(r'перебо\w*|блокад\w*|прекрати\w*|останов\w* добыч\w*', text):
            events.append(('oil', 'supply_disruption', 'GLOBAL_OIL', 'Нарушение нефтяных поставок'))
        if matches(r'\bсша\b|\bapi\b|\beia\b', text) and matches(r'запас\w* нефти', text):
            events.append(('oil', 'oil_inventories', 'US_OIL', 'Статистика запасов нефти США, отдельная группа'))
    for company in companies:
        for event, pattern in [('dividends', r'дивиденд\w*'),
                               ('results', r'мсфо|рсбу|чист\w* прибыл\w*|чист\w* убыт\w*|отчетност\w*'),
                               ('sanctions', r'санкци\w*|эмбарго'),
                               ('capital_action', r'байб[эе]к|обратн\w* выкуп\w*|допэмисси\w*|национализаци\w*')]:
            if matches(pattern, text):
                events.append(('corporate', event, company, f'Существенное корпоративное событие {company}'))
    for category, event, obj, reason in events:
        result['groups'].append(f'{category}:{event}:{obj}:{stage}')
        result['categories'].append(category)
        result['reasons'].append(reason)
    for key in ('categories', 'groups', 'reasons'):
        result[key] = sorted(set(result[key]))
    if events and stage != 'unspecified':
        result['decision'] = 'include'
    elif cbr or ru or geopolitical or oil or companies or matches(r'\bпутин\w*|\bмишустин\w*', text):
        result['decision'] = 'uncertain'
        result['reasons'] = ['Есть потенциально связанный объект, но значимое событие по заголовку не установлено']
    elif gas:
        result['reasons'] = ['Газовый рынок без установленной связи с российскими эмитентами/экспортом; вне основного набора']
    else:
        result['reasons'] = ['Не найден канал связи с основным набором RTS/MIX']
    result['reason'] = '; '.join(result['reasons'])
    return result


def compatible_groups(left, right):
    """Возвращает пересечение групп двух классификаций left/right только при include."""
    if not left or not right or left.get('decision') != 'include' or right.get('decision') != 'include':
        return []
    return sorted(set(left['groups']) & set(right['groups']))
