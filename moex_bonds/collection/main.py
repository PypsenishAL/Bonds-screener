"""Collect TQCB bonds, the CBR curve and weekly RUONIA: python main.py."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
import json
from pathlib import Path
import re
import sys

import pandas as pd

from moex_bonds.sources.moex_client import MSK, MoexClient, tables
from moex_bonds.sources.cbr_zcyc import run_zcyc
from moex_bonds.sources.cbr_ruonia import run_ruonia
from moex_bonds.storage.daily_storage import DEFAULT_DAILY_OUTPUT, output_lock, reset_daily_output

MARKET = 'engines/stock/markets/bonds'
HISTORY = 'history/' + MARKET
SPREADS = ['ZSPREAD', 'ZSPREADATWAPRICE', 'ZSPREADATPREVWAPRICE', 'ZSPREADBP', 'GSPREADBP']
ENDPOINTS = [
    MARKET + '/securities.json', MARKET + '/boards/TQCB/securities.json',
    'securities/{SECID}.json', 'securities/{SECID}/bondization.json',
    MARKET + '/securities/{SECID}.json',
    MARKET + '/boards/{BOARDID}/securities/{SECID}.json',
    MARKET + '/boards/{BOARDID}/securities/{SECID}/orderbook.json',
    HISTORY + '/yields/{SECID}.json',
    HISTORY + '/boards/{BOARDID}/yields/{SECID}.json',
    HISTORY + '/boards/{BOARDID}/securities/{SECID}.json',
]
METRICS = {
    'SECID': ['SECID'], 'ISIN': ['ISIN'], 'Название': ['SHORTNAME', 'SECNAME', 'NAME'],
    'Эмитент': ['EMITTER', 'EMITTERNAME', 'ISSUERNAME', 'EMITENTNAME', 'EMITTER_ID'],
    'Тип облигации': ['TYPE', 'TYPENAME', 'GROUP', 'GROUPNAME', 'BONDTYPE', 'BONDSUBTYPE'],
    'Тип купона': ['COUPON_DETAILS', 'COUPONTYPE', 'COUPON_TYPE', 'COUPONKIND', 'COUPONFORMULA', 'FLOATERTYPE', 'FLOATER'],
    'Валюта номинала': ['FACEUNIT'], 'Номинал': ['FACEVALUE', 'INITIALFACEVALUE'],
    'Ставка купона': ['COUPONPERCENT'], 'Выплата купона': ['COUPONVALUE', 'VALUE'],
    'Период купона': ['COUPONPERIOD', 'COUPONFREQUENCY'], 'Погашение': ['MATDATE', 'MATURITYDATE'],
    'Оферта': ['OFFERDATE', 'BUYBACKDATE', 'OFFERSTARTDATE', 'OFFERENDDATE'],
    'Амортизация': ['AMORTDATE', 'AMORTIZATION', 'AMORTIZATIONS', 'ISAMORTIZED'],
    'Кредитный рейтинг': ['CREDITRATING', 'RATING', 'RATINGLEVEL'],
    'Отрасль': ['SECTOR', 'SECTORNAME', 'INDUSTRY', 'INDUSTRYNAME'],
    'Последняя цена': ['LAST'], 'Средневзвешенная цена': ['WAPRICE', 'WAPRICE', 'WAPRICEMARKET'],
    'Bid': ['BID'], 'Ask': ['ASK', 'OFFER'], 'Закрытие': ['CLOSE', 'CLOSEPRICE', 'PREVPRICE', 'LEGALCLOSEPRICE'],
    'НКД': ['ACCRUEDINT'], 'Биржевая доходность (не всегда YTM)': ['YIELD', 'YIELDATWAPRICE', 'YIELDATPREVWAPRICE'],
    'Эффективная доходность': ['EFFECTIVEYIELD', 'EFFECTIVEYIELDWAPRICE'],
    'Дюрация (тип требует проверки)': ['DURATION', 'DURATIONWAPRICE', 'DURATIONYEAR', 'MODIFIEDDURATION'],
    'Обновление': ['UPDATETIME', 'SYSTIME', 'TRADEMOMENT'],
    'Оборот RUB': ['VALTODAY_RUR'], 'Оборот в валюте расчётов': ['VALTODAY', 'VALUE'],
    'Сделки': ['NUMTRADES'], 'Объём': ['VOLTODAY', 'VOLUME'],
    'Последняя сделка': ['TIME', 'TRADEMOMENT'],
    'Объём лучшего Bid': ['BIDDEPTH'], 'Объём лучшего Ask': ['OFFERDEPTH', 'ASKDEPTH'],
}
UNITS = {'ZSPREADBP': 'б.п. (суффикс BP; методология не подтверждена)',
         'GSPREADBP': 'б.п. (суффикс BP; методология не подтверждена)',
         'VALTODAY_RUR': 'RUB', 'FACEVALUE': 'FACEUNIT', 'COUPONPERCENT': '%',
         'COUPONPERIOD': 'дни; сверить со словарём', 'NUMTRADES': 'сделок'}
SOURCES = {
    'reference': 'https://iss.moex.com/iss/reference/',
    'history_yields_board': 'https://iss.moex.com/iss/reference/495',
    'columns': 'https://iss.moex.com/iss/reference/381',
    'asts_spreads': 'https://www.moex.com/n95330',
}


def present(value):
    return value is not None and not pd.isna(value) and str(value).strip() != ''


def availability(label, field, source, rows, total):
    """Coverage counts instruments, never repeated rows or numeric truthiness."""
    values = {str(row['SECID']): row[field] for row in rows if field in row and present(row[field])}
    nonzero = {key: value for key, value in values.items()
               if not isinstance(value, (int, float)) or value != 0}
    found = any(field in row for row in rows)
    status = ('Не удалось проверить' if total == 0 else
              'Поле не найдено' if not found else
              'Поле есть, но значения пустые' if not values else
              'Доступно' if len(values) == total else 'Частично доступно')
    example = next(iter(values.items()), ('', ''))
    return dict(indicator=label, api_field=field, source=source, nonempty_count=len(values),
                nonzero_count=len(nonzero), total_instruments=total,
                coverage=f'{len(values)}/{total}', status=status, units=UNITS.get(field, 'не подтверждены'),
                example_secid=example[0], example_value=example[1])


def age_seconds(value, received):
    """Only full timestamps; time of day alone cannot identify a trade date."""
    if not present(value) or not re.match(r'^\d{4}-\d\d-\d\d[ T]', str(value)):
        return None
    try:
        stamp = datetime.fromisoformat(str(value))
        stamp = stamp.replace(tzinfo=MSK) if stamp.tzinfo is None else stamp
        age = (datetime.fromisoformat(received) - stamp).total_seconds()
        return round(age, 3) if age >= 0 else None
    except ValueError:
        return None


def export_csv(rows, path, columns=None):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if frame.empty and columns:
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, index=False, encoding='utf-8-sig')


def flatten_listing(payload):
    frames = tables(payload)
    result = frames.get('securities', pd.DataFrame()).copy()
    if result.empty:
        return result
    keys = ['SECID'] + (['BOARDID'] if 'BOARDID' in result else [])
    result = result.rename(columns={c: 'securities.' + c for c in result if c not in keys})
    for section in ('marketdata', 'marketdata_yields'):
        other = frames.get(section, pd.DataFrame())
        if not other.empty and all(k in other for k in keys):
            other = other.rename(columns={c: section + '.' + c for c in other if c not in keys})
            result = result.merge(other.drop_duplicates(keys), on=keys, how='left', validate='many_to_one')
    return result


def observe(payload, secid, source, observations, sample, freshness, received):
    for section, frame in tables(payload).items():
        if section.endswith('.cursor'):
            continue
        records = frame.to_dict('records')
        if section == 'description' and {'name', 'value'}.issubset(frame.columns):
            records = [{str(row['name']).upper(): row['value'] for row in records}]
        for row in records:
            # Never merge one board's quote into another board's quote.
            if row.get('SECID', secid) != secid:
                continue
            normalized = {str(k).upper(): v for k, v in row.items()}
            if normalized.get('BOARDID', sample.get('BOARDID')) != sample.get('BOARDID'):
                continue
            observations.append(dict(secid=secid, source=source, section=section, row=normalized))
            if section not in ('orderbook', 'coupons', 'amortizations', 'offers'):
                for field, value in normalized.items():
                    sample[f'{source}.{section}.{field}'] = value
            for field, value in normalized.items():
                if field in ('SYSTIME', 'UPDATETIME', 'TIME', 'TRADEMOMENT', 'ZCYCMOMENT'):
                    freshness.append(dict(SECID=secid, source=source, section=section, field=field,
                        value=value, received_at_msk=received,
                        observed_age_seconds=age_seconds(value, received),
                        meaning='возраст метки; не измерение транспортной задержки API'))
        # Empty blocks still prove which fields exist, but do not prove values.
        if not records:
            observations.append(dict(secid=secid, source=source, section=section,
                                     row={str(c).upper(): None for c in frame.columns}))


def make_availability(observations, n, failures):
    grouped = defaultdict(list)
    for item in observations:
        grouped[item['source'] + '#' + item['section']].append({'SECID': item['secid'], **item['row']})
    discovered = {field for rows in grouped.values() for row in rows for field in row if 'SPREAD' in field}
    metrics = {**METRICS, **{field: [field] for field in sorted(set(SPREADS) | discovered)}}
    output = []
    for label, fields in metrics.items():
        matched = False
        for source, rows in grouped.items():
            for field in dict.fromkeys(fields):
                # Calendar VALUE is a payment, market VALUE is turnover.
                if field == 'VALUE' and ((label == 'Выплата купона') != ('#coupons' in source)):
                    continue
                if any(field in row for row in rows):
                    output.append(availability(label, field, source, rows, n))
                    matched = True
        if not matched:
            output.append(dict(indicator=label, api_field='|'.join(fields), source='',
                nonempty_count=0, nonzero_count=0, total_instruments=n, coverage=f'0/{n}',
                status=('Не удалось проверить' if n == 0 else 'Ошибка запроса' if failures else 'Поле не найдено'),
                units='не подтверждены', example_secid='', example_value=''))
    return output


COMPLETE = {'empty_terminal', 'cursor_complete'}
DAY_HISTORY = HISTORY + '/boards/TQCB/securities.json'
DAY_YIELDS = HISTORY + '/boards/TQCB/yields.json'
FIXED = 'Облигация с фиксированным (известным) купоном'


def request_diagnostic(client):
    """Keep transport failures distinct from successful but empty ISS blocks."""
    if not client.records:
        return 'Нет записи HTTP-запроса'
    record = client.records[-1]
    status = record.get('http_status')
    transport = 'HTTP-ответ не получен' if status is None else f'HTTP {status}'
    detail = record.get('error') or record.get('outcome', '')
    return f"{transport}; {detail}; URL: {record.get('url', '')}"


def latest_trade_date(client, requested=None):
    """Use published MOEX dates, never a weekday/calendar guess."""
    if requested is not None:
        return requested.isoformat()
    payload = client.get(HISTORY + '/boards/TQCB/dates.json',
                         {'tradingsession': 3}, purpose='latest_published_date')
    if payload is None:
        raise ValueError('Не удалось выполнить запрос торговой даты: ' + request_diagnostic(client))
    dates = tables(payload).get('dates', pd.DataFrame())
    if dates.empty or 'till' not in dates:
        raise ValueError('Ответ MOEX не содержит непустую dates.till; '
                         f'секции: {list(payload)}; ' + request_diagnostic(client))
    parsed = pd.to_datetime(dates['till'], errors='coerce').dropna()
    if parsed.empty or parsed.max().date() > datetime.now(MSK).date():
        raise ValueError('Некорректная дата в dates.till')
    return parsed.max().date().isoformat()


def complete_day(client, path, section, day):
    params = {'date': day, 'limit': 100}
    if section == 'history':
        params['tradingsession'] = 3
    frame, state = client.collect(path, section, params, purpose='daily_' + section)
    if state not in COMPLETE:
        raise ValueError(f'Неполная дневная выгрузка {section}: {state}; ' + request_diagnostic(client))
    if frame.empty:
        return frame
    if not {'SECID', 'BOARDID', 'TRADEDATE'}.issubset(frame.columns):
        raise ValueError(f'Нет дневных ключей в {section}')
    if not frame.TRADEDATE.eq(day).all() or not frame.BOARDID.eq('TQCB').all():
        raise ValueError(f'В {section} получена другая дата или режим торгов')
    if section == 'history' and 'TRADINGSESSION' in frame:
        frame = frame[pd.to_numeric(frame.TRADINGSESSION, errors='coerce').eq(3)].copy()
    return frame


def fixed_ruble(sample):
    """A cheap local gate; final corporate/rating/offer filters stay in daily.py."""
    return (sample.get('securities.BONDTYPE') == FIXED
            and sample.get('securities.FACEUNIT') in ('SUR', 'RUB')
            and sample.get('securities.CURRENCYID') in ('SUR', 'RUB'))


def day_universe(history, snapshot):
    """Keep one row per actual daily identity, with dated fields taking priority."""
    if history.empty:
        return []
    if history.duplicated(['SECID', 'BOARDID']).any():
        raise ValueError('Повтор SECID/BOARDID в полных дневных итогах')
    static = flatten_listing({'securities': snapshot.get('securities', {'columns': [], 'data': []})}) if snapshot else pd.DataFrame()
    indexed = {(r['SECID'], r['BOARDID']): r for r in static.to_dict('records')}
    samples = []
    for record in history.to_dict('records'):
        key = (record['SECID'], record['BOARDID'])
        sample = dict(indexed.get(key, {}))
        sample.update(SECID=key[0], BOARDID=key[1])
        for field, value in record.items():
            if field not in ('SECID', 'BOARDID'):
                sample['securities.' + field] = value
        samples.append(sample)
    return samples


def run(args):
    from moex_bonds.storage.history_storage import history_output
    history_output(args.output, getattr(args, 'history_output', None))
    with output_lock(args.output) as output:
        reset_daily_output(output)
        return collect_daily(args, output)


def collect_reference(client, sample, observations=None, freshness=None, *, strict=False):
    """Shared description/calendar loader; mutate one sample, return its events."""
    observations = observations if observations is not None else []
    freshness = freshness if freshness is not None else []
    secid = sample['SECID']
    events = []
    path = f'securities/{secid}.json'
    payload = client.get(path, purpose='instrument_description')
    description = tables(payload).get('description')
    sample['reference.description_status'] = 'ok' if description is not None and not description.empty else 'error'
    if strict and (description is None or description.empty):
        raise ValueError('Не получено описание выпуска для догрузки: ' + secid)
    observe(payload, secid, 'securities/{SECID}.json', observations, sample, freshness,
            client.records[-1]['received_at_msk'])
    path = f'securities/{secid}/bondization.json'
    payload = client.get(path, purpose='instrument_calendar')
    if strict and payload is None:
        raise ValueError('Не получен календарь выпуска для догрузки: ' + secid)
    if payload is not None:
        for section in ('coupons', 'amortizations', 'offers'):
            frame, state = client.collect(path, section, first=payload,
                start_key=section + '.start', purpose='calendar_pagination')
            sample[f'calendar.{section}.pagination'] = state
            sample[f'calendar.{section}.rows'] = len(frame)
            for row in frame.to_dict('records'):
                events.append(dict(probe_secid=secid, section=section, **row))
    return events


def collect_daily(args, output):
    client = MoexClient(output, pause=args.pause, timeout=args.timeout)
    now = datetime.now(MSK)
    manifest = dict(mode='day', started_at_msk=now.isoformat(), as_of_date=now.date().isoformat(),
                    args=vars(args), collection_status='incomplete', selected_count=0,
                    scope='Все дневные записи TQCB; итоговые фильтры применяются отдельно',
                    history_downloaded=False, adv20_basis='Не загружается в дневном режиме',
                    cbr_zcyc={'status': 'not_requested'}, cbr_ruonia={'status': 'not_requested'},
                    daily_export={'status': 'pending' if getattr(args, 'daily', False) else 'not_requested'},
                    storage_mode='overwrite', output_dir=str(output))
    # Invalidate the old success before the first network request. Readers
    # must check this run's status before opening its tables.
    (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    samples, quotes, events, observations, freshness = [], [], [], [], []
    result = 2
    cbr_pool, cbr_futures = None, {}
    try:
        day = latest_trade_date(client, getattr(args, 'date', None))
        manifest['trade_date'] = day
        print('MOEX: один торговый день', day, '(последние опубликованные итоги)', flush=True)
        hist = complete_day(client, DAY_HISTORY, 'history', day)
        if hist.empty:
            raise ValueError(f'Нет опубликованных дневных итогов за {day}; другая дата не подставлена')
        # Share the exact confirmed MOEX date. A separate session keeps the
        # CBR request independent while MOEX calendars/metadata are collected.
        cbr_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='cbr-daily')
        cbr_futures['cbr_zcyc'] = cbr_pool.submit(run_zcyc, date.fromisoformat(day), output, timeout=args.timeout)
        cbr_futures['cbr_ruonia'] = cbr_pool.submit(run_ruonia, date.fromisoformat(day), output, timeout=args.timeout)
        print('ЦБ: кривая за ту же дату', day, '(параллельная загрузка)', flush=True)
        print('ЦБ: недельная RUONIA по дату', day, '(параллельная загрузка)', flush=True)
        yields = complete_day(client, DAY_YIELDS, 'history_yields', day)
        snapshot = client.get(MARKET + '/boards/TQCB/securities.json', purpose='static_snapshot')
        samples = day_universe(hist, snapshot)
        requested = set(re.split(r'[,\s]+', args.secids.strip())) if args.secids else set()
        if requested:
            missing = requested - {r['SECID'] for r in samples}
            if missing:
                raise ValueError('Нет на выбранную дату: ' + ', '.join(sorted(missing)))
            samples = [r for r in samples if r['SECID'] in requested]
            manifest['scope'] = 'Явный список --secids на TQCB'
        chosen = {r['SECID'] for r in samples}
        for section, frame, endpoint in [('history', hist, DAY_HISTORY), ('history_yields', yields, DAY_YIELDS)]:
            for record in frame.to_dict('records'):
                if record['SECID'] not in chosen:
                    continue
                quotes.append(dict(probe_secid=record['SECID'], probe_board=record['BOARDID'],
                                   endpoint=endpoint, source_section=section, **record))
                observations.append(dict(secid=record['SECID'], source=endpoint, section=section, row=record))
        candidates = [r for r in samples if fixed_ruble(r)]
        print(f'Все выпуски за дату: {len(samples)}; фиксированные рублёвые кандидаты: {len(candidates)}', flush=True)
        for i, sample in enumerate(candidates):
            secid = sample['SECID']
            print(f'Справка и календари [{i+1}/{len(candidates)}] {secid}', flush=True)
            events.extend(collect_reference(client, sample, observations, freshness))
        manifest['collection_status'] = 'ok'
        result = 0
    except ValueError as error:
        manifest['error'] = str(error)
        print('Сбор не завершён:', error, file=sys.stderr, flush=True)
    finally:
        if cbr_pool is not None:
            try:
                for name, future in cbr_futures.items():
                    try:
                        manifest[name] = future.result()
                    except Exception as error:
                        manifest[name] = dict(status='error', trade_date=manifest['trade_date'],
                                              error=f'{type(error).__name__}: {error}')
                    status = manifest[name]
                    print(name + ':', status['status'], status.get('error', str(output / (name + '.csv'))), flush=True)
            finally:
                cbr_pool.shutdown(wait=True)
        manifest.update(finished_at_msk=datetime.now(MSK).isoformat(), selected_count=len(samples),
                        day_quote_rows=len(quotes), calendar_rows=len(events))
        export_csv(samples, output / 'bonds_universe.csv', ['SECID', 'BOARDID'])
        export_csv(quotes, output / 'day_quotes.csv', ['SECID', 'BOARDID', 'TRADEDATE', 'endpoint', 'source_section'])
        export_csv(events, output / 'bond_events.csv', ['probe_secid', 'section'])
        fields = make_availability(observations, len(samples), [])
        export_csv(fields, output / 'fields_availability.csv')
        client.save()
        (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        write_day_report(output, manifest, client)
    print('Дневная выгрузка:', output, flush=True)
    if result != 0:
        return result
    exported = finish_daily(args, output)
    return exported or (0 if all(manifest[name]['status'] == 'ok' for name in ('cbr_zcyc', 'cbr_ruonia')) else 4)


def finish_daily(args, output):
    """The audit stays saved if the separate daily enrichment/export fails."""
    if not getattr(args, 'daily', False):
        return 0
    path = output / 'run_manifest.json'
    manifest = json.loads(path.read_text(encoding='utf-8'))
    result = 0
    try:
        from moex_bonds.collection.daily import run_daily
        as_of = datetime.fromisoformat(manifest['as_of_date']).date() if manifest.get('as_of_date') else None
        daily = run_daily(output, as_of=as_of, pause=max(args.pause, .5), timeout=args.timeout,
                          xlsx=not getattr(args, 'csv_only', False),
                          history_dir=getattr(args, 'history_output', None))
        manifest['daily_export'] = dict(status='ok', **daily['metadata'])
        manifest['history_downloaded'] = bool(daily['metadata'].get('history_update', {}).get('recovered_dates'))
        if daily['metadata'].get('history_update', {}).get('status') == 'error':
            result = 5
        print('FINAL DAILY TABLE:', output / 'daily' / ('bonds_daily.csv' if getattr(args, 'csv_only', False) else 'bonds_daily.xlsx'), flush=True)
    except Exception as error:
        manifest['daily_export'] = dict(status='error', error=f'{type(error).__name__}: {error}')
        print('DAILY EXPORT FAILED (audit files saved):', error, file=sys.stderr, flush=True)
        result = 3
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    return result


def write_day_report(output, manifest, client):
    lines = ['# Выгрузка за один торговый день', '',
             'Дата данных: ' + str(manifest.get('trade_date', 'не определена')),
             'Статус сбора: ' + manifest['collection_status'],
             'Хранение: постоянная папка; предыдущие дневные файлы очищаются перед сбором.',
             'Режим: TQCB. ' + manifest['scope'],
             'Выпусков: ' + str(manifest['selected_count']),
             'Дневных строк history + history_yields: ' + str(manifest['day_quote_rows']), '',
             'Лимита 20 выпусков нет. Все страницы дневных итогов загружаются по одной общей дате.',
             'Дату по умолчанию определяет MOEX dates.till для полных итогов (tradingsession=3).',
             'Если today ещё не опубликован, дата файла будет предыдущей опубликованной датой.',
             'При --date без данных дата не заменяется другой.', '',
             'Кривая ЦБ за ту же дату: ' + manifest['cbr_zcyc']['status'],
             'Отдельная таблица: cbr_zcyc.csv (дата, срок в годах, % годовых).',
             'Исходный HTML и диагностика ЦБ: _support/cbr_zcyc*.',
             'При отсутствии/ошибке ЦБ другая дата не подставляется; MOEX и фильтрация продолжаются.', '',
             'Недельная RUONIA: ' + manifest['cbr_ruonia']['status'],
             'Отдельная таблица: cbr_ruonia.csv; интервал — дата торгов минус 7 дней по дату торгов включительно.',
             'Выходные не заполняются. Дата ставки и дата публикации сохранены отдельно.',
             'Исходный HTML и диагностика: _support/cbr_ruonia*.', '',
             'Дневные таблицы содержат только эту дату; пропуски общей базы догружаются в data/history_data.',
             'Первоначальное окно 180 дат собирает history.py; без инициализации базы скрытого полного сбора нет.',
             'Справка эмитента и календари — текущие метаданные; рыночные показатели — только за дату данных.',
             'Итоговый отбор и Smart-Lab остаются в daily.py; исходная совокупность — bonds_universe.csv.',
             'Охват TQCB не означает все режимы MOEX. Неликвидные строки с нулём сделок сохраняются.', '',
             'HTTP-попыток: ' + str(len(client.records))]
    if manifest.get('error'):
        lines += ['', 'Ошибка: ' + manifest['error']]
    if manifest['cbr_zcyc'].get('error'):
        lines += ['', 'ЦБ: ' + manifest['cbr_zcyc']['error']]
    if manifest['cbr_ruonia'].get('error'):
        lines += ['', 'RUONIA: ' + manifest['cbr_ruonia']['error']]
    (output / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args():
    from datetime import date
    parser = argparse.ArgumentParser(description='Все облигации TQCB за один опубликованный торговый день')
    parser.add_argument('--date', type=date.fromisoformat, help='Дата YYYY-MM-DD; иначе последняя опубликованная MOEX')
    parser.add_argument('--secids', default='', help='Необязательный явный список SECID через запятую/пробел')
    parser.add_argument('--output', default=str(DEFAULT_DAILY_OUTPUT),
                        help='Постоянная папка; дневные данные перезаписываются. По умолчанию data/daily_data')
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--history-output', help='Папка исторической базы для добавления даты; по умолчанию data/history_data')
    parser.add_argument('--pause', type=float, default=.25)
    parser.add_argument('--no-daily', dest='daily', action='store_false', default=True,
                        help='Сохранить MOEX, кривую ЦБ и RUONIA без итоговой таблицы и Smart-Lab')
    parser.add_argument('--csv-only', action='store_true', help='Итоговый CSV без генератора Excel')
    publishing = parser.add_mutually_exclusive_group()
    publishing.add_argument("--publish", action="store_true", help="Collect, analyze and publish to GitHub")
    publishing.add_argument("--publish-retry", action="store_true", help="Retry saved Git delivery only")
    args = parser.parse_args()
    if args.timeout <= 0 or args.pause < 0:
        parser.error('timeout > 0, pause >= 0')
    if args.date and args.date > datetime.now(MSK).date():
        parser.error('--date не может быть в будущем')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.publish or args.publish_retry:
        from moex_bonds.publication.workflow import run as publish_run
        raise SystemExit(publish_run(args))
    raise SystemExit(run(args))
