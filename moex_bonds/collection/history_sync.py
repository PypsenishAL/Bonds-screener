"""Recover skipped board dates without publishing any past-day daily tables."""
from datetime import date, timedelta
import json
from pathlib import Path
import shutil

import pandas as pd

from moex_bonds.collection.daily import (build_daily, filter_reasons, format_daily_row, sample_value,
                   number, HISTORY_ENDPOINT, YIELDS_ENDPOINT)
from moex_bonds.storage.daily_storage import checked_child
from moex_bonds.collection.history import RequestPool
from moex_bonds.collection.main import complete_day, collect_reference, day_universe, fixed_ruble, DAY_HISTORY, DAY_YIELDS
from moex_bonds.sources.smartlab import SmartLabClient


def reusable_reference(sample):
    complete = ('empty_terminal', 'cursor_complete')
    return (sample.get('reference.description_status') != 'error'
            and sample.get('calendar.offers.pagination') in complete
            and number(sample.get('calendar.offers.rows')) is not None
            and sample.get('calendar.coupons.pagination') in complete)


def fetch_day(client, day):
    hist = complete_day(client, DAY_HISTORY, 'history', day)
    yields = complete_day(client, DAY_YIELDS, 'history_yields', day)
    if any(not {'SECID', 'BOARDID', 'TRADEDATE'}.issubset(table) for table in (hist, yields)):
        raise ValueError(f'Неподтверждённая дневная схема TQCB за {day}')
    if not yields.empty and not {'YIELDDATETYPE', 'YIELDDATE'}.issubset(yields):
        raise ValueError(f'Не подтверждена схема доходности TQCB за {day}')
    if hist.empty and not yields.empty:
        raise ValueError(f'Доходности без дневных итогов TQCB за {day}')
    return day, hist, yields


def recover_days(output, dataset, frame, metadata, context, pause, timeout, workers):
    """Caller holds the base lock. Never publish a partial date/checkpoint."""
    day = dataset['metadata']['trade_date']
    calendar = sorted(metadata.get('trading_dates', []))
    actual_last = None if frame.empty else frame.TRADEDATE.max()
    if actual_last and actual_last not in calendar:
        raise ValueError('Последняя фактическая дата CSV не подтверждена календарём базы')
    # Old versions may have appended later dates while skipping days in between.
    # The original rebuild's selection date is the last proven contiguous point.
    anchor = metadata.get('synchronized_through') or metadata.get('selection_date') or max(calendar)
    if anchor not in calendar:
        raise ValueError('Дата полной синхронизации отсутствует в календаре базы')
    start, end = date.fromisoformat(anchor), date.fromisoformat(day)
    missing = [(start + timedelta(days=i)).isoformat() for i in range(1, (end-start).days)]
    result = dict(recovered_dates=[], synchronized_through=max(anchor, day),
                  previous_last_row_date=actual_last, sync_from=anchor)
    subset = dataset['metadata'].get('replace_secids')
    if subset is not None:
        result['synchronized_through'] = anchor
    # The latest table has already applied these references. If a transient
    # failure excluded an issue there, do not certify that date or skip its retry.
    for sample in context['samples']:
        if not fixed_ruble(sample):
            continue
        kind, isin = sample_value(sample, 'TYPE'), sample_value(sample, 'ISIN')
        if kind and kind not in ('corporate_bond', 'exchange_bond'):
            continue
        if isinstance(isin, str) and not isin.startswith('RU'):
            continue
        if not reusable_reference(sample) or context['smartlab'].get(sample['SECID'], {}).get('status') != 'ok':
            raise ValueError('Неполная текущая справка; повторите дневной сбор: ' + sample['SECID'])
    if not missing:
        return [], [], result

    support = checked_child(output, '_support')
    support.mkdir(exist_ok=True)
    audit = checked_child(support, 'daily_sync')
    if audit.exists():
        if not audit.is_dir():
            raise ValueError('На месте аудита синхронизации обнаружен файл')
        shutil.rmtree(checked_child(support, 'daily_sync'))
    audit.mkdir()
    pool = RequestPool(audit / 'moex', workers, pause, timeout)
    try:
        print(f'История: проверка пропусков {missing[0]} — {missing[-1]}', flush=True)
        # Cache only current/reference fields, never use them as dated quotes.
        refs, events, smart = {}, {}, {}
        cache = checked_child(support, 'sync_reference.json')
        for path in (support / 'selection.json', cache):
            if path.exists():
                saved = json.loads(path.read_text(encoding='utf-8'))
                refs.update({(s['SECID'], s['BOARDID']): s for s in saved.get('samples', [])})
                smart.update(saved.get('smartlab', {}))
                for event in saved.get('events', []):
                    events.setdefault(event['probe_secid'], []).append(event)
        current_ids = {s['SECID'] for s in context['samples']}
        refs.update({(s['SECID'], s['BOARDID']): s for s in context['samples']})
        smart.update(context['smartlab'])
        for secid in current_ids:
            events[secid] = []
        for event in context['events']:
            events.setdefault(event['probe_secid'], []).append(event)
        smart_client = SmartLabClient(audit, pause=pause, timeout=timeout)
        def load_reference(client, sample):
            sample = dict(sample)
            ev = collect_reference(client, sample, strict=True)
            if (sample.get('calendar.offers.pagination') not in ('empty_terminal', 'cursor_complete')
                    or sample.get('calendar.coupons.pagination') not in ('empty_terminal', 'cursor_complete')):
                raise ValueError('Неполный календарь выпуска ' + sample['SECID'])
            return sample, ev

        first_known = frame.groupby(['SECID', 'BOARDID']).TRADEDATE.min().to_dict()
        observed_in_gap = set()
        rows, reports = [], []
        # Bound full-market tables in memory regardless of the length of the gap.
        for offset in range(0, len(missing), workers):
            blocks = pool.map(fetch_day, missing[offset:offset+workers])
            recovered = [(d, h, y) for d, h, y in blocks if not h.empty]
            result['recovered_dates'].extend(d for d, _, _ in recovered)
            if not recovered:
                continue
            samples_by_day, unresolved = {}, {}
            for d, hist, _ in recovered:
                samples = []
                for dated in day_universe(hist, None):
                    if subset is not None and dated['SECID'] not in subset:
                        continue
                    key = dated['SECID'], dated['BOARDID']
                    sample = dict(refs.get(key, {}), **dated)
                    samples.append(dated)
                    if fixed_ruble(sample) and (key not in refs or sample['SECID'] not in events
                                                or not reusable_reference(sample)):
                        unresolved[key] = sample
                samples_by_day[d] = samples

            for sample, ev in pool.map(load_reference, unresolved.values()):
                refs[sample['SECID'], sample['BOARDID']] = sample
                events[sample['SECID']] = ev
            needed = {s['SECID'] for samples in samples_by_day.values() for s in samples
                      if fixed_ruble(s) and str(sample_value(refs.get((s['SECID'], s['BOARDID']), s), 'ISIN') or '').startswith('RU')
                      and sample_value(refs.get((s['SECID'], s['BOARDID']), s), 'TYPE') in ('corporate_bond', 'exchange_bond')}
            for secid in sorted(needed):
                if smart.get(secid, {}).get('status') != 'ok':
                    smart[secid] = smart_client.fetch(secid)
                if smart[secid].get('status') != 'ok':
                    raise ValueError('Не получена справка для пропущенных дат: ' + secid)
            all_events = [e for ev in events.values() for e in ev]
            # Already eligible identities keep explicit missing-record rows while
            # their current reference fields still pass the common eligibility gate.
            for d, hist, yields in recovered:
                samples = [dict(refs.get((s['SECID'], s['BOARDID']), {}), **s) for s in samples_by_day[d]]
                quotes = [dict(r, endpoint=endpoint) for table, endpoint in
                          ((hist, HISTORY_ENDPOINT), (yields, YIELDS_ENDPOINT)) for r in table.to_dict('records')]
                dated_rows, excluded = build_daily(samples, quotes, all_events, smart, date.fromisoformat(d))
                present = set(zip(hist.SECID, hist.BOARDID))
                known = {key for key, first in first_known.items() if first <= d} | observed_in_gap
                for key in sorted(known - present):
                    if subset is not None and key[0] not in subset:
                        continue
                    sample = refs.get(key)
                    if not sample:
                        continue
                    reference = {field: sample_value(sample, field) for field in ('MATDATE', 'FACEUNIT', 'CURRENCYID', 'BONDTYPE')}
                    if filter_reasons(sample, reference, smart.get(key[0], {}), events.get(key[0], []), date.fromisoformat(d)):
                        continue
                    dated_rows.append(format_daily_row(sample, {}, pd.DataFrame(), events.get(key[0], []),
                                                       smart[key[0]], date.fromisoformat(d)))
                observed_in_gap.update((r['SECID'], r['BOARDID']) for r in dated_rows)
                rows.extend(dated_rows)
                reports.append(dict(trade_date=d, included=len(dated_rows), excluded=len(excluded)))
            print(f"История: обработано {min(offset+workers, len(missing))}/{len(missing)} календарных дат", flush=True)
        all_events = [e for ev in events.values() for e in ev]
        (audit / 'coverage.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
        cache.write_text(json.dumps(dict(samples=list(refs.values()), events=all_events, smartlab=smart),
                                    ensure_ascii=False, default=str), encoding='utf-8')
        return rows, result['recovered_dates'], result
    finally:
        pool.close()
