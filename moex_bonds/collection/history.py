"""Rebuild the current daily screen over 180 published TQCB trading dates."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
import json
from pathlib import Path
import re
import threading

import pandas as pd

from moex_bonds.collection.daily import (build_daily, format_daily_row, unique_history, iso,
                   HISTORY_ENDPOINT, YIELDS_ENDPOINT)
from moex_bonds.storage.daily_storage import DEFAULT_DAILY_OUTPUT, output_lock
from moex_bonds.storage.history_storage import history_output, reset_history, publish_history, write_manifest
from moex_bonds.sources.moex_client import MoexClient, tables

BOARD_HISTORY = 'history/engines/stock/markets/bonds/boards/TQCB/securities.json'
COMPLETE = ('empty_terminal', 'cursor_complete')


class RequestPool:
    """One requests.Session/audit stream per thread, reused across all tasks."""
    def __init__(self, output, workers, pause, timeout):
        self.output, self.pause, self.timeout = Path(output), pause, timeout
        self.clients, self.local, self.lock = [], threading.local(), threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='bond-history')

    def client(self):
        if not hasattr(self.local, 'client'):
            with self.lock:
                client = MoexClient(self.output / f'worker_{len(self.clients)+1}', self.pause, self.timeout)
                self.clients.append(client)
                self.local.client = client
        return self.local.client

    def map(self, function, items):
        return list(self.executor.map(lambda item: function(self.client(), item), items))

    def close(self):
        self.executor.shutdown(wait=True)
        for client in self.clients:
            try:
                client.save()
            finally:
                client.session.close()


def published_day(client, day):
    payload = client.get(BOARD_HISTORY,
        {'date': day, 'tradingsession': 3, 'limit': 1, 'start': 0,
         'iss.only': 'history', 'history.columns': 'TRADEDATE,BOARDID,TRADINGSESSION'},
        purpose='history_calendar')
    frame = tables(payload).get('history')
    if frame is None or not {'TRADEDATE', 'BOARDID'}.issubset(frame):
        raise ValueError(f'Не подтверждён календарь TQCB за {day}: нет ответа/дневной схемы')
    if not frame.empty:
        if not frame.TRADEDATE.eq(day).all() or not frame.BOARDID.eq('TQCB').all():
            raise ValueError('Календарь вернул другую дату или режим')
        if 'TRADINGSESSION' in frame and not pd.to_numeric(frame.TRADINGSESSION, errors='coerce').eq(3).all():
            raise ValueError('Календарь вернул другую торговую сессию')
    return not frame.empty


def trading_dates(client, end, count):
    """Check actual board records, including exceptional weekend sessions.

    A one-row existence probe intentionally does not paginate the whole market.
    An empty valid block is a non-trading date; any request/schema error aborts.
    """
    dates, offset = [], 0
    while len(dates) < count:
        if offset > count * 5 + 366:
            raise ValueError('Не удалось подтвердить требуемое число торговых дат')
        size = min(32, count - len(dates))
        days = [(end - timedelta(days=n)).isoformat() for n in range(offset, offset + size)]
        flags = (client.map(published_day, days) if isinstance(client, RequestPool)
                 else [published_day(client, day) for day in days])
        if offset == 0 and not flags[0]:
            raise ValueError('Дата сохранённого дневного запуска не подтверждена как торговая')
        dates.extend(day for day, exists in zip(days, flags) if exists)
        offset += size
        print(f'Календарь TQCB: {len(dates)}/{count} торговых дат', flush=True)
    return sorted(dates[:count])


def build_history(samples, histories, events, smartlab, dates):
    """Format only observed lifetimes; do not backfill pre-observation dates."""
    frame = pd.DataFrame(histories)
    hist = unique_history(frame, HISTORY_ENDPOINT)
    yields = unique_history(frame, YIELDS_ENDPOINT)
    history_groups = ({key: group for key, group in hist.groupby(['SECID', 'BOARDID'])}
                      if not hist.empty else {})
    yield_groups = ({key: group for key, group in yields.groupby(['SECID', 'BOARDID', 'TRADEDATE'])}
                    if not yields.empty else {})
    events_by_id = {}
    for event in events:
        events_by_id.setdefault(event.get('probe_secid'), []).append(event)
    rows, coverage = [], []
    for sample in samples:
        secid, board = sample['SECID'], sample['BOARDID']
        group = history_groups.get((secid, board), pd.DataFrame())
        indexed = {r['TRADEDATE']: r for r in group.to_dict('records')}
        first = min(indexed) if indexed else None
        prior = []
        for day in dates:
            if not first or day < first:
                continue
            h = indexed.get(day, {})
            prior.append(h or dict(TRADEDATE=day, VALUE=None))
            ys = yield_groups.get((secid, board, day), pd.DataFrame())
            if not ys.empty:
                ys = ys[ys.YIELDDATETYPE.eq('MATDATE') & ys.YIELDDATE.eq(iso(h.get('MATDATE')))]
            if len(ys) > 1:
                raise ValueError(f'Неоднозначная доходность {secid} за {day}')
            row = format_daily_row(sample, h, ys, events_by_id.get(secid, []),
                                   smartlab.get(secid, {}), date.fromisoformat(day), prior[-20:])
            rows.append(row)
        coverage.append(dict(SECID=secid, BOARDID=board, first_observation=first,
                             observed_days=len(indexed), expected_days=sum(d >= first for d in dates) if first else 0,
                             missing_days=sum(d not in indexed for d in dates if first and d >= first),
                             lifetime_basis='первая подтверждённая запись TQCB в окне; более ранние даты не создаются'))
    return rows, coverage


def collect_instrument(client, task):
    sample, start, end, dates = task
    secid, board = sample['SECID'], sample['BOARDID']
    rows = []
    for template in (HISTORY_ENDPOINT, YIELDS_ENDPOINT):
        section = 'history_yields' if template == YIELDS_ENDPOINT else 'history'
        params = {'from': start, 'till': end, 'limit': 100}
        if section == 'history':
            params['tradingsession'] = 3
        frame, state = client.collect(template.format(SECID=secid, BOARDID=board), section,
                                      params, purpose='requested_history')
        if state not in COMPLETE:
            raise ValueError(f'Неполная история {secid} {section}: {state}')
        if not {'SECID', 'BOARDID', 'TRADEDATE'}.issubset(frame):
            raise ValueError(f'История {secid} {section} не содержит дневные ключи')
        if section == 'history_yields' and not {'YIELDDATETYPE', 'YIELDDATE'}.issubset(frame):
            raise ValueError('История доходности не содержит тип и дату расчёта')
        if (not frame.SECID.eq(secid).all() or not frame.BOARDID.eq(board).all()
                or not frame.TRADEDATE.isin(dates).all()):
            raise ValueError('История вернула другой SECID, BOARDID или неподтверждённую дату')
        if section == 'history':
            if 'TRADINGSESSION' in frame:
                frame = frame[pd.to_numeric(frame.TRADINGSESSION, errors='coerce').eq(3)]
            if not frame.TRADEDATE.eq(end).any():
                raise ValueError(f'Нет известной записи дневного запуска {secid} за {end}')
        for record in frame.to_dict('records'):
            rows.append(dict(record, endpoint=template, source_section=section))
    print(f'История {secid}: {len(rows)} исходных строк', flush=True)
    return rows


def run_history(args):
    source = Path(args.input).resolve()
    output = history_output(source, args.output)
    with output_lock(source), output_lock(output):
        manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('mode') != 'day' or manifest.get('collection_status') != 'ok'
                or not manifest.get('trade_date')):
            raise ValueError('--input должен указывать на завершённый дневной запуск main.py')
        end = date.fromisoformat(manifest['trade_date'])
        samples = pd.read_csv(source / 'bonds_universe.csv').to_dict('records')
        events = pd.read_csv(source / 'bond_events.csv').to_dict('records')
        quotes = pd.read_csv(source / 'day_quotes.csv').to_dict('records')
        smart_path = source / 'daily/_support/smartlab_enrichment.json'
        if not smart_path.exists():
            raise ValueError('Нет справки итогового отбора: сначала main.py или daily.py')
        smart = json.loads(smart_path.read_text(encoding='utf-8'))
        selected, excluded = build_daily(samples, quotes, events, smart, end)
        keys = {(r['SECID'], r['BOARDID']) for r in selected}
        samples = [s for s in samples if (s['SECID'], s['BOARDID']) in keys]
        wanted = set(re.split(r'[,\s]+', args.secids.strip())) if args.secids else set()
        if wanted - {s['SECID'] for s in samples}:
            raise ValueError('--secids содержит бумаги, не прошедшие итоговый дневной отбор')
        if wanted:
            samples = [s for s in samples if s['SECID'] in wanted]
        if not samples:
            raise ValueError('Нет бумаг, прошедших итоговые фильтры ежедневного скрипта')
        reset_history(output)
        info = dict(mode='history', source_run=str(source), selection_date=end.isoformat(),
                    history_days=args.history_days, window_unit='trading_days', instruments=len(samples),
                    status='incomplete', rating_basis='Текущая справка Smart-Lab; не исторический рейтинг',
                    universe_basis='Бумаги текущего дневного отбора; не исторический состав рынка',
                    storage_mode='overwrite', excluded_count=len(excluded))
        pool = RequestPool(output / '_support/moex', getattr(args, 'workers', 4), args.pause, args.timeout)
        try:
            dates = trading_dates(pool, end, args.history_days)
            info.update(from_date=dates[0], till=dates[-1], trading_dates=dates)
            info['synchronized_through'] = dates[-1]
            chunks = pool.map(collect_instrument, [(s, dates[0], dates[-1], set(dates)) for s in samples])
            raw = [r for chunk in chunks for r in chunk]
            pd.DataFrame(raw).to_csv(output / '_support/historical_sample.csv', index=False, encoding='utf-8-sig')
            rows, coverage = build_history(samples, raw, events, smart, dates)
            pd.DataFrame(coverage).to_csv(output / 'historical_coverage.csv', index=False, encoding='utf-8-sig')
            (output / '_support/selection.json').write_text(json.dumps(dict(samples=samples, smartlab=smart),
                ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            info['status'] = 'ok'
            publish_history(output, rows, info)
        except Exception as error:
            info.update(status='incomplete', error=f'{type(error).__name__}: {error}')
            write_manifest(output, info)
            print('История не завершена:', error, flush=True)
            return 2
        finally:
            pool.close()
        print(f'Историческая база: {output}; {len(rows)} строк', flush=True)
        return 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=str(DEFAULT_DAILY_OUTPUT), help='Папка завершённого дневного запуска')
    parser.add_argument('--history-days', type=int, default=180, help='Число опубликованных торговых дат, по умолчанию 180')
    parser.add_argument('--secids', default='', help='Необязательный поднабор итогового дневного отбора')
    parser.add_argument('--output', help='Постоянная отдельная папка, по умолчанию data/history_data')
    parser.add_argument('--workers', type=int, default=4, help='Параллельные соединения, 1..16')
    parser.add_argument('--pause', type=float, default=.25, help='Пауза между запросами одного соединения')
    parser.add_argument('--timeout', type=float, default=15)
    args = parser.parse_args()
    if not 1 <= args.history_days <= 10000 or not 1 <= args.workers <= 16 or args.pause < 0 or args.timeout <= 0:
        parser.error('history-days: 1..10000, workers: 1..16, pause >= 0, timeout > 0')
    return args


if __name__ == '__main__':
    raise SystemExit(run_history(parse_args()))
