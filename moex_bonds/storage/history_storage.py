"""Persistent bond table, separate from replaceable daily outputs."""
import json
from pathlib import Path
import shutil

import pandas as pd

from moex_bonds.storage.daily_storage import PROJECT_DIR, checked_output, checked_child, output_lock

from moex_bonds.paths import HISTORY_DIR

DEFAULT_HISTORY_OUTPUT = HISTORY_DIR
KEYS = ['SECID', 'BOARDID', 'TRADEDATE']


class HistoryReader:
    """Read-only projected batch queries; cached supersets serve repeated requests.

    This reader never calls collectors, append_daily or publish_history. It scans
    CSV chunks once per new projection/range, rather than once per instrument.
    Calendar discovery rides on the same scan, including non-trading issue rows.
    """
    def __init__(self, path, identities=None):
        self.path = Path(path).resolve()
        self.identities = identities
        self.columns = set(pd.read_csv(self.path, nrows=0).columns)
        if not set(KEYS).issubset(self.columns):
            raise ValueError('В истории отсутствует ключ SECID/BOARDID/TRADEDATE')
        self.read_passes, self.rows_scanned, self.queries = 0, 0, []
        self.cache, self.observed_dates = [], set()
        self.calendar_through = None
        self.manifest = {}
        manifest = self.path.parent / 'run_manifest.json'
        if manifest.exists():
            self.manifest = json.loads(manifest.read_text(encoding='utf-8'))
            if self.manifest.get('mode') == 'history' and self.manifest.get('status') != 'ok':
                raise ValueError('Историческая база не завершена')
        self.calendar_source = ('history_manifest' if self.manifest.get('trading_dates')
                                else 'observed_board_dates_in_csv')

    def query(self, secids, start, end, fields):
        wanted, fields = set(secids), set(fields) | set(KEYS)
        if not end or (start is not None and start > end):
            raise ValueError('Неверный диапазон дат истории')
        missing = fields - self.columns - {'ADV20_RUB'}
        if missing:
            raise ValueError('В истории отсутствуют поля: ' + ', '.join(sorted(missing)))
        self.queries.append(dict(secids=sorted(wanted), start=start, end=end, fields=sorted(fields)))
        for cached in self.cache:
            if (wanted <= cached['secids'] and fields <= cached['fields']
                    and (cached['start'] is None or (start is not None and cached['start'] <= start))
                    and cached['end'] >= end):
                frame = cached['frame']
                mask = frame.SECID.isin(wanted) & frame.TRADEDATE.le(end)
                if start:
                    mask &= frame.TRADEDATE.ge(start)
                return frame.loc[mask, sorted(fields)].copy()
        parts = []
        self.read_passes += 1
        usecols = sorted(fields & self.columns)
        for chunk in pd.read_csv(self.path, usecols=usecols, chunksize=50000,
                                 dtype={key: str for key in KEYS}):
            self.rows_scanned += len(chunk)
            parsed = pd.to_datetime(chunk.TRADEDATE, format='%Y-%m-%d', errors='coerce')
            if parsed.isna().any():
                raise ValueError('Неверная дата в истории')
            if self.identities:
                board_mask = chunk.BOARDID.isin(set(self.identities.values()))
            else:
                board_mask = pd.Series(True, index=chunk.index)
            self.observed_dates.update(chunk.loc[board_mask & chunk.TRADEDATE.le(end), 'TRADEDATE'])
            mask = chunk.SECID.isin(wanted) & chunk.TRADEDATE.le(end)
            if self.identities:
                mask &= chunk.BOARDID.eq(chunk.SECID.map(self.identities))
            if start:
                mask &= chunk.TRADEDATE.ge(start)
            parts.append(chunk.loc[mask])
        self.calendar_through = max(self.calendar_through or end, end)
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)
        for field in fields - set(frame):
            frame[field] = None
        if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
            raise ValueError('Пустой или повторный ключ SECID/BOARDID/TRADEDATE в истории')
        self.cache.append(dict(secids=wanted, start=start, end=end, fields=fields, frame=frame))
        return frame[sorted(fields)].copy()

    def calendar_dates(self, end):
        dates = self.manifest.get('trading_dates')
        if dates:
            parsed = pd.to_datetime(dates, format='%Y-%m-%d', errors='coerce')
            if parsed.isna().any() or len(set(dates)) != len(dates):
                raise ValueError('Некорректный календарь истории')
            return sorted(d for d in dates if d <= end)
        if not self.calendar_through or self.calendar_through < end:
            self.query([], None, end, [])
        return sorted(d for d in self.observed_dates if d <= end)


def history_output(source, output=None):
    source = Path(source).resolve()
    target = checked_output(output or DEFAULT_HISTORY_OUTPUT)
    if target == source or target in source.parents or source in target.parents:
        raise ValueError('История и дневная выгрузка должны быть в отдельных, не вложенных папках')
    return target


def reset_history(output):
    """Clear only owned outputs after validating every recursive target."""
    base = checked_output(output)
    files = [checked_child(base, name) for name in ('bonds_history.csv', 'historical_coverage.csv', 'run_manifest.json')]
    support = checked_child(base, '_support')
    if any(p.exists() and not p.is_file() for p in files) or (support.exists() and not support.is_dir()):
        raise ValueError('Неверный тип файла/каталога истории; очистка отменена')
    write_manifest(base, dict(mode='history', status='incomplete', storage_mode='overwrite'))
    for path in files:
        if path.name != 'run_manifest.json':
            path.unlink(missing_ok=True)
    if support.exists():
        shutil.rmtree(checked_child(base, '_support'))
    support.mkdir()


def write_manifest(output, metadata):
    path = checked_child(Path(output).resolve(), 'run_manifest.json')
    temp = checked_child(Path(output).resolve(), 'run_manifest.json.tmp')
    try:
        temp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def publish_history(output, rows, metadata):
    """Caller owns the directory lock. Stage the complete CSV before replacing it."""
    from moex_bonds.collection.daily import COLUMNS
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = rows.copy() if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    columns = [c[0] for c in COLUMNS]
    frame = frame.reindex(columns=columns)
    if frame[KEYS].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Пустой или повторный ключ SECID/BOARDID/TRADEDATE в истории')
    frame = frame.sort_values(['TRADEDATE', 'SECID', 'BOARDID'])
    target = checked_child(output, 'bonds_history.csv')
    temp = checked_child(output, 'bonds_history.csv.tmp')
    try:
        frame.to_csv(temp, index=False, encoding='utf-8-sig')
        write_manifest(output, dict(metadata, status='publishing'))
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)
    metadata = dict(metadata, mode='history', rows=len(frame), key=KEYS)
    write_manifest(output, metadata)
    return metadata


def append_daily(output, dataset, *, sync_context=None, pause=.25, timeout=15, workers=4):
    """Replace a date, or only explicitly requested identities for subset runs."""
    output = checked_output(output)
    manifest_path = output / 'run_manifest.json'
    if not manifest_path.exists():
        return dict(status='not_initialized', output=str(output))
    with output_lock(output):
        metadata = json.loads(manifest_path.read_text(encoding='utf-8'))
        if metadata.get('mode') != 'history' or metadata.get('status') != 'ok':
            raise ValueError('База истории не завершена; повторите history.py')
        day = dataset['metadata']['trade_date']
        dates = metadata.get('trading_dates', [])
        if not dates or day < min(dates):
            raise ValueError('Дневная дата раньше начала базы; повторите history.py с нужным окном')
        from moex_bonds.collection.daily import COLUMNS
        frame = pd.read_csv(output / 'bonds_history.csv',
                            dtype={key: str for key, _, kind in COLUMNS if kind in ('text', 'date')})
        rows = list(dataset['rows'])
        if any(r.get('TRADEDATE') != day for r in rows):
            raise ValueError('Дата дневных строк не совпадает с общей датой')
        subset = dataset['metadata'].get('replace_secids')
        dates_added, sync_info = [day], {}
        if sync_context is not None:
            from moex_bonds.collection.history_sync import recover_days
            recovered, recovered_dates, sync_info = recover_days(output, dataset, frame, metadata,
                                                                 sync_context, pause, timeout, workers)
            rows = recovered + rows
            dates_added = recovered_dates + [day]
        remove = frame.TRADEDATE.isin(dates_added)
        if subset is not None:
            if not subset or any(r.get('SECID') not in subset for r in rows):
                raise ValueError('Некорректный явный подсписок дневного обновления')
            remove &= frame.SECID.isin(subset)
        merged = pd.concat([frame[~remove], pd.DataFrame(rows)], ignore_index=True)
        metadata.update(status='ok', trading_dates=sorted(set(dates + dates_added)),
                        till=max(dates + dates_added), last_daily_date=day, **sync_info,
                        daily_membership='каждая добавленная дата — собственный текущий отбор')
        publish_history(output, merged, metadata)
        return dict(status='ok', output=str(output), trade_date=day, rows_added=len(rows), **sync_info)
