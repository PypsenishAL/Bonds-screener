"""Download the published RUONIA week ending on the MOEX trading date.

python -m moex_bonds.sources.cbr_ruonia --input data/daily_data
python -m moex_bonds.sources.cbr_ruonia --date 2026-10-26
"""
import argparse
from datetime import date, datetime, timedelta
import json
import math
from pathlib import Path
import re
import time

from bs4 import BeautifulSoup
import pandas as pd
import requests

from moex_bonds.sources.moex_client import MSK
from moex_bonds.storage.daily_storage import DEFAULT_DAILY_OUTPUT, matching_daily_manifest, output_lock

URL = 'https://www.cbr.ru/hd_base/ruonia/dynamics/'
# Stable pandas/Streamlit keys; the original CBR labels and units travel in status.
SOURCE_COLUMNS = [
    ('rate_date', 'Дата ставки', 'date', None),
    ('ruonia_pct', 'Ставка RUONIA, % годовых', 'number', '% годовых'),
    ('volume_bln_rub', 'Объем сделок RUONIA, млрд руб.', 'number', 'млрд руб.'),
    ('trades_count', 'Количество сделок, ед.', 'integer', 'ед.'),
    ('participants_count', 'Количество участников RUONIA, совершавших сделки в данный день, ед.', 'integer', 'ед.'),
    ('min_rate_pct', 'Минимальная процентная ставка, % годовых', 'number', '% годовых'),
    ('p25_rate_pct', '25-й процентиль по процентным ставкам, % годовых', 'number', '% годовых'),
    ('p75_rate_pct', '75-й процентиль по процентным ставкам, % годовых', 'number', '% годовых'),
    ('max_rate_pct', 'Максимальная процентная ставка, % годовых', 'number', '% годовых'),
    ('calculation_status', 'Статус расчета', 'text', None),
    ('publication_date', 'Дата публикации', 'date', None),
]
COLUMNS = [('trade_date', 'Дата торгов MOEX', 'date', None)] + SOURCE_COLUMNS + [
    ('data_gaps', 'Пропуски в источнике (имена полей)', 'text', None)]
MISSING = {'', '-', '—', '–', 'н/д', 'n/a'}


class NoRuoniaForPeriod(ValueError):
    """The exact requested week has no published observations."""


def clean(text):
    return re.sub(r'\s+', ' ', text).strip()


def normalized(text):
    return clean(text).lower().replace('ё', 'е')


def parse_value(text, kind):
    text = clean(text)
    if text.lower() in MISSING:
        return None
    if kind == 'date':
        return datetime.strptime(text, '%d.%m.%Y').date().isoformat()
    if kind == 'text':
        return text
    value = float(text.replace(' ', '').replace(',', '.'))
    if not math.isfinite(value):
        raise ValueError('RUONIA: недопустимое числовое значение')
    if kind == 'integer':
        if value < 0 or not value.is_integer():
            raise ValueError('RUONIA: количество должно быть целым неотрицательным числом')
        return int(value)
    return value


def parse_ruonia(html, start, end):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = [t for t in soup.find_all('table')
                  if 'ставка ruonia' in normalized(t.get_text(' ', strip=True))]
    if len(candidates) != 1:
        if not candidates and re.search(r'нет\s+информации|нет\s+данных|данные\s+отсутствуют',
                                        soup.get_text(' ', strip=True), re.I):
            raise NoRuoniaForPeriod(f'Нет публикаций RUONIA за {start} — {end}')
        raise ValueError('Не найдена единственная таблица RUONIA; проверьте HTML ЦБ')
    table = candidates[0]
    headers = [normalized(c.get_text(' ', strip=True)) for c in table.find_all('th')]
    mapping = {normalized(label): (key, kind) for key, label, kind, _ in SOURCE_COLUMNS}
    if len(headers) != len(mapping) or set(headers) != set(mapping):
        raise ValueError('Изменилась схема таблицы RUONIA')
    rows, dates = [], set()
    for tr in table.find_all('tr'):
        cells = tr.find_all('td')
        if not cells:
            continue
        if len(cells) != len(headers):
            raise ValueError('Неполная строка RUONIA')
        row = {mapping[header][0]: parse_value(cell.get_text(' ', strip=True), mapping[header][1])
               for header, cell in zip(headers, cells)}
        rate_date = row['rate_date']
        if not rate_date or not start.isoformat() <= rate_date <= end.isoformat():
            raise ValueError('RUONIA: дата ставки отсутствует или выходит за запрошенный интервал')
        if rate_date in dates:
            raise ValueError('Повторная дата ставки RUONIA: ' + rate_date)
        if row['volume_bln_rub'] is not None and row['volume_bln_rub'] < 0:
            raise ValueError('RUONIA: отрицательный объём сделок')
        dates.add(rate_date)
        row['data_gaps'] = '; '.join(key for key, _, _, _ in SOURCE_COLUMNS if row[key] is None)
        rows.append(row)
    if not rows:
        raise NoRuoniaForPeriod(f'Нет публикаций RUONIA за {start} — {end}')
    return sorted(rows, key=lambda row: row['rate_date'])


def run_ruonia(day, output, timeout=15, retries=2):
    """Save all available weekly observations, raw HTML and explicit status."""
    if timeout <= 0 or retries < 0:
        raise ValueError('timeout > 0, retries >= 0')
    start = day - timedelta(days=7)
    output = Path(output).resolve()
    support = output / '_support'
    support.mkdir(parents=True, exist_ok=True)
    target, raw = output / 'cbr_ruonia.csv', support / 'cbr_ruonia.html'
    staged, status_file = support / 'pending_cbr_ruonia.csv', support / 'cbr_ruonia_status.json'
    params = {'UniDbQuery.Posted': 'True', 'UniDbQuery.From': start.strftime('%d.%m.%Y'),
              'UniDbQuery.To': day.strftime('%d.%m.%Y')}
    metadata = dict(status='pending', schema_version=1, trade_date=day.isoformat(),
                    date_from=start.isoformat(), date_to=day.isoformat(),
                    interval='trade_date - 7 calendar days through trade_date, both inclusive',
                    rows=0, latest_rate_date=None, lag_days=None, has_rate_on_trade_date=False,
                    source_url=requests.Request('GET', URL, params=params).prepare().url,
                    started_at_msk=datetime.now(MSK).isoformat(), attempts=[],
                    publication_basis='Снимок на время загрузки; дата публикации может быть позднее даты торгов',
                    columns=[dict(key=key, label=label, type=kind, unit=unit) for key, label, kind, unit in COLUMNS])

    def save_status():
        status_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    for path in (target, raw, staged):
        path.unlink(missing_ok=True)
    save_status()
    try:
        with requests.Session() as session:
            session.headers.update({'User-Agent': 'BondDaily/1.0 (public CBR daily data)'})
            for attempt in range(retries + 1):
                record = dict(attempt=attempt + 1)
                metadata['attempts'].append(record)
                try:
                    response = session.get(URL, params=params, timeout=timeout)
                    record.update(http_status=response.status_code, url=response.url)
                    raw.write_bytes(response.content)
                    response.raise_for_status()
                except requests.RequestException as error:
                    record['error'] = f'{type(error).__name__}: {error}'
                    retryable = (not isinstance(error, requests.HTTPError)
                                 or record.get('http_status') == 429 or record.get('http_status', 0) >= 500)
                    if retryable and attempt < retries:
                        time.sleep(2 ** attempt)
                        continue
                    raise
                rows = parse_ruonia(response.content, start, day)
                rows = [dict(trade_date=day.isoformat(), **row) for row in rows]
                frame = pd.DataFrame(rows, columns=[c[0] for c in COLUMNS])
                for key, _, kind, _ in COLUMNS:
                    if kind == 'integer':
                        frame[key] = frame[key].astype('Int64')
                frame.to_csv(staged, index=False, encoding='utf-8-sig')
                staged.replace(target)
                rate_dates = [r['rate_date'] for r in rows if r['ruonia_pct'] is not None]
                latest = max(rate_dates) if rate_dates else None
                metadata.update(status='ok', rows=len(rows), table_file=target.name,
                                actual_date_from=rows[0]['rate_date'], actual_date_to=rows[-1]['rate_date'],
                                latest_rate_date=latest, valid_rate_count=len(rate_dates),
                                lag_days=(day - date.fromisoformat(latest)).days if latest else None,
                                has_rate_on_trade_date=day.isoformat() in rate_dates,
                                data_gaps={r['rate_date']: r['data_gaps'] for r in rows if r['data_gaps']})
                break
    except NoRuoniaForPeriod as error:
        metadata.update(status='unavailable', error=str(error))
    except (requests.RequestException, ValueError, OSError) as error:
        metadata.update(status='error', error=f'{type(error).__name__}: {error}')
        target.unlink(missing_ok=True)
    finally:
        staged.unlink(missing_ok=True)
    metadata['finished_at_msk'] = datetime.now(MSK).isoformat()
    save_status()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Папка завершённого дневного запуска main.py')
    source.add_argument('--date', type=date.fromisoformat, help='Дата торгов, конец недельного интервала')
    parser.add_argument('--output', help='Постоянная папка; по умолчанию --input или общая daily_data')
    parser.add_argument('--timeout', type=float, default=15)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error('timeout > 0')
    if args.input:
        manifest = json.loads((Path(args.input) / 'run_manifest.json').read_text(encoding='utf-8'))
        if manifest.get('mode') != 'day' or manifest.get('collection_status') != 'ok':
            parser.error('--input должен указывать на завершённый дневной сбор MOEX')
        day = date.fromisoformat(manifest['trade_date'])
    else:
        day = args.date
    if day > datetime.now(MSK).date():
        parser.error('Дата не может быть в будущем')
    output = Path(args.output or args.input or DEFAULT_DAILY_OUTPUT).resolve()
    with output_lock(output):
        try:
            manifest = matching_daily_manifest(output, day)
        except ValueError as error:
            parser.error(str(error))
        result = run_ruonia(day, output, timeout=args.timeout)
        if manifest is not None:
            manifest['cbr_ruonia'] = result
            (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"RUONIA {result['date_from']} — {day}: {result['status']}; {output}", flush=True)
    if result.get('error'):
        print(result['error'], flush=True)
    return 0 if result['status'] == 'ok' else 4


if __name__ == '__main__':
    raise SystemExit(main())
