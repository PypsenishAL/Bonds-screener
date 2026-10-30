"""Download CBR zero-coupon yields for the exact MOEX trading date.

python -m moex_bonds.sources.cbr_zcyc --input data/daily_data
python -m moex_bonds.sources.cbr_zcyc --date 2026-10-27
"""
import argparse
from datetime import date, datetime
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

URL = 'https://www.cbr.ru/hd_base/zcyc_params/'
TERMS = (.25, .5, .75, 1, 2, 3, 5, 7, 10, 15, 20, 30)
COLUMNS = [('trade_date', 'Дата данных'), ('maturity_years', 'Срок до погашения, лет'),
           ('yield_pct', 'Доходность, % годовых')]


class NoCurveForDate(ValueError):
    """The requested date has no published curve; never use a nearby date."""


def numeric(text):
    value = float(re.sub(r'\s+', '', text).replace(',', '.'))
    if not math.isfinite(value):
        raise ValueError('Нечисловое/бесконечное значение кривой ЦБ')
    return value


def parse_zcyc(html, day):
    soup = BeautifulSoup(html, 'html.parser')
    candidates = [t for t in soup.find_all('table')
                  if 'Дата' in t.get_text() and 'Срок до погашения' in t.get_text()]
    if len(candidates) != 1:
        if not candidates and re.search(r'данные\s+отсутствуют|нет\s+данных', soup.get_text(), re.I):
            raise NoCurveForDate(f'ЦБ не опубликовал кривую за {day}')
        raise ValueError('Не найдена единственная таблица кривой ЦБ; проверьте исходный HTML')
    table = candidates[0]
    headers = [row.find_all('th') for row in table.find_all('tr')]
    terms = next((cells for cells in headers if len(cells) == len(TERMS)), None)
    if terms is None:
        raise ValueError('Не найдены 12 сроков в заголовке кривой ЦБ')
    terms = [numeric(cell.get_text(strip=True)) for cell in terms]
    if len(set(terms)) != len(TERMS) or set(terms) != set(TERMS):
        raise ValueError('Неожиданные или повторные сроки кривой ЦБ')
    matched = []
    for row in table.find_all('tr'):
        cells = row.find_all('td')
        if cells and cells[0].get_text(strip=True) == day.strftime('%d.%m.%Y'):
            matched.append(cells)
    if not matched:
        raise NoCurveForDate(f'ЦБ не опубликовал кривую за {day}; другая дата не подставлена')
    if len(matched) != 1:
        raise ValueError(f'Повторные строки кривой ЦБ за {day}')
    cells = matched[0]
    if len(cells) != len(TERMS) + 1:
        raise ValueError('Неполная строка кривой ЦБ')
    rows = [dict(trade_date=day.isoformat(), maturity_years=term,
                 yield_pct=numeric(cell.get_text(strip=True)))
            for term, cell in zip(terms, cells[1:])]
    return sorted(rows, key=lambda row: row['maturity_years'])


def run_zcyc(day, output, timeout=15, retries=2):
    """Save a separate CSV + audit; return explicit ok/unavailable/error status."""
    if timeout <= 0 or retries < 0:
        raise ValueError('timeout > 0, retries >= 0')
    output = Path(output).resolve()
    support = output / '_support'
    support.mkdir(parents=True, exist_ok=True)
    target, raw = output / 'cbr_zcyc.csv', support / 'cbr_zcyc.html'
    status_file = support / 'cbr_zcyc_status.json'
    params = {'UniDbQuery.Posted': 'True', 'UniDbQuery.From': day.strftime('%d.%m.%Y'),
              'UniDbQuery.To': day.strftime('%d.%m.%Y')}
    metadata = dict(status='pending', trade_date=day.isoformat(), rows=0,
                    source_url=requests.Request('GET', URL, params=params).prepare().url,
                    units='Доходность в % годовых: 13.09 означает 13.09%; срок в годах',
                    started_at_msk=datetime.now(MSK).isoformat(), attempts=[])

    def save_status():
        status_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    # Do not let an earlier successful response masquerade as this attempt.
    target.unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
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
                                 or record.get('http_status') == 429
                                 or record.get('http_status', 0) >= 500)
                    if retryable and attempt < retries:
                        time.sleep(2 ** attempt)
                        continue
                    raise
                # HTTP 200 is not enough: validate header, date and all 12 values.
                rows = parse_zcyc(response.content, day)
                pd.DataFrame([[row[key] for key, _ in COLUMNS] for row in rows],
                             columns=[label for _, label in COLUMNS]).to_csv(
                                 target, index=False, encoding='utf-8-sig')
                metadata.update(status='ok', rows=len(rows), table_file=target.name)
                break
    except NoCurveForDate as error:
        metadata.update(status='unavailable', error=str(error))
    except (requests.RequestException, ValueError) as error:
        metadata.update(status='error', error=f'{type(error).__name__}: {error}')
    metadata['finished_at_msk'] = datetime.now(MSK).isoformat()
    save_status()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='Папка дневного запуска main.py; дата берётся из manifest')
    source.add_argument('--date', type=date.fromisoformat, help='Точная дата YYYY-MM-DD')
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
        result = run_zcyc(day, output, timeout=args.timeout)
        if manifest is not None:
            manifest['cbr_zcyc'] = result
            (output / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"CBR {day}: {result['status']}; {output}", flush=True)
    if result.get('error'):
        print(result['error'], flush=True)
    return 0 if result['status'] == 'ok' else 4


if __name__ == '__main__':
    raise SystemExit(main())
