"""Create one filtered daily bond table from a saved MOEX probe run + Smart-Lab.

python -m moex_bonds.collection.daily --input data/daily_data [--as-of 2026-10-27]
"""
import argparse
from datetime import date, datetime
import json
from pathlib import Path
import re
import subprocess
import os

import pandas as pd

from moex_bonds.sources.moex_client import MSK
from moex_bonds.storage.daily_storage import output_lock
from moex_bonds.sources.smartlab import SmartLabClient, parse_smartlab

HISTORY_ENDPOINT = 'history/engines/stock/markets/bonds/boards/{BOARDID}/securities/{SECID}.json'
YIELDS_ENDPOINT = 'history/engines/stock/markets/bonds/boards/{BOARDID}/yields/{SECID}.json'
FIXED = 'Облигация с фиксированным (известным) купоном'
LEVELS = ['D', 'C', 'CC', 'CCC-', 'CCC', 'CCC+', 'B-', 'B', 'B+',
          'BB-', 'BB', 'BB+', 'BBB-', 'BBB', 'BBB+', 'A-', 'A', 'A+', 'AA-', 'AA', 'AA+', 'AAA']
OFFERS = ['OFFERDATE', 'BUYBACKDATE', 'CALLOPTIONDATE', 'PUTOPTIONDATE',
          'OFFERSTARTDATE', 'OFFERENDDATE', 'OFFERDATESTART', 'OFFERDATEEND']
# key, reader-facing header, format. Raw ISS percentages stay in percentage points.
COLUMNS = [
    ('SECID', 'SECID', 'text'), ('NAME', 'Название бумаги', 'text'),
    ('ISSUER', 'Эмитент', 'text'), ('RATING', 'Кредитный рейтинг', 'text'),
    ('SECTOR', 'Сектор', 'text'), ('WAPRICE', 'Цена WAPRICE, % номинала', 'number'),
    ('ACCINT', 'НКД, руб.', 'number'), ('YTM_WAPRICE', 'YTM по WAPRICE, % годовых', 'number'),
    ('ZSPREADATWAPRICE', 'ZSPREADATWAPRICE, п.п.', 'number'),
    ('ZSPREAD', 'ZSPREAD, п.п.', 'number'), ('DURATION_DAYS', 'Дюрация MOEX, дней', 'integer'),
    ('DURATION_YEARS', 'Дюрация MOEX, лет', 'number'), ('MATDATE', 'Дата погашения', 'date'),
    ('VALUE_RUB', 'Оборот за день, руб.', 'integer'), ('GSPREADBP', 'G-spread, б.п.', 'number'),
    ('NEXTCOUPON', 'Следующий купон', 'date'), ('COUPONPERCENT', 'Ставка купона, % годовых', 'number'),
    ('COUPONVALUE', 'Купон на бумагу, руб.', 'number'), ('FACEVALUE', 'Номинал, руб.', 'number'),
    ('NUMTRADES', 'Число сделок', 'integer'), ('VOLUME', 'Объём, бумаг', 'integer'),
    ('ADV20_RUB', 'Средний оборот 20 дат, руб.', 'integer'),
    ('CLOSE', 'Цена закрытия, % номинала', 'number'), ('DAYS_TO_MATURITY', 'До погашения, дней', 'integer'),
    ('TRADEDATE', 'Дата данных', 'date'), ('BOARDID', 'Режим торгов', 'text'),
    ('TRADE_STATUS', 'Статус торгов', 'text'), ('DATA_GAPS', 'Пропуски и замечания', 'text'),
    ('SMARTLAB_AT', 'Справка Smart-Lab получена (МСК)', 'text'), ('SMARTLAB_URL', 'Карточка Smart-Lab', 'text'),
]


def valid(value):
    return value is not None and not pd.isna(value) and str(value).strip() not in ('', '0000-00-00', '—', '-')


def number(value):
    return float(value) if valid(value) and pd.notna(pd.to_numeric(value, errors='coerce')) else None


def iso(value):
    if not valid(value):
        return None
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10]).isoformat()
        except ValueError:
            pass
    stamp = pd.to_datetime(value, errors='coerce')
    return stamp.date().isoformat() if pd.notna(stamp) else None


def rating_grade(raw):
    if not valid(raw):
        return None
    grades = []
    for token in re.split(r'[/;,|\n]+', str(raw)):
        token = re.sub(r'\s+', '', token).upper()
        token = re.sub(r'^RU|\(RU\)$|\.RU$', '', token)
        if token not in LEVELS:
            return None
        grades.append(LEVELS.index(token))
    return min(grades) if grades else None


def previous_trade_date(samples, as_of, run_date):
    if as_of != run_date:
        raise ValueError('Дата запуска MOEX не совпадает с --as-of; сначала обновите основной сбор')
    dates = [iso(row.get('securities.PREVDATE')) for row in samples]
    dates = [d for d in dates if d and d < as_of.isoformat()]
    if not dates:
        raise ValueError('Нет подтверждённой PREVDATE предыдущего торгового дня MOEX')
    return date.fromisoformat(max(dates))


def unique_history(frame, endpoint):
    if frame.empty or 'endpoint' not in frame:
        return pd.DataFrame()
    # Daily bulk endpoints return the same ISS blocks as per-security history.
    bulk = endpoint.replace('/securities/{SECID}.json', '/securities.json').replace('/yields/{SECID}.json', '/yields.json')
    actual_bulk = bulk.replace('{BOARDID}', 'TQCB')
    frame = frame[frame.endpoint.isin([endpoint, bulk, actual_bulk])].copy()
    frame['endpoint'] = endpoint
    if frame.empty:
        return frame
    # ISS TRADINGSESSION=3 is the full-day record. Never sum separate sessions.
    if 'TRADINGSESSION' in frame and frame.TRADINGSESSION.notna().any():
        frame = frame[frame.TRADINGSESSION.eq(3)].copy()
    keys = ['SECID', 'BOARDID', 'TRADEDATE']
    if endpoint == YIELDS_ENDPOINT:
        keys += ['YIELDDATETYPE', 'YIELDDATE']
    ignore = {'window_days', 'probe_secid', 'probe_board'}
    compare = [c for c in frame if c not in ignore]
    for _, rows in frame.groupby(keys, dropna=False):
        if len(rows[compare].drop_duplicates()) > 1:
            raise ValueError('Конфликт исторических значений по дневному ключу')
    return frame.drop_duplicates(keys)


def sample_value(row, field):
    for key in ('securities.' + field, 'securities/{SECID}.json.description.' + field):
        if valid(row.get(key)):
            return row[key]
    return None


def filter_reasons(sample, dayrow, smart, events, day, past_rows=()):
    reasons = []
    isin = sample_value(sample, 'ISIN')
    kind = sample_value(sample, 'TYPE')
    if not isinstance(isin, str) or not isin.startswith('RU') or kind not in ('corporate_bond', 'exchange_bond'):
        reasons.append('Не российский корпоративный выпуск (ISIN/TYPE)')
    if sample_value(sample, 'FACEUNIT') not in ('RUB', 'SUR') or sample_value(sample, 'CURRENCYID') not in ('RUB', 'SUR'):
        reasons.append('Номинал или расчёты не в рублях')
    if sample_value(sample, 'BONDTYPE') != FIXED:
        reasons.append('BONDTYPE не фиксированный известный купон')
    if not dayrow:
        reasons.append('Нет дневной записи на общую дату данных')
    else:
        maturity = iso(dayrow.get('MATDATE'))
        cutoff = (pd.Timestamp(day) + pd.DateOffset(months=3)).date().isoformat()
        if not maturity or maturity < cutoff:
            reasons.append('Погашение менее чем через 3 календарных месяца или дата неизвестна')
        for field in ('FACEUNIT', 'CURRENCYID'):
            if dayrow.get(field) not in ('RUB', 'SUR'):
                reasons.append('Историческая валюта не подтверждена как RUB')
                break
        if valid(dayrow.get('BONDTYPE')) and dayrow['BONDTYPE'] != FIXED:
            reasons.append('Исторический BONDTYPE не фиксированный известный купон')
    complete = ('empty_terminal', 'cursor_complete')
    if sample.get('calendar.offers.pagination') not in complete or number(sample.get('calendar.offers.rows')) is None:
        reasons.append('Календарь оферт не подтверждён полностью')
    # Inspect every source-prefixed offer field, including historical/past offers.
    has_offer = any(valid(v) for k, v in sample.items() if k.rsplit('.', 1)[-1].upper() in OFFERS)
    has_offer |= any(valid((dayrow or {}).get(k)) for k in OFFERS)
    has_offer |= (number(sample.get('calendar.offers.rows')) or 0) > 0
    has_offer |= any(e.get('section') == 'offers' for e in events)
    has_offer |= any(any(valid(row.get(k)) for k in OFFERS)
                     or re.search(r'оферт|offer|\bput\b|\bcall\b',
                                  str(row.get('BONDSUBTYPE', '')) + ' ' + str(row.get('YIELDDATETYPE', '')), re.I)
                     for row in past_rows)
    subtype = str(sample_value(sample, 'BONDSUBTYPE') or '')
    has_offer |= bool(re.search(r'оферт|\bput\b|\bcall\b', subtype, re.I))
    offer_type = smart.get('offer_type')
    has_offer |= valid(smart.get('offer_date'))
    if smart.get('status') == 'ok' and offer_type != 'Оферты отсутствуют':
        if valid(offer_type):
            has_offer = True
        else:
            reasons.append('Тип оферты Smart-Lab не подтверждён')
    if has_offer:
        reasons.append('Оферта / put / call (включая прошлые)')
    issue_description = sample_value(sample, 'ISSUENAME')
    if not valid(issue_description):
        reasons.append('Нет описания выпуска для проверки субординации')
    issue_text = ' '.join(str(v) for k, v in sample.items()
                          if k.rsplit('.', 1)[-1] in ('ISSUENAME', 'NAME', 'SECNAME', 'SHORTNAME',
                                                       'BONDTYPE', 'BOND_TYPE', 'BONDSUBTYPE', 'BOND_SUBTYPE'))
    issue_text += ' ' + str(smart.get('title', '')) + ' ' + str(smart.get('bond_type', ''))
    issue_text = re.sub(r'несубординир\w*|non[- ]subordinated', '', issue_text, flags=re.I)
    sub_fields = {'ISSUBORDINATED', 'IS_SUBORDINATED', 'SUBORDINATED', 'ISSUBORD', 'SUBORDINATION'}
    sub_flag = any(k.rsplit('.', 1)[-1].upper() in sub_fields and valid(v)
                   and str(v).strip().lower() in ('1', '1.0', 'true', 'yes', 'да')
                   for k, v in sample.items())
    if sub_flag or re.search(r'суборд|\bsubordinat|perpetual|бессрочн', issue_text, re.I):
        reasons.append('Субординация / бессрочный выпуск')
    if smart.get('status') != 'ok':
        reasons.append('Не получена карточка Smart-Lab: ' + str(smart.get('error', '')))
    for field, label in [('issuer', 'эмитент'), ('sector', 'сектор')]:
        if not valid(smart.get(field)):
            reasons.append('Не подтверждён ' + label + ' Smart-Lab')
    grade = rating_grade(smart.get('rating'))
    if grade is None:
        reasons.append('Рейтинг отсутствует или не распознан')
    elif grade < rating_grade('BB'):
        reasons.append('Рейтинг ниже BB')
    return reasons


def build_daily(samples, histories, events, smartlab, day):
    frame = pd.DataFrame(histories)
    hist = unique_history(frame, HISTORY_ENDPOINT)
    yields = unique_history(frame, YIELDS_ENDPOINT)
    final, excluded = [], []
    events_by_secid = {}
    for event in events:
        events_by_secid.setdefault(event.get('probe_secid'), []).append(event)
    identities = [(s['SECID'], s['BOARDID']) for s in samples]
    if len(set(identities)) != len(identities):
        raise ValueError('Повторный SECID/BOARDID во входной выборке')
    if len({s['SECID'] for s in samples}) != len(samples):
        raise ValueError('Для одной строки на SECID требуется единственный режим торгов')
    for sample in samples:
        secid, board = sample['SECID'], sample['BOARDID']
        def select(source, on_day=True):
            if source.empty:
                return source
            mask = source.SECID.eq(secid) & source.BOARDID.eq(board)
            mask &= source.TRADEDATE.eq(day.isoformat()) if on_day else source.TRADEDATE.le(day.isoformat())
            return source[mask]
        records = select(hist)
        h = records.iloc[0].to_dict() if not records.empty else {}
        smart = smartlab.get(secid, {})
        ev = events_by_secid.get(secid, [])
        past = select(hist, on_day=False).to_dict('records') + select(yields, on_day=False).to_dict('records')
        reasons = filter_reasons(sample, h, smart, ev, day, past)
        ys = select(yields)
        if not ys.empty and ys.YIELDDATETYPE.ne('MATDATE').any():
            reasons.append('История доходности рассчитана к оферте/другой дате')
        if not ys.empty:
            ys = ys[ys.YIELDDATETYPE.eq('MATDATE') & ys.YIELDDATE.eq(iso(h.get('MATDATE')))]
        if len(ys) > 1:
            reasons.append('Неоднозначная доходность к погашению')
        if reasons:
            excluded.append(dict(SECID=secid, NAME=sample_value(sample, 'SHORTNAME'),
                                 RATING=smart.get('rating'), reasons='; '.join(reasons),
                                 SMARTLAB_URL=smart.get('url'), SMARTLAB_AT=smart.get('retrieved_at_msk')))
            continue
        row = format_daily_row(sample, h, ys, ev, smart, day, select(hist, on_day=False).to_dict('records'))
        final.append(row)
    return sorted(final, key=lambda r: (-(r['VALUE_RUB'] or 0), r['SECID'])), excluded


def format_daily_row(sample, h, ys, ev, smart, day, past_history=()):
    """Shared dated fields for daily snapshots and the current-screen history."""
    gaps = []
    secid, board = sample['SECID'], sample['BOARDID']
    y = ys.iloc[0].to_dict() if len(ys) == 1 else {}
    stale = iso(y.get('TRADEMOMENT')) != day.isoformat()
    if y and stale:
        gaps.append('устаревшая/неизвестная дата сделки в yields; показатели не перенесены')
        y = {}
    if not y:
        gaps.append('нет свежей строки yields к погашению')
    coupon_dates = sorted({iso(e.get('coupondate')) for e in ev
                           if e.get('section') == 'coupons' and iso(e.get('coupondate'))
                           and iso(e.get('coupondate')) > day.isoformat()})
    complete_coupons = sample.get('calendar.coupons.pagination') in ('empty_terminal', 'cursor_complete')
    next_coupon = coupon_dates[0] if coupon_dates and complete_coupons else None
    if not next_coupon:
        gaps.append('нет подтверждённого следующего купона в полном календаре')
    daily20 = sorted(past_history, key=lambda r: r['TRADEDATE'])[-20:]
    amounts = [number(r.get('VALUE')) for r in daily20]
    rub = all(r.get(c) in ('SUR', 'RUB') for r in daily20 for c in ('FACEUNIT', 'CURRENCYID'))
    adv = sum(amounts) / 20 if len(amounts) == 20 and all(a is not None for a in amounts) and rub else None
    duration = number(y.get('DURATIONWAPRICE'))
    if duration is None:
        duration = number(h.get('DURATION'))
    ytm = number(y.get('EFFECTIVEYIELDWAPRICE'))
    if y and number(y.get('WAPRICE')) != number(h.get('WAPRICE')):
        ytm = None
        gaps.append('WAPRICE истории и yields не совпали; YTM не перенесена')
    row = dict(SECID=secid, NAME=sample_value(sample, 'SHORTNAME'), ISSUER=smart.get('issuer'),
               RATING=smart.get('rating'), SECTOR=smart.get('sector'), WAPRICE=number(h.get('WAPRICE')),
               ACCINT=number(h.get('ACCINT')), YTM_WAPRICE=ytm,
               ZSPREADATWAPRICE=number(h.get('ZSPREADATWAPRICE')), ZSPREAD=number(h.get('ZSPREAD')),
               DURATION_DAYS=duration, DURATION_YEARS=duration / 365 if duration is not None else None,
               MATDATE=iso(h.get('MATDATE')), VALUE_RUB=number(h.get('VALUE')),
               GSPREADBP=number(y.get('GSPREADBP')), NEXTCOUPON=next_coupon,
               COUPONPERCENT=number(h.get('COUPONPERCENT')), COUPONVALUE=number(h.get('COUPONVALUE')),
               FACEVALUE=number(h.get('FACEVALUE')), NUMTRADES=number(h.get('NUMTRADES')),
               VOLUME=number(h.get('VOLUME')), ADV20_RUB=adv, CLOSE=number(h.get('CLOSE')),
               DAYS_TO_MATURITY=(date.fromisoformat(iso(h.get('MATDATE'))) - day).days if iso(h.get('MATDATE')) else None,
               TRADEDATE=day.isoformat(), BOARDID=board,
               SMARTLAB_AT=smart.get('retrieved_at_msk'), SMARTLAB_URL=smart.get('url'))
    for field in ('WAPRICE', 'ACCINT', 'YTM_WAPRICE', 'ZSPREADATWAPRICE', 'ZSPREAD',
                  'DURATION_DAYS', 'VALUE_RUB', 'GSPREADBP', 'ADV20_RUB'):
        if row[field] is None:
            gaps.append('нет ' + field)
    if row['NUMTRADES'] == 0:
        gaps.append('нет сделок на дату данных')
    row['DATA_GAPS'] = '; '.join(gaps)
    count = number(h.get('NUMTRADES'))
    status = ('no_history_record' if not h else 'no_trades' if count == 0
              else 'unknown_trade_count' if count is None or count < 0 else 'traded')
    last = iso(h.get('LASTTRADEDATE'))
    if status == 'traded' and last and last != day.isoformat():
        status = 'stale_trade_date'
    row['TRADE_STATUS'] = status
    if status != 'traded':
        for field in ('WAPRICE', 'CLOSE', 'YTM_WAPRICE', 'ZSPREAD', 'ZSPREADATWAPRICE',
                      'GSPREADBP', 'DURATION_DAYS', 'DURATION_YEARS'):
            row[field] = None
        row['DATA_GAPS'] += '; ' + status + ': свежие цена/доходность/спреды не подтверждены'
    return row


def export_xlsx(output):
    runtime = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/node'
    node = Path(os.environ.get('BONDS_NODE', runtime / 'bin/node.exe'))
    modules = Path(os.environ.get('BONDS_NODE_MODULES', runtime / 'node_modules'))
    if not node.exists() or not (modules / '@oai/artifact-tool').exists():
        raise RuntimeError('Для XLSX нужен @oai/artifact-tool; CSV/JSON уже сохранены. См. README.')
    # The import entrypoint is resolved explicitly; no dependency-directory mutations.
    package = json.loads((modules / '@oai/artifact-tool/package.json').read_text(encoding='utf-8'))
    entry = package.get('exports', {}).get('.', {})
    if isinstance(entry, dict):
        entry = entry.get('import') or entry.get('default')
    if isinstance(entry, dict):
        entry = entry.get('default')
    if not isinstance(entry, str):
        entry = package.get('module') or package.get('main')
    if not entry:
        raise RuntimeError('Не найден публичный entrypoint @oai/artifact-tool')
    env = {**os.environ, 'BONDS_ARTIFACT_MODULE': (modules / '@oai/artifact-tool' / entry).resolve().as_uri()}
    subprocess.run([str(node), str(Path(__file__).with_name('export_daily.mjs')), str(output)], check=True, env=env)


def run_daily(input_dir, output=None, as_of=None, pause=.5, timeout=15, reuse_smartlab=False, xlsx=True,
              history_dir=None):
    source = Path(input_dir).resolve()
    manifest = json.loads((source / 'run_manifest.json').read_text(encoding='utf-8'))
    as_of = as_of or datetime.now(MSK).date()
    day_mode = manifest.get('mode') == 'day'
    if day_mode and manifest.get('collection_status') != 'ok':
        raise ValueError('Дневной сбор не завершён; итоговая таблица не формируется')
    samples_file = 'bonds_universe.csv' if day_mode else 'bonds_sample.csv'
    samples = pd.read_csv(source / samples_file).to_dict('records')
    day = (date.fromisoformat(manifest['trade_date']) if day_mode else
           previous_trade_date(samples, as_of, date.fromisoformat(manifest['as_of_date'])))
    output = Path(output).resolve() if output else source / 'daily'
    if output == source:
        raise ValueError('Итоговая папка должна отличаться от исходного запуска')
    output.mkdir(parents=True, exist_ok=True)
    support = output / '_support'
    support.mkdir(exist_ok=True)
    cache = support / 'smartlab_enrichment.json'
    smartlab = json.loads(cache.read_text(encoding='utf-8')) if reuse_smartlab and cache.exists() else {}
    client = SmartLabClient(support, pause=pause, timeout=timeout)
    for i, sample in enumerate(samples):
        secid = sample['SECID']
        # Do not spend external requests on the rest of the all-market universe.
        if day_mode and not (sample_value(sample, 'BONDTYPE') == FIXED
                             and sample_value(sample, 'FACEUNIT') in ('SUR', 'RUB')
                             and sample_value(sample, 'CURRENCYID') in ('SUR', 'RUB')
                             and sample_value(sample, 'TYPE') in ('corporate_bond', 'exchange_bond')
                             and str(sample_value(sample, 'ISIN') or '').startswith('RU')):
            smartlab[secid] = {'status': 'not_requested', 'error': 'не пройден локальный фильтр MOEX'}
            continue
        if secid not in smartlab:
            print(f'Smart-Lab [{i+1}/{len(samples)}] {secid}', flush=True)
            smartlab[secid] = client.fetch(secid)
            cache.write_text(json.dumps(smartlab, ensure_ascii=False, indent=2), encoding='utf-8')
    histories_file = 'day_quotes.csv' if day_mode else 'historical_sample.csv'
    histories = pd.read_csv(source / histories_file).to_dict('records')
    events = pd.read_csv(source / 'bond_events.csv').to_dict('records')
    rows, excluded = build_daily(samples, histories, events, smartlab, day)
    history_update = {'status': 'not_requested'}
    if day_mode:
        from moex_bonds.storage.history_storage import append_daily, history_output
        try:
            subset_run = bool(manifest.get('args', {}).get('secids'))
            history_update = append_daily(history_output(source, history_dir),
                                          dict(rows=rows, metadata=dict(trade_date=day.isoformat(),
                                               replace_secids=[s['SECID'] for s in samples] if subset_run else None)),
                                          sync_context=dict(samples=samples, events=events, smartlab=smartlab),
                                          pause=pause, timeout=timeout)
        except (OSError, ValueError, RuntimeError) as error:
            history_update = dict(status='error', error=f'{type(error).__name__}: {error}')
            print('Не удалось обновить историческую базу:', error, flush=True)
    # Once new output publication starts, an old XLSX must not masquerade as
    # the result of this attempt if the exporter fails or CSV-only was chosen.
    previous_xlsx = output / 'bonds_daily.xlsx'
    previous_xlsx.unlink(missing_ok=True)
    (support / 'previous_bonds_daily.xlsx').unlink(missing_ok=True)
    headers = [c[1] for c in COLUMNS]
    pd.DataFrame([[r.get(c[0]) for c in COLUMNS] for r in rows], columns=headers).to_csv(
        output / 'bonds_daily.csv', index=False, encoding='utf-8-sig')
    metadata = dict(trade_date=day.isoformat(), as_of=as_of.isoformat(), source_run=str(source),
                    history_update=history_update,
                    input_count=len(samples), final_count=len(rows), excluded_count=len(excluded),
                    smartlab_ok=sum(smartlab.get(s['SECID'], {}).get('status') == 'ok' for s in samples),
                    smartlab_errors=sum(smartlab.get(s['SECID'], {}).get('status') != 'ok' for s in samples),
                    scope=manifest.get('scope', 'Входная выборка TQCB; не весь рынок'),
                    history_mode='одна дата; ADV20 не загружался' if day_mode else 'сохранённые исторические окна',
                    rating_basis='Текущая справка Smart-Lab; не исторический рейтинг на дату торгов',
                    duration_basis='Биржевая дюрация в днях; годы = дни/365; не модифицированная',
                    subordination_basis='По доступным типам, заголовку и описанию выпуска; полнота юридических условий не гарантируется',
                    country_basis='Российский выпуск: RU ISIN + corporate_bond/exchange_bond; резидентство эмитента отдельно не проверялось',
                    xlsx_status='not_requested' if not xlsx else 'pending')
    dataset = dict(metadata=metadata, columns=COLUMNS, rows=rows, excluded=excluded)
    target = support / 'daily_dataset.json'
    target.write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    if xlsx:
        try:
            export_xlsx(output)
            metadata['xlsx_status'] = 'ok'
        except (RuntimeError, subprocess.CalledProcessError) as error:
            metadata['xlsx_status'] = str(error)
            raise
        finally:
            target.write_text(json.dumps(dataset, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(f'Daily {day}: {len(rows)} included / {len(samples)}; output: {output}', flush=True)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Папка дневного запуска main.py или прежнего аудита')
    parser.add_argument('--output')
    parser.add_argument('--history-output', help='Общая база; по умолчанию data/history_data')
    parser.add_argument('--as-of', type=date.fromisoformat, help='Дата расчёта по Москве; по умолчанию сегодня')
    parser.add_argument('--pause', type=float, default=.5)
    parser.add_argument('--timeout', type=float, default=15)
    parser.add_argument('--reuse-smartlab', action='store_true', help='Явно переиспользовать ранее полученную справку с прежним timestamp')
    parser.add_argument('--csv-only', action='store_true')
    args = parser.parse_args()
    if args.pause < 0 or args.timeout <= 0:
        parser.error('pause >= 0, timeout > 0')
    with output_lock(args.input):
        dataset = run_daily(args.input, args.output, args.as_of, args.pause, args.timeout, args.reuse_smartlab,
                            not args.csv_only, args.history_output)
    return 5 if dataset['metadata']['history_update']['status'] == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
