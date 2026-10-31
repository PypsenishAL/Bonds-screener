"""Daily source preparation and deterministic weighted comparison baskets."""
import json
from pathlib import Path
from moex_bonds.paths import CONFIG_FILE
import re

import numpy as np
import pandas as pd

from moex_bonds.analysis.spread_recovery import validate_recovery_config


DIRECTIONS = ('self', 'issuer', 'peers', 'market')
RATINGS = ['AAA', 'AA+', 'AA', 'AA-', 'A+', 'A', 'A-', 'BBB+', 'BBB', 'BBB-',
           'BB+', 'BB', 'BB-', 'B+', 'B', 'B-', 'CCC+', 'CCC', 'CCC-', 'CC', 'C', 'D']
REQUIRED = ['SECID', 'ISSUER', 'RATING', 'SECTOR', 'TRADEDATE', 'BOARDID',
            'ZSPREADATWAPRICE', 'DURATION_YEARS', 'NUMTRADES', 'TRADE_STATUS']
MEMBER_COLUMNS = ['analysis_date', 'SECID', 'comparison', 'member_SECID', 'member_BOARDID',
                  'distance', 'weight', 'member_z_bps', 'target_issuer', 'member_issuer',
                  'target_rating', 'member_rating', 'rating_diff', 'target_duration',
                  'member_duration', 'duration_diff', 'target_sector', 'member_sector',
                  'sector_diff', 'target_adv20', 'member_adv20', 'liquidity_log_ratio',
                  'liquidity_omitted', 'rating_distance', 'duration_distance', 'sector_distance',
                  'liquidity_distance', 'target_adv_source', 'member_adv_source',
                  'target_adv_last_value_date', 'member_adv_last_value_date', 'flags']


def finite(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def text_key(value):
    return ' '.join(str(value).split()).casefold() if pd.notna(value) else ''


def load_config(path=None):
    config = json.loads(CONFIG_FILE.read_text(encoding='utf-8'))
    if path:
        overrides = json.loads(Path(path).read_text(encoding='utf-8'))
        def merge(target, source):
            if not isinstance(source, dict):
                raise ValueError('Конфиг должен быть объектом JSON')
            if 'recovery_fraction' in source:
                raise ValueError('recovery_fraction относится к прежней методологии; используйте target_scale')
            unknown = source.keys() - target.keys()
            if unknown:
                raise ValueError('Неизвестные настройки: ' + ', '.join(sorted(unknown)))
            for key, value in source.items():
                if key == 'issuer_aliases':
                    if not isinstance(value, dict):
                        raise ValueError('issuer_aliases должен быть словарём')
                    target[key].update(value)
                elif isinstance(target[key], dict):
                    merge(target[key], value)
                else:
                    target[key] = value
        merge(config, overrides)
    validate_recovery_config(config['recovery'])
    for key in ('temperature', 'duration_unit_years', 'mad_scale'):
        if finite(config[key]) is None or config[key] <= 0:
            raise ValueError(key + ' должен быть положительным')
    if (not isinstance(config['windows'], list) or not config['windows']
            or any(type(w) is not int or w < 1 for w in config['windows'])
            or len(set(config['windows'])) != len(config['windows'])):
        raise ValueError('Окна должны быть уникальными положительными целыми числами')
    if type(config['min_observations']) is not int or config['min_observations'] < 1:
        raise ValueError('min_observations должен быть положительным целым')
    classification = config['classification']
    ordinary = classification['min_ordinary_observations']
    if type(ordinary) is not int or ordinary < config['min_observations']:
        raise ValueError('classification.min_ordinary_observations должен быть целым не меньше min_observations')
    primary = classification['primary_window']
    if type(primary) is not int or primary not in config['windows'] or primary < ordinary:
        raise ValueError('classification.primary_window должен входить в windows и быть не короче обычного минимума')
    threshold = classification['percentile_threshold']
    if type(threshold) not in (int, float) or finite(threshold) is None or not 0 < threshold < 100:
        raise ValueError('classification.percentile_threshold должен быть конечным числом строго между 0 и 100')
    robust_threshold = classification['robust_score_threshold']
    if type(robust_threshold) not in (int, float) or finite(robust_threshold) is None or robust_threshold <= 0:
        raise ValueError('classification.robust_score_threshold должен быть конечным положительным числом')
    if finite(config['min_weight_coverage']) is None or not 0 < config['min_weight_coverage'] <= 1:
        raise ValueError('min_weight_coverage должен быть в (0, 1]')
    for kind, basket in config['baskets'].items():
        for key in ('aR', 'aD', 'aS', 'aL', 'max_duration_diff'):
            if finite(basket[key]) is None or basket[key] < 0:
                raise ValueError(f'baskets.{kind}.{key} должен быть неотрицательным')
        for key in ('max_members', 'max_rating_diff'):
            val = basket[key]
            if val is not None and (type(val) is not int or val < (1 if key == 'max_members' else 0)):
                raise ValueError('Неверное ограничение ' + key)
    for key in ('structured_issuers', 'structured_secids'):
        if not isinstance(config[key], list) or any(not isinstance(v, str) or not v.strip() for v in config[key]):
            raise ValueError(key + ' должен быть списком непустых строк')
    aliases = {text_key(k): text_key(v) for k, v in config['issuer_aliases'].items()}
    for origin in aliases:
        seen, current = set(), origin
        while current in aliases:
            if current in seen:
                raise ValueError('Цикл в issuer_aliases')
            seen.add(current)
            current = aliases[current]
    return config


def read_daily(path, analysis_date=None):
    """Read only Итог and its field dictionary, finding the SECID header."""
    with pd.ExcelFile(path, engine='openpyxl') as book:
        dictionary = pd.read_excel(book, sheet_name='Поля', header=None)
        mapping = {'SECID': 'SECID'}
        for value in dictionary.to_numpy().ravel():
            if isinstance(value, str):
                match = re.fullmatch(r'(.+?)\s*\[([A-Z][A-Z0-9_]*)\]', value.strip())
                if match:
                    label, key = match.groups()
                    if label in mapping and mapping[label] != key:
                        raise ValueError('Неоднозначное поле в листе Поля: ' + label)
                    mapping[label] = key
        frame = pd.read_excel(book, sheet_name='Итог', header=None)
    header_rows = [i for i, row in frame.iterrows() if 'SECID' in row.tolist()]
    if len(header_rows) != 1:
        raise ValueError('Не найден единственный заголовок SECID на листе Итог')
    pos = header_rows[0]
    headers = [mapping.get(str(v).strip(), str(v).strip()) for v in frame.loc[pos]]
    if len(headers) != len(set(headers)):
        raise ValueError('Повторные поля дневной выгрузки')
    frame = frame.loc[pos + 1:].copy()
    frame.columns = headers
    frame = frame.dropna(how='all').reset_index(drop=True)
    if not set(REQUIRED).issubset(frame):
        raise ValueError('Не хватает обязательных полей дневной выгрузки: ' + ', '.join(sorted(set(REQUIRED)-set(frame))))
    if frame.empty or frame.SECID.isna().any() or frame.SECID.astype(str).str.strip().eq('').any() or frame.SECID.duplicated().any():
        raise ValueError('Пустой или повторный SECID в дневной выгрузке')
    if frame.BOARDID.isna().any():
        raise ValueError('Отсутствует BOARDID дневного выпуска')
    dates = pd.to_datetime(frame.TRADEDATE, errors='coerce')
    if dates.isna().any() or dates.dt.normalize().nunique() != 1:
        raise ValueError('Нужна одна подтверждённая дата TRADEDATE ежедневной выгрузки')
    day = dates.iloc[0].date().isoformat()
    if analysis_date and analysis_date != day:
        raise ValueError('Указанная дата не соответствует дате ежедневной выгрузки')
    frame.TRADEDATE = day
    return frame, day


def prepare_daily(frame, config):
    frame = frame.copy()
    aliases = {text_key(k): text_key(v) for k, v in config['issuer_aliases'].items()}
    def issuer(value):
        key = text_key(value)
        while key in aliases:
            key = aliases[key]
        return key
    def rating(value):
        token = str(value).strip().upper().replace('−', '-').replace('–', '-').replace('—', '-')
        token = re.sub(r'^RU|\(RU\)$|\.RU$', '', token)
        return RATINGS.index(token) if token in RATINGS else np.nan
    frame['issuer_key'] = frame.ISSUER.map(issuer)
    frame['rating_step'] = frame.RATING.map(rating)
    frame['sector_key'] = frame.SECTOR.map(text_key)
    for source, target in [('ZSPREADATWAPRICE', 'z_bps'), ('DURATION_YEARS', 'duration'), ('ADV20_RUB', 'adv20')]:
        frame[target] = frame[source].map(finite) if source in frame else np.nan
    frame['z_bps'] = frame.z_bps * 100
    count = pd.to_numeric(frame.NUMTRADES, errors='coerce')
    frame['confirmed_trade'] = frame.TRADE_STATUS.eq('traded') & np.isfinite(count) & count.gt(0)
    # Product labels refer to original names: aliasing СберИОС to its real
    # issuer must not classify ordinary Сбербанк ПАО bonds as structured.
    structured = {text_key(v) for v in config['structured_issuers']}
    frame['structured'] = frame.ISSUER.map(text_key).isin(structured) | frame.SECID.isin(config['structured_secids'])
    # Explicit product fields, when provided by a richer daily source.
    for field in ('BONDTYPE', 'PRODUCT_TYPE', 'BONDSUBTYPE'):
        if field in frame:
            frame['structured'] |= frame[field].fillna('').str.contains(r'структур|structured', case=False)
    frame['adv_source'] = np.where(frame.adv20.notna(), 'daily', 'missing')
    frame['adv_n_values'] = np.nan
    frame['adv_last_value_date'] = None
    return frame


def fill_liquidity(daily, reader, day):
    frame = daily.copy()
    ids = frame.loc[frame.adv20.isna(), 'SECID'].tolist()
    if not ids:
        return frame
    history = reader.query(ids, None, day, ['ADV20_RUB', 'VALUE_RUB'])
    groups = {secid: rows.sort_values('TRADEDATE') for secid, rows in history.groupby('SECID')}
    for index, row in frame[frame.SECID.isin(ids)].iterrows():
        past = groups.get(row.SECID, pd.DataFrame(columns=history.columns))
        values = past.VALUE_RUB.map(finite)
        published = past[values.notna()].tail(20)
        frame.at[index, 'adv_n_values'] = len(published)
        frame.at[index, 'adv_last_value_date'] = published.TRADEDATE.iloc[-1] if len(published) else None
        today = past[past.TRADEDATE.eq(day)]
        ready = finite(today.ADV20_RUB.iloc[0]) if len(today) else None
        if ready is not None:
            frame.at[index, 'adv20'] = ready
            frame.at[index, 'adv_source'] = 'history_adv20'
        elif len(published) == 20:
            frame.at[index, 'adv20'] = published.VALUE_RUB.astype(float).mean()
            frame.at[index, 'adv_source'] = 'last_20_published_values'
        else:
            frame.at[index, 'adv_source'] = 'insufficient_20_values'
    return frame


def softmax_weights(distances, temperature):
    d = np.asarray(distances, dtype=float)
    shifted = np.exp(-(d - d.min()) / temperature)
    return shifted / shifted.sum()


def own_reason(row):
    if not row['confirmed_trade']:
        return 'unconfirmed_trade'
    if finite(row['z_bps']) is None:
        return 'missing_zspreadatwaprice'
    return ''


def basket_reason(row, settings):
    reason = own_reason(row)
    if reason:
        return reason
    if row['structured']:
        return 'structured_product'
    if not row['issuer_key']:
        return 'missing_issuer'
    if finite(row['duration']) is None or row['duration'] <= 0:
        return 'missing_positive_duration'
    if (settings['aR'] > 0 or settings['max_rating_diff'] is not None) and finite(row['rating_step']) is None:
        return 'missing_rating'
    if settings['aS'] > 0 and not row['sector_key']:
        return 'missing_sector'
    return ''


def select_baskets(daily, config):
    members, rejected, current = [], [], {}
    rows = daily.to_dict('records')
    day = rows[0]['TRADEDATE']
    for target in rows:
        reason = own_reason(target)
        current[(target['SECID'], 'self')] = dict(current_metric_bps=None if reason else target['z_bps'],
            benchmark_bps=None, gap_bps=None, reason=reason, n_members=0, n_issuers=0, n_eff=None, flags='')
        for kind, settings in config['baskets'].items():
            target_reason = basket_reason(target, settings)
            candidates = []
            if not target_reason:
                for other in rows:
                    secid = other['SECID']
                    why = 'self_excluded' if secid == target['SECID'] else basket_reason(other, settings)
                    same = target['issuer_key'] == other['issuer_key']
                    dd = abs(target['duration'] - other['duration']) if finite(other['duration']) is not None else np.nan
                    rr = abs(target['rating_step'] - other['rating_step'])
                    if not why and (same != (kind == 'issuer')):
                        why = 'different_issuer' if kind == 'issuer' else 'same_issuer'
                    if not why and dd > settings['max_duration_diff'] + 1e-12:
                        why = 'duration_outside_limit'
                    if not why and settings['max_rating_diff'] is not None and rr > settings['max_rating_diff']:
                        why = 'rating_outside_limit'
                    if why:
                        rejected.append(dict(analysis_date=day, SECID=target['SECID'], comparison=kind,
                                             candidate_SECID=secid, reason=why))
                        continue
                    sector_diff = target['sector_key'] != other['sector_key'] if target['sector_key'] and other['sector_key'] else None
                    adv_i, adv_j = finite(target['adv20']), finite(other['adv20'])
                    omitted = adv_i is None or adv_j is None or adv_i <= 0 or adv_j <= 0
                    liquidity_log_ratio = None if omitted else abs(np.log(adv_i) - np.log(adv_j))
                    components = dict(rating_distance=settings['aR'] * rr if settings['aR'] else 0.,
                        duration_distance=settings['aD'] * dd / config['duration_unit_years'],
                        sector_distance=settings['aS'] * bool(sector_diff),
                        liquidity_distance=0. if omitted else settings['aL'] * liquidity_log_ratio)
                    candidates.append(dict(analysis_date=day, SECID=target['SECID'], comparison=kind,
                        member_SECID=secid, member_BOARDID=other['BOARDID'], distance=sum(components.values()),
                        member_z_bps=other['z_bps'], target_issuer=target['issuer_key'], member_issuer=other['issuer_key'],
                        target_rating=target['RATING'], member_rating=other['RATING'], rating_diff=rr,
                        target_duration=target['duration'], member_duration=other['duration'], duration_diff=dd,
                        target_sector=target['SECTOR'], member_sector=other['SECTOR'], sector_diff=sector_diff,
                        target_adv20=adv_i, member_adv20=adv_j, liquidity_log_ratio=liquidity_log_ratio,
                        liquidity_omitted=omitted, target_adv_source=target['adv_source'], member_adv_source=other['adv_source'],
                        target_adv_last_value_date=target['adv_last_value_date'],
                        member_adv_last_value_date=other['adv_last_value_date'],
                        flags='liquidity_omitted' if omitted else '', **components))
            candidates.sort(key=lambda c: (c['distance'], c['member_SECID']))
            selected, issuers = [], set()
            for item in candidates:
                why = ''
                if settings['max_members'] and len(selected) >= settings['max_members']:
                    why = 'outside_nearest_limit'
                if why:
                    rejected.append(dict(analysis_date=day, SECID=target['SECID'], comparison=kind,
                        candidate_SECID=item['member_SECID'], reason=why))
                else:
                    selected.append(item)
                    issuers.add(item['member_issuer'])
            weights = softmax_weights([m['distance'] for m in selected], config['temperature']) if selected else []
            for item, weight in zip(selected, weights):
                item['weight'] = float(weight)
            members.extend(selected)
            benchmark = float(sum(m['weight'] * m['member_z_bps'] for m in selected)) if selected else None
            gap = target['z_bps'] - benchmark if benchmark is not None else None
            current[(target['SECID'], kind)] = dict(current_metric_bps=gap, benchmark_bps=benchmark,
                gap_bps=gap, reason=target_reason or ('' if selected else 'no_eligible_analogs'),
                n_members=len(selected), n_issuers=len(issuers), n_eff=float(1 / np.square(weights).sum()) if selected else None,
                flags='liquidity_omitted' if any(m['liquidity_omitted'] for m in selected) else '')
    return (pd.DataFrame(members, columns=MEMBER_COLUMNS), current,
            pd.DataFrame(rejected, columns=['analysis_date','SECID','comparison','candidate_SECID','reason']))
