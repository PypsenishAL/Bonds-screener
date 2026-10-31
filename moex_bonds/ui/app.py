"""Local read-only workspace for the saved Z-spread analysis.

Launch: python -m streamlit run app.py
Collectors and analysis commands are never invoked by this application.
"""
from dataclasses import dataclass
import base64
from functools import lru_cache
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import re
from zipfile import BadZipFile

import numpy as np
import pandas as pd
import streamlit as st

from moex_bonds.analysis.spread_model import DIRECTIONS, RATINGS, read_daily, text_key
from moex_bonds.analysis.spread_results import BASE_KEYS as KEYS
from moex_bonds.analysis.spread_recovery import (RECOVERY_RULE_VERSION, EPISODE_KEYS, EPISODE_COLUMNS,
                             SUMMARY_FIELDS, validate_recovery_config, recovery_statistics)

from moex_bonds.paths import PROJECT_DIR, DAILY_DIR, ANALYSIS_DIR, recorded_source_hash

ROOT = PROJECT_DIR
ANALYSIS = ANALYSIS_DIR
DAILY = DAILY_DIR
RULE = 'upper_tail_midrank_v1'
ROBUST_RULE = 'upper_tail_midrank_robust_v2'
ANOMALY_BASES = {'percentile_only': 'Только процентиль', 'robust_only': 'Только robust score',
                 'both': 'Оба критерия', 'none': 'Нет аномалии'}
KINDS = {'peers': 'Аналоги', 'issuer': 'Эмитент', 'market': 'Рынок'}
STATUSES = {'candidate': 'Выделен анализом', 'not_candidate': 'Сигнал не выявлен',
            'not_fully_assessed': 'Недостаточно данных для вывода', 'not_saved': 'Статус не сохранён'}
ASSESSMENTS = {'evaluated': 'Оценено', 'provisional': 'Предварительно', 'unavailable': 'Недоступно'}
MISSING = 'Нет данных'
DARK_CHART = {'background':'#0F141B','view':{'stroke':None},
    'font':'Inter, Arial, sans-serif',
    'axis':{'gridColor':'#2B3746','labelColor':'#A8B3C2','titleColor':'#E6EDF3'},
    'legend':{'labelColor':'#A8B3C2','titleColor':'#E6EDF3'},
    'title':{'color':'#E6EDF3','subtitleColor':'#A8B3C2'}}


class DataUnavailable(ValueError):
    """Stable saved input cannot be interpreted safely."""


class PublicationBusy(DataUnavailable):
    """The publisher is updating the saved result set."""


@dataclass
class Snapshot:
    """Shared snapshots are read-only; consumers must copy before modifying tables."""
    manifest: dict
    tables: dict
    path: Path
    legacy: bool = False
    exclusions_reason: str = ''


def _json(raw):
    value = json.loads(raw.decode('utf-8-sig'))
    if not isinstance(value, dict):
        raise DataUnavailable('Манифест должен содержать объект JSON')
    return value


def _csv(raw):
    frame = pd.read_csv(BytesIO(raw), dtype={'SECID': str, 'BOARDID': str, 'member_SECID': str,
                                           'member_BOARDID': str, 'candidate_SECID': str})
    boolean = ('calendar_confirmed', 'upper_tail_anomaly', 'robust_anomaly', 'candidate_qualified', 'structured', 'liquidity_omitted')
    for column in frame:
        if column in boolean or any(column.endswith('__'+name) for name in boolean):
            def nullable(value):
                if pd.isna(value):
                    return pd.NA
                token = str(value).lower()
                if token not in ('true', 'false'):
                    raise DataUnavailable('Неверное логическое значение: '+column)
                return token == 'true'
            frame[column] = frame[column].map(nullable).astype('boolean')
    return frame


def _guard(root):
    if (root / '.daily_write.lock').exists():
        raise PublicationBusy('Идёт обновление результатов')
    raw = (root / 'run_manifest.json').read_bytes()
    if _json(raw).get('status') != 'ok':
        raise PublicationBusy('Публикация результатов ещё не завершена')
    return raw


def load_snapshot(root=ANALYSIS, day=None):
    """Capture a consistent partition without writing to any input directory."""
    root = Path(root)
    if not day or not re.fullmatch(r'\d{4}-\d{2}-\d{2}', str(day)):
        raise DataUnavailable('Не выбрана сохранённая дата анализа')
    folder = root / 'snapshots' / day
    for attempt in range(3):
        try:
            before = _guard(root)
            manifest_bytes = (folder / 'run_manifest.json').read_bytes()
            manifest = _json(manifest_bytes)
            if manifest.get('status') != 'ok' or manifest.get('analysis_date') != day:
                raise DataUnavailable('Снимок не завершён или его дата не совпадает')
            table_keys = dict(KEYS)
            if 'recovery_rule_version' in manifest:
                if manifest['recovery_rule_version'] != RECOVERY_RULE_VERSION:
                    raise DataUnavailable('Версия истории восстановления не поддерживается')
                validate_recovery_config(manifest.get('config', {}).get('recovery'))
                table_keys['recovery_episodes'] = EPISODE_KEYS
            raw = {name: (folder / (name+'.csv')).read_bytes() for name in table_keys}
            exclusion_path = folder / 'candidate_exclusions.csv'
            exclusions = exclusion_path.read_bytes() if exclusion_path.exists() else None
            after = _guard(root)
            if before != after or manifest_bytes != (folder / 'run_manifest.json').read_bytes():
                raise PublicationBusy('Результаты изменились во время чтения')
            tables = {name: _csv(value) for name, value in raw.items()}
            required = {
                'summary': ['NAME','ISSUER','RATING','SECTOR','BOARDID','DURATION_YEARS','adv20'],
                'metrics': ['window_start','window_end','current_metric_bps','hist_median_bps','excess_bps','n_obs'],
                'comparison_history': ['z_bps','metric_bps','benchmark_bps','coverage','reason'],
                'basket_members': ['member_BOARDID','weight']}
            for name, columns in required.items():
                if not set(columns).issubset(tables[name]):
                    raise DataUnavailable('Не хватает полей для просмотра: '+name)
            schema = manifest.get('result_schema_version', 1)
            if schema not in (1, 2, 3):
                raise DataUnavailable('Версия схемы снимка не поддерживается')
            legacy = schema == 1
            if not legacy and manifest.get('classification_rule_version') != {2: RULE, 3: ROBUST_RULE}[schema]:
                raise DataUnavailable('Версия правила классификации не поддерживается')
            root_manifest = _json(before)
            if not legacy and root_manifest.get('analysis_date') == day and root_manifest.get('analysis_run_id') != manifest.get('analysis_run_id'):
                raise DataUnavailable('Манифесты одного дня относятся к разным расчётам')
            identity = ('analysis_run_id', 'result_schema_version', 'classification_rule_version')
            for name, frame in tables.items():
                keys = table_keys[name]
                if not set(keys).issubset(frame) or frame[keys].isna().any().any() or frame.duplicated(keys).any():
                    raise DataUnavailable('Пустой или повторный ключ: '+name)
                if not frame.analysis_date.eq(day).all():
                    raise DataUnavailable('Смешаны даты: '+name)
                for field in identity:
                    if legacy:
                        if field in frame and frame[field].notna().any():
                            raise DataUnavailable('Смешаны старые и новые идентификаторы расчёта')
                    elif field not in frame or not frame[field].notna().all() or not frame[field].eq(manifest.get(field)).all():
                        raise DataUnavailable('Смешаны идентификаторы расчёта: '+name)
            summary, metrics, history = (tables[n] for n in ('summary','metrics','comparison_history'))
            if summary.empty or metrics.empty or history.empty or len(summary) != manifest.get('instruments'):
                raise DataUnavailable('Неполный набор результатов')
            if 'BOARDID' not in summary or summary.BOARDID.isna().any():
                raise DataUnavailable('В сводке отсутствует режим торгов')
            windows = manifest.get('config', {}).get('windows', [])
            expected = {(s, k, w) for s in summary.SECID for k in DIRECTIONS for w in windows}
            if set(zip(metrics.SECID, metrics.comparison, metrics.window)) != expected:
                raise DataUnavailable('Неполные направления или окна статистики')
            ids = set(summary.SECID)
            for name in ('metrics','comparison_history','basket_members'):
                if not set(tables[name].SECID).issubset(ids):
                    raise DataUnavailable('Неизвестный выпуск в '+name)
            members = tables['basket_members']
            if not members.empty:
                boards = summary.set_index('SECID').BOARDID
                if not set(members.member_SECID).issubset(ids) or not members.member_BOARDID.eq(members.member_SECID.map(boards)).all():
                    raise DataUnavailable('Режим или идентификатор аналога не совпадает со сводкой')
            if 'recovery_episodes' in tables:
                _validate_recovery(tables, manifest)
            if schema == 3:
                _validate_anomalies(tables, manifest)
            exclusions_reason = ''
            try:
                tables['candidate_exclusions'] = _csv(exclusions) if exclusions else pd.DataFrame()
                if exclusions is None:
                    exclusions_reason = 'Причины исключения не сохранены'
                else:
                    excluded = tables['candidate_exclusions']
                    keys = ['analysis_date','SECID','comparison','candidate_SECID']
                    if not set(keys+['reason']).issubset(excluded) or excluded[keys].isna().any().any() or excluded.duplicated(keys).any():
                        raise DataUnavailable('Неверные поля исключений')
                    if not excluded.analysis_date.eq(day).all() or not set(excluded.SECID).issubset(ids) or not set(excluded.candidate_SECID).issubset(ids) or not set(excluded.comparison).issubset(DIRECTIONS):
                        raise DataUnavailable('Исключения относятся к другому снимку')
                    for field in identity:
                        if legacy:
                            if field in excluded and excluded[field].notna().any():
                                raise DataUnavailable('Смешаны идентификаторы исключений')
                        elif field not in excluded or not excluded[field].notna().all() or not excluded[field].eq(manifest.get(field)).all():
                            raise DataUnavailable('Смешаны идентификаторы исключений')
            except (ValueError, OSError):
                tables['candidate_exclusions'] = pd.DataFrame()
                exclusions_reason = 'Не удалось прочитать причины исключения'
            return Snapshot(manifest, tables, folder, legacy, exclusions_reason)
        except PublicationBusy:
            if attempt == 2:
                raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            if isinstance(error, DataUnavailable):
                raise
            raise DataUnavailable('Не удалось прочитать сохранённый анализ: '+str(error)) from error


def _validate_anomalies(tables, manifest):
    """Validate saved criteria and their summary copies without recalculating history."""
    metrics, summary = tables['metrics'], tables['summary']
    fields = ['upper_tail_anomaly', 'robust_anomaly', 'anomaly_basis', 'assessment_status']
    if not set(fields).issubset(metrics):
        raise DataUnavailable('Не хватает полей основания аномальности')
    threshold = manifest.get('config', {}).get('classification', {}).get('robust_score_threshold')
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise DataUnavailable('Не сохранён допустимый порог robust score')
    if not set(metrics.assessment_status.dropna()).issubset(ASSESSMENTS) or metrics.assessment_status.isna().any():
        raise DataUnavailable('Неверный статус оценки аномальности')
    evaluated = metrics.assessment_status.eq('evaluated')
    if (metrics.loc[evaluated, 'upper_tail_anomaly'].isna().any()
            or metrics.loc[~evaluated, ['upper_tail_anomaly', 'robust_anomaly']].notna().any().any()):
        raise DataUnavailable('Признаки аномальности не соответствуют доступности оценки')
    expected = pd.Series(pd.NA, index=metrics.index, dtype='string')
    for percentile_flag, robust_flag, basis in [(True, False, 'percentile_only'),
            (False, True, 'robust_only'), (True, True, 'both'), (False, False, 'none')]:
        mask = metrics.upper_tail_anomaly.eq(percentile_flag) & metrics.robust_anomaly.eq(robust_flag)
        expected.loc[mask.fillna(False)] = basis
    if not metrics.anomaly_basis.fillna('').eq(expected.fillna('')).all():
        raise DataUnavailable('Основание аномальности не соответствует признакам')
    for kind in DIRECTIONS:
        for window in manifest['config']['windows']:
            selected = metrics[metrics.comparison.eq(kind) & metrics.window.eq(window)].set_index('SECID')
            for field in ('robust_anomaly', 'anomaly_basis'):
                column = f'{kind}__{window}__{field}'
                if column not in summary:
                    raise DataUnavailable('Не хватает полей основания аномальности в сводке')
                actual = summary[column].reset_index(drop=True)
                saved = summary.SECID.map(selected[field]).reset_index(drop=True)
                if not actual.isna().eq(saved.isna()).all() or not actual.astype('string').fillna('').eq(saved.astype('string').fillna('')).all():
                    raise DataUnavailable('Основание аномальности в сводке не совпадает с метриками')


def _validate_recovery(tables, manifest):
    """Check saved recovery contracts, without recalculating spread histories."""
    summary, cases = tables['summary'], tables['recovery_episodes']
    if (not set(SUMMARY_FIELDS+['recovery_rule_version']).issubset(summary)
            or not set(EPISODE_COLUMNS).issubset(cases)):
        raise DataUnavailable('Не хватает полей истории восстановления')
    if (not summary.recovery_rule_version.eq(RECOVERY_RULE_VERSION).all()
            or not cases.recovery_rule_version.eq(RECOVERY_RULE_VERSION).all()):
        raise DataUnavailable('Смешаны версии истории восстановления')
    cfg = manifest['config']['recovery']
    identities = set(map(tuple, summary[['SECID','BOARDID']].itertuples(index=False, name=None)))
    if not set(map(tuple, cases[['SECID','BOARDID']].itertuples(index=False, name=None))) <= identities:
        raise DataUnavailable('Неизвестная бумага или режим в истории восстановления')
    outcomes = {'pending','incomplete','recovered','not_recovered'}
    if not set(cases.outcome).issubset(outcomes):
        raise DataUnavailable('Неизвестный результат восстановления')
    numeric = [f for f in EPISODE_COLUMNS if f.endswith('_bps') or f in
               ('reference_n_obs','initial_excess_scales','days_to_recovery','n_expected','n_valid')]
    for field in numeric:
        cases[field] = pd.to_numeric(cases[field], errors='raise')
        if np.isinf(cases[field]).any():
            raise DataUnavailable('Неконечное значение истории восстановления')
    def same(left, right):
        if pd.isna(left) or right is None:
            return pd.isna(left) and right is None
        try:
            return bool(np.isfinite(float(left)) and np.isclose(float(left), right, rtol=1e-9, atol=1e-9))
        except (TypeError, ValueError):
            return False
    for case in cases.to_dict('records'):
        entry = pd.to_datetime(case['entry_date'], format='%Y-%m-%d', errors='raise')
        deadline = (entry+pd.DateOffset(months=cfg['horizon_months'])).date().isoformat()
        if (case['entry_date'] > manifest['analysis_date'] or case['horizon_end'] != deadline
                or case['scale_bps'] <= 0 or case['initial_excess_bps'] <= 0
                or (case['initial_excess_scales'] < cfg['entry_scale']
                    and not np.isclose(case['initial_excess_scales'], cfg['entry_scale'], rtol=1e-12, atol=1e-12))
                or not same(case['initial_excess_bps'], case['entry_z_bps']-case['reference_bps'])
                or not same(case['initial_excess_scales'], case['initial_excess_bps']/case['scale_bps'])
                or not same(case['target_bps'], case['entry_z_bps']-cfg['recovery_fraction']*case['initial_excess_bps'])):
            raise DataUnavailable('Несогласованный ориентир или срок восстановления')
        for field in ('reference_n_obs','n_expected','n_valid'):
            if pd.isna(case[field]) or case[field] < 0 or case[field] != int(case[field]):
                raise DataUnavailable('Неверное число наблюдений восстановления')
        if (not cfg['min_observations'] <= case['reference_n_obs'] <= cfg['reference_window']
                or case['n_valid'] > case['n_expected']):
            raise DataUnavailable('Неверное покрытие истории восстановления')
        known = case['outcome'] in ('recovered','not_recovered')
        hit = case['first_recovery_date']
        has_hit = pd.notna(hit)
        if case['hit_timing'] not in ('complete_path','after_gap','not_observed'):
            raise DataUnavailable('Неизвестное качество даты восстановления')
        if has_hit:
            hit_date = pd.to_datetime(hit, format='%Y-%m-%d', errors='raise')
            if not case['entry_date'] < hit <= min(deadline, manifest['analysis_date']):
                raise DataUnavailable('Дата восстановления вне срока')
            expected_days = (hit_date-entry).days if case['hit_timing'] == 'complete_path' else None
            if case['hit_timing'] == 'not_observed' or not same(case['days_to_recovery'], expected_days):
                raise DataUnavailable('Несогласованное время восстановления')
        elif case['hit_timing'] != 'not_observed' or pd.notna(case['days_to_recovery']):
            raise DataUnavailable('Время восстановления без наблюдения')
        terminal = [case[f] for f in ('end_observation_date','end_z_bps','reduction_bps')]
        if deadline > manifest['analysis_date'] and any(pd.notna(v) for v in terminal):
            raise DataUnavailable('Будущий конечный результат восстановления')
        if known and (deadline > manifest['analysis_date'] or any(pd.isna(v) for v in terminal)
                or case['n_valid'] != case['n_expected']
                or (case['outcome'] == 'recovered') != has_hit):
            raise DataUnavailable('Неподтверждённый известный результат восстановления')
        if pd.notna(case['end_observation_date']):
            pd.to_datetime(case['end_observation_date'], format='%Y-%m-%d', errors='raise')
            if (not case['entry_date'] < case['end_observation_date'] <= min(deadline, manifest['analysis_date'])
                    or not same(case['reduction_bps'], case['entry_z_bps']-case['end_z_bps'])):
                raise DataUnavailable('Несогласованный конечный результат восстановления')
        if pd.isna(case['max_worsening_bps']) or case['max_worsening_bps'] < 0:
            raise DataUnavailable('Неверное дальнейшее расширение спреда')
    for row in summary.to_dict('records'):
        subset = cases[cases.SECID.eq(row['SECID']) & cases.BOARDID.eq(row['BOARDID'])]
        stats = recovery_statistics(subset)
        if any(not same(row['recovery__'+field], value) for field,value in stats.items()):
            raise DataUnavailable('Сводка восстановления не совпадает с подробной таблицей')
        status = row['recovery__status']
        if ((stats['n_known'] and status != 'available')
                or (len(subset) and not stats['n_known'] and status != 'no_known_outcomes')
                or (not len(subset) and status not in ('no_cases','unavailable'))):
            raise DataUnavailable('Неверный статус сводки восстановления')


def recovery_view(snapshot, identity):
    """Return saved recovery facts for one stable bond identity."""
    if 'recovery_rule_version' not in snapshot.manifest:
        return dict(status='not_saved', message='История восстановления не рассчитана',
                    criterion='', episodes=pd.DataFrame(), summary=None)
    selected = snapshot.tables['summary']
    selected = selected[selected.SECID.eq(identity[0]) & selected.BOARDID.eq(identity[1])]
    if selected.empty:
        raise DataUnavailable('Бумаги нет в сводке восстановления')
    row = selected.iloc[0]
    cfg = snapshot.manifest['config']['recovery']
    fraction = f"{100*cfg['recovery_fraction']:g}%"
    criterion = (f"Восстановление минимум {fraction} начального расширения за "
                 f"{cfg['horizon_months']} календарных мес. Ориентир: "
                 f"{cfg['reference_window']} торговых дат.")
    known, recovered = int(row['recovery__n_known']), int(row['recovery__n_recovered'])
    message = (f'Спред восстановился минимум на {fraction} в {recovered} из {known} '
               'прошлых случаев с известным результатом') if known else 'Недостаточно истории'
    cases = snapshot.tables['recovery_episodes']
    cases = cases[cases.SECID.eq(identity[0]) & cases.BOARDID.eq(identity[1])].copy()
    return dict(status=row['recovery__status'], message=message, criterion=criterion,
                episodes=cases, summary=row)


@lru_cache(maxsize=2)
def _parse_workbook(raw, day):
    return read_daily(BytesIO(raw), day)[0]


@lru_cache(maxsize=2)
def _shared_snapshot(root, day, fingerprint):
    return load_snapshot(root, day)


def file_stamp(path):
    stat = Path(path).stat()
    return (stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)


@lru_cache(maxsize=2)
def _book_file(path, expected, day, stamp):
    raw = Path(path).read_bytes()
    if file_stamp(path) != stamp or hashlib.sha256(raw).hexdigest() != expected:
        raise DataUnavailable('Дневная книга относится к другой версии расчёта')
    return _parse_workbook(raw,day)


@st.cache_data(max_entries=4,show_spinner=False)
def cached_candidates(key, window, _snapshot, _enrichment):
    # cache_data returns independent frames; shared snapshot/book inputs stay read-only.
    return candidate_view(_snapshot,window,_enrichment)


def load_enrichment(snapshot, daily=DAILY, publication=None):
    path = Path(daily) / 'daily' / 'bonds_daily.xlsx'
    try:
        expected = recorded_source_hash(snapshot.manifest.get('source_sha256', {}), path)
        if publication is not None:
            binding = publication['source_bindings']['daily_workbook']
            expected = snapshot.manifest.get('source_sha256', {}).get(binding['source_key'])
            if expected != binding['sha256']:
                raise DataUnavailable('Invalid portable workbook binding')
        if not expected:
            raise DataUnavailable('Не сохранена привязка дневной книги к этому анализу')
        return _book_file(str(path.resolve()),expected,snapshot.manifest['analysis_date'],file_stamp(path)), ''
    except (OSError, ValueError, KeyError, BadZipFile) as error:
        return pd.DataFrame(), str(error)


def load_curve(day, daily=DAILY):
    root = Path(daily)
    paths = [root/'run_manifest.json', root/'_support/cbr_zcyc_status.json', root/'cbr_zcyc.csv']
    try:
        if (root/'.daily_write.lock').exists():
            raise DataUnavailable('КБД обновляется')
        raw = [path.read_bytes() for path in paths]
        if (root/'.daily_write.lock').exists() or raw != [path.read_bytes() for path in paths]:
            raise DataUnavailable('КБД изменилась во время чтения')
        daily_manifest, status = map(_json, raw[:2])
        embedded = daily_manifest.get('cbr_zcyc', {})
        if daily_manifest.get('collection_status') != 'ok' or daily_manifest.get('trade_date') != day or any(not isinstance(item,dict) or item.get('status') != 'ok' or item.get('trade_date') != day for item in (status, embedded)):
            raise DataUnavailable('КБД за выбранную дату не сохранена')
        frame = pd.read_csv(BytesIO(raw[2])).rename(columns={'Дата данных':'date', 'Срок до погашения, лет':'maturity', 'Доходность, % годовых':'yield_pct'})
        terms = [.25,.5,.75,1.,2.,3.,5.,7.,10.,15.,20.,30.]
        if not {'date','maturity','yield_pct'}.issubset(frame) or len(frame) != 12 or not frame.date.eq(day).all():
            raise DataUnavailable('Неполная кривая ЦБ')
        frame[['maturity','yield_pct']] = frame[['maturity','yield_pct']].apply(pd.to_numeric, errors='coerce')
        if not np.isfinite(frame[['maturity','yield_pct']]).all().all() or sorted(frame.maturity) != terms:
            raise DataUnavailable('Неверные сроки или значения КБД')
        return frame.sort_values('maturity'), ''
    except (OSError, ValueError, KeyError) as error:
        return pd.DataFrame(), str(error)


def issuer_keys(values, manifest):
    config = manifest.get('config', {})
    if not isinstance(config.get('issuer_aliases'), dict):
        raise DataUnavailable('В снимке не сохранены правила объединения эмитентов')
    aliases = {text_key(k):text_key(v) for k,v in config['issuer_aliases'].items()}
    def canonical(value):
        key, seen = text_key(value), set()
        while key in aliases:
            if key in seen:
                raise DataUnavailable('Цикл в сохранённых названиях эмитентов')
            seen.add(key)
            key = aliases[key]
        return key
    return values.map(canonical)


def candidate_view(snapshot, window, enrichment=None):
    if window not in snapshot.manifest.get('config', {}).get('windows', []):
        raise DataUnavailable('Это окно не сохранено в выбранном снимке')
    frame = snapshot.tables['summary'].copy()
    metrics = snapshot.tables['metrics']
    criteria = {}
    for kind in DIRECTIONS:
        rows = metrics[metrics.comparison.eq(kind) & metrics.window.eq(window)].set_index('SECID')
        frame[kind+'_delta'] = frame.SECID.map(rows.excess_bps)
        if snapshot.manifest.get('result_schema_version', 1) == 3:
            basis = frame.SECID.map(rows.anomaly_basis)
            criteria[kind] = basis.map(ANOMALY_BASES).fillna('Проверка неполная')
        else:
            criteria[kind] = pd.Series('Не сохранено', index=frame.index)
        if kind == 'self':
            frame['z_bps'] = frame.SECID.map(rows.current_metric_bps)
            frame['change_1d_bps'] = frame.SECID.map(rows.get('change_1d_bps', pd.Series(dtype=float)))
    criteria = pd.DataFrame(criteria, index=frame.index)
    frame['anomaly_label'] = criteria.apply(lambda row: row.iloc[0] if row.nunique() == 1 else
        '; '.join(f"{'К себе' if kind == 'self' else KINDS[kind]}: {value}" for kind, value in row.items()), axis=1)
    status = f'candidate__{window}__status'
    frame['candidate_status'] = 'not_saved' if snapshot.legacy or status not in frame else frame[status].fillna('not_saved')
    frame['status_label'] = frame.candidate_status.map(STATUSES).fillna(STATUSES['not_saved'])
    def explanation(row):
        basis = row.get(f'candidate__{window}__basis')
        incomplete = row.get(f'candidate__{window}__incomplete_directions')
        text = []
        for value, prefix in [(basis,'По '),(incomplete,'Нет полного вывода по ')]:
            if pd.notna(value):
                text.extend(prefix+{'peers':'аналогам','issuer':'эмитенту'}.get(k,k)
                            for k in str(value).split(';') if k)
        return ' · '.join(text)
    frame['basis_label'] = frame.apply(explanation,axis=1)
    history = snapshot.tables['comparison_history']
    history = history[history.comparison.eq('self') & history.observation_date.le(snapshot.manifest['analysis_date'])]
    spark = {secid:[None if pd.isna(v) else float(v) for v in rows.sort_values('observation_date').tail(30).z_bps]
             for secid,rows in history.groupby('SECID',sort=False)}
    frame['z_history_30'] = frame.SECID.map(spark).map(lambda values:values if isinstance(values,list) else [])
    frame['z_sparkline'] = frame.z_history_30.map(spread_sparkline)
    for name in ('YTM_WAPRICE','MATDATE','COUPONPERCENT','COUPONVALUE','WAPRICE','ACCINT'):
        frame[name] = pd.NaT if name == 'MATDATE' else np.nan
    if enrichment is not None and not enrichment.empty:
        right = enrichment.copy()
        left_keys = ['SECID','BOARDID']
        right = right[right.TRADEDATE.eq(snapshot.manifest['analysis_date'])].set_index(left_keys)
        index = pd.MultiIndex.from_frame(frame[left_keys])
        for name in ('YTM_WAPRICE','MATDATE','COUPONPERCENT','COUPONVALUE','WAPRICE','ACCINT'):
            if name in right:
                frame[name] = right[name].reindex(index).to_numpy()
    frame['MATDATE'] = pd.to_datetime(frame.MATDATE, errors='coerce')
    return frame


def filter_view(frame, filters):
    mask = pd.Series(True, index=frame.index)
    query = str(filters.get('search', '')).strip()
    if query:
        text = frame[['NAME','SECID','ISSUER']].fillna('').astype(str).agg(' '.join, axis=1)
        mask &= text.str.contains(query, case=False, regex=False)
    for field, key in [('candidate_status','status'),('RATING','rating'),('SECTOR','sector'),('ISSUER','issuer')]:
        values = filters.get(key)
        if values is not None:
            mask &= frame[field].fillna(MISSING).isin(values)
    bounds = filters.get('duration')
    if bounds is not None:
        numeric = pd.to_numeric(frame.DURATION_YEARS, errors='coerce')
        mask &= numeric.between(*bounds) | (numeric.isna() & filters.get('include_missing_duration', True))
    minimum = filters.get('min_liquidity', 0) or 0
    if minimum > 0:
        mask &= pd.to_numeric(frame.adv20, errors='coerce').ge(minimum)
    return frame.loc[mask].copy().reset_index(drop=True)


def sort_view(frame, field='NAME', ascending=True):
    field = field if field in frame else 'NAME'
    fields = [field] if field == 'SECID' else [field,'SECID']
    order = [ascending] if field == 'SECID' else [ascending,True]
    return frame.sort_values(fields, ascending=order, kind='stable', na_position='last').reset_index(drop=True)


def browsing_views(full, filters, signals=True):
    background = filter_view(full,{k:v for k,v in filters.items() if k != 'status'})
    table = (background[background.candidate_status.eq('candidate')].copy() if signals
             else filter_view(background,{'status':filters.get('status')}))
    return background, table.reset_index(drop=True)


def active_filter_count(filters, bounds, signals=True):
    count = int(bool(str(filters.get('search') or '').strip()))
    count += sum(bool(filters.get(k)) for k in ('rating','sector','issuer'))
    count += int(not signals and bool(filters.get('status')))
    count += int((filters.get('min_liquidity') or 0) > 0)
    count += int((bounds is not None and tuple(filters.get('duration') or bounds) != tuple(bounds))
                 or not filters.get('include_missing_duration',True))
    return count


def shortlist_notice(full, table):
    if not full.candidate_status.ne('not_saved').any():
        return 'Статус отбора не сохранён'
    if not full.candidate_status.eq('candidate').any():
        return 'Среди бумаг с сохранённым статусом выделенных нет'
    return 'Нет выделенных бумаг по выбранным фильтрам' if table.empty else ''


def headline_values(snapshot, secid, basket, window):
    metrics = snapshot.tables['metrics']
    rows = metrics[metrics.SECID.eq(secid) & metrics.window.eq(window)].set_index('comparison')
    own, relative = rows.loc['self'], rows.loc[basket]
    return (own.current_metric_bps,relative.get('benchmark_bps',np.nan),
            relative.current_metric_bps,relative.excess_bps)


def selected_identity(frame, positions):
    if not positions or not 0 <= positions[0] < len(frame):
        return None
    row = frame.iloc[positions[0]]
    return str(row.SECID), str(row.BOARDID)


def history_view(snapshot, secid, basket, window):
    metrics = snapshot.tables['metrics']
    rows = metrics[metrics.SECID.eq(secid) & metrics.window.eq(window)].set_index('comparison')
    own, premium = rows.loc['self'], rows.loc[basket]
    history = snapshot.tables['comparison_history']
    history = history[history.SECID.eq(secid)]
    start, end = own.window_start, snapshot.manifest['analysis_date']
    history = history[history.observation_date.between(start, own.window_end) | history.observation_date.eq(end)]
    upper = history[history.comparison.eq('self')][['observation_date','z_bps','reason']].rename(columns={'z_bps':'own_z','reason':'own_reason'})
    lower = history[history.comparison.eq(basket)][['observation_date','metric_bps','benchmark_bps','coverage','reason']].rename(columns={'metric_bps':'premium','reason':'premium_reason'})
    frame = upper.merge(lower, on='observation_date', how='outer', validate='one_to_one').sort_values('observation_date').reset_index(drop=True)
    return frame, own, premium


LABELS = {
    'NAME':'Бумага', 'status_label':'Статус', 'RATING':'Рейтинг', 'DURATION_YEARS':'Дюрация, лет',
    'z_bps':'Z-спред, б.п.', 'self_delta':'К себе: Δ, б.п.', 'peers_delta':'К аналогам: Δ, б.п.',
    'market_delta':'К рынку: Δ, б.п.', 'issuer_delta':'К эмитенту: Δ, б.п.',
    'adv20':'Оборот 20 дат, ₽', 'change_1d_bps':'ΔZ за день, б.п.', 'YTM_WAPRICE':'YTM, %',
    'MATDATE':'Погашение', 'COUPONPERCENT':'Купон, %', 'COUPONVALUE':'Купон на бумагу, ₽',
    'SECID':'SECID', 'ISSUER':'Эмитент', 'SECTOR':'Сектор', 'WAPRICE':'Цена, % номинала',
    'VALUE_RUB':'Оборот за день, ₽', 'ACCINT':'НКД, ₽','basis_label':'Основание','z_sparkline':'Z-спред · 30 дат',
    'anomaly_label':'Критерий аномальности'}
COMMON = ['NAME','anomaly_label','status_label','RATING','DURATION_YEARS','z_bps','z_sparkline','self_delta','peers_delta','market_delta','issuer_delta','basis_label']
MODE_COLUMNS = {'Торговля':['adv20','change_1d_bps'], 'Инвестирование':['YTM_WAPRICE','MATDATE','COUPONPERCENT']}
REASONS = {
    'missing_zspreadatwaprice':'Нет подтверждённого Z-спреда по средневзвешенной цене',
    'missing_confirmed_target_spread':'Нет подтверждённого спреда бумаги',
    'no_eligible_analogs':'Нет подходящих аналогов', 'insufficient_history':'Недостаточно истории',
    'insufficient_basket_weight_coverage':'Недостаточное покрытие весов корзины',
    'calendar_not_confirmed_through_analysis_date':'Календарь не подтверждён до даты анализа',
    'previous_trading_date_unavailable':'Нет подтверждённого значения за предыдущую торговую дату',
    'current_metric_unavailable':'Текущий показатель недоступен', 'zero_mad':'Исторический MAD равен нулю',
    'self_excluded':'Сама выбранная бумага', 'different_issuer':'Другой эмитент',
    'same_issuer':'Тот же эмитент', 'duration_outside_limit':'Разница дюрации выше ограничения',
    'rating_outside_limit':'Разница рейтинга выше ограничения', 'outside_nearest_limit':'Вне числа ближайших',
    'less_near_issue_of_same_issuer':'Выбран более близкий выпуск этого эмитента',
    'missing_rating':'Нет рейтинга', 'missing_duration':'Нет дюрации', 'missing_issuer':'Нет эмитента',
    'structured_product':'Структурный продукт', 'no_trades':'Нет сделок за дату',
    'current_reference_not_point_in_time':'Использована справка на дату загрузки',
    'basket_composition_differs_from_current':'В отдельные даты доступна только часть текущего состава корзины',
    'calendar_shorter_than_window':'Доступный календарь короче выбранного окна'}


def reason_text(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return ''
    return '; '.join(REASONS.get(code, code) for code in str(value).split(';') if code)


def assessment_details(snapshot, secid, window, basket):
    metrics = snapshot.tables['metrics']
    selected = metrics[metrics.SECID.eq(secid) & metrics.window.eq(window)]
    records = []
    criteria = {}
    has_robust = snapshot.manifest.get('result_schema_version', 1) == 3
    for kind in DIRECTIONS:
        row = selected[selected.comparison.eq(kind)].iloc[0]
        label = 'Собственная история' if kind == 'self' else KINDS[kind]
        def flag_text(value):
            return 'Недоступно' if pd.isna(value) else 'Да' if value else 'Нет'
        criteria[label] = {'Процентильная аномалия': flag_text(row.get('upper_tail_anomaly')),
            'Аномалия по robust score': flag_text(row.get('robust_anomaly')) if has_robust else 'Не сохранено',
            'Основание аномальности': (ANOMALY_BASES.get(row.get('anomaly_basis'), 'Проверка неполная')
                                      if has_robust else 'Не сохранено')}
        explanation = '; '.join(filter(None, [ASSESSMENTS.get(row.get('assessment_status'), 'Оценка не сохранена'),
            reason_text(row.get('current_reason')), reason_text(row.get('hist_reason')), reason_text(row.get('scale_reason'))]))
        records.append({'Сравнение': label, 'Показатель': 'Аномальность', 'Пояснение': explanation})
    for kind in ('self',basket):
        row = selected[selected.comparison.eq(kind)].iloc[0]
        for field,label in [('current_reason','Текущее значение'),('hist_reason','История'),
                            ('change_1d_reason','Изменение за день'),('quality_flags','Качество истории'),
                            ('scale_reason','Исторический масштаб')]:
            explanation = reason_text(row.get(field))
            if explanation:
                records.append({'Сравнение':'Собственная история' if kind == 'self' else KINDS[kind],
                                'Показатель':label,'Пояснение':explanation})
        quantities = [('current_metric_bps','Текущий Z-спред бумаги')] if kind == 'self' else [
            ('benchmark_bps','Спред корзины'),('current_metric_bps','Текущая премия'),('excess_bps','Отклонение от медианы')]
        for field,label in quantities:
            if pd.isna(row.get(field)):
                explanation = (reason_text(row.get('hist_reason')) if field == 'excess_bps' else '') or reason_text(row.get('current_reason'))
                fallback = {'benchmark_bps':'Спред корзины не сохранён','excess_bps':'Отклонение от медианы не сохранено'}
                records.append({'Сравнение':'Собственная история' if kind == 'self' else KINDS[kind],
                                'Показатель':label,'Пояснение':explanation or fallback.get(field,label+' не сохранён' if kind == 'self' else 'Текущая премия не сохранена')})
    details = pd.DataFrame(records, columns=['Сравнение','Показатель','Пояснение'])
    for field in ('Процентильная аномалия', 'Аномалия по robust score', 'Основание аномальности'):
        details[field] = details.apply(lambda r: criteria[r['Сравнение']][field] if r['Показатель'] == 'Аномальность' else '', axis=1)
    return details


def number(value, digits=1, signed=False):
    try:
        value = float(value)
        if not math.isfinite(value):
            return '—'
        if round(value,digits) == 0: value = 0.
        return format(value, ('+' if signed and value != 0 else '')+',.'+str(digits)+'f').replace(',', '\u2009').replace('.', ',').replace('-', '−')
    except (TypeError, ValueError):
        return '—'


def turnover(value):
    if pd.isna(value): return '—'
    for scale,label in ((1e9,'млрд ₽'),(1e6,'млн ₽'),(1e3,'тыс. ₽')):
        if abs(value) >= scale: return number(value/scale,1)+' '+label
    return number(value,0)+' ₽'


def compact_label(value):
    if not isinstance(value,str): return '—'
    if ': ' in value and '; ' in value:
        values = [v.split(': ',1)[-1] for v in value.split('; ')]
        assessed = set(values)-{'Проверка неполная','Не сохранено'}
        return (compact_label(next(iter(assessed))) if len(assessed)==1 else 'Разные')+('*' if len(assessed)<len(set(values)) else '')
    return {'Оба критерия':'Оба', 'Только robust score':'Robust', 'Только процентиль':'Процентиль',
            'Выделен анализом':'Выделен', 'Недостаточно данных для вывода':'Нет вывода',
            'Сигнал не выявлен':'Нет сигнала','Только процентиль':'Процент.'}.get(value,value)


def style_table(frame):
    numeric = [c for c in frame if pd.api.types.is_numeric_dtype(frame[c])]
    def delta(c): return c.endswith('_delta') or c in ('change_1d_bps',)
    def rounded(v,c):
        text = number(v,3,delta(c))
        if ',' in text:
            text = text.rstrip('0')
            if text.endswith(','): text += '0'
        return text
    formats = {c:(lambda v,c=c:rounded(v,c)) for c in numeric}
    if 'basis_label' in frame:
        formats['basis_label'] = lambda v:v.replace('По аналогам','Аналоги').replace('По эмитенту','Эмитент').replace('Нет полного вывода по эмитенту','эмитент: нет вывода').replace('Нет полного вывода по аналогам','аналоги: нет вывода')
    for c in ('adv20','VALUE_RUB','member_adv20','Оборот 20 дат, ₽'):
        if c in frame: formats[c] = turnover
    for c in ('anomaly_label','status_label'):
        if c in frame: formats[c] = compact_label
    if 'basis_label' in frame:
        formats['basis_label'] = lambda v:' + '.join(k for text,k in [('По аналогам','Аналоги'),('По эмитенту','Эмитент')] if text in v) or 'Нет вывода'
    styled = frame.style.format(formats,na_rep='—')
    for c in numeric:
        if delta(c):
            styled = styled.map(lambda v:'color: #7CB99A' if pd.notna(v) and v>0 else ('color: #D18A8A' if pd.notna(v) and v<0 else ''),subset=[c])
    return styled


def ru_axis(digits=0):
    return "replace(replace(replace(format(datum.value, ',."+str(digits)+"f'), /,/g, '\u2009'), '.', ','), '-', '−')"


def table_height(count, selected=False):
    return 35*9+2


def spread_sparkline(values):
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite: return None
    lo,hi = min(finite),max(finite)
    segments, points, current = [],[],[]
    for index,value in enumerate(values):
        if value is None or not math.isfinite(float(value)):
            if current: segments.append(current); current=[]
            continue
        x = 3+114*index/max(1,len(values)-1)
        y = 14 if hi == lo else 25-22*(float(value)-lo)/(hi-lo)
        point = f'{x:.2f},{y:.2f}'
        current.append(point)
        points.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1"/>')
    if current: segments.append(current)
    paths = ''.join('<path d="M'+' L'.join(segment)+'" fill="none"/>' for segment in segments)
    color = '#7CB99A' if len(finite)>1 and finite[-1]>finite[-2] else '#D18A8A' if len(finite)>1 and finite[-1]<finite[-2] else '#A8B3C2'
    svg = '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="28" viewBox="0 0 120 28"><g stroke="'+color+'" fill="'+color+'" stroke-width="1.4">'+paths+''.join(points)+'</g></svg>'
    return 'data:image/svg+xml;base64,'+base64.b64encode(svg.encode()).decode()


def _records(frame):
    return json.loads(frame.replace([np.inf,-np.inf], np.nan).to_json(orient='records', date_format='iso'))


def _milliseconds(value):
    if isinstance(value, (int,float,np.number)):
        return float(value) if math.isfinite(float(value)) else None
    try:
        return float(pd.Timestamp(value, tz='UTC').timestamp()*1000)
    except (TypeError, ValueError, OverflowError):
        return None


def history_domain(state, context, full):
    if state.get('history_context') != context:
        state['history_context'] = context
        state['history_domain'] = list(full)
        state['history_revision'] = state.get('history_revision', 0)+1
    return state.get('history_domain', list(full))


def capture_zoom(state, context, selection, full):
    if state.get('history_context') != context:
        return
    zoom = selection.get('zoom_x', {})
    if not zoom:
        domain = list(full)
    else:
        values = zoom.get('observation_date', zoom.get('x'))
        if not isinstance(values, (list,tuple)) or len(values) != 2:
            return
        domain = [_milliseconds(v) for v in values]
        if any(v is None for v in domain) or domain[0] >= domain[1]:
            return
    if domain != state.get('history_domain'):
        state['history_domain'] = domain
        state['history_revision'] = state.get('history_revision', 0)+1


def _stat_line(row):
    return (f"Сейчас {number(row.get('current_metric_bps'))} · медиана {number(row.get('hist_median_bps'))} · "
            f"Δ {number(row.get('excess_bps'), signed=True)} б.п. · наблюдений {number(row.get('n_obs'),0)} · "
            + ASSESSMENTS.get(row.get('assessment_status'), 'Оценка не сохранена'))


def build_history_spec(frame, own, premium, day, basket, domain=None):
    records = _records(frame)
    for row in records:
        row.update(own_text=number(row['own_z']), premium_text=number(row['premium'], signed=True),
                   basket_text=number(row['benchmark_bps']), is_current=row['observation_date'] == day,
                   coverage_text=number(100*row['coverage'],0)+'%' if row.get('coverage') is not None else '—',
                   explanation=reason_text(row.get('premium_reason')) or reason_text(row.get('own_reason')))
    x = {'field':'observation_date','type':'temporal','title':None,
         'axis':{'format':'%d.%m','labelAngle':0,'tickCount':{'interval':'month','step':1},'labelOverlap':True}, 'scale':{'nice':False,'padding':20}}
    encoded_domain = [pd.to_datetime(value,unit='ms',utc=True).isoformat() for value in domain] if domain else None
    if encoded_domain:
        x['scale']['domain'] = encoded_domain
    tooltip = [{'field':'observation_date','type':'temporal','title':'Дата','format':'%d.%m.%Y'},
        {'field':'own_text','title':'Z-спред бумаги, б.п.'}, {'field':'basket_text','title':'Спред корзины, б.п.'},
        {'field':'premium_text','title':'Премия, б.п.'}, {'field':'coverage_text','title':'Покрытие весов'},
        {'field':'explanation','title':'Примечание'}]
    def panel(field, title, row, prefix):
        panel_x = {**x,'axis':{**x['axis'],'labels':field == 'premium','ticks':field == 'premium'}}
        y = {'field':field,'type':'quantitative','title':title,'scale':{'zero':False,'padding':30},'axis':{'labelExpr':ru_axis()}}
        layers = [
            {'mark':{'type':'line','color':'#60A5FA','strokeWidth':2,'invalid':'break-paths-show-domains'},
             'encoding':{'x':panel_x,'y':y}},
            {'transform':[{'filter':f'datum.is_current && isValid(datum.{field})'}],
             'mark':{'type':'point','filled':True,'color':'#60A5FA','size':75},'encoding':{'x':panel_x,'y':y}},
            {'transform':[{'filter':f'datum.is_current && isValid(datum.{field})'}],
             'mark':{'type':'text','align':'right','dx':-8,'dy':-12,'fontSize':12,'fontWeight':600,'color':'#E6EDF3'},
             'encoding':{'x':panel_x,'y':y,'text':{'field':'own_text' if field=='own_z' else 'premium_text'}}},
            {'name':prefix+'_hit','mark':{'type':'point','opacity':0,'aria':False},'encoding':{'x':panel_x,'y':{'value':0},'tooltip':tooltip}},
            {'transform':[{'filter':{'param':'hover_date','empty':False}}],
             'mark':{'type':'rule','color':'#A8B3C2'},'encoding':{'x':panel_x,'tooltip':tooltip}}]
        finite = [r[field] for r in records if r.get(field) is not None]
        if finite:
            layers.insert(0,{'mark':{'type':'area','color':'#60A5FA','opacity':.07,'invalid':'break-paths-show-domains'},
                           'encoding':{'x':panel_x,'y':y,'y2':{'datum':min(finite)}}})
        median = row.get('hist_median_bps')
        if number(median) != '—':
            median_data = {'values':[{'median':float(median),'label':'Медиана '+number(median)}]}
            layers.insert(1, {'data':median_data,
                'mark':{'type':'rule','strokeDash':[5,4],'color':'#A8B3C2'},
                'encoding':{'y':{'field':'median','type':'quantitative'},'tooltip':[{'field':'label','title':'Ориентир, б.п.'}]}})
            layers.append({'data':median_data,'mark':{'type':'text','align':'left','dy':-8,'color':'#A8B3C2','fontSize':11},
                           'encoding':{'x':{'value':8},'y':{'field':'median','type':'quantitative'},'text':{'field':'label'}}})
        if field == 'premium':
            layers.insert(0, {'data':{'values':[{'zero':0}]},'mark':{'type':'rule','color':'#455468'},
                'encoding':{'y':{'field':'zero','type':'quantitative'}}})
        if not any(r.get(field) is not None for r in records):
            layers.append({'data':{'values':[{'message':'Нет подтверждённых значений'}]},
                'mark':{'type':'text','color':'#A8B3C2','fontSize':13},'encoding':{'text':{'field':'message'}}})
        return {'height':210,'title':{'text':title,'subtitle':_stat_line(row),'anchor':'start'},'layer':layers}
    zoom = {'name':'zoom_x','select':{'type':'interval','encodings':['x']},
            'bind':'scales','views':['own_hit','premium_hit']}
    if encoded_domain:
        zoom['value'] = {'x':encoded_domain}
    return {'$schema':'https://vega.github.io/schema/vega-lite/v6.json',
        'data':{'values':records},
        'params':[zoom, {'name':'hover_date','select':{'type':'point','fields':['observation_date'],
            'on':'pointermove','nearest':True,'clear':'pointerout'},'views':['own_hit','premium_hit']}],
        'vconcat':[panel('own_z','Z-спред бумаги, б.п.',own,'own'),
                   panel('premium','Премия: '+KINDS[basket].lower()+', б.п.',premium,'premium')],
        'resolve':{'scale':{'x':'shared','y':'independent'}}, 'spacing':18,
        'config':DARK_CHART}


def scatter_spec(view, full, snapshot, identity, scope='Выборка', show_all=False):
    selected = None if identity is None else full[full.SECID.eq(identity[0]) & full.BOARDID.eq(identity[1])]
    selected = None if selected is None or selected.empty else selected.iloc[0]
    members = snapshot.tables['basket_members']
    peer_ids = set() if selected is None else set(members.loc[members.SECID.eq(selected.SECID) & members.comparison.eq('peers'), 'member_SECID'])
    issuer_ids, grouping_reason = set(), ''
    if selected is not None:
        try:
            canonical = issuer_keys(full.ISSUER, snapshot.manifest)
            key = issuer_keys(pd.Series([selected.ISSUER]), snapshot.manifest).iloc[0]
            if not key:
                raise DataUnavailable('Для выбранной бумаги не указан эмитент')
            issuer_ids = set(full.loc[canonical.eq(key),'SECID'])
        except DataUnavailable as error:
            grouping_reason = str(error)
    if scope == 'Эмитент':
        plotted = full[full.SECID.isin(issuer_ids)].copy()
    else:
        overlays = peer_ids | issuer_ids | ({selected.SECID} if selected is not None else set())
        plotted = pd.concat([view,full[full.SECID.isin(overlays)]],ignore_index=True).drop_duplicates(['SECID','BOARDID']).copy()
    role, roles = [], []
    for row in plotted.itertuples():
        all_roles = []
        if identity == (row.SECID,row.BOARDID): all_roles.append('Выбранная бумага')
        if row.SECID in issuer_ids: all_roles.append('Выпуски эмитента')
        if row.SECID in peer_ids: all_roles.append('Аналоги')
        role.append(all_roles[0] if all_roles else 'Выборка')
        roles.append(', '.join(all_roles) or 'Выборка')
    plotted['role'], plotted['roles'] = role, roles
    valid = np.isfinite(pd.to_numeric(plotted.DURATION_YEARS, errors='coerce')) & np.isfinite(pd.to_numeric(plotted.z_bps, errors='coerce'))
    missing = int((~valid).sum())
    plotted = plotted[valid]
    plotted = plotted.copy()
    plotted['duration_text'] = plotted.DURATION_YEARS.map(lambda v:number(v,2))
    plotted['spread_text'] = plotted.z_bps.map(number)
    plotted['turnover_text'] = plotted.adv20.map(turnover)
    plotted['selected_order'] = plotted.role.eq('Выбранная бумага').astype(int)
    scale = {'zero':False,'padding':20}
    if not show_all and len(plotted)>2:
        lo,hi = plotted.z_bps.quantile([.01,.99]).tolist()
        if lo < hi: scale.update(domain=[lo,hi],nice=False)
    legend = {'orient':'top','direction':'horizontal','columns':4,'labelLimit':200,'symbolSize':90}
    ticks = np.arange(0,math.ceil(float(plotted.DURATION_YEARS.max())*2)/2+.5,.5).tolist() if not plotted.empty else [0,.5,1.]
    spec = {'data':{'values':_records(plotted[['SECID','BOARDID','NAME','ISSUER','RATING','SECTOR','DURATION_YEARS','z_bps','adv20','status_label','role','roles','duration_text','spread_text','turnover_text','selected_order']])},
        'height':350, 'mark':{'type':'point','filled':True,'size':85,'clip':True},
        'params':[{'name':'bond_pick','select':{'type':'point','fields':['SECID','BOARDID'],'toggle':False,'clear':False},'views':['market_points']}],
        'encoding':{'x':{'field':'DURATION_YEARS','type':'quantitative','title':'Дюрация, лет','axis':{'values':ticks,'labelExpr':ru_axis(1)}},
            'y':{'field':'z_bps','type':'quantitative','title':'Z-спред, б.п.','scale':scale,'axis':{'labelExpr':ru_axis()}},
            'color':{'field':'role','type':'nominal','title':None,'legend':legend,
                'scale':{'domain':['Выбранная бумага','Выпуски эмитента','Аналоги','Выборка'],
                         'range':['#60A5FA','#C4B5FD','#22D3EE','#69788D']}},
            'shape':{'field':'role','type':'nominal','title':None,'legend':legend,
                'scale':{'domain':['Выбранная бумага','Выпуски эмитента','Аналоги','Выборка'],
                         'range':['diamond','square','triangle-up','circle']}},
            'fillOpacity':{'condition':{'test':"datum.role === 'Аналоги'",'value':0},'value':1},
            'stroke':{'condition':[{'test':"datum.role === 'Выбранная бумага'",'value':'#60A5FA'},
                                   {'test':"datum.role === 'Выпуски эмитента'",'value':'#C4B5FD'},
                                   {'test':"datum.role === 'Аналоги'",'value':'#22D3EE'}],'value':'#69788D'},
            'tooltip':[{'field':f,'title':LABELS.get(f,f)} for f in ['NAME','SECID','ISSUER','RATING','SECTOR']]+[
                {'field':'duration_text','title':'Дюрация, лет'},{'field':'spread_text','title':'Z-спред, б.п.'},
                {'field':'turnover_text','title':'Оборот 20 дат, ₽'},{'field':'status_label','title':'Статус'},{'field':'roles','title':'Роль'}]},
        'config':DARK_CHART}
    spec['encoding'].update(size={'condition':{'test':"datum.role === 'Выбранная бумага'",'value':220},'value':75},
        opacity={'condition':{'test':"datum.role === 'Выбранная бумага'",'value':1},'value':.45},
        order={'field':'selected_order','type':'quantitative'})
    spec['layer'] = [{'name':'market_points','mark':spec.pop('mark')},
        {'transform':[{'filter':"datum.role === 'Выбранная бумага'"}],
         'mark':{'type':'text','align':{'expr':f"datum.DURATION_YEARS > {max(ticks)/2} ? 'right' : 'left'"},
                 'dx':{'expr':f"datum.DURATION_YEARS > {max(ticks)/2} ? -10 : 10"},
                 'dy':-12,'clip':False,'fontSize':12,'fontWeight':600},
         'encoding':{'text':{'field':'NAME'},'color':{'value':'#E6EDF3'},'opacity':{'value':1},'size':{'value':12}}}]
    return spec, missing, grouping_reason


def _available_dates(root):
    dates = []
    for folder in (Path(root)/'snapshots').glob('????-??-??'):
        try:
            marker = _json((folder/'run_manifest.json').read_bytes())
            if marker.get('status') == 'ok' and marker.get('analysis_date') == folder.name:
                dates.append(folder.name)
        except (OSError,ValueError):
            continue
    return sorted(dates, reverse=True)


def _selection(event):
    return event.get('selection', {}) if event else {}


def render_app(analysis=ANALYSIS, daily=DAILY, publication=None):
    import streamlit as st
    st.set_page_config(page_title='Анализ облигаций', page_icon='📈', layout='wide')
    # Composed Vega views add axes outside their data width; preserve the SVG viewBox.
    st.html('<style>@font-face {font-family:Inter;src:url("app/static/fonts/Inter.ttf");font-weight:100 900;font-feature-settings:"tnum";}'
            '[data-testid="stVegaLiteChart"] svg {max-width:100%;height:auto;}'
            '[data-testid="stMain"] {font-variant-numeric:tabular-nums;}'
            '[data-testid="stMainBlockContainer"] {padding-top:4rem;max-width:1600px;margin-inline:auto;}'
            '[data-testid="stMetricValue"] {font-size:1.35rem;}'
            '#bond-detail {display:block;scroll-margin-top:5rem;}'
            '.vg-tooltip {background:#171F2A!important;color:#E6EDF3!important;border-color:#2B3746!important;}</style>')
    state = st.session_state
    # Keep conditional widget settings when a lazy section is not rendered.
    for key in ('market_scope','scatter_show_all','premium_basket','detail_section'):
        if key in state: state[key] = state[key]
    if not state.get('_url_loaded'):
        state['_url_loaded'] = True
        if st.query_params.get('mode') in MODE_COLUMNS: state['mode'] = st.query_params['mode']
        if st.query_params.get('window') in ('90','150'): state['window'] = int(st.query_params['window'])
        secid,board = st.query_params.get('secid'),st.query_params.get('board')
        if secid and board: state['selected_bond'] = (secid,board)
    if publication is not None and state.get('_publication_id') != publication['analysis_run_id']:
        for key in ('analysis_date', 'archive_date', '_snapshot', '_snapshot_token'):
            state.pop(key, None)
        state['_publication_id'] = publication['analysis_run_id']
    dates = _available_dates(analysis)
    st.subheader('Анализ облигаций')
    mode = st.segmented_control('Режим',list(MODE_COLUMNS),default='Торговля',required=True,key='mode',label_visibility='collapsed')
    if not dates:
        st.info('Нет завершённых сохранённых анализов. Подготовьте данные существующими командами проекта.')
        return
    if state.get('analysis_date') and state.analysis_date not in dates:
        dates.append(state.analysis_date)
    state.setdefault('analysis_date',dates[0])
    state.setdefault('sort_field','peers_delta')
    state.setdefault('sort_ascending',False)
    context_line = st.empty()
    toolbar = st.container(horizontal=True,vertical_alignment='center')
    with toolbar:
        signals = st.segmented_control('Список',['Выделены анализом','Все выпуски'],default='Выделены анализом',required=True,
                                      key='list_scope',label_visibility='collapsed') == 'Выделены анализом'
        filter_slot = st.container(width='content')
        if len(dates)>1:
            parameters = st.popover('Дата анализа',key='analysis_parameters')
        else: parameters = None
    if parameters is not None:
        with parameters:
            def archive_changed():
                state['analysis_date'] = state['archive_date']
            st.selectbox('Дата анализа',dates,index=dates.index(state.analysis_date),key='archive_date',
                         format_func=lambda d:pd.Timestamp(d).strftime('%d.%m.%Y'),on_change=archive_changed)
    day = state['analysis_date']
    try:
        guard_before = _guard(Path(analysis))
        token = hashlib.sha256(guard_before+(Path(analysis)/'snapshots'/day/'run_manifest.json').read_bytes()).hexdigest()
        folder = Path(analysis)/'snapshots'/day
        fingerprint = token + str([(p.name,file_stamp(p)) for p in sorted(folder.glob('*.csv'))])
        snapshot = _shared_snapshot(str(Path(analysis).resolve()), day, fingerprint)
        if _guard(Path(analysis)) != guard_before:
            raise PublicationBusy('Результаты изменились во время чтения')
        state.pop('_snapshot', None)
    except PublicationBusy as error:
        st.warning(str(error))
        return
    except (OSError,ValueError) as error:
        st.error(str(error))
        return
    windows = [w for w in (90,150) if w in snapshot.manifest['config']['windows']]
    with toolbar:
        if state.get('window') not in windows:
            state['window'] = snapshot.manifest['config'].get('classification',{}).get('primary_window', windows[0])
        window = st.segmented_control('Окно, торговых дат',windows,required=True,key='window')
    context_line.caption(f'Данные за {pd.Timestamp(day):%d.%m.%Y} · окно {window} торговых дат')
    enrichment, enrichment_reason = load_enrichment(snapshot, daily, publication=publication)
    book_path = Path(daily)/'daily'/'bonds_daily.xlsx'
    book_stamp = file_stamp(book_path) if book_path.exists() else None
    full = cached_candidates((str(Path(analysis).resolve()),day,fingerprint,str(book_path.resolve()),book_stamp,enrichment_reason),window,snapshot,enrichment)
    options = {}
    unavailable = {}
    for field,key in [('candidate_status','status'),('RATING','rating'),('SECTOR','sector'),('ISSUER','issuer')]:
        values = full[field].fillna(MISSING).unique().tolist()
        if field == 'RATING': values.sort(key=lambda v:RATINGS.index(v) if v in RATINGS else len(RATINGS))
        else: values.sort()
        unavailable[key] = set(state.get('filter_'+key, []))-set(values)
        options[key] = list(dict.fromkeys(values+state.get('filter_'+key, [])))
    duration = pd.to_numeric(full.DURATION_YEARS,errors='coerce').dropna()
    bounds = (float(duration.min()),float(duration.max())) if not duration.empty else None
    def reset_filters():
        state['filter_search'] = ''
        state['filter_min_liquidity'] = 0.
        state['include_missing_duration'] = True
        for key in options: state['filter_'+key] = []
        if bounds: state['filter_duration'] = bounds
    filters = {key:state.get('filter_'+key) for key in ('search','status','rating','sector','issuer','duration','min_liquidity')}
    filters['include_missing_duration'] = state.get('include_missing_duration',True)
    active_count = active_filter_count(filters,bounds,signals)
    with filter_slot.popover(f'Фильтры ({active_count})',key='browsing_filters'):
        filters['search'] = st.text_input('Найти бумагу',key='filter_search',placeholder='Название, SECID, эмитент')
        for key,label in [('status','Статус отбора'),('rating','Рейтинг'),('sector','Сектор'),('issuer','Эмитент')]:
            chosen = st.multiselect(label,options[key],default=[],key='filter_'+key,placeholder='Все',
                help='Пустой список означает все значения.',
                disabled=key == 'status' and signals,
                format_func=(lambda v:STATUSES.get(v,v)) if key == 'status' else str)
            filters[key] = chosen or None
            if unavailable[key]:
                labels = [STATUSES.get(v,v) if key == 'status' else str(v) for v in sorted(unavailable[key])]
                st.caption(label+': нет на выбранную дату — '+', '.join(labels))
        if bounds:
            saved = state.get('filter_duration',bounds)
            lo,hi = min(bounds[0],saved[0]),max(bounds[1],saved[1])
            filters['duration'] = st.slider('Дюрация, лет',lo,hi,value=saved,key='filter_duration') if lo < hi else bounds
        filters['include_missing_duration'] = st.checkbox('Включать без дюрации',value=True,key='include_missing_duration')
        filters['min_liquidity'] = st.number_input('Мин. оборот за 20 дат, ₽',min_value=0.,step=1000000.,key='filter_min_liquidity')
        if filters['min_liquidity'] > 0:
            st.caption('Бумаги без значения оборота исключены.')
        if signals: st.caption('В этом списке показаны только выделенные анализом бумаги; статусный фильтр сохранён для всех выпусков.')
        st.button('Сбросить фильтры',on_click=reset_filters,width='stretch',key='reset_filters')
    background, filtered = browsing_views(full,filters,signals)
    sort_options = [name for name in LABELS if name in full and name != 'z_sparkline']
    sort_summary = st.empty()
    controls = toolbar
    with controls.popover('Сортировка',key='sorting_controls'):
        sort_field = st.selectbox('Сортировать по',sort_options,key='sort_field',format_func=lambda f:LABELS[f])
        sort_ascending = st.radio('Порядок',[True,False],format_func=lambda value:'По возрастанию' if value else 'По убыванию',horizontal=True,key='sort_ascending')
    base = [c for c in COMMON if not (signals and c == 'status_label')]+MODE_COLUMNS[mode]
    with controls.popover('Колонки'):
        profile = st.segmented_control('Профиль',['Минимум','Стандарт','Всё'],default='Стандарт',required=True,key='column_profile')
        extra = st.multiselect('Дополнительные', [c for c in LABELS if c in full and c not in base],
            format_func=lambda f:LABELS[f], key='columns_'+mode)
    if profile == 'Минимум': base = ['NAME','RATING','z_bps','peers_delta']+MODE_COLUMNS[mode][:1]
    elif profile == 'Всё': base = [c for c in LABELS if c in full and not (signals and c == 'status_label')]
    columns = list(dict.fromkeys(base+extra))
    view = sort_view(filtered, sort_field, sort_ascending)
    sort_name = 'Отклонение премии к аналогам' if sort_field == 'peers_delta' else LABELS[sort_field]
    sort_summary.caption(f'Показано {len(view)} из {len(full)} · {sort_name} '+('↑' if sort_ascending else '↓'),
                         help='Отклонение текущей премии к аналогам от её исторической медианы.' if sort_field == 'peers_delta' else None)
    unsaved = int(full.candidate_status.eq('not_saved').sum())
    if 0 < unsaved < len(full):
        st.caption(f'Для {unsaved} из {len(full)} выпусков статус не сохранён')
    cfg = {}
    for field in columns:
        if field == 'z_sparkline':
            cfg[field] = st.column_config.ImageColumn(LABELS[field],width=120,help='Сохранённый Z-спред за последние 30 дат снимка; разрывы означают пропуски, шкала своя для каждой строки.')
        elif field == 'MATDATE': cfg[field] = st.column_config.DateColumn(LABELS[field],format='DD.MM.YYYY')
        elif field == 'anomaly_label':
            cfg[field] = st.column_config.TextColumn('Критерий аномальности', width='medium',
                help='Оба — два критерия; Robust — только robust score; Процент. — только процентиль; Разные — выводы направлений различаются; * — часть направлений недоступна. Полная расшифровка — в карточке и вкладке «Расчёт».')
        elif field in ['NAME','status_label','RATING','SECID','ISSUER','SECTOR','basis_label']:
            cfg[field] = st.column_config.TextColumn(LABELS[field],pinned=field == 'NAME',width='small' if field in ('basis_label','status_label') else None)
        else:
            help_text = f'Отклонение от исторической медианы за {window} торговых дат; положительное значение — выше медианы.' if field.endswith('_delta') else None
            cfg[field] = st.column_config.NumberColumn(LABELS[field],help=help_text)
    identity = state.get('selected_bond')
    selected_rows = [] if identity is None else view.index[view.SECID.eq(identity[0]) & view.BOARDID.eq(identity[1])].tolist()
    signature = str((snapshot.manifest.get('analysis_run_id'),day,window,mode,signals,columns,sort_field,sort_ascending,identity,list(zip(view.SECID,view.BOARDID))))
    table_key = 'candidates_'+hashlib.sha256(signature.encode()).hexdigest()[:16]
    def select_row():
        positions = _selection(state.get(table_key)).get('rows',[])
        if positions == selected_rows[:1]:
            return  # Column-only events retain the current row selection.
        chosen = selected_identity(view,positions)
        if chosen is not None: state['selected_bond'] = chosen
    st.dataframe(style_table(view[columns]),column_config=cfg,hide_index=True,height=table_height(len(view),identity is not None),row_height=35,placeholder='—',
        key=table_key,on_select=select_row,selection_mode='single-row',
        selection_default={'selection':{'rows':selected_rows[:1]}})
    notice = shortlist_notice(full,view) if signals else ('Нет бумаг по выбранным фильтрам' if view.empty else '')
    if notice: st.info(notice)
    if signals and view.empty:
        def show_all(): state['list_scope'] = 'Все выпуски'
        st.button('Все выпуски',on_click=show_all,key='show_all_issues')
    if mode == 'Инвестирование' and enrichment_reason:
        st.caption('Дневные характеристики недоступны: '+enrichment_reason)
    identity = state.get('selected_bond')
    selected = full.iloc[:0] if identity is None else full[full.SECID.eq(identity[0]) & full.BOARDID.eq(identity[1])]
    url_state = {'mode':mode,'window':str(window)}
    if identity: url_state.update(secid=identity[0],board=identity[1])
    for key in ('secid','board','mode','window'):
        if key in url_state:
            if st.query_params.get(key) != url_state[key]: st.query_params[key] = url_state[key]
        elif key in st.query_params: del st.query_params[key]
    st.divider()
    if identity is None:
        st.caption('Выберите облигацию в таблице или на диаграмме, чтобы открыть подробности.')
        leaders = sort_view(full[full.candidate_status.eq('candidate')],'peers_delta',False).head(3)
        st.markdown(f'**Выделено анализом: {int(full.candidate_status.eq("candidate").sum())}** · топ по отклонению премии к аналогам')
        for item in leaders.itertuples():
            st.caption(str(item.NAME)+' · '+number(item.peers_delta,signed=True)+' б.п.')
    elif selected.empty:
        st.subheader(identity[0])
        st.info('Бумаги нет в анализе за выбранную дату')
    else:
        row = selected.iloc[0]
        st.html('<span id="bond-detail"></span>')
        st.subheader(str(row.NAME))
        st.caption(f'{row.SECID} · {row.ISSUER}')
        with st.container(horizontal=True):
            rating = str(row.RATING)
            rating_color = 'blue' if rating.startswith('AA') else 'violet' if rating.startswith('A') else 'orange' if rating.startswith('BBB') else 'red' if rating.startswith(('BB','B','C','D')) else 'gray'
            for badge,color in ((rating,rating_color),(str(row.SECTOR),'gray'),(row.status_label,'blue' if row.candidate_status=='candidate' else 'gray'),(row.basis_label,'gray')):
                if badge: st.badge(badge,color=color)
        if not ((view.SECID == identity[0]) & (view.BOARDID == identity[1])).any():
            st.info('Выбранная бумага вне текущего списка')
        facts = (f'Z-спред: {number(row.z_bps)} б.п. · дюрация: {number(row.DURATION_YEARS,2)} лет · '
                 f'оборот 20 дат: {turnover(row.adv20)} · ΔZ за день: {number(row.change_1d_bps,signed=True)} б.п.') if mode == 'Торговля' else (
                 f'YTM: {number(row.YTM_WAPRICE,2)}% · погашение: '+(pd.Timestamp(row.MATDATE).strftime('%d.%m.%Y') if pd.notna(row.MATDATE) else '—')+
                 f' · купон: {number(row.COUPONPERCENT,2)}% / {number(row.COUPONVALUE,2)} ₽ · Z-спред: {number(row.z_bps)} б.п.')
        with st.popover('Характеристики'):
            st.write('Критерий аномальности: '+row.anomaly_label)
        st.caption(facts)
        basket = st.segmented_control('Премия к корзине',list(KINDS),default='peers',required=True,format_func=lambda k:KINDS[k],key='premium_basket')
        values = headline_values(snapshot,identity[0],basket,window)
        history, own, premium = history_view(snapshot,identity[0],basket,window)
        metric_columns = st.columns(4)
        with st.container():
            for index,label in enumerate(['Z-спред бумаги','Спред корзины','Текущая премия','Медиана премии']):
                with metric_columns[index], st.container(border=True,height=110):
                    if index < 3:
                        daily_delta = row.change_1d_bps if index == 0 else np.nan
                        st.metric(label,number(values[index],signed=index == 2)+' б.п.',
                                  delta=(number(daily_delta,signed=True)+' б.п. за день') if pd.notna(daily_delta) else None,
                                  delta_arrow='up' if pd.notna(daily_delta) and daily_delta>0 else 'down' if pd.notna(daily_delta) and daily_delta<0 else 'off',
                                  delta_color='green' if pd.notna(daily_delta) and daily_delta>0 else 'red' if pd.notna(daily_delta) and daily_delta<0 else 'off',
                                  help=['Сохранённый текущий спред выпуска.','Взвешенный спред выбранной корзины.','Спред бумаги минус спред корзины.'][index])
                    else:
                        delta = values[3]
                        st.metric(label,number(premium.get('hist_median_bps'))+' б.п.',
                                  delta=number(delta,signed=True)+' б.п. к медиане' if pd.notna(delta) else None,
                                  delta_arrow='up' if pd.notna(delta) and delta>0 else ('down' if pd.notna(delta) and delta<0 else 'off'),
                                  delta_color='green' if pd.notna(delta) and delta>0 else ('red' if pd.notna(delta) and delta<0 else 'off'),
                                  help=f'Историческая медиана премии за {window} дат; дельта — отклонение текущей премии.')
        @st.fragment
        def history_fragment():
            context = str((snapshot.manifest.get('analysis_run_id'),day,identity,window))
            full_domain = [_milliseconds(own.window_start), _milliseconds(day)]
            chart_title,chart_action = st.columns([3,1])
            chart_title.markdown('#### История спреда и премии')
            if chart_action.button('Сбросить масштаб',key='reset_history_zoom',help='Вернуть полный диапазон выбранного окна на обеих панелях'):
                state['history_domain'] = list(full_domain)
                state['history_revision'] = state.get('history_revision',0)+1
            domain = history_domain(state,context,full_domain)
            chart_key = 'history_'+hashlib.sha256(str((context,basket,state.get('history_revision'))).encode()).hexdigest()[:16]
            def zoom_changed():
                capture_zoom(state,context,_selection(state.get(chart_key)),full_domain)
            st.vega_lite_chart(build_history_spec(history,own,premium,day,basket,domain),width='stretch',
                key=chart_key,on_select=zoom_changed,selection_mode=['zoom_x'],alt='Сверху история Z-спреда бумаги, снизу её премия к выбранной корзине. Общие даты и масштаб.')
            st.caption('Сохранённые значения · '+pd.Timestamp(day).strftime('%d.%m.%Y')+f' · окно {window} дат')
        history_fragment()
        reason = reason_text(premium.get('current_reason')) or reason_text(premium.get('hist_reason'))
        detail_section = st.segmented_control('Подробности',['Расчёт','Восстановление','Данные'],default='Расчёт',required=True,key='detail_section')
        if detail_section == 'Расчёт':
            st.write('Состав и веса корзины зафиксированы на дату анализа. Масштаб не меняет окно расчёта.')
            if reason: st.write('Сравнение: '+reason)
            details = assessment_details(snapshot,identity[0],window,basket)
            if not details.empty: st.dataframe(details,hide_index=True)
        elif detail_section == 'Восстановление':
            if mode == 'Торговля': _render_recovery(st, snapshot, identity)
            else: st.caption('История восстановления доступна в режиме «Торговля».')
        elif detail_section == 'Данные':
            _render_data(st,snapshot,day,selected,enrichment_reason)
    comparison_section = st.segmented_control('Обзор',['Аналоги','Бумага среди других выпусков','КБД'],default='Аналоги',required=True,key='comparison_section')
    if comparison_section == 'Аналоги':
        _render_members(st,snapshot,identity,window)
    elif comparison_section == 'Бумага среди других выпусков':
        if not selected.empty:
            st.caption('Выбрана: '+str(selected.iloc[0].NAME))
            st.markdown('[↑ К карточке бумаги](#bond-detail)')
        st.caption('Каждая точка — облигация. Правее — больше дюрация, выше — шире текущий Z-спред. Нажмите точку, чтобы открыть выпуск.')
        scope = st.segmented_control('Область диаграммы',['Выборка','Эмитент'],default='Выборка',required=True,key='market_scope')
        show_all = st.toggle('Показать все выбросы',value=False,key='scatter_show_all',help='Без ограничения по квантилям 1–99%. Исходные значения не изменяются.')
        if scope == 'Эмитент' and (identity is None or selected.empty):
            st.info('Выберите бумагу для просмотра эмитента')
        else:
            spec,missing,grouping_reason = scatter_spec(background,full,snapshot,identity,scope,show_all)
            domain = spec['encoding']['y']['scale'].get('domain')
            outside = [r for r in spec['data']['values'] if domain and not domain[0] <= r['z_bps'] <= domain[1]]
            market_key = 'market_'+hashlib.sha256(str((day,window,identity,scope,show_all,list(background.SECID))).encode()).hexdigest()[:16]
            def market_pick():
                picks = _selection(state.get(market_key)).get('bond_pick',[])
                if picks:
                    candidate = (picks[0].get('SECID'),picks[0].get('BOARDID'))
                    if ((full.SECID == candidate[0]) & (full.BOARDID == candidate[1])).any(): state['selected_bond'] = candidate
            st.vega_lite_chart(spec,width='stretch',key=market_key,on_select=market_pick,selection_mode=['bond_pick'],alt='Дюрация и Z-спред загруженной выборки; выберите точку для открытия бумаги.')
            st.caption(f'Загруженная выборка · без координат: {missing} · за границами 1–99%: {len(outside)}' if domain else f'Загруженная выборка · все значения · без координат: {missing}')
            if identity and any((r['SECID'],r['BOARDID']) == identity for r in outside):
                st.info('Выбранная бумага за границами видимой области. Включите «Показать все выбросы».')
            with st.expander('О расчёте диаграммы'):
                st.write('Показана загруженная выборка после фильтров, с выбранной бумагой, аналогами и выпусками эмитента. Высота точки сама по себе не определяет недооценённость.')
                st.write('Премия к рынку берётся из сохранённой взвешенной корзины. Линия медиан по дюрации не является этим показателем.')
                if grouping_reason: st.write(grouping_reason)
            if not selected.empty and (pd.isna(selected.iloc[0].z_bps) or pd.isna(selected.iloc[0].DURATION_YEARS)):
                st.info('Нет данных для точки выбранной бумаги на диаграмме')
    elif comparison_section == 'КБД':
        curve, curve_reason = load_curve(day,daily)
        if curve.empty: st.info(curve_reason or 'КБД за выбранную дату не сохранена')
        else:
            st.subheader('КБД · '+pd.Timestamp(day).strftime('%d.%m.%Y'))
            curve = curve.assign(maturity_text=curve.maturity.map(lambda v:number(v,2)),yield_text=curve.yield_pct.map(lambda v:number(v,2)))
            st.vega_lite_chart({'data':{'values':_records(curve)},'height':300,'config':DARK_CHART,
                'mark':{'type':'line','point':True,'color':'#60A5FA'},
                'encoding':{'x':{'field':'maturity','type':'quantitative','title':'Срок до погашения, лет','axis':{'labelExpr':ru_axis()}},
                    'y':{'field':'yield_pct','type':'quantitative','title':'Доходность, % годовых','scale':{'zero':False},'axis':{'labelExpr':ru_axis(1)}},
                    'tooltip':[{'field':'maturity_text','title':'Срок, лет'},{'field':'yield_text','title':'Доходность, %'},{'field':'date','title':'Дата'}]}},width='stretch')
            st.caption('Точки опубликованы Банком России. Соединяющие отрезки служат ориентиром.')
    if selected.empty:
        with st.expander('Данные и ограничения'):
            _render_data(st,snapshot,day,selected,enrichment_reason)


def _render_data(st,snapshot,day,selected,enrichment_reason):
        for key,value in {'Дата анализа':pd.Timestamp(day).strftime('%d.%m.%Y'),'Расчёт':snapshot.manifest.get('analysis_run_id','Старая схема'),
                  'Версия схемы':snapshot.manifest.get('result_schema_version',1),'Правило':snapshot.manifest.get('classification_rule_version','Не сохранено')}.items():
            st.markdown(f'**{key}:** {value}')
        st.caption('Показаны сохранённые итоги торгов. Рейтинг, сектор и эмитент относятся к справке на дату загрузки.')
        if enrichment_reason: st.caption('Дополнительные характеристики: '+enrichment_reason)
        if not selected.empty:
            for field,label in [('adv_source','Источник оборота'),('adv_last_value_date','Последняя дата оборота'),('adv_n_values','Наблюдений оборота'),('DATA_GAPS','Пропуски данных'),('SMARTLAB_AT','Дата справочника')]:
                st.markdown('**'+label+':** '+str(selected.iloc[0].get(field,'—')))


def _render_members(st, snapshot, identity, window):
    if identity is None:
        st.caption('Выберите облигацию в верхней таблице или на диаграмме, чтобы увидеть состав сравнения.')
        return
    kind = st.session_state.get('premium_basket','peers')
    st.caption('Состав сравнения: '+KINDS[kind])
    frame = snapshot.tables['basket_members']
    members = frame[frame.SECID.eq(identity[0]) & frame.comparison.eq(kind)].copy()
    if members.empty:
        metrics = snapshot.tables['metrics']
        row = metrics[metrics.SECID.eq(identity[0]) & metrics.comparison.eq(kind) & metrics.window.eq(window)]
        st.info(reason_text(row.iloc[0].get('current_reason')) if not row.empty and reason_text(row.iloc[0].get('current_reason')) else 'Состав сравнения не сохранён')
        return
    metrics = snapshot.tables['metrics']
    metric = metrics[metrics.SECID.eq(identity[0]) & metrics.comparison.eq(kind) & metrics.window.eq(window)].iloc[0]
    st.caption(f"Участников: {len(members)} · эмитентов: {number(metric.get('n_issuers'),0)} · эффективное число: {number(metric.get('n_eff'),2)} · спред корзины: {number(metric.get('benchmark_bps'))} б.п.")
    if len(members) == 1: st.caption('Один сопоставимый выпуск')
    summary = snapshot.tables['summary'].set_index('SECID')
    members['NAME'] = members.member_SECID.map(summary.NAME).fillna(members.member_SECID)
    members['member_issuer'] = members.member_SECID.map(summary.ISSUER)
    members['weight_pct'] = members.weight*100
    labels = {'NAME':'Бумага','member_SECID':'SECID','member_issuer':'Эмитент','weight_pct':'Вес, %',
        'member_z_bps':'Z-спред, б.п.','member_rating':'Рейтинг','member_duration':'Дюрация, лет',
        'member_sector':'Сектор','member_adv20':'Оборот 20 дат, ₽'}
    columns = [c for c in labels if c in members]
    event = st.dataframe(style_table(members[columns].rename(columns=labels)),hide_index=True,on_select='rerun',selection_mode='single-row',
                        height=table_height(len(members)),row_height=35,
                        key=f'members_{snapshot.manifest["analysis_date"]}_{identity[0]}_{kind}',placeholder='—')
    picks = _selection(event).get('rows',[])
    if picks:
        member = members.iloc[picks[0]]
        with st.expander('Почему выбран',expanded=True):
            for field,label in [('rating_diff','Разница рейтинга, ступеней'),('duration_diff','Разница дюрации, лет'),
                ('sector_diff','Секторы различаются'),('liquidity_log_ratio','Различие ликвидности'),('distance','Суммарное расстояние')]:
                if field in member:
                    st.write(label+': '+(number(member[field],2) if isinstance(member[field],(int,float,np.number)) else str(member[field])))
    with st.expander('Исключённые из сравнения'):
        exclusions = snapshot.tables['candidate_exclusions']
        if snapshot.exclusions_reason: st.caption(snapshot.exclusions_reason)
        elif exclusions.empty: st.caption('Исключённых выпусков нет')
        else:
            excluded = exclusions[exclusions.SECID.eq(identity[0]) & exclusions.comparison.eq(kind)].copy()
            excluded['Причина'] = excluded.reason.map(reason_text)
            counts = excluded.groupby('Причина',sort=False).size().rename('Выпусков').reset_index()
            st.dataframe(counts,hide_index=True)


def _render_recovery(st, snapshot, identity):
    st.subheader('История восстановления собственного спреда')
    view = recovery_view(snapshot, identity)
    if view['status'] == 'not_saved':
        st.caption(view['message'])
        return
    st.caption(view['criterion'])
    st.write(view['message'])
    row = view['summary']
    labels = ['Известных исходов', 'Восстановилось',
              'Сужение · медиана', 'Срок · медиана', 'Расширение · медиана']
    share = row['recovery__share']
    known = int(row['recovery__n_known'])
    recovered = int(row['recovery__n_recovered'])
    values = [str(known),
        f'{recovered} из {known}',
        number(row['recovery__median_reduction_bps'], signed=True)+' б.п.' if pd.notna(row['recovery__median_reduction_bps']) else '—',
        number(row['recovery__median_days'], 1)+' дней' if pd.notna(row['recovery__median_days']) else '—',
        number(row['recovery__median_worsening_bps'], 1)+' б.п.' if pd.notna(row['recovery__median_worsening_bps']) else '—']
    if known < 5:
        st.caption('Малая выборка: менее 5 известных исходов. Смотрите число восстановлений и общее число случаев.')
    elif pd.notna(share):
        st.caption('Доля восстановлений: '+number(100*share,1)+'%')
    for column,label,value in zip(st.columns(5),labels,values):
        with column, st.container(border=True):
            st.metric(label,value)
    st.caption(f"Срок наблюдения ещё не прошёл: {int(row['recovery__n_pending'])} · "
               f"Неполных наблюдений: {int(row['recovery__n_incomplete'])}")
    reasons = {
        'calendar_unconfirmed':'Торговый календарь не подтверждён на дату анализа',
        'insufficient_reference':'Недостаточно данных для исторического ориентира',
        'zero_mad':'В части окон спред не имел измеримого обычного разброса',
        'missing_current_spread':'В части дат нет подтверждённого спреда',
        'unobserved_onset':'Начало части расширений не наблюдалось',
        'awaiting_horizon':'Для части случаев срок наблюдения ещё не прошёл',
        'missing_forward_spread':'В периоде наблюдения есть пропуски спреда',
        'maturity_unknown':'Дата погашения неизвестна',
        'maturity_within_horizon':'Погашение попадает в срок наблюдения',
        'no_threshold_crossing':'Отдельных заметных расширений спреда не найдено'}
    with st.container():
        reason = row['recovery__reason']
        if pd.notna(reason) and reason:
            st.caption('; '.join(reasons.get(code, code) for code in reason.split(';')))
        details = view['episodes']
        if details.empty:
            st.caption('Прошлых случаев для показа нет')
        else:
            columns = {'entry_date':'Начало', 'initial_excess_bps':'Расширение, б.п.',
                'target_bps':'Целевой спред, б.п.', 'horizon_end':'Конец срока',
                'first_recovery_date':'Первое наблюдаемое восстановление',
                'days_to_recovery':'Время, дней', 'reduction_bps':'Сужение к концу срока, б.п.',
                'max_worsening_bps':'Дальнейшее расширение, б.п.', 'outcome':'Результат',
                'hit_timing':'Наблюдение восстановления'}
            shown = details[list(columns)].copy()
            shown['outcome'] = shown.outcome.map({'recovered':'Восстановился',
                'not_recovered':'Не восстановился', 'pending':'Срок ещё не прошёл',
                'incomplete':'Недостаточно данных'})
            shown['hit_timing'] = shown.hit_timing.map({'complete_path':'Без пропусков до восстановления',
                'after_gap':'Наблюдалось после пропуска', 'not_observed':'Не наблюдалось'})
            st.dataframe(style_table(shown.rename(columns=columns)), hide_index=True, width='stretch')


if __name__ == '__main__':
    render_app()
