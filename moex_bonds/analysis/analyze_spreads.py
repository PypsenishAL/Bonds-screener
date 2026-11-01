"""Compare WAPRICE Z-spreads using frozen weighted baskets and read-only history.

python analyze_spreads.py --daily data/daily_data/daily/bonds_daily.xlsx --history data/history_data/bonds_history.csv
"""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
from pathlib import Path
import uuid

import pandas as pd

from moex_bonds.storage.history_storage import HistoryReader
MSK = ZoneInfo('Europe/Moscow')
from moex_bonds.analysis.spread_history import calculate_history
from moex_bonds.analysis.spread_recovery import calculate_recovery, RECOVERY_RULE_VERSION
from moex_bonds.analysis.spread_classification import (classify_metrics, aggregate_candidates,
                                   CLASSIFICATION_RULE_VERSION, RESULT_SCHEMA_VERSION)
from moex_bonds.analysis.spread_model import DIRECTIONS, load_config, prepare_daily, read_daily, fill_liquidity, select_baskets
from moex_bonds.analysis.spread_results import KEYS, build_summary, file_hash, publish, validate_output
from moex_bonds.analysis.spread_investing import calculate_investing, RULE_VERSION as INVESTING_RULE, CONFIG as INVESTING_CONFIG


from moex_bonds.paths import PROJECT_DIR, DAILY_DIR, HISTORY_DIR, ANALYSIS_DIR, CONFIG_FILE

ROOT = Path(__file__).resolve().parent
IMPLEMENTATION_FILES = {name: ROOT / name for name in
    ('analyze_spreads.py', 'spread_classification.py', 'spread_model.py', 'spread_history.py',
     'spread_results.py', 'spread_recovery.py', 'spread_investing.py')}
IMPLEMENTATION_FILES.update({name: PROJECT_DIR / 'moex_bonds/storage' / name
    for name in ('history_storage.py', 'daily_storage.py')})
IMPLEMENTATION_FILES['paths.py'] = PROJECT_DIR / 'moex_bonds/paths.py'


def make_report(daily, members, metrics, candidates, metadata):
    windows = metadata['config']['windows']
    rule = metadata['config']['classification']
    primary = rule['primary_window']
    header = '| Направление | Текущий показатель | ' + ' | '.join(f'Историческая оценка {w}' for w in windows) + ' |'
    lines = [f"# Анализ Z-спредов за {metadata['analysis_date']}", '',
        f"Выпусков: {len(daily)}. Подтверждённый текущий спред: {metadata['confirmed_current_spreads']}.",
        'Единицы: ZSPREADATWAPRICE × 100 → б.п. Дюрация — биржевая DURATION_YEARS.',
        'Состав и веса корзин зафиксированы на дату анализа; текущий день исключён из исторических норм.',
        'Рейтинги, секторы и эмитенты — текущая справка, не историческая. Отбор относится к текущей выборке.', '',
        header, '|---|---:|' + '---:|' * len(windows)]
    for kind in DIRECTIONS:
        counts = metadata['coverage'][kind]
        values = [str(counts['historical'][str(w)]) for w in windows]
        lines.append(f"| {kind} | {counts['current']}/{len(daily)} | " + ' | '.join(values) + ' |')
    lines += ['', 'Причины недоступности текущих сравнений:', '']
    for kind in DIRECTIONS:
        reasons = metadata['coverage'][kind]['current_missing_reasons']
        lines.append(f'- {kind}: ' + (', '.join(f'{key}: {value}' for key, value in reasons.items()) or 'нет'))
    lines += ['', '## Аномально высокие спреды и кандидаты', '',
        f"Основное окно: {primary} торговых дат. Процентильная аномалия: P > {rule['percentile_threshold']} и показатель выше медианы; "
        f"не менее {rule['min_ordinary_observations']} наблюдений, доступны текущий показатель и подтверждённый календарь.",
        f"От {metadata['config']['min_observations']} до {rule['min_ordinary_observations'] - 1} наблюдений — provisional: "
        'статистика сохраняется, обычная оценка отсутствует; иначе при недостатке данных — unavailable.',
        f"Дополнительный критерий: robust_score ≥ {rule['robust_score_threshold']:g}. "
        'Модуль score не используется. Один robust-сигнал не создаёт кандидата; MAD=0 оставляет robust-проверку недоступной.',
        'Кандидат: issuer ИЛИ peers с обычной аномалией, положительной премией и минимум двумя аналогами в той же корзине. '
        'Один аналог — paired; self и market служат пояснениями. MAD=0 не отменяет процентильную оценку.',
        'candidate — есть подходящее направление; not_candidate — оба проверены и не подходят; '
        'not_fully_assessed — положительного результата нет и хотя бы одно направление оценено не полностью.',
        'Кандидат относительно эмитента не подтверждает дешевизну к внешним аналогам. '
        'Окна оцениваются независимо, их согласие не используется как голос.', '',
        '| Окно | candidate | not_candidate | not_fully_assessed |', '|---:|---:|---:|---:|']
    for window in windows:
        counts = candidates.loc[candidates.window.eq(window), 'candidate_status'].value_counts()
        lines.append(f'| {window} | ' + ' | '.join(str(counts.get(s, 0)) for s in
            ('candidate', 'not_candidate', 'not_fully_assessed')) + ' |')
    lines += ['', '| Сравнение | Окно | evaluated | provisional | unavailable | Аномалий |',
              '|---|---:|---:|---:|---:|---:|']
    for kind in DIRECTIONS:
        for window in windows:
            selected = metrics[metrics.comparison.eq(kind) & metrics.window.eq(window)]
            counts = selected.assessment_status.value_counts()
            lines.append(f'| {kind} | {window} | ' + ' | '.join(str(counts.get(s, 0)) for s in
                ('evaluated', 'provisional', 'unavailable')) + f' | {int(selected.upper_tail_anomaly.sum())} |')
    chosen_candidates = candidates[candidates.window.eq(primary) & candidates.candidate_status.eq('candidate')]
    lines += ['', '### Основание аномальности', '',
              '| Сравнение | Окно | Только процентиль | Только robust score | Оба | Нет | Неполная проверка |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for kind in DIRECTIONS:
        for window in windows:
            basis = metrics.loc[metrics.comparison.eq(kind) & metrics.window.eq(window), 'anomaly_basis']
            counts = basis.value_counts()
            lines.append(f'| {kind} | {window} | ' + ' | '.join(str(counts.get(label, 0)) for label in
                ('percentile_only', 'robust_only', 'both', 'none')) + f' | {int(basis.isna().sum())} |')
    lines += ['', '| SECID кандидата | Основание | Неполные направления |', '|---|---|---|']
    for row in chosen_candidates.itertuples():
        lines.append(f'| {row.SECID} | {row.candidate_basis} | {row.candidate_incomplete_directions} |')
    lines += ['', 'basket_composition_differs_from_current предупреждает о допустимых исторических наблюдениях '
              'на неполном составе текущей корзины внутри соответствующего окна; правило отбора не меняется.',
              '', '## Примеры: аналоги → расстояния → веса → результат', '']
    for kind in ('issuer', 'peers', 'market'):
        examples = metrics[(metrics.comparison == kind) & metrics.current_metric_bps.notna() & metrics.window.eq(primary)]
        if examples.empty:
            continue
        examples = examples.copy()
        examples['has_history'] = examples.excess_bps.notna()
        chosen = examples.sort_values(['has_history', 'n_eff', 'SECID'], ascending=[False, False, True]).iloc[0]
        selected = members[(members.SECID == chosen.SECID) & (members.comparison == kind)]
        lines += [f'### {kind}: {chosen.SECID}', '',
                  '| Аналог | Эмитент | d | Вес | Спред, б.п. |', '|---|---|---:|---:|---:|']
        for row in selected.itertuples():
            lines.append(f'| {row.member_SECID} | {row.member_issuer} | {row.distance:.6f} | {row.weight:.6f} | {row.member_z_bps:.2f} |')
        lines += ['', f'Спред бумаги {chosen.current_z_bps:.2f}; корзина {chosen.benchmark_bps:.2f}; '
                  f'премия {chosen.current_metric_bps:.2f} б.п.; N_eff {chosen.n_eff:.3f}.']
        if chosen.has_history:
            lines += [f'Медиана премии за {primary} дат {chosen.hist_median_bps:.2f}; '
                      f'excess {chosen.excess_bps:.2f} б.п.; наблюдений {chosen.n_obs}.']
        anomaly = '' if pd.isna(chosen.upper_tail_anomaly) else str(bool(chosen.upper_tail_anomaly)).lower()
        qualified = '' if pd.isna(chosen.candidate_qualified) else str(bool(chosen.candidate_qualified)).lower()
        lines += [f'Статус: {chosen.assessment_status}; аномалия: {anomaly}; '
                  f'состав: {chosen.evidence_status}; квалификация кандидата: {qualified}.']
        lines += ['']
    lines += ['## Проверки и воспроизводимость', '',
        f"Чтений CSV истории: {metadata['history_read_passes']}; источник календаря: {metadata['calendar_source']}.",
        'SHA-256 входных файлов до и после расчёта совпали. Источники открывались только для чтения.',
        'Результаты обновляются заменой раздела даты; удалённые из корзины участники не остаются в результатах.',
        f"N_eff = 1 / Σw². robust_score = excess / ({metadata['config']['mad_scale']} × MAD); "
        'при MAD=0 robust_score, robust_anomaly и общее основание пусты; процентильная оценка сохраняется.',
        f"analysis_run_id: {metadata['analysis_run_id']}; classification_rule_version: {metadata['classification_rule_version']}; "
        f"result_schema_version: {metadata['result_schema_version']}.",
        '', 'Фактический конфиг и SHA-256 реализации сохранены в run_manifest.json. Версию старой строки '
        'определяет манифест снимка её analysis_date. candidate_exclusions.csv содержит исключённые аналоги.', '']
    return '\n'.join(lines)


def run_analysis(daily_path, history_path, output, config_path=None, analysis_date=None):
    daily_path, history_path = Path(daily_path).resolve(), Path(history_path).resolve()
    sources = list(dict.fromkeys([daily_path, history_path, CONFIG_FILE]
        + ([Path(config_path).resolve()] if config_path else [])
        + ([history_path.parent / 'run_manifest.json'] if (history_path.parent / 'run_manifest.json').exists() else [])))
    output = validate_output(output, sources)
    source_hashes = {str(path): file_hash(path) for path in sources}
    implementation_hashes = {name: file_hash(path) for name, path in IMPLEMENTATION_FILES.items()}
    config = load_config(config_path)
    raw, day = read_daily(daily_path, analysis_date)
    daily = prepare_daily(raw, config)
    reader = HistoryReader(history_path, identities=daily.set_index('SECID').BOARDID.to_dict())
    recovery_fields = ['ZSPREADATWAPRICE', 'NUMTRADES', 'TRADE_STATUS']
    if 'WAPRICE' in reader.columns:
        recovery_fields.append('WAPRICE')
    recovery_history = reader.query(daily.SECID.tolist(), None, day, recovery_fields)
    daily = fill_liquidity(daily, reader, day)
    members, current, exclusions = select_baskets(daily, config)
    history, metrics = calculate_history(daily, members, current, reader, day, config)
    metrics = classify_metrics(metrics, config)
    candidates = aggregate_candidates(metrics, config)
    summary = build_summary(daily, metrics, candidates, config)
    investing = calculate_investing(daily)
    summary = summary.merge(investing, on=['analysis_date','SECID','BOARDID'], how='left', validate='one_to_one')
    calendar = reader.calendar_dates(day)
    recovery_calendar_confirmed = (reader.manifest.get('mode') == 'history'
        and reader.manifest.get('status') == 'ok'
        and bool(reader.manifest.get('trading_dates')) and day in calendar)
    recovery_summary, recovery_episodes, recovery_series = calculate_recovery(
        daily, recovery_history, calendar, day, config['recovery'], recovery_calendar_confirmed)
    summary = summary.merge(recovery_summary, on=['analysis_date', 'SECID', 'BOARDID'],
                            how='left', validate='one_to_one')
    recovery_coverage = dict(
        calendar_confirmed=bool(recovery_calendar_confirmed), calendar_dates=len(calendar),
        history_start=calendar[0] if calendar else None,
        episode_count=int(recovery_summary.recovery__n_episodes.sum()),
        recovered_cases=int(recovery_summary.recovery__n_recovered.sum()),
        open_cases=int(recovery_summary.recovery__n_open.sum()),
        ended_cases=int(recovery_summary.recovery__n_ended.sum()),
        mature_known_cases=int(recovery_summary.recovery__n_mature_known.sum()),
        mature_unknown_cases=int(recovery_summary.recovery__n_mature_unknown.sum()),
        bond_statuses=recovery_summary.recovery__status.value_counts().to_dict(),
        daily_override_count=recovery_episodes.attrs['daily_override_count'],
        reference_eligible_bonds=sum(r['eligible_dates'] > 0 for r in recovery_episodes.attrs['eligibility']))
    for path in sources:
        if source_hashes[str(path)] != file_hash(path):
            raise RuntimeError('Входной источник изменился во время расчёта: ' + str(path))
    for name, digest in implementation_hashes.items():
        if file_hash(IMPLEMENTATION_FILES[name]) != digest:
            raise RuntimeError('Реализация изменилась во время расчёта: ' + name)
    coverage = {}
    for kind in DIRECTIONS:
        stats = metrics[metrics.comparison.eq(kind)]
        once = stats[stats.window.eq(config['classification']['primary_window'])]
        coverage[kind] = dict(current=int(once.current_metric_bps.notna().sum()),
            historical={str(w): int(stats.loc[stats.window.eq(w), 'hist_median_bps'].notna().sum()) for w in config['windows']},
            historical_with_current={str(w): int(stats.loc[stats.window.eq(w), 'excess_bps'].notna().sum()) for w in config['windows']},
            current_missing_reasons={str(k): int(v) for k,v in once.loc[once.current_metric_bps.isna(), 'current_reason'].value_counts().items()})
    metadata = dict(status='ok', mode='zspread_analysis', analysis_date=day, created_at=datetime.now(MSK).isoformat(),
        instruments=len(daily), confirmed_current_spreads=coverage['self']['current'],
        source_sha256=source_hashes, input_unchanged=True, config=config,
        analysis_run_id=uuid.uuid4().hex, classification_rule_version=CLASSIFICATION_RULE_VERSION,
        result_schema_version=RESULT_SCHEMA_VERSION, implementation_sha256=implementation_hashes,
        row_manifest='snapshots/{analysis_date}/run_manifest.json',
        history_read_passes=reader.read_passes, history_rows_scanned=reader.rows_scanned,
        history_queries=reader.queries, calendar_source=reader.calendar_source,
        rating_basis='Текущая справка Smart-Lab, не point-in-time',
        structured_basket_policy='exclude_explicit_structured_products',
        coverage=coverage, result_keys=KEYS, recovery_rule_version=RECOVERY_RULE_VERSION,
        recovery_coverage=recovery_coverage, investing_rule_version=INVESTING_RULE,
        investing_config=INVESTING_CONFIG,
        investing_coverage=investing['investing__status'].value_counts().to_dict())
    tables = dict(summary=summary, basket_members=members, comparison_history=history, metrics=metrics,
                  recovery_episodes=recovery_episodes, recovery_history=recovery_series)
    for frame in [*tables.values(), exclusions]:
        for field in ('analysis_run_id', 'classification_rule_version', 'result_schema_version'):
            frame[field] = metadata[field]
    report = make_report(daily, members, metrics, candidates, metadata)
    recovery = config['recovery']
    report += ('\n\n## История восстановления собственного Z-спреда\n\n'
        f"Ориентир: {recovery['reference_window']} опубликованных торговых дат; "
        f"минимум {recovery['min_observations']} значений. Вход: {recovery['entry_scale']} масштаба MAD. "
        f"Цель: медиана + {recovery['target_scale']:g} × 1.4826 × MAD, фиксируется при входе. "
        f"Достижение учитывается сразу; {recovery['horizon_months']} календарных мес. — отдельная проверка.\n\n"
        f"Календарь подтверждён: {recovery_calendar_confirmed}; доступно дат: {len(calendar)}; "
        f"бумаг с пригодным ориентиром: {recovery_coverage['reference_eligible_bonds']}.\n\n"
        f"Найдено расширений: {recovery_coverage['episode_count']}; "
        f"восстановлений: {recovery_coverage['recovered_cases']}; "
        f"открытых: {recovery_coverage['open_cases']}; "
        f"погашенных без наблюдаемого возврата: {recovery_coverage['ended_cases']}.\n\n"
        f"Трёхмесячная оценка среди достаточно старых случаев: "
        f"известна у {recovery_coverage['mature_known_cases']}, "
        f"неизвестна у {recovery_coverage['mature_unknown_cases']}. "
        'Изменение WAPRICE — в п.п. номинала на точных датах, без купонов и НКД.\n\n'
        f"Текущая дневная строка заменила отличающуюся историческую: "
        f"{recovery_coverage['daily_override_count']}. Детали: recovery_episodes.csv.\n")
    publish(output, tables,
            exclusions, metadata, report, sources)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--daily', type=Path, default=DAILY_DIR / 'daily/bonds_daily.xlsx', help='XLSX: листы Итог и Поля')
    parser.add_argument('--history', type=Path, default=HISTORY_DIR / 'bonds_history.csv', help='Исторический CSV, только чтение')
    parser.add_argument('--output', type=Path, default=ANALYSIS_DIR, help='Отдельная папка результатов')
    parser.add_argument('--config', type=Path, help='JSON с переопределением настроек spread_config.json')
    parser.add_argument('--date', help='Проверяемая дата YYYY-MM-DD; должна совпасть с TRADEDATE выгрузки')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        result = run_analysis(args.daily, args.history, args.output, args.config, args.date)
    except (ValueError, RuntimeError, OSError) as error:
        raise SystemExit(f'Анализ не выполнен: {error}') from error
    print(json.dumps(dict(analysis_date=result['analysis_date'], output=str(args.output.resolve()),
                          coverage=result['coverage'], history_read_passes=result['history_read_passes']),
                     ensure_ascii=False, indent=2))
