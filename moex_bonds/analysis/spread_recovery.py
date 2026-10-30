"""Pure, causal recovery histories of each bond's own WAPRICE z-spread."""
import numpy as np
import pandas as pd


RECOVERY_RULE_VERSION = 'own_spread_recovery_v1'
IDENTITY = ['analysis_date', 'SECID', 'BOARDID']
EPISODE_KEYS = IDENTITY + ['entry_date']
EPISODE_COLUMNS = EPISODE_KEYS + [
    'reference_bps', 'scale_bps', 'reference_n_obs', 'entry_z_bps',
    'initial_excess_bps', 'initial_excess_scales', 'target_bps', 'horizon_end',
    'end_observation_date', 'end_z_bps', 'first_recovery_date', 'hit_timing',
    'days_to_recovery', 'reduction_bps', 'max_worsening_bps', 'n_expected',
    'n_valid', 'outcome', 'reason', 'recovery_rule_version']
STAT_FIELDS = ['n_known', 'n_recovered', 'n_pending', 'n_incomplete', 'share',
               'median_reduction_bps', 'median_days', 'median_worsening_bps']
SUMMARY_FIELDS = ['recovery__' + f for f in ['status', 'reason'] + STAT_FIELDS]
CONFIG_KEYS = {'reference_window', 'min_observations', 'entry_scale',
               'recovery_fraction', 'horizon_months'}


def validate_recovery_config(config):
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError('Неверные поля recovery config')
    for field in ('reference_window', 'min_observations', 'horizon_months'):
        if type(config[field]) is not int or config[field] < 1:
            raise ValueError('recovery.' + field + ' должен быть положительным целым')
    if config['min_observations'] > config['reference_window']:
        raise ValueError('recovery.min_observations больше reference_window')
    for field in ('entry_scale', 'recovery_fraction'):
        value = config[field]
        if type(value) not in (int, float) or not np.isfinite(value) or value <= 0:
            raise ValueError('recovery.' + field + ' должен быть конечным и положительным')
    if config['recovery_fraction'] > 1:
        raise ValueError('recovery.recovery_fraction должен быть в (0, 1]')
    return config


def recovery_statistics(episodes):
    """Aggregate saved case outcomes, also used for viewer consistency checks."""
    known = episodes[episodes.outcome.isin(['recovered', 'not_recovered'])]
    recovered = known[known.outcome.eq('recovered')]
    def median(frame, field):
        return float(frame[field].median()) if len(frame) and frame[field].notna().any() else None
    return dict(n_known=len(known), n_recovered=len(recovered),
        n_pending=int(episodes.outcome.eq('pending').sum()),
        n_incomplete=int(episodes.outcome.eq('incomplete').sum()),
        share=len(recovered) / len(known) if len(known) else None,
        median_reduction_bps=median(known, 'reduction_bps'),
        median_days=median(recovered, 'days_to_recovery'),
        median_worsening_bps=median(known, 'max_worsening_bps'))


def _finite(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _source_tuple(row):
    return (_finite(row.get('ZSPREADATWAPRICE')), row.get('TRADE_STATUS'),
            _finite(row.get('NUMTRADES')))


def calculate_recovery(daily, history, calendar, analysis_date, recovery_config,
                       calendar_confirmed):
    """Return one summary per daily identity and a possibly empty case table.

    Inputs and their dates are read-only. No reference or case uses a future row.
    Daily raw fields replace the current historical row even when invalid.
    """
    cfg = validate_recovery_config(recovery_config)
    day = pd.Timestamp(analysis_date).date().isoformat()
    ids = ['SECID', 'BOARDID']
    if daily[ids].isna().any().any() or daily.duplicated(ids).any():
        raise ValueError('Пустой или повторный ключ дневного выпуска')
    if not daily.TRADEDATE.eq(day).all():
        raise ValueError('Дневные строки recovery относятся к другой дате')
    dates = sorted(d for d in calendar if d <= day)
    if len(set(dates)) != len(dates):
        raise ValueError('Повторные даты календаря recovery')
    if calendar_confirmed and day not in dates:
        raise ValueError('Календарь recovery не подтверждает дату анализа')
    empty = pd.DataFrame(columns=EPISODE_COLUMNS)
    summaries, all_cases, diagnostics = [], [], []
    overrides = 0
    wanted = set(map(tuple, daily[ids].itertuples(index=False, name=None)))
    source = history[history.TRADEDATE.le(day)].copy()
    mask = np.array([tuple(key) in wanted for key in source[ids].itertuples(index=False, name=None)], dtype=bool)
    source = source.loc[mask]
    if source.duplicated(ids + ['TRADEDATE']).any():
        raise ValueError('Повторный ключ SECID/BOARDID/TRADEDATE recovery')
    groups = {key: frame for key, frame in source.groupby(ids, sort=False)}
    for current in daily.to_dict('records'):
        key = (current['SECID'], current['BOARDID'])
        identity = dict(analysis_date=day, SECID=key[0], BOARDID=key[1])
        reasons, cases, eligible = set(), [], 0
        if not calendar_confirmed:
            reasons.add('calendar_unconfirmed')
        else:
            raw = groups.get(key, source.iloc[:0]).set_index('TRADEDATE')
            if day in raw.index and _source_tuple(raw.loc[day]) != _source_tuple(current):
                overrides += 1
            aligned = raw.reindex(dates).copy()
            for field in ('ZSPREADATWAPRICE', 'TRADE_STATUS', 'NUMTRADES'):
                aligned.loc[day, field] = current[field]
            numeric = pd.to_numeric(aligned.ZSPREADATWAPRICE, errors='coerce').to_numpy(dtype=float)
            counts = pd.to_numeric(aligned.NUMTRADES, errors='coerce').to_numpy(dtype=float)
            valid = (aligned.TRADE_STATUS.eq('traded').to_numpy()
                     & np.isfinite(numeric) & np.isfinite(counts) & (counts > 0))
            spreads = np.where(valid, numeric * 100, np.nan)
            previous, active = None, None
            entries = []
            window = cfg['reference_window']
            for i, z in enumerate(spreads):
                condition, reference = None, None
                past = spreads[max(0, i-window):i]
                past = past[np.isfinite(past)]
                if i < window or len(past) < cfg['min_observations']:
                    reasons.add('insufficient_reference')
                elif not np.isfinite(z):
                    reasons.add('missing_current_spread')
                else:
                    m0 = float(np.median(past))
                    s0 = float(1.4826 * np.median(np.abs(past - m0)))
                    if s0 <= 0:
                        reasons.add('zero_mad')
                    else:
                        eligible += 1
                        condition = bool(z > m0 and z - m0 >= cfg['entry_scale'] * s0)
                        reference = (m0, s0, len(past))
                if active is not None:
                    if np.isfinite(z) and z <= active[1]:
                        active = None
                elif condition:
                    if previous is False:
                        entries.append((i, *reference))
                        active = (i, reference[0])
                    elif previous is None:
                        reasons.add('unobserved_onset')
                previous = condition

            maturity = pd.to_datetime(current.get('MATDATE'), errors='coerce')
            for start, m0, s0, n_reference in entries:
                entry = pd.Timestamp(dates[start])
                deadline = (entry + pd.DateOffset(months=cfg['horizon_months'])).date().isoformat()
                stop = int(np.searchsorted(dates, min(deadline, day), side='right'))
                forward = spreads[start+1:stop]
                good = np.isfinite(forward)
                z0 = float(spreads[start])
                excess = z0 - m0
                target = z0 - cfg['recovery_fraction'] * excess
                hits = np.flatnonzero(good & (forward <= target))
                hit = int(hits[0]) if len(hits) else None
                first_date = dates[start+1+hit] if hit is not None else None
                complete_path = hit is not None and bool(good[:hit+1].all())
                hit_timing = ('complete_path' if complete_path else 'after_gap') if hit is not None else 'not_observed'
                if pd.isna(maturity):
                    outcome, reason = 'incomplete', 'maturity_unknown'
                elif maturity.date().isoformat() <= deadline:
                    outcome, reason = 'incomplete', 'maturity_within_horizon'
                elif deadline > day:
                    outcome, reason = 'pending', 'awaiting_horizon'
                elif not good.all():
                    outcome, reason = 'incomplete', 'missing_forward_spread'
                else:
                    outcome, reason = ('recovered' if hit is not None else 'not_recovered'), ''
                end_date, end_z, reduction = None, None, None
                if deadline <= day and stop > start+1:
                    endpoint = dates[stop-1]
                    if (np.isfinite(spreads[stop-1]) and pd.notna(maturity)
                            and endpoint < maturity.date().isoformat()):
                        end_date, end_z = endpoint, float(spreads[stop-1])
                        reduction = z0 - end_z
                observed = forward[:hit+1] if hit is not None else forward
                observed = observed[np.isfinite(observed)]
                worsening = max(0., float(observed.max() - z0)) if len(observed) else 0.
                cases.append(dict(**identity, entry_date=dates[start], reference_bps=m0,
                    scale_bps=s0, reference_n_obs=n_reference, entry_z_bps=z0,
                    initial_excess_bps=excess, initial_excess_scales=excess/s0,
                    target_bps=target, horizon_end=deadline, end_observation_date=end_date,
                    end_z_bps=end_z, first_recovery_date=first_date, hit_timing=hit_timing,
                    days_to_recovery=(pd.Timestamp(first_date)-entry).days if complete_path else None,
                    reduction_bps=reduction, max_worsening_bps=worsening,
                    n_expected=len(forward), n_valid=int(good.sum()), outcome=outcome,
                    reason=reason, recovery_rule_version=RECOVERY_RULE_VERSION))
                if reason:
                    reasons.add(reason)
        table = pd.DataFrame(cases, columns=EPISODE_COLUMNS)
        stats = recovery_statistics(table)
        status = ('unavailable' if not calendar_confirmed else
                  'available' if stats['n_known'] else 'no_known_outcomes' if cases else
                  'no_cases' if eligible else 'unavailable')
        # The inevitable initial warm-up is not an exclusion once eligibility exists.
        if eligible:
            reasons.discard('insufficient_reference')
        if not cases and not reasons:
            reasons.add('no_threshold_crossing')
        summaries.append(dict(**identity, recovery__status=status,
            recovery__reason=';'.join(sorted(reasons)), recovery_rule_version=RECOVERY_RULE_VERSION,
            **{'recovery__'+field: value for field, value in stats.items()}))
        diagnostics.append(dict(**identity, eligible_dates=eligible))
        all_cases.extend(cases)
    episodes = pd.DataFrame(all_cases, columns=EPISODE_COLUMNS)
    episodes.attrs['daily_override_count'] = overrides
    episodes.attrs['eligibility'] = diagnostics
    summary = pd.DataFrame(summaries, columns=IDENTITY + SUMMARY_FIELDS + ['recovery_rule_version'])
    return summary, episodes
