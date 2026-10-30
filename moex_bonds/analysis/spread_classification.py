"""Pure classification of existing unrounded spread statistics; no data loading."""
import numpy as np
import pandas as pd


CLASSIFICATION_RULE_VERSION = 'upper_tail_midrank_robust_v2'
RESULT_SCHEMA_VERSION = 3
CANDIDATE_KEYS = ['analysis_date', 'SECID', 'window']
CANDIDATE_FIELDS = ['candidate_status', 'candidate_basis', 'candidate_incomplete_directions']


def classify_metrics(metrics, config):
    """Return a copy with nullable decisions; leave all original statistics intact."""
    result = metrics.copy(deep=True)
    rule = config['classification']
    current = pd.to_numeric(result.current_metric_bps, errors='coerce')
    median = pd.to_numeric(result.hist_median_bps, errors='coerce')
    percentile = pd.to_numeric(result.percentile, errors='coerce')
    available = (result.calendar_confirmed.eq(True) & np.isfinite(current)
                 & np.isfinite(median) & np.isfinite(percentile)
                 & result.n_obs.ge(config['min_observations']))
    ordinary = available & result.n_obs.ge(rule['min_ordinary_observations'])
    result['assessment_status'] = 'unavailable'
    result.loc[available & ~ordinary, 'assessment_status'] = 'provisional'
    result.loc[ordinary, 'assessment_status'] = 'evaluated'
    result['upper_tail_anomaly'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    result.loc[ordinary, 'upper_tail_anomaly'] = (
        percentile.gt(rule['percentile_threshold']) & current.gt(median))[ordinary]
    score = pd.to_numeric(result.robust_score, errors='coerce')
    robust_available = ordinary & np.isfinite(score)
    result['robust_anomaly'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    result.loc[robust_available, 'robust_anomaly'] = score.ge(rule['robust_score_threshold'])[robust_available]
    result['anomaly_basis'] = pd.Series(pd.NA, index=result.index, dtype='string')
    for percentile_flag, robust_flag, basis in [(True, False, 'percentile_only'),
            (False, True, 'robust_only'), (True, True, 'both'), (False, False, 'none')]:
        matching = result.upper_tail_anomaly.eq(percentile_flag) & result.robust_anomaly.eq(robust_flag)
        result.loc[matching.fillna(False), 'anomaly_basis'] = basis
    result['evidence_status'] = 'no_basket'
    result.loc[result.n_members.eq(1), 'evidence_status'] = 'paired'
    result.loc[result.n_members.ge(2), 'evidence_status'] = 'basket'
    result.loc[result.comparison.eq('self'), 'evidence_status'] = 'self'
    qualified = ordinary & result.comparison.isin(['issuer', 'peers']) & result.n_members.ge(2)
    result['candidate_qualified'] = pd.Series(pd.NA, index=result.index, dtype='boolean')
    result.loc[qualified, 'candidate_qualified'] = (
        result.upper_tail_anomaly & pd.to_numeric(result.gap_bps, errors='coerce').gt(0))[qualified]
    return result


def aggregate_candidates(metrics, config):
    """Resolve issuer/peers independently for every analysis_date + SECID + window."""
    if metrics.duplicated(CANDIDATE_KEYS + ['comparison']).any():
        raise ValueError('Повторный ключ классифицированных метрик')
    rows = []
    for key, frame in metrics.groupby(CANDIDATE_KEYS, sort=False, dropna=False):
        by_kind = frame.set_index('comparison')
        basis, incomplete = [], []
        for kind in ('issuer', 'peers'):
            value = by_kind.loc[kind, 'candidate_qualified'] if kind in by_kind.index else pd.NA
            if pd.isna(value):
                incomplete.append(kind)
            elif value:
                basis.append(kind)
        rows.append(dict(zip(CANDIDATE_KEYS, key),
            candidate_status='candidate' if basis else 'not_fully_assessed' if incomplete else 'not_candidate',
            candidate_basis=';'.join(basis), candidate_incomplete_directions=';'.join(incomplete)))
    return pd.DataFrame(rows, columns=CANDIDATE_KEYS + CANDIDATE_FIELDS)
