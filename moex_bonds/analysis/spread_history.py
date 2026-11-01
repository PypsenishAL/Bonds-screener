"""Historical gaps of the baskets frozen at the daily analysis date."""
import numpy as np
import pandas as pd

from moex_bonds.analysis.spread_model import DIRECTIONS, finite, own_reason


HISTORY_COLUMNS = ['analysis_date', 'observation_date', 'SECID', 'comparison', 'z_bps',
                   'benchmark_bps', 'metric_bps', 'coverage', 'n_available', 'reason']


def historical_stats(values, current, minimum, mad_scale=1.4826):
    past = np.asarray([v for v in values if finite(v) is not None], dtype=float)
    result = dict(n_obs=len(past), hist_mean_bps=None, hist_median_bps=None, excess_bps=None,
                  percentile=None, mad_bps=None, robust_score=None, hist_reason='', scale_reason='')
    if len(past) < minimum:
        result['hist_reason'] = 'insufficient_history'
        return result
    median = float(np.median(past))
    mad = float(np.median(np.abs(past - median)))
    result.update(hist_mean_bps=float(past.mean()), hist_median_bps=median, mad_bps=mad)
    if mad == 0:
        result['scale_reason'] = 'zero_mad'
    if finite(current) is not None:
        result.update(excess_bps=float(current - median),
                      percentile=float(100 * ((past < current).sum() + .5 * (past == current).sum()) / len(past)),
                      robust_score=float((current - median) / (mad_scale * mad)) if mad else None)
    return result


def calculate_history(daily, members, current, reader, day, config):
    # If a manifest/calendar has already been read, request only the dates needed
    # by the largest window; otherwise discover actual board dates in this scan.
    known = reader.manifest.get('trading_dates') or (sorted(reader.observed_dates) if reader.calendar_through else [])
    known_past = sorted(d for d in known if d < day)
    size = max(config['windows'])
    start = known_past[-size] if len(known_past) >= size else None
    source = reader.query(daily.SECID.tolist(), start, day,
                          ['ZSPREADATWAPRICE', 'NUMTRADES', 'TRADE_STATUS'])
    published_calendar = reader.calendar_dates(day)
    calendar_confirmed = day in published_calendar
    calendar = [d for d in published_calendar if d < day]
    past_dates = calendar[-size:]
    previous = calendar[-1] if calendar and calendar_confirmed else None
    numeric = pd.to_numeric(source.ZSPREADATWAPRICE, errors='coerce')
    counts = pd.to_numeric(source.NUMTRADES, errors='coerce')
    good = source.TRADE_STATUS.eq('traded') & np.isfinite(counts) & counts.gt(0) & np.isfinite(numeric)
    source = source.loc[good].copy()
    source['z_bps'] = numeric[good] * 100
    # A SECID must keep its daily board; standalone readers also fail closed.
    identities = daily.set_index('SECID').BOARDID.to_dict()
    source = source[source.BOARDID.eq(source.SECID.map(identities))]
    if source.duplicated(['SECID', 'TRADEDATE']).any():
        raise ValueError('Неоднозначная история SECID/TRADEDATE')
    spreads = {(r.SECID, r.TRADEDATE): r.z_bps for r in source.itertuples() if r.TRADEDATE < day}
    for row in daily.to_dict('records'):
        if not own_reason(row):
            spreads[(row['SECID'], day)] = row['z_bps']
    groups = {(secid, kind): frame.to_dict('records')
              for (secid, kind), frame in members.groupby(['SECID', 'comparison'])}
    history_rows, metrics = [], []
    for target in daily.to_dict('records'):
        secid = target['SECID']
        for direction in DIRECTIONS:
            basket = groups.get((secid, direction), [])
            now = current[(secid, direction)]
            series = {}
            partial_composition_dates = set()
            for observed in past_dates + [day]:
                z = spreads.get((secid, observed))
                benchmark, metric = None, None
                coverage, available_count = 0., 0
                reason = ''
                if direction == 'self':
                    metric = z
                    coverage, available_count = (1., 1) if z is not None else (0., 0)
                    if z is None:
                        reason = 'missing_confirmed_target_spread'
                elif not basket:
                    reason = now['reason']
                else:
                    available = [(m['weight'], spreads[(m['member_SECID'], observed)])
                                 for m in basket if (m['member_SECID'], observed) in spreads]
                    coverage = min(1., max(0., sum(w for w, _ in available)))
                    available_count = len(available)
                    if coverage + 1e-12 >= config['min_weight_coverage']:
                        benchmark = sum(w * value for w, value in available) / coverage
                        if z is not None:
                            metric = z - benchmark
                        else:
                            reason = 'missing_confirmed_target_spread'
                    else:
                        reason = 'insufficient_basket_weight_coverage'
                        if z is None:
                            reason += ';missing_confirmed_target_spread'
                series[observed] = metric
                if finite(metric) is not None and available_count < len(basket):
                    partial_composition_dates.add(observed)
                history_rows.append(dict(analysis_date=day, observation_date=observed, SECID=secid,
                    comparison=direction, z_bps=z, benchmark_bps=benchmark, metric_bps=metric,
                    coverage=coverage, n_available=available_count, reason=reason))
            change = None
            if finite(now['current_metric_bps']) is not None and finite(series.get(previous)) is not None:
                change = now['current_metric_bps'] - series[previous]
            change_reason = ('calendar_not_confirmed_through_analysis_date' if not calendar_confirmed else
                             '' if change is not None else 'current_metric_unavailable'
                             if finite(now['current_metric_bps']) is None else 'previous_trading_date_unavailable')
            for window in config['windows']:
                dates = calendar[-window:]
                stats = historical_stats([series.get(d) for d in dates], now['current_metric_bps'],
                                         config['min_observations'], config['mad_scale'])
                if not calendar_confirmed:
                    for field in ('hist_mean_bps', 'hist_median_bps', 'excess_bps', 'percentile', 'mad_bps', 'robust_score'):
                        stats[field] = None
                    stats['hist_reason'] = 'calendar_not_confirmed_through_analysis_date'
                    stats['scale_reason'] = ''
                quality = [flag for flag in [now['flags'], stats['scale_reason'],
                           'current_reference_not_point_in_time',
                           'calendar_shorter_than_window' if len(dates) < window else '',
                           'basket_composition_differs_from_current'
                           if partial_composition_dates.intersection(dates) else '',
                           'structured_product' if target['structured'] else ''] if flag]
                metrics.append(dict(analysis_date=day, SECID=secid, comparison=direction, window=window,
                    current_z_bps=target['z_bps'] if not own_reason(target) else None,
                    current_metric_bps=now['current_metric_bps'], benchmark_bps=now['benchmark_bps'],
                    gap_bps=now['gap_bps'], n_members=now['n_members'], n_issuers=now['n_issuers'], n_eff=now['n_eff'],
                    change_1d_bps=change, change_1d_reason=change_reason, previous_trading_date=previous,
                    current_reason=now['reason'], reason=';'.join(filter(None, [now['reason'], stats['hist_reason']])),
                    window_start=dates[0] if dates else None, window_end=dates[-1] if dates else None,
                    n_window_dates=len(dates), calendar_confirmed=calendar_confirmed,
                    adv20_rub=target['adv20'], adv_source=target['adv_source'],
                    adv_last_value_date=target['adv_last_value_date'], adv_n_values=target['adv_n_values'],
                    source_data_gaps=target.get('DATA_GAPS'), smartlab_at=target.get('SMARTLAB_AT'),
                    quality_flags=';'.join(quality), **stats))
    return pd.DataFrame(history_rows, columns=HISTORY_COLUMNS), pd.DataFrame(metrics)
