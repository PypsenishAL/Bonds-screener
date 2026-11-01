"""Offline sensitivity study, not a production scorer or a strategy backtest."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from moex_bonds.analysis.spread_model import prepare_daily, softmax_weights

OUT = Path(__file__).parent / 'validation'
OUT.mkdir(exist_ok=True)
CONFIG = json.loads((ROOT / 'config/spread_config.json').read_text(encoding='utf-8'))
SOURCES = [ROOT / 'config/spread_config.json', ROOT / 'data/history_data/bonds_history.csv']
SOURCES += sorted((ROOT / 'data/spread_analysis/snapshots').glob('*/summary.csv'))
def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCES}
BEFORE = hashes()
HISTORY = pd.read_csv(SOURCES[1])
assert not HISTORY.duplicated(['SECID', 'BOARDID', 'TRADEDATE']).any()

def aggregate(z, distances, method):
    w = softmax_weights(distances, 1.)
    if method == 'mean':
        return float(np.mean(z))
    if method == 'median':
        return float(np.median(z))
    if method == 'weighted_mean':
        return float(w @ z)
    ix = np.argsort(z, kind='stable')
    cumulative = np.cumsum(w[ix])
    k = int(np.searchsorted(cumulative, .5))
    if k + 1 < len(ix) and abs(cumulative[k] - .5) < 1e-12:
        return float((z[ix[k]] + z[ix[k + 1]]) / 2)
    return float(z[ix[k]])

def candidates(frame, target, exact, maxdiff, sector):
    c = frame.loc[frame.issuer_key.ne(target.issuer_key) & frame.group.eq(target.group)].copy()
    c['rd'] = abs(c.rating_step - target.rating_step)
    c['dd'] = abs(c.duration - target.duration)
    c = c.loc[c.rd.le(0 if exact else 1) & c.dd.le(maxdiff)].copy()
    if sector:
        c = c.loc[c.sector_key.eq(target.sector_key)].copy()
    if c.empty:
        return c
    c['distance'] = (1.5 * c.rd + c.dd / .5 + .75 * c.sector_key.ne(target.sector_key)
                     + .25 * abs(np.log(c.adv20 / target.adv20)))
    return c.sort_values(['distance', 'SECID']).drop_duplicates('issuer_key').head(10)

VARIANTS = [('loose', False, 1.5, False), ('exact', True, 1.5, False),
            ('exact_1y', True, 1., False), ('exact_halfyear', True, .5, False),
            ('exact_sector', True, 1.5, True)]
METHODS = ['mean', 'median', 'weighted_mean', 'weighted_median']
rows, coverage, flags = [], [], []
for source in SOURCES[2:]:
    s = pd.read_csv(source)
    day = str(s.analysis_date.iloc[0])
    h = HISTORY.loc[HISTORY.TRADEDATE.eq(day), ['SECID','BOARDID','NUMTRADES','YTM_WAPRICE','MATDATE']]
    s = s.merge(h, on=['SECID','BOARDID'], how='left', validate='one_to_one')
    s['ADV20_RUB'] = s.adv20
    d = prepare_daily(s, CONFIG)
    d['group'] = d.RATING.str.extract(r'^(AAA|AA|A|BBB|BB)')[0]
    valid = (d.confirmed_trade & ~d.structured & d.z_bps.notna() & np.isfinite(d.YTM_WAPRICE)
             & d.duration.gt(0) & d.adv20.gt(0) & d.rating_step.notna()
             & d.issuer_key.ne('') & d.sector_key.ne(''))
    base = d.loc[valid].copy()
    # Frozen preliminary screen, independent of the final local baskets.
    # Each issuer has one vote; excluded issuer never sets its own screen threshold.
    base['screen'] = False
    for i, t in base.iterrows():
        other = base.loc[base.RATING.eq(t.RATING) & base.issuer_key.ne(t.issuer_key)]
        reps = other.groupby('issuer_key')[['z_bps','YTM_WAPRICE']].median()
        for field in ['z_bps','YTM_WAPRICE']:
            if len(reps) < 5:
                continue
            med = float(reps[field].median())
            scale = float(1.4826 * (reps[field] - med).abs().median())
            if scale > 0 and (t[field] - med) / scale > 6:
                base.at[i, 'screen'] = True
                flags.append(dict(day=day,SECID=t.SECID,rating=t.RATING,field=field,
                                  value=float(t[field]),threshold=med+6*scale))
    for screen in [False, True]:
        pool = base.loc[~base.screen] if screen else base
        for variant, exact, diff, sector in VARIANTS:
            counts = {}
            for t in pool.itertuples():
                c = candidates(pool, t, exact, diff, sector)
                n = len(c)
                if n < 3:
                    continue
                counts[t.group] = counts.get(t.group,0)+1
                z, dist = c.z_bps.to_numpy(), c.distance.to_numpy()
                w = softmax_weights(dist,1.)
                for method in METHODS:
                    b = aggregate(z,dist,method)
                    removed = [aggregate(np.delete(z,k),np.delete(dist,k),method) for k in range(n)]
                    rows.append(dict(day=day,screen=screen,variant=variant,method=method,
                        SECID=t.SECID,rating=t.RATING,group=t.group,duration=t.duration,
                        ytm=t.YTM_WAPRICE,z=t.z_bps,n=n,neff=1/float(w@w),
                        benchmark=b,premium=t.z_bps-b,loo_shift=max(abs(v-b) for v in removed),
                        conservative_premium=t.z_bps-max([b]+removed),
                        loo_positive=bool(t.z_bps>max(removed)),
                        peers=';'.join(c.SECID),different_rating=int(c.rd.gt(0).sum())))
            coverage.append(dict(day=day,screen=screen,variant=variant,total=len(s),valid=len(base),
                                 pool=len(pool),scored=sum(counts.values()),groups=counts))

r = pd.DataFrame(rows)
r.to_csv(OUT/'variants.csv',index=False)
pd.DataFrame(flags).to_csv(OUT/'preliminary_flags.csv',index=False)
stats=[]
for key,g in r.groupby(['day','screen','variant','method']):
    stats.append(dict(zip(['day','screen','variant','method'],key)) | dict(n=len(g),
        median_loo_shift=float(g.loo_shift.median()),p90_loo_shift=float(g.loo_shift.quantile(.9)),
        neff_atleast3=int(g.neff.ge(3).sum()),positive=int(g.premium.gt(0).sum()),
        positive_after_loo=int(g.loo_positive.sum())))

# Counterexamples isolate mechanisms; they are not a trained return model.
counterexamples={
    'near100_far200_210': {m:aggregate(np.array([100.,200.,210.]),np.array([0.,2.,3.]),m) for m in METHODS},
    'fair_rating_notches': {'AA+_spread':100,'AA_spread':150,'AA-_spread':200,
                           'AA+_premium_vs_AA':-50,'AA-_premium_vs_AA':50,
                           'exact_rating_premia_with_equal_peers':0},
    'contamination_100_500_600_median':aggregate(np.array([100.,500.,600.]),np.zeros(3),'median')}
assert counterexamples['near100_far200_210']['median']==200
assert counterexamples['near100_far200_210']['weighted_median']==100
assert counterexamples['contamination_100_500_600_median']==500
assert BEFORE == hashes(), 'Source files changed'
report=dict(source_hashes=BEFORE,source_unchanged=True,coverage=coverage,stats=stats,
            flags=flags,counterexamples=counterexamples,
            limitations=['Two current-reference snapshots; not a point-in-time backtest.',
                         'Preliminary 6-MAD screen is experimental, not validated credit-risk detection.',
                         'Missing positive ADV20 excluded; source fallback policy retained.',
                         'Leave-one-out uses fixed representatives, does not refill removed issuer.'])
(OUT/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(coverage=[x for x in coverage if x['screen']],
    stats=[x for x in stats if x['screen'] and x['variant']=='exact'],
    flags=flags,counterexamples=counterexamples,source_unchanged=True),ensure_ascii=False,indent=2))
