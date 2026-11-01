"""Versioned buy-and-hold comparisons, independent of trading signals.

Inputs are prepared daily rows; this module never reads files or fetches data.
The frozen issuer-balanced screen is evaluated before selecting any baskets.
"""
import json
import re

import numpy as np
import pandas as pd

from moex_bonds.analysis.spread_model import RATINGS, finite, softmax_weights

RULE_VERSION = 'conservative_exact_rating_v1'
CONFIG = dict(screen_min_issuers=5, screen_mad_scale=1.4826, screen_threshold=6.,
              max_members=10, min_members=4, min_effective=3.,
              duration_windows=[1.5, 1.], duration_unit=.5,
              sector_penalty=.75, liquidity_penalty=.25, min_score_cohort=10)
STATUSES = {'relative_advantage':'Устойчивая доплата', 'no_advantage':'Доплата не подтверждена',
            'provisional':'Предварительное сравнение', 'insufficient_peers':'Мало сопоставимых эмитентов',
            'screen_incomplete':'Проверка экстремумов неполная', 'requires_review':'Требует проверки',
            'invalid_data':'Неполные исходные данные', 'not_saved':'Расчёт не сохранён'}
FIELDS = ['rule_version','group','status','reason','screen_status','screen_reason',
          'screen_n_issuers','screen_z_limit','screen_ytm_limit','median_bps','weighted_bps',
          'ordinary_premium_bps','premium_bps','reference_max_bps','n_issuers','n_eff',
          'narrow_n_issuers','narrow_n_eff','rank','cohort_size','score','audit']


def _group(step):
    if finite(step) is None or int(step) != step or not 0 <= step < len(RATINGS):
        return ''
    return re.sub(r'[+-]$', '', RATINGS[int(step)])


def _reason(row):
    if not row.get('confirmed_trade', False): return 'unconfirmed_trade'
    if row.get('structured', False): return 'structured_product'
    for field in ('z_bps','YTM_WAPRICE','duration','adv20','rating_step'):
        value = finite(row.get(field))
        if value is None or (field in ('duration','adv20') and value <= 0):
            return 'missing_' + field
    if _group(row.get('rating_step')) not in ('AAA','AA','A','BBB','BB'): return 'unsupported_rating'
    for field in ('issuer_key','sector_key'):
        if not row.get(field): return 'missing_' + field
    return ''


def compare(target, pool, duration_window):
    """Select distinct external issuers and evaluate both aggregators and deletions."""
    c = pool.loc[pool.issuer_key.ne(target.issuer_key)
                 & pool.rating_step.eq(target.rating_step)].copy()
    c['duration_diff'] = abs(c.duration - target.duration)
    c = c.loc[c.duration_diff.le(duration_window + 1e-12)].copy()
    c['distance'] = (c.duration_diff / CONFIG['duration_unit']
        + CONFIG['sector_penalty'] * c.sector_key.ne(target.sector_key)
        + CONFIG['liquidity_penalty'] * abs(np.log(c.adv20 / target.adv20)))
    c = c.sort_values(['distance','SECID']).drop_duplicates('issuer_key').head(CONFIG['max_members'])
    n = len(c)
    result = dict(window=duration_window, n=n, n_eff=0., usable=False,
                  median_bps=None, weighted_bps=None, ceiling_bps=None, members=[])
    if not n: return result
    z, distances = c.z_bps.to_numpy(float), c.distance.to_numpy(float)
    w = softmax_weights(distances, 1.)
    med, mean = float(np.median(z)), float(w @ z)
    ceiling = max(med,mean)
    if n >= 2:
        for k in range(n):
            rest_z, rest_d = np.delete(z,k), np.delete(distances,k)
            ceiling = max(ceiling,float(np.median(rest_z)),float(softmax_weights(rest_d,1.) @ rest_z))
    neff = float(1 / (w @ w))
    result.update(n_eff=neff, usable=n >= CONFIG['min_members'] and neff >= CONFIG['min_effective']-1e-12,
                  median_bps=med, weighted_bps=mean, ceiling_bps=ceiling)
    result['members'] = [dict(SECID=r.SECID, issuer=r.issuer_key, rating=RATINGS[int(r.rating_step)],
        sector=r.sector_key, duration=float(r.duration), z_bps=float(r.z_bps),
        distance=float(r.distance), weight=float(weight)) for r,weight in zip(c.itertuples(),w)]
    return result


def calculate_investing(daily):
    """One saved row per daily identity, with explicit incomplete-screen status."""
    d = daily.copy().reset_index(drop=True)
    for field in ('z_bps','YTM_WAPRICE','duration','adv20','rating_step'):
        d[field]=pd.to_numeric(d[field],errors='coerce') if field in d else np.nan
    if d.empty or d.duplicated(['SECID','BOARDID']).any():
        raise ValueError('Investing needs nonempty unique daily identities')
    reasons = d.apply(_reason,axis=1)
    base = d.loc[reasons.eq('')].copy()
    screens = {}
    for t in base.itertuples():
        others = base.loc[base.rating_step.eq(t.rating_step) & base.issuer_key.ne(t.issuer_key)]
        reps = others.groupby('issuer_key')[['z_bps','YTM_WAPRICE']].median()
        completed, flagged, limits, missing = [], [], {}, []
        for field in ('z_bps','YTM_WAPRICE'):
            if len(reps) < CONFIG['screen_min_issuers']:
                missing.append(field + ':few_issuers'); continue
            med = float(reps[field].median())
            scale = float(CONFIG['screen_mad_scale'] * (reps[field]-med).abs().median())
            if not np.isfinite(scale) or scale <= 0:
                missing.append(field + ':zero_mad'); continue
            limit = med + CONFIG['screen_threshold'] * scale
            limits[field] = limit
            completed.append(field)
            if getattr(t,field) > limit: flagged.append(field)
        screens[t.SECID] = dict(status='flagged' if flagged else 'complete' if len(completed)==2 else 'incomplete',
            reason=';'.join(flagged or missing),n=len(reps),limits=limits)
    # Screen membership is frozen. Incomplete but unflagged rows can be analogues;
    # incompleteness is disclosed in the saved audit and never awards a main score.
    blocked = {secid for secid,s in screens.items() if s['status']=='flagged'}
    pool = base.loc[~base.SECID.isin(blocked)]
    output=[]
    for i,t in d.iterrows():
        row = dict(SECID=t.SECID,BOARDID=t.BOARDID,analysis_date=t.TRADEDATE)
        values = {name:np.nan for name in FIELDS}
        values.update(rule_version=RULE_VERSION,group=_group(t.get('rating_step')),
            status='invalid_data',reason=reasons.iloc[i],screen_status='not_run',screen_reason='',
            audit='[]',cohort_size=0)
        if reasons.iloc[i]:
            output.append(row | {'investing__'+k:v for k,v in values.items()}); continue
        screen=screens[t.SECID]
        values.update(screen_status=screen['status'],screen_reason=screen['reason'],
            screen_n_issuers=screen['n'],screen_z_limit=screen['limits'].get('z_bps'),
            screen_ytm_limit=screen['limits'].get('YTM_WAPRICE'))
        if t.SECID in blocked:
            values.update(status='requires_review',reason=screen['reason'])
        else:
            comparisons=[compare(t,pool,w) for w in CONFIG['duration_windows']]
            broad,narrow=comparisons
            # Record incomplete-screen membership of every representative.
            for comparison in comparisons:
                for m in comparison['members']:
                    m['screen_status']=screens[m['SECID']]['status']
            values.update(n_issuers=broad['n'],n_eff=broad['n_eff'],narrow_n_issuers=narrow['n'],
                narrow_n_eff=narrow['n_eff'],median_bps=broad['median_bps'],weighted_bps=broad['weighted_bps'],
                ordinary_premium_bps=t.z_bps-broad['weighted_bps'] if broad['n'] else np.nan,
                audit=json.dumps(comparisons,ensure_ascii=False,allow_nan=False))
            if not broad['usable']:
                values.update(status='insufficient_peers',reason='broad_coverage')
            else:
                ceiling=max(c['ceiling_bps'] for c in comparisons if c['usable'])
                premium=float(t.z_bps-ceiling)
                if abs(premium)<1e-10: premium=0.
                values.update(reference_max_bps=ceiling,premium_bps=premium)
                if not narrow['usable']:
                    values.update(status='provisional',reason='narrow_coverage')
                elif screen['status']!='complete':
                    values.update(status='screen_incomplete',reason=screen['reason'])
                else:
                    values.update(status='relative_advantage' if premium>0 else 'no_advantage',
                        reason='comparison_sensitive' if premium<=0 and values['ordinary_premium_bps']>0 else '')
        output.append(row | {'investing__'+k:v for k,v in values.items()})
    result=pd.DataFrame(output)
    eligible=result['investing__status'].isin(['relative_advantage','no_advantage'])
    for group in result['investing__group'].unique():
        mask=eligible & result['investing__group'].eq(group)
        count=int(mask.sum())
        result.loc[result['investing__group'].eq(group),'investing__cohort_size']=count
        rank=result.loc[mask,'investing__premium_bps'].rank(method='average',ascending=False)
        result.loc[mask,'investing__rank']=rank
        if count>=CONFIG['min_score_cohort']:
            result.loc[mask,'investing__score']=100*(count-rank+.5)/count
    return result


def validate_saved(summary, manifest):
    """Validate optional versioned extension; old snapshots remain readable."""
    version=manifest.get('investing_rule_version')
    if version is None:
        if any(c.startswith('investing__') for c in summary):
            raise ValueError('Инвестиционные поля без версии расчёта')
        return
    if version!=RULE_VERSION or manifest.get('investing_config')!=CONFIG:
        raise ValueError('Версия инвестиционного расчёта не поддерживается')
    columns={'investing__'+name for name in FIELDS}
    if not columns.issubset(summary): raise ValueError('Неполные инвестиционные поля')
    if not summary['investing__rule_version'].eq(version).all(): raise ValueError('Смешаны инвестиционные правила')
    if not summary['investing__status'].isin(set(STATUSES)-{'not_saved'}).all():
        raise ValueError('Неверный инвестиционный статус')
    eligible=summary['investing__status'].isin(['relative_advantage','no_advantage'])
    for field in ('premium_bps','reference_max_bps','n_eff','narrow_n_eff','n_issuers','narrow_n_issuers'):
        values=pd.to_numeric(summary.loc[eligible,'investing__'+field],errors='coerce')
        if not np.isfinite(values).all(): raise ValueError('Неполное инвестиционное сравнение')
    if (summary.loc[eligible,'investing__n_issuers'].lt(CONFIG['min_members']).any()
            or summary.loc[eligible,'investing__narrow_n_issuers'].lt(CONFIG['min_members']).any()
            or summary.loc[eligible,'investing__n_eff'].lt(CONFIG['min_effective']-1e-12).any()
            or summary.loc[eligible,'investing__narrow_n_eff'].lt(CONFIG['min_effective']-1e-12).any()):
        raise ValueError('Недостаточная инвестиционная корзина')
    positive=summary['investing__premium_bps'].gt(0)
    if not positive.loc[eligible].eq(summary.loc[eligible,'investing__status'].eq('relative_advantage')).all():
        raise ValueError('Статус не соответствует инвестиционной премии')
    if summary.loc[~eligible,['investing__rank','investing__score']].notna().any().any():
        raise ValueError('Инвестиционная оценка присвоена неполным данным')
    if not summary.loc[eligible,'investing__screen_status'].eq('complete').all():
        raise ValueError('Инвестиционная проверка не завершена')
    for group,g in summary.groupby('investing__group'):
        main=g.loc[eligible.reindex(g.index)]
        if not g['investing__cohort_size'].eq(len(main)).all(): raise ValueError('Неверная инвестиционная когорта')
        expected=main['investing__premium_bps'].rank(method='average',ascending=False)
        if not np.allclose(main['investing__rank'],expected,equal_nan=True): raise ValueError('Неверное инвестиционное место')
        scores=100*(len(main)-expected+.5)/len(main) if len(main)>=CONFIG['min_score_cohort'] else np.full(len(main),np.nan)
        if not np.allclose(main['investing__score'],scores,equal_nan=True): raise ValueError('Неверный инвестиционный балл')
