"""Summarize already calculated variants; never rerun collection."""
from pathlib import Path
import json
import pandas as pd

OUT=Path(__file__).parent/'validation'
r=pd.read_csv(OUT/'variants.csv')
reports=[]
all_candidates=[]
for day, daily in r.loc[r.screen].groupby('day'):
    base=daily.loc[daily.variant.eq('exact') & daily.method.isin(['median','weighted_mean'])]
    p=base.pivot(index='SECID',columns='method',values='conservative_premium').dropna()
    info=base.loc[base.method.eq('weighted_mean')].set_index('SECID')
    p=p.join(info[['group','rating','duration','n','neff','ytm']])
    p['base_quality']=p.n.ge(4)&p.neff.ge(3)
    p['robust_premium']=p[['median','weighted_mean']].min(axis=1)
    # Narrower window can only lower the assessment, never increase it.
    narrow=daily.loc[daily.variant.eq('exact_1y') & daily.method.isin(['median','weighted_mean'])]
    nw=narrow.pivot(index='SECID',columns='method',values='conservative_premium')
    ni=narrow.loc[narrow.method.eq('weighted_mean')].set_index('SECID')
    nw=nw.loc[ni.index[ni.n.ge(4)&ni.neff.ge(3)]]
    p['narrow_available']=p.index.isin(nw.index)
    for secid,v in nw.iterrows():
        if secid in p.index:
            p.at[secid,'robust_premium']=min(p.at[secid,'robust_premium'],v.min())
    p['stable_positive']=p.base_quality & p.narrow_available & p.robust_premium.gt(0)
    p['day']=day
    all_candidates.append(p.reset_index())
    changes=[]
    for group,g in p.loc[p.base_quality].groupby('group'):
        before=info.loc[g.index].copy()
        med=daily.loc[daily.variant.eq('exact')&daily.method.eq('median')].set_index('SECID')
        old=set(med.loc[g.index].premium.nlargest(min(5,len(g))).index)
        new=set(g.robust_premium.nlargest(min(5,len(g))).index)
        changes.append(dict(group=group,n=len(g),top5_overlap=len(old&new),
                            stable_positive=int(g.stable_positive.sum())))
    reports.append(dict(day=day,comparison_count=len(p),base_quality=int(p.base_quality.sum()),
        narrow_available=int((p.narrow_available&p.base_quality).sum()),
        stable_positive=int(p.stable_positive.sum()),groups=changes))
pd.concat(all_candidates).to_csv(OUT/'refined_candidates.csv',index=False)
(OUT/'refined_summary.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(reports,ensure_ascii=False,indent=2))
