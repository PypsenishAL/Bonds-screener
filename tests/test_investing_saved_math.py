"""Independent arithmetic check of the shipped saved investment audit."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]


class SavedInvestingMathTests(unittest.TestCase):
    def test_saved_benchmarks_premiums_ranks_and_scores(self):
        root=ROOT/'data/published/spread_analysis'
        if not (root/'run_manifest.json').exists(): self.skipTest('No published snapshot')
        manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8-sig'))
        if not manifest.get('investing_rule_version'): self.skipTest('Legacy snapshot')
        frame=pd.read_csv(root/'snapshots'/manifest['analysis_date']/'summary.csv')
        checked=0
        for _,row in frame.iterrows():
            usable=[]
            for comparison in json.loads(row['investing__audit']):
                members=comparison['members']
                if not members: continue
                self.assertEqual(len(members),len({m['issuer'] for m in members}))
                z=np.array([m['z_bps'] for m in members],dtype=float)
                dist=np.array([m['distance'] for m in members],dtype=float)
                def weighted(x,d):
                    weights=np.exp(-(d-d.min()));weights/=weights.sum()
                    return float(sum(x*weights))
                self.assertAlmostEqual(comparison['median_bps'],float(np.median(z)),places=8)
                self.assertAlmostEqual(comparison['weighted_bps'],weighted(z,dist),places=8)
                candidates=[float(np.median(z)),weighted(z,dist)]
                if len(z)>1:
                    for i in range(len(z)):
                        rest=np.delete(z,i);rd=np.delete(dist,i)
                        candidates.extend([float(np.median(rest)),weighted(rest,rd)])
                ceiling=max(candidates)
                self.assertAlmostEqual(comparison['ceiling_bps'],ceiling,places=8)
                if comparison['usable']: usable.append(ceiling)
                checked+=1
            if usable:
                spread=float(row['ZSPREADATWAPRICE'])*100
                self.assertAlmostEqual(row['investing__premium_bps'],spread-max(usable),places=8)
        self.assertGreater(checked,0)
        ranked=frame[frame['investing__rank'].notna()]
        for group,rows in ranked.groupby('investing__group'):
            values=rows['investing__premium_bps'].to_numpy()
            for _,r in rows.iterrows():
                value=r['investing__premium_bps']
                rank=(values>value).sum()+((values==value).sum()+1)/2
                self.assertAlmostEqual(r['investing__rank'],rank)
                self.assertEqual(r['investing__cohort_size'],len(values))
                if len(values)>=manifest['investing_config']['min_score_cohort']:
                    score=100*((values<value).sum()+.5*(values==value).sum())/len(values)
                    self.assertAlmostEqual(r['investing__score'],score)
                else: self.assertTrue(pd.isna(r['investing__score']))
