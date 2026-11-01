import unittest
import numpy as np
import pandas as pd

from moex_bonds.analysis.spread_model import prepare_daily, load_config
from moex_bonds.analysis.spread_investing import calculate_investing, compare, RULE_VERSION, CONFIG, validate_saved


def universe():
    return prepare_daily(pd.DataFrame([dict(SECID=f'B{i}',BOARDID='TQCB',NAME=f'Bond{i}',
        ISSUER=f'Issuer{i}',RATING='AA',SECTOR='Industry',TRADEDATE='2026-10-30',
        TRADE_STATUS='traded',NUMTRADES=10,ZSPREADATWAPRICE=1+i*.1,
        DURATION_YEARS=2.,ADV20_RUB=1e7,YTM_WAPRICE=16+i*.1,MATDATE='2029-10-30')
        for i in range(9)]),load_config())


class InvestingTests(unittest.TestCase):
    def test_scores_use_midrank_in_full_category(self):
        d=universe();more=d.iloc[:3].copy()
        more['SECID']='X'+more.SECID;more['issuer_key']='x'+more.issuer_key
        result=calculate_investing(pd.concat([d,more],ignore_index=True))
        main=result.loc[result['investing__rank'].notna()]
        self.assertEqual(len(main),12)
        values=main['investing__premium_bps'].to_numpy()
        for _,row in main.iterrows():
            x=row['investing__premium_bps']
            expected=100*((values<x).sum()+.5*(values==x).sum())/len(values)
            self.assertAlmostEqual(row['investing__score'],expected)

    def test_positive_ordinary_but_negative_robust_has_explanation(self):
        d=universe();d.loc[4,'z_bps']=145.
        r=calculate_investing(d).set_index('SECID').loc['B4']
        self.assertGreater(r['investing__ordinary_premium_bps'],0)
        self.assertLessEqual(r['investing__premium_bps'],0)
        self.assertEqual(r['investing__reason'],'comparison_sensitive')
        self.assertTrue(pd.notna(r['investing__rank']))

    def test_excel_object_numeric_columns_are_supported(self):
        d=universe()
        for field in ('z_bps','YTM_WAPRICE','duration','adv20','rating_step'):
            d[field]=d[field].astype(object)
        result=calculate_investing(d)
        self.assertEqual(result['investing__rank'].notna().sum(),9)

    def test_roundtrip_and_tampered_investment_metadata(self):
        from io import StringIO
        r=calculate_investing(universe())
        saved=pd.read_csv(StringIO(r.to_csv(index=False)))
        manifest=dict(investing_rule_version=RULE_VERSION,investing_config=CONFIG)
        validate_saved(saved,manifest)
        bad=saved.copy();bad.loc[0,'investing__rank']=999
        with self.assertRaises(ValueError): validate_saved(bad,manifest)
        with self.assertRaises(ValueError): validate_saved(saved,{})
        bad=saved.copy();bad.loc[0,'investing__premium_bps']=np.inf
        with self.assertRaises(ValueError): validate_saved(bad,manifest)

    def test_exact_rating_unique_issuers_and_conservative_deletions(self):
        d=universe()
        d.loc[1,'issuer_key']=d.loc[2,'issuer_key']
        d.loc[3,'rating_step']+=1
        result=compare(d.iloc[0],d,1.5)
        self.assertNotIn('B3',[r['SECID'] for r in result['members']])
        self.assertEqual(len(result['members']),len({r['issuer'] for r in result['members']}))
        self.assertGreaterEqual(result['ceiling_bps'],result['weighted_bps'])
        self.assertGreaterEqual(result['ceiling_bps'],result['median_bps'])

    def test_complete_screen_and_fixed_category_scores(self):
        d=universe()
        r=calculate_investing(d).set_index('SECID')
        self.assertTrue(r['investing__screen_status'].eq('complete').all())
        self.assertTrue(r['investing__score'].isna().all()) # cohort below10
        self.assertTrue(r['investing__cohort_size'].eq(9).all())
        self.assertTrue(r['investing__rank'].notna().all())
        self.assertTrue(r['investing__rule_version'].eq(RULE_VERSION).all())
        self.assertGreater(r.loc['B8','investing__premium_bps'],0)

    def test_frozen_extreme_excluded_from_every_basket(self):
        d=universe()
        d.loc[8,['z_bps','YTM_WAPRICE']]=[10000,200]
        r=calculate_investing(d).set_index('SECID')
        self.assertEqual(r.loc['B8','investing__status'],'requires_review')
        self.assertTrue(pd.isna(r.loc['B8','investing__rank']))
        self.assertFalse(r.drop(index='B8')['investing__audit'].str.contains('B8').any())

    def test_incomplete_screen_cannot_score(self):
        d=universe()
        d['z_bps']=100.
        r=calculate_investing(d)
        self.assertTrue(r['investing__screen_status'].eq('incomplete').all())
        self.assertTrue(r['investing__score'].isna().all())
        self.assertTrue(r['investing__rank'].isna().all())

    def test_missing_liquidity_blocks_target_and_analogue(self):
        d=universe();d.loc[0,'adv20']=np.nan
        r=calculate_investing(d).set_index('SECID')
        self.assertEqual(r.loc['B0','investing__status'],'invalid_data')
        self.assertFalse(r.drop(index='B0')['investing__audit'].str.contains('B0').any())

    def test_no_cross_category_cohort(self):
        a=universe();b=universe()
        b['SECID']='X'+b.SECID;b['issuer_key']='x'+b.issuer_key
        b['RATING']='BBB';b['rating_step']=8;b['z_bps']+=500
        r=calculate_investing(pd.concat([a,b],ignore_index=True))
        self.assertEqual(set(r['investing__group']),{'AA','BBB'})
        self.assertTrue(r['investing__cohort_size'].eq(9).all())


if __name__=='__main__': unittest.main()
