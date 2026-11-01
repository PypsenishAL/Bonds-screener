import unittest
import json
from pathlib import Path
import pandas as pd
from moex_bonds.ui.investing import (mode_defaults, investing_view, sort_investing,
    selection_ranks, add_investing_labels, sync_duration_filter, LABELS)


class InvestingViewTests(unittest.TestCase):
    def test_default_duration_range_follows_category_but_user_range_is_kept(self):
        state={}
        sync_duration_filter(state,(.5,6.))
        self.assertEqual(state['filter_duration'],(.5,6.))
        sync_duration_filter(state,(.2,4.))
        self.assertEqual(state['filter_duration'],(.2,4.))
        state['filter_duration']=(1.,3.)
        sync_duration_filter(state,(.1,5.))
        self.assertEqual(state['filter_duration'],(1.,3.))

    def test_real_app_filter_places_sort_direction_and_plain_copy(self):
        from streamlit.testing.v1 import AppTest
        root=Path(__file__).resolve().parents[1]
        manifest=root/'data/published/spread_analysis/run_manifest.json'
        if not manifest.exists(): self.skipTest('No published snapshot')
        publication=root/'data/published'
        code=f"from pathlib import Path\nfrom moex_bonds.ui.app import render_app\nrender_app(Path({str(publication/'spread_analysis')!r}),Path({str(publication/'daily_data')!r}))"
        at=AppTest.from_string(code)
        at.query_params['mode']='Инвестирование'
        at.run(timeout=30)
        self.assertFalse(at.exception)
        at.selectbox(key='investing_group').set_value('A').run(timeout=30)
        at.number_input(key='investing_min_ytm').set_value(15.).run(timeout=30)
        self.assertEqual(tuple(at.session_state['filter_duration']),
                         tuple(at.session_state['_duration_default_bounds']))
        at.multiselect(key='columns_Инвестирование').set_value(['SECID','investing__rank']).run(timeout=30)
        self.assertFalse(at.exception)
        table=at.dataframe[0].value
        main=table[table['investing__rank'].notna()]
        if not main.empty:
            self.assertEqual(main.investing_selection_rank.min(),1)
            self.assertTrue(main['investing__rank'].is_monotonic_increasing)
            labels=json.loads(at.dataframe[0].proto.columns)
            self.assertEqual(labels['investing_selection_rank']['label'],'Место в выборке')
            saved=main.set_index('SECID')['investing__score'].to_dict()
            at.selectbox(key='sort_field').set_value('YTM_WAPRICE').run(timeout=30)
            self.assertFalse(at.session_state['sort_ascending'])
            at.selectbox(key='sort_field').set_value('investing__rank').run(timeout=30)
            self.assertTrue(at.session_state['sort_ascending'])
            at.button(key='reset_investing_sort').click().run(timeout=30)
            table=at.dataframe[0].value
            self.assertTrue(table['investing__rank'].dropna().is_monotonic_increasing)
            for _,r in table[table.SECID.isin(saved)].iterrows():
                if pd.notna(saved[r.SECID]): self.assertAlmostEqual(r['investing__score'],saved[r.SECID])
            at.session_state['selected_bond']=(main.SECID.iloc[0],'TQCB')
            at.run(timeout=30)
            self.assertFalse(at.exception)
        text=' '.join(str(e.value) for kind in ('caption','markdown','info') for e in getattr(at,kind))
        for phrase in ('Премия — доплата','не вероятность прибыли','Проверка экстремумов: complete','Устойчивая доплата'):
            self.assertNotIn(phrase,text)

    def test_mode_switch_resets_order_and_does_not_inherit_trading_shortlist(self):
        state={'sort_field':'adv20','sort_ascending':True,'list_scope':'Выделены анализом'}
        mode_defaults(state,'Инвестирование')
        self.assertEqual(state['sort_field'],'investing_selection_rank')
        self.assertTrue(state['sort_ascending'])
        state['sort_field']='YTM_WAPRICE'
        mode_defaults(state,'Инвестирование')
        self.assertEqual(state['sort_field'],'YTM_WAPRICE')
        mode_defaults(state,'Торговля')
        self.assertEqual(state['sort_field'],'peers_delta')

    def test_filters_do_not_change_saved_score_and_long_bonds_can_be_shown(self):
        f=pd.DataFrame(dict(SECID=['A','B','C'],RATING=['AAA','AAA','BB'],
            MATDATE=['2029-10-30','2040-10-30','2028-10-30'],YTM_WAPRICE=[16,18,40],
            investing__group=['AAA','AAA','BB'],investing__score=[55,90,95],
            investing__status=['relative_advantage']*3,investing__premium_bps=[20,30,100],adv20=[1,2,3]))
        result=investing_view(f,'2026-10-30','AAA',False,0.)
        self.assertEqual(result.SECID.tolist(),['A'])
        self.assertEqual(result['investing__score'].tolist(),[55])
        self.assertEqual(len(investing_view(f,'2026-10-30','AAA',True,0.)),2)
        self.assertEqual(len(f),3)

    def test_review_status_never_dominates_main_order(self):
        f=pd.DataFrame(dict(SECID=['BAD','GOOD'],investing__group=['AAA','AAA'],
            investing__status=['requires_review','relative_advantage'],
            investing__premium_bps=[10000,30],adv20=[1,1]))
        self.assertEqual(sort_investing(f).SECID.tolist(),['GOOD','BAD'])

    def test_requested_rank_order_has_no_hidden_status_priority(self):
        f=pd.DataFrame(dict(SECID=['A','B','C','D'],investing__group=['A']*4,
            investing__status=['relative_advantage']*2+['no_advantage']*2,
            investing__rank=[1.,2.,3.,4.],investing__premium_bps=[80,20,-10,-20],adv20=[1]*4))
        self.assertEqual(sort_investing(f,'investing__rank',False).SECID.tolist(),['D','C','B','A'])
        self.assertEqual(sort_investing(f).SECID.tolist(),['A','B','C','D'])

    def test_selection_rank_starts_at_one_without_changing_saved_places(self):
        f=pd.DataFrame(dict(investing__group=['A','A','A'],investing__rank=[3.,4.,None]))
        result=selection_ranks(f)
        self.assertEqual(result.investing_selection_rank.iloc[:2].tolist(),[1.,2.])
        self.assertTrue(pd.isna(result.investing_selection_rank.iloc[2]))
        self.assertEqual(f['investing__rank'].iloc[:2].tolist(),[3.,4.])
        self.assertNotIn('investing_selection_rank',f)

    def test_plain_labels_and_comparison_sensitive_status(self):
        f=pd.DataFrame(dict(RATING=['A'],investing__status=['no_advantage'],
            investing__group=['A'],investing__reason=['comparison_sensitive']))
        result=add_investing_labels(f)
        self.assertEqual(result.investing_status_label.iloc[0],'Зависит от выбора аналогов')
        self.assertNotIn('доплата',str(LABELS).lower())
