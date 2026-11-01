import unittest
import pandas as pd
from moex_bonds.ui.investing import mode_defaults, investing_view, sort_investing


class InvestingViewTests(unittest.TestCase):
    def test_mode_switch_resets_order_and_does_not_inherit_trading_shortlist(self):
        state={'sort_field':'adv20','sort_ascending':True,'list_scope':'Выделены анализом'}
        mode_defaults(state,'Инвестирование')
        self.assertEqual(state['sort_field'],'investing__premium_bps')
        self.assertFalse(state['sort_ascending'])
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
