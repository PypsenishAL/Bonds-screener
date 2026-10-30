"""Own-spread cases with an independent synthetic calendar and explicit paths."""
import unittest

import numpy as np
import pandas as pd

from moex_bonds.analysis.spread_recovery import calculate_recovery, RECOVERY_RULE_VERSION


CONFIG = dict(reference_window=90, min_observations=60, entry_scale=1.,
              recovery_fraction=.5, horizon_months=3)


def recovery_fixture(tail=None, periods=201):
    dates = pd.bdate_range('2026-01-01', periods=periods).strftime('%Y-%m-%d').tolist()
    values = ([140., 150., 160.] * 30 + [140.] +
              (tail if tail is not None else [210., 250., 220., 180., 150.]))
    values = (values + [150.] * periods)[:periods]
    history = pd.DataFrame([dict(SECID='A', BOARDID='TQCB', TRADEDATE=date,
        ZSPREADATWAPRICE=value/100, TRADE_STATUS='traded', NUMTRADES=1)
        for date, value in zip(dates, values)])
    daily = history.tail(1).copy()
    daily['MATDATE'] = '2028-01-01'
    return daily, history, dates


class RecoveryTests(unittest.TestCase):
    def calculate(self, daily, history, dates, **config):
        return calculate_recovery(daily, history, dates, dates[-1],
                                  dict(CONFIG, **config), True)

    def test_fixed_target_first_hit_and_summary(self):
        daily, history, dates = recovery_fixture()
        before = history.copy(deep=True)
        summary, cases = self.calculate(daily, history, dates)
        self.assertEqual(len(cases), 1)
        case = cases.iloc[0]
        self.assertEqual(case.entry_date, dates[91])
        self.assertEqual(case.reference_bps, 150.)
        self.assertEqual(case.initial_excess_bps, 60.)
        self.assertEqual(case.target_bps, 180.)
        self.assertEqual(case.first_recovery_date, dates[94])
        self.assertEqual(case.hit_timing, 'complete_path')
        self.assertEqual(case.days_to_recovery,
                         (pd.Timestamp(dates[94])-pd.Timestamp(dates[91])).days)
        self.assertEqual(case.max_worsening_bps, 40.)
        self.assertEqual(case.outcome, 'recovered')
        self.assertEqual(case.reduction_bps, 60.)
        self.assertEqual(summary.iloc[0]['recovery__share'], 1.)
        self.assertEqual(summary.iloc[0]['recovery__n_known'], 1)
        self.assertEqual(summary.iloc[0].recovery_rule_version, RECOVERY_RULE_VERSION)
        pd.testing.assert_frame_equal(history, before)

    def test_peak_narrowing_is_not_entry_recovery_and_no_expiry_restart(self):
        daily, history, dates = recovery_fixture([210., 250.] + [220.] * 109)
        summary, cases = self.calculate(daily, history, dates)
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases.iloc[0].outcome, 'not_recovered')
        self.assertAlmostEqual(cases.iloc[0].reduction_bps, -10.)
        self.assertEqual(summary.iloc[0]['recovery__share'], 0.)
        self.assertTrue(pd.isna(summary.iloc[0]['recovery__median_days']))

    def test_early_hit_is_pending_until_full_horizon_and_no_future_leakage(self):
        daily, history, dates = recovery_fixture(periods=100)
        history = pd.concat([history, history.tail(1).assign(TRADEDATE='2029-01-01',
                             ZSPREADATWAPRICE=999)], ignore_index=True)
        summary, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.iloc[0].outcome, 'pending')
        self.assertEqual(cases.iloc[0].first_recovery_date, dates[94])
        for field in ('end_z_bps', 'end_observation_date', 'reduction_bps'):
            self.assertTrue(pd.isna(cases.iloc[0][field]))
        self.assertEqual(summary.iloc[0]['recovery__n_pending'], 1)
        self.assertTrue(pd.isna(summary.iloc[0]['recovery__share']))

    def test_gap_before_hit_preserves_observed_hit_but_not_exact_time(self):
        daily, history, dates = recovery_fixture()
        history.loc[93, ['ZSPREADATWAPRICE', 'NUMTRADES']] = [None, 0]
        summary, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.iloc[0].outcome, 'incomplete')
        self.assertEqual(cases.iloc[0].hit_timing, 'after_gap')
        self.assertEqual(cases.iloc[0].first_recovery_date, dates[94])
        self.assertTrue(pd.isna(cases.iloc[0].days_to_recovery))
        self.assertEqual(summary.iloc[0]['recovery__n_known'], 0)

    def test_gap_after_hit_does_not_erase_observed_hit_time(self):
        daily, history, dates = recovery_fixture()
        history = history.drop(index=110)
        _, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.iloc[0].outcome, 'incomplete')
        self.assertEqual(cases.iloc[0].hit_timing, 'complete_path')
        self.assertTrue(pd.notna(cases.iloc[0].days_to_recovery))

    def test_maturity_precedes_pending_including_unknown_maturity(self):
        for maturity in ('2026-06-01', None):
            daily, history, dates = recovery_fixture(periods=100)
            daily['MATDATE'] = maturity
            summary, cases = self.calculate(daily, history, dates)
            self.assertEqual(cases.iloc[0].outcome, 'incomplete')
            self.assertTrue(pd.isna(cases.iloc[0].end_z_bps))
            self.assertEqual(summary.iloc[0]['recovery__n_pending'], 0)

    def test_daily_invalid_value_replaces_valid_history_not_fallback(self):
        daily, history, dates = recovery_fixture()
        daily['ZSPREADATWAPRICE'] = None
        daily['NUMTRADES'] = 0
        summary, cases = self.calculate(daily, history, dates, horizon_months=1)
        # Force analysis day to be the terminal observation of the first case.
        deadline = cases.iloc[0].horizon_end
        dates = [d for d in dates if d <= deadline]
        daily['TRADEDATE'] = dates[-1]
        _, cases = self.calculate(daily, history, dates, horizon_months=1)
        self.assertEqual(cases.iloc[0].outcome, 'incomplete')
        self.assertTrue(pd.isna(cases.iloc[0].end_z_bps))
        self.assertEqual(cases.attrs['daily_override_count'], 1)

    def test_left_censored_start_and_gap_before_entry_are_not_cases(self):
        for index in (90,):
            daily, history, dates = recovery_fixture()
            history.loc[index, 'NUMTRADES'] = 0
            summary, cases = self.calculate(daily, history, dates)
            self.assertTrue(cases.empty)
            self.assertIn('unobserved_onset', summary.iloc[0]['recovery__reason'])
        daily, history, dates = recovery_fixture([220.] * 110)
        history.loc[90:, 'ZSPREADATWAPRICE'] = 2.2
        _, cases = self.calculate(daily.assign(ZSPREADATWAPRICE=2.2), history, dates)
        self.assertTrue(cases.empty)

    def test_no_rearming_at_half_recovery_but_full_reset_allows_new_case(self):
        daily, history, dates = recovery_fixture([210., 180., 210., 180., 150., 150., 240., 150.])
        _, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.entry_date.tolist(), [dates[91], dates[97]])
        self.assertEqual(cases.iloc[0].horizon_end,
                         (pd.Timestamp(dates[91])+pd.DateOffset(months=3)).date().isoformat())

    def test_initial_history_and_zero_mad_have_explicit_unavailability(self):
        daily, history, dates = recovery_fixture(periods=50)
        summary, cases = self.calculate(daily, history, dates)
        self.assertTrue(cases.empty)
        self.assertEqual(summary.iloc[0]['recovery__status'], 'unavailable')
        daily, history, dates = recovery_fixture()
        history.ZSPREADATWAPRICE = 1.5
        daily.ZSPREADATWAPRICE = 1.5
        summary, cases = self.calculate(daily, history, dates)
        self.assertTrue(cases.empty)
        self.assertIn('zero_mad', summary.iloc[0]['recovery__reason'])

    def test_unconfirmed_calendar_returns_one_unavailable_row_per_bond(self):
        daily, history, dates = recovery_fixture()
        summary, cases = calculate_recovery(daily, history, dates, dates[-1], CONFIG, False)
        self.assertEqual(summary.iloc[0]['recovery__reason'], 'calendar_unconfirmed')
        self.assertEqual(summary.iloc[0]['recovery__n_known'], 0)
        self.assertTrue(cases.empty)
        self.assertIn('entry_date', cases)

    def test_empty_history_preserves_schema_and_returns_unavailable_summaries(self):
        daily, history, dates = recovery_fixture()
        for confirmed in (True, False):
            summary, cases = calculate_recovery(daily, history.iloc[:0], dates,
                                                dates[-1], CONFIG, confirmed)
            self.assertEqual(len(summary), 1)
            self.assertEqual(summary.iloc[0]['recovery__status'], 'unavailable')
            self.assertTrue(cases.empty)
            self.assertIn('entry_date', cases)

    def test_no_cases_has_a_reason_after_reference_warmup(self):
        daily, history, dates = recovery_fixture()
        # A varying but never above-threshold oscillation, with a wider scale.
        history.loc[:, 'ZSPREADATWAPRICE'] = [1.4, 1.5, 1.6] * 67
        daily.ZSPREADATWAPRICE = history.iloc[-1].ZSPREADATWAPRICE
        summary, cases = self.calculate(daily, history, dates, entry_scale=100.)
        self.assertTrue(cases.empty)
        self.assertEqual(summary.iloc[0]['recovery__status'], 'no_cases')
        self.assertEqual(summary.iloc[0]['recovery__reason'], 'no_threshold_crossing')

    def test_hit_then_renewed_widening_retains_success_but_negative_terminal_reduction(self):
        daily, history, dates = recovery_fixture([210., 180.] + [230.] * 109)
        _, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.iloc[0].outcome, 'recovered')
        self.assertAlmostEqual(cases.iloc[0].reduction_bps, -20.)
        self.assertEqual(cases.iloc[0].max_worsening_bps, 0.)

    def test_board_identity_and_duplicate_keys(self):
        daily, history, dates = recovery_fixture()
        other = history.assign(BOARDID='OTHER', ZSPREADATWAPRICE=999)
        _, cases = self.calculate(daily, pd.concat([history, other]), dates)
        self.assertEqual(cases.iloc[0].entry_z_bps, 210.)
        with self.assertRaisesRegex(ValueError, 'ключ'):
            self.calculate(daily, pd.concat([history, history.iloc[[1]]]), dates)

    def test_config_changes_target_and_horizon(self):
        daily, history, dates = recovery_fixture()
        _, cases = self.calculate(daily, history, dates, recovery_fraction=.25, horizon_months=1)
        self.assertEqual(cases.iloc[0].target_bps, 195.)
        self.assertEqual(cases.iloc[0].horizon_end,
                         (pd.Timestamp(dates[91])+pd.DateOffset(months=1)).date().isoformat())

    def test_month_end_deadline_clamps_and_uses_published_endpoint(self):
        daily, history, _ = recovery_fixture()
        start = pd.bdate_range(end='2025-01-31', periods=92)[0]
        dates = pd.bdate_range(start, periods=len(history)).strftime('%Y-%m-%d').tolist()
        history.TRADEDATE = dates
        daily.TRADEDATE = dates[-1]
        _, cases = self.calculate(daily, history, dates)
        self.assertEqual(cases.iloc[0].entry_date, '2025-01-31')
        self.assertEqual(cases.iloc[0].horizon_end, '2025-04-30')
        self.assertEqual(cases.iloc[0].end_observation_date, '2025-04-30')


if __name__ == '__main__':
    unittest.main()
