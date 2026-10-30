"""Offline acceptance examples for the upper-tail rule and candidate aggregation."""
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from moex_bonds.analysis.spread_history import historical_stats
from moex_bonds.analysis.spread_model import load_config
from moex_bonds.analysis.spread_classification import classify_metrics, aggregate_candidates


def metric(comparison='issuer', n=60, current=100, window=90, **changes):
    row = dict(analysis_date='2026-10-27', SECID='A', comparison=comparison, window=window,
               current_metric_bps=current, gap_bps=current, n_members=2, calendar_confirmed=True,
               benchmark_bps=0, n_issuers=2, n_eff=2, change_1d_bps=0, change_1d_reason='',
               quality_flags='', current_reason='', reason='',
               **historical_stats(range(n), current, 30))
    row.update(changes)
    return row


class ClassificationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def classify(self, rows):
        return classify_metrics(pd.DataFrame(rows), self.config)

    def test_strict_midrank_threshold_ties_and_unrounded_values(self):
        rows = [metric(current=53.5), metric(current=54),
                metric(**historical_stats([15] * 60, 15, 30), current_metric_bps=15),
                metric(percentile=90.00000001), metric(percentile=100, hist_median_bps=100)]
        result = self.classify(rows)
        self.assertEqual(result.percentile.tolist()[:3], [90, 90 + 100 / 120, 50])
        self.assertEqual(result.upper_tail_anomaly.tolist(), [False, True, False, True, False])

    def test_robust_basis_boundaries_and_percentile_only_candidates(self):
        rows = [metric(percentile=p, robust_score=s) for p, s in
                [(95, 3.499999999), (50, 3.5), (95, 3.500000001),
                 (50, -4), (95, None), (50, float('inf'))]]
        result = self.classify(rows)
        self.assertIn('robust_anomaly', result)
        self.assertEqual(result.robust_anomaly.iloc[:4].tolist(), [False, True, True, False])
        self.assertTrue(result.robust_anomaly.iloc[4:].isna().all())
        self.assertEqual(result.anomaly_basis.iloc[:4].tolist(),
                         ['percentile_only', 'robust_only', 'both', 'none'])
        self.assertTrue(result.anomaly_basis.iloc[4:].isna().all())
        self.assertEqual(result.candidate_qualified.tolist(), [True, False, True, False, True, False])
        self.config['classification']['robust_score_threshold'] = 4.
        self.assertFalse(self.classify([metric(robust_score=3.5)]).robust_anomaly.iloc[0])

    def test_robust_uses_ordinary_assessment_in_all_directions_and_windows(self):
        for kind in ('self', 'issuer', 'peers', 'market'):
            for window in (90, 150):
                with self.subTest(kind=kind, window=window):
                    result = self.classify([metric(kind, window=window, n=n, robust_score=4.)
                                            for n in (29, 30, 59, 60)])
                    self.assertIn('robust_anomaly', result)
                    self.assertTrue(result.robust_anomaly.iloc[:3].isna().all())
                    self.assertTrue(result.anomaly_basis.iloc[:3].isna().all())
                    self.assertTrue(result.robust_anomaly.iloc[3])
        result = self.classify([metric(calendar_confirmed=False, robust_score=4.),
                                metric(**historical_stats([15]*60, 100, 30))])
        self.assertTrue(result.robust_anomaly.isna().all())
        self.assertTrue(result.candidate_qualified.iloc[1])

    def test_observation_boundaries_preserve_descriptive_statistics(self):
        result = self.classify([metric(n=n) for n in [29, 30, 59, 60]])
        self.assertEqual(result.assessment_status.tolist(), ['unavailable', 'provisional', 'provisional', 'evaluated'])
        self.assertTrue(result.upper_tail_anomaly.iloc[:3].isna().all())
        self.assertTrue(result.candidate_qualified.iloc[:3].isna().all())
        self.assertTrue(result.upper_tail_anomaly.iloc[3])
        self.assertTrue(result.hist_median_bps.iloc[1:3].notna().all())

    def test_zero_mad_current_and_calendar_availability(self):
        result = self.classify([metric(**historical_stats([15] * 60, 100, 30)),
                                metric(current=None, current_reason='missing_zspreadatwaprice',
                                       reason='missing_zspreadatwaprice'),
                                metric(calendar_confirmed=False, hist_reason='calendar_not_confirmed_through_analysis_date',
                                       reason='calendar_not_confirmed_through_analysis_date')])
        self.assertTrue(result.upper_tail_anomaly.iloc[0])
        self.assertTrue(pd.isna(result.robust_score.iloc[0]))
        self.assertEqual(result.assessment_status.tolist(), ['evaluated', 'unavailable', 'unavailable'])
        self.assertTrue(result.upper_tail_anomaly.iloc[1:].isna().all())
        self.assertIn('calendar_not_confirmed', result.reason.iloc[2])

    def test_candidate_requires_same_basket_positive_gap_and_two_members(self):
        rows = [metric(gap_bps=-1), metric(n_members=1), metric(n_members=2),
                metric('self'), metric('market'), metric(n_members=0)]
        result = self.classify(rows)
        self.assertEqual(result.evidence_status.tolist(), ['basket', 'paired', 'basket', 'self', 'basket', 'no_basket'])
        self.assertFalse(result.candidate_qualified.iloc[0])
        self.assertTrue(pd.isna(result.candidate_qualified.iloc[1]))
        self.assertTrue(result.candidate_qualified.iloc[2])
        self.assertTrue(result.candidate_qualified.iloc[3:].isna().all())

    def test_dataframe_contract_preserves_index_inputs_and_statistics(self):
        source = pd.DataFrame([metric(), metric('peers')], index=[7, 2])
        before = source.copy(deep=True)
        result = classify_metrics(source, self.config)
        pd.testing.assert_frame_equal(source, before)
        pd.testing.assert_frame_equal(result[source.columns], before)
        self.assertIsNot(source, result)
        classified = result.copy(deep=True)
        aggregate_candidates(result, self.config)
        pd.testing.assert_frame_equal(result, classified)

    def test_window_aggregation_partial_directions_and_no_cross_basket_votes(self):
        pairs = [(True, False), (True, None), (False, None), (False, False), (True, True)]
        rows = []
        for i, pair in enumerate(pairs):
            for kind, value in zip(['issuer', 'peers'], pair):
                rows.append(metric(kind, SECID=str(i), percentile=100 if value else 50,
                                   calendar_confirmed=value is not None))
        result = aggregate_candidates(self.classify(rows).sample(frac=1, random_state=3), self.config).set_index('SECID')
        self.assertEqual(result.index.nunique(), 5)
        self.assertEqual(result.loc['0', 'candidate_status'], 'candidate')
        self.assertEqual(result.loc['0', 'candidate_basis'], 'issuer')
        self.assertEqual(result.loc['1', 'candidate_status'], 'candidate')
        self.assertEqual(result.loc['1', 'candidate_incomplete_directions'], 'peers')
        self.assertEqual(result.loc['2', 'candidate_status'], 'not_fully_assessed')
        self.assertEqual(result.loc['3', 'candidate_status'], 'not_candidate')
        self.assertEqual(result.loc['4', 'candidate_basis'], 'issuer;peers')
        self.assertEqual(result.loc['2', 'candidate_basis'], '')
        # Self/market cannot qualify a target, and a missing peers row is incomplete.
        missing = aggregate_candidates(self.classify([metric('self'), metric('issuer', percentile=50)]), self.config).iloc[0]
        self.assertEqual(missing.candidate_status, 'not_fully_assessed')
        self.assertEqual(missing.candidate_incomplete_directions, 'peers')

    def test_custom_threshold_minimum_and_independent_windows(self):
        self.config['classification'].update(primary_window=150, percentile_threshold=95, min_ordinary_observations=80)
        rows = [metric(n=79, window=150), metric(n=80, window=150, SECID='B'),
                metric(n=80, window=90, SECID='B', percentile=95),
                metric(n=80, window=150, SECID='C', percentile=95.000001)]
        result = self.classify(rows)
        self.assertEqual(result.assessment_status.tolist(), ['provisional', 'evaluated', 'evaluated', 'evaluated'])
        self.assertTrue(pd.isna(result.upper_tail_anomaly.iloc[0]))
        self.assertEqual(result.upper_tail_anomaly.iloc[1:].tolist(), [True, False, True])
        aggregated = aggregate_candidates(result, self.config).set_index(['SECID', 'window'])
        self.assertEqual(aggregated.loc[('B', 150), 'candidate_status'], 'candidate')
        self.assertEqual(aggregated.loc[('B', 90), 'candidate_status'], 'not_fully_assessed')


class ConfigurationTests(unittest.TestCase):
    def test_defaults_and_invalid_classification_settings(self):
        self.assertEqual(load_config()['classification'], dict(primary_window=90, percentile_threshold=90,
                         min_ordinary_observations=60, robust_score_threshold=3.5))
        invalid = [dict(primary_window=100), dict(primary_window=True), dict(min_ordinary_observations=91),
                   dict(min_ordinary_observations=29), dict(min_ordinary_observations=60.5),
                   dict(min_ordinary_observations=True), dict(percentile_threshold=0),
                   dict(percentile_threshold=100), dict(percentile_threshold=float('nan')),
                   dict(percentile_threshold=float('inf')), dict(percentile_threshold=True),
                   dict(percentile_threshold='95'), dict(unknown=1)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for changes in invalid:
                with self.subTest(changes=changes):
                    path.write_text(json.dumps(dict(classification=changes)))
                    with self.assertRaises(ValueError):
                        load_config(path)
            path.write_text(json.dumps(dict(windows=[150, 90], classification=dict(primary_window=150, percentile_threshold=95, min_ordinary_observations=80))))
            self.assertEqual(load_config(path)['classification']['primary_window'], 150)

    def test_robust_threshold_validation_and_override(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(dict(classification=dict(robust_score_threshold=4))))
            self.assertEqual(load_config(path)['classification']['robust_score_threshold'], 4)
            for value in (0, -1, True, '3.5', float('nan'), float('inf')):
                with self.subTest(value=value):
                    path.write_text(json.dumps(dict(classification=dict(robust_score_threshold=value))))
                    with self.assertRaises(ValueError):
                        load_config(path)


if __name__ == '__main__':
    unittest.main()
