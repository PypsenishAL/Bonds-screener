"""Specification examples and offline integration checks for spread analysis."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from openpyxl import Workbook

from moex_bonds.storage.history_storage import HistoryReader
from moex_bonds.analysis.spread_model import (load_config, prepare_daily, read_daily, select_baskets,
                          fill_liquidity, softmax_weights)
from moex_bonds.analysis.spread_history import calculate_history, historical_stats
from moex_bonds.analysis.analyze_spreads import run_analysis
from moex_bonds.analysis.spread_classification import classify_metrics, aggregate_candidates
from moex_bonds.analysis.spread_results import build_summary


def bond(secid, issuer='Issuer', z=2.7, duration=2, rating='AAA', sector='Transport', **kwargs):
    return dict(SECID=secid, NAME=secid, ISSUER=issuer, RATING=rating, SECTOR=sector,
                TRADEDATE='2026-10-27', BOARDID='TQCB', ZSPREADATWAPRICE=z,
                DURATION_YEARS=duration, ADV20_RUB=100, VALUE_RUB=100,
                NUMTRADES=1, TRADE_STATUS='traded', DATA_GAPS='', SMARTLAB_AT='', **kwargs)


def write_daily(path, rows):
    """Workbook fixture is independent of the production reader/exporter."""
    wb = Workbook()
    main = wb.active
    main.title = 'Итог'
    main.append(['Unrelated title'])
    main.append([])
    fields = list(rows[0])
    headers = [key if key == 'SECID' else 'Заголовок ' + key for key in fields]
    main.append(headers)
    for row in rows:
        main.append([row.get(key) for key in fields])
    dictionary = wb.create_sheet('Поля')
    dictionary.append(['Примечание'])
    dictionary.append(['Поле', 'Источник'])
    for key, header in zip(fields, headers):
        dictionary.append([f'{header} [{key}]', 'MOEX'])
    excluded = wb.create_sheet('Исключены')
    excluded.append(['MUST_NOT_APPEAR'])
    wb.save(path)


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def select(self, rows):
        daily = prepare_daily(pd.DataFrame(rows), self.config)
        return select_baskets(daily, self.config)

    def test_read_russian_headers_date_validation_and_all_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'daily.xlsx'
            rows = [bond('A'), bond('B', z=None)]
            write_daily(path, rows)
            frame, day = read_daily(path)
            self.assertEqual(day, '2026-10-27')
            self.assertEqual(frame.SECID.tolist(), ['A', 'B'])
            self.assertTrue(pd.isna(frame.iloc[1].ZSPREADATWAPRICE))
            with self.assertRaisesRegex(ValueError, 'дат'):
                read_daily(path, '2026-10-26')
            write_daily(path, [bond('A'), dict(bond('B'), TRADEDATE='2026-10-26')])
            with self.assertRaises(ValueError):
                read_daily(path)

    def test_units_aliases_and_no_self_in_issuer(self):
        daily = prepare_daily(pd.DataFrame([bond('A', issuer='  РЖД '),
                                           bond('B', issuer='ржд оао', z=.76)]), self.config)
        self.assertEqual(daily.iloc[1].z_bps, 76)
        members, current, rejected = select_baskets(daily, self.config)
        chosen = members[(members.SECID == 'A') & (members.comparison == 'issuer')]
        self.assertEqual(chosen.member_SECID.tolist(), ['B'])
        self.assertAlmostEqual(current[('A', 'issuer')]['benchmark_bps'], 76)
        self.assertFalse((members.SECID == members.member_SECID).any())

    def test_distance_softmax_and_multiple_issues_per_other_issuer(self):
        rows = [bond('A'), bond('B', issuer='Other', duration=2.5, rating='AA+'),
                bond('C', issuer='Other', duration=2.5, rating='AA+'),
                bond('D', issuer='Third', duration=2, sector='Bank')]
        members, current, _ = self.select(rows)
        peers = members[(members.SECID == 'A') & (members.comparison == 'peers')]
        self.assertEqual(peers.member_SECID.tolist(), ['D', 'B', 'C'])
        by_id = peers.set_index('member_SECID')
        self.assertAlmostEqual(by_id.loc['B', 'distance'], 2.5)
        self.assertAlmostEqual(by_id.loc['D', 'distance'], .75)
        self.assertAlmostEqual(peers.weight.sum(), 1)
        self.assertAlmostEqual(current[('A', 'peers')]['n_eff'], 1 / (peers.weight ** 2).sum())
        np.testing.assert_allclose(softmax_weights([10000, 10001], 1), [1/(1+np.exp(-1)), 1/(1+np.exp(1))])

    def test_missing_liquidity_omits_term_missing_sector_only_blocks_peers(self):
        rows = [bond('A'), dict(bond('B', issuer='Other', sector=None), ADV20_RUB=0)]
        members, current, rejected = self.select(rows)
        market = members[(members.SECID == 'A') & (members.comparison == 'market')]
        self.assertEqual(market.member_SECID.tolist(), ['B'])
        self.assertTrue(market.iloc[0].liquidity_omitted)
        self.assertEqual(current[('A', 'peers')]['reason'], 'no_eligible_analogs')
        self.assertTrue(rejected.reason.str.contains('missing_sector').any())

    def test_unknown_rating_not_guessed_and_issuer_can_work_without_rating(self):
        members, current, _ = self.select([bond('A', rating='unknown'), bond('B'), bond('C', issuer='Other')])
        self.assertEqual(current[('A', 'issuer')]['n_members'], 1)
        self.assertEqual(current[('A', 'market')]['reason'], 'missing_rating')
        daily = prepare_daily(pd.DataFrame([bond('A', rating='AA−'), bond('B', rating='AA-')]), self.config)
        self.assertEqual(daily.iloc[0].rating_step, daily.iloc[1].rating_step)

    def test_no_trades_negative_spread_and_structured_products(self):
        rows = [bond('A', z=-.1), dict(bond('B'), NUMTRADES=0, TRADE_STATUS='no_trades'),
                bond('C', issuer='СберИОС'), bond('D', issuer='Other', duration=4)]
        members, current, _ = self.select(rows)
        self.assertEqual(current[('A', 'self')]['current_metric_bps'], -10)
        self.assertEqual(current[('B', 'self')]['reason'], 'unconfirmed_trade')
        self.assertEqual(current[('C', 'issuer')]['reason'], 'structured_product')
        self.assertEqual(current[('A', 'market')]['reason'], 'no_eligible_analogs')
        self.assertNotIn('B', set(members.member_SECID))

    def test_limits_are_not_relaxed_and_market_not_truncated(self):
        rows = [bond('A')] + [bond(f'B{i:02}', issuer=f'Issuer{i:02}') for i in range(12)]
        members, current, _ = self.select(rows)
        self.assertEqual(current[('A', 'peers')]['n_members'], 10)
        self.assertEqual(current[('A', 'market')]['n_members'], 12)
        _, current, _ = self.select([bond('A'), bond('B', issuer='Other', rating='A+')])
        self.assertEqual(current[('A', 'peers')]['n_members'], 0)

    def test_ten_nearest_repeated_issuer_and_unlimited_market(self):
        for kind, issuer in [('issuer', 'Issuer'), ('peers', 'Other')]:
            with self.subTest(kind=kind):
                rows = [bond('A')] + [bond(f'B{i:02}', issuer=issuer) for i in range(11)]
                members, current, rejected = self.select(rows)
                chosen = members[members.SECID.eq('A') & members.comparison.eq(kind)]
                self.assertEqual(chosen.member_SECID.tolist(), [f'B{i:02}' for i in range(10)])
                self.assertEqual(current[('A', kind)]['n_issuers'], 1)
                self.assertAlmostEqual(chosen.weight.sum(), 1.)
                self.assertFalse(rejected.reason.eq('less_near_issue_of_same_issuer').any())
                if kind == 'peers':
                    self.assertEqual(current[('A', 'market')]['n_members'], 11)

    def test_verified_emitter_aliases_do_not_enter_other_issuer_baskets(self):
        for left, right in [('ТГК-14', 'ПАО "ТГК-14"'),
                            ('iКаршеринг Руссия', 'Каршеринг Руссия'),
                            ('ПАО "ПСБ"', 'Промсвязьбанк')]:
            with self.subTest(issuer=left):
                _, current, _ = self.select([bond('A', issuer=left), bond('B', issuer=right)])
                self.assertEqual(current[('A', 'issuer')]['n_members'], 1)
                self.assertEqual(current[('A', 'peers')]['n_members'], 0)
                self.assertEqual(current[('A', 'market')]['n_members'], 0)

    def test_structured_alias_preserves_separate_product_class(self):
        frame = prepare_daily(pd.DataFrame([bond('A', issuer='СберИОС'),
                                           bond('B', issuer='Сбербанк ПАО')]), self.config)
        self.assertEqual(frame.iloc[0].issuer_key, frame.iloc[1].issuer_key)
        self.assertTrue(frame.iloc[0].structured)
        self.assertFalse(frame.iloc[1].structured)


class ReaderTests(unittest.TestCase):
    def test_batch_query_cache_projection_future_and_adv_missing_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bonds_history.csv'
            dates = pd.bdate_range(end='2026-10-27', periods=23).strftime('%Y-%m-%d').tolist()
            rows = [dict(bond('A'), TRADEDATE=day, ADV20_RUB=None,
                         VALUE_RUB=(None if i == 1 else 0 if i == 2 else 100)) for i, day in enumerate(dates)]
            rows.append(dict(bond('A'), TRADEDATE='2026-10-28', VALUE_RUB=1000000))
            pd.DataFrame(rows).to_csv(path, index=False)
            before = path.read_bytes()
            reader = HistoryReader(path, identities={'A': 'TQCB'})
            daily = prepare_daily(pd.DataFrame([dict(bond('A'), ADV20_RUB=None)]), load_config())
            filled = fill_liquidity(daily, reader, '2026-10-27')
            self.assertEqual(filled.iloc[0].adv20, 100)  # zero is outside the last 20 published values
            self.assertEqual(filled.iloc[0].adv_last_value_date, '2026-10-27')
            self.assertEqual(filled.iloc[0].adv_n_values, 20)
            frame = reader.query(['A'], None, '2026-10-27', ['VALUE_RUB', 'ADV20_RUB'])
            self.assertEqual(reader.read_passes, 1)
            self.assertNotIn('2026-10-28', set(frame.TRADEDATE))
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn('ISSUER', frame)

    def test_adv_explicit_zero_is_counted_and_19_values_are_insufficient(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bonds_history.csv'
            dates = pd.bdate_range(end='2026-10-27', periods=20).strftime('%Y-%m-%d')
            rows = [dict(bond(secid), TRADEDATE=day, ADV20_RUB=None, VALUE_RUB=0 if i == 0 else 100)
                    for secid in ['A', 'B'] for i, day in enumerate(dates) if secid == 'A' or i > 0]
            pd.DataFrame(rows).to_csv(path, index=False)
            daily = prepare_daily(pd.DataFrame([dict(bond(s), ADV20_RUB=None) for s in ['A', 'B']]), load_config())
            filled = fill_liquidity(daily, HistoryReader(path), '2026-10-27').set_index('SECID')
            self.assertEqual(filled.loc['A', 'adv20'], 95)
            self.assertTrue(pd.isna(filled.loc['B', 'adv20']))
            self.assertEqual(filled.loc['B', 'adv_n_values'], 19)

    def test_duplicate_history_key_fails_and_board_is_respected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            rows = [bond('A'), dict(bond('A'), BOARDID='OTHER')]
            pd.DataFrame(rows).to_csv(path, index=False)
            reader = HistoryReader(path, identities={'A': 'TQCB'})
            self.assertEqual(len(reader.query(['A'], None, '2026-10-27', ['VALUE_RUB'])), 1)
            pd.DataFrame([bond('A'), bond('A')]).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, 'ключ'):
                HistoryReader(path).query(['A'], None, '2026-10-27', ['VALUE_RUB'])


class HistoryTests(unittest.TestCase):
    def test_composition_warning_is_limited_to_usable_observations_in_each_window(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            config = load_config()
            dates = pd.bdate_range(end='2026-10-27', periods=161).strftime('%Y-%m-%d').tolist()
            rows = [bond('A'), bond('B', issuer='Other', z=2.2), bond('C', issuer='Third', z=2.2)]
            daily = prepare_daily(pd.DataFrame(rows), config)
            members, current, _ = select_baskets(daily, config)
            mask = members.SECID.eq('A') & members.comparison.isin(['peers', 'market'])
            members.loc[mask & members.member_SECID.eq('B'), 'weight'] = .8
            members.loc[mask & members.member_SECID.eq('C'), 'weight'] = .2
            history_rows = [dict(r, TRADEDATE=day) for day in dates for r in rows
                            if not (r['SECID'] == 'C' and day == dates[20])
                            and not (r['SECID'] == 'B' and day == dates[-4])
                            and not (r['SECID'] == 'A' and day == dates[-3])]
            pd.DataFrame(history_rows).to_csv(path, index=False)
            _, metrics = calculate_history(daily, members, current, HistoryReader(path), dates[-1], config)
            result = metrics[metrics.SECID.eq('A') & metrics.comparison.eq('peers')].set_index('window')
            warning = 'basket_composition_differs_from_current'
            self.assertNotIn(warning, result.loc[90, 'quality_flags'])
            self.assertIn(warning, result.loc[150, 'quality_flags'])
            self.assertEqual(result.loc[90, 'n_obs'], 88)
            self.assertEqual(result.loc[150, 'n_obs'], 148)
            self.assertTrue(metrics.calendar_confirmed.all())
            self.assertAlmostEqual(result.loc[150, 'hist_median_bps'], 50)
            # A missing member only matters when the metric was actually used.
            config['min_weight_coverage'] = .9
            _, strict = calculate_history(daily, members, current, HistoryReader(path), dates[-1], config)
            selected = strict[strict.SECID.eq('A') & strict.comparison.eq('peers')]
            self.assertFalse(selected.quality_flags.str.contains(warning).any())

    def test_flat_seven_member_basket_has_midrank_50_and_zero_mad(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            rows = [bond('A')] + [bond(f'B{i}', issuer=f'Other{i}', z=2.2) for i in range(7)]
            dates = pd.bdate_range(end='2026-10-27', periods=40).strftime('%Y-%m-%d').tolist()
            pd.DataFrame([dict(r, TRADEDATE=day) for day in dates for r in rows]).to_csv(path, index=False)
            daily = prepare_daily(pd.DataFrame(rows), load_config())
            members, current, _ = select_baskets(daily, load_config())
            _, metrics = calculate_history(daily, members, current, HistoryReader(path), '2026-10-27', load_config())
            result = metrics[(metrics.SECID == 'A') & (metrics.comparison == 'peers')].iloc[0]
            self.assertEqual(result.percentile, 50)
            self.assertEqual(result.mad_bps, 0)
            self.assertTrue(pd.isna(result.robust_score))

    def test_calendar_does_not_jump_over_missing_entire_market_date(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            dates = pd.bdate_range(end='2026-10-27', periods=40).strftime('%Y-%m-%d').tolist()
            pd.DataFrame([dict(bond('A'), TRADEDATE=day) for day in dates if day != dates[-2]]).to_csv(path, index=False)
            (path.parent / 'run_manifest.json').write_text(json.dumps(dict(mode='history', status='ok', trading_dates=dates)))
            daily = prepare_daily(pd.DataFrame([bond('A')]), load_config())
            members, current, _ = select_baskets(daily, load_config())
            _, metrics = calculate_history(daily, members, current, HistoryReader(path), dates[-1], load_config())
            result = metrics[(metrics.SECID == 'A') & (metrics.comparison == 'self')].iloc[0]
            self.assertTrue(pd.isna(result.change_1d_bps))
            self.assertEqual(result.previous_trading_date, dates[-2])
            # A stale calendar cannot certify yesterday or a complete 90/150-date window.
            (path.parent / 'run_manifest.json').write_text(json.dumps(dict(mode='history', status='ok', trading_dates=dates[:-2])))
            _, metrics = calculate_history(daily, members, current, HistoryReader(path), dates[-1], load_config())
            result = metrics[(metrics.SECID == 'A') & (metrics.comparison == 'self')].iloc[0]
            self.assertTrue(pd.isna(result.change_1d_bps))
            self.assertTrue(pd.isna(result.hist_median_bps))
            self.assertEqual(result.hist_reason, 'calendar_not_confirmed_through_analysis_date')
            self.assertFalse(metrics.calendar_confirmed.any())
            classified = classify_metrics(metrics, load_config())
            self.assertTrue(classified.assessment_status.eq('unavailable').all())
            self.assertTrue(classified.upper_tail_anomaly.isna().all())

    def test_reference_example_ties_zero_mad_and_minimum(self):
        stats = historical_stats([15] * 30, 270 - 220, 30)
        self.assertEqual(stats['excess_bps'], 35)
        self.assertEqual(stats['hist_median_bps'], 15)
        self.assertEqual(stats['percentile'], 100)
        self.assertIsNone(stats['robust_score'])
        self.assertEqual(historical_stats([15] * 30, 15, 30)['percentile'], 50)
        short = historical_stats([15] * 29, 50, 30)
        self.assertEqual(short['n_obs'], 29)
        self.assertIsNone(short['hist_median_bps'])

    def test_fixed_weights_80_percent_calendar_windows_and_no_gap_skip(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'history.csv'
            dates = pd.bdate_range(end='2026-10-27', periods=161).strftime('%Y-%m-%d').tolist()
            daily = prepare_daily(pd.DataFrame([bond('A'), bond('B', issuer='Other', z=2.2),
                                                bond('C', issuer='Third', z=2.2)]), load_config())
            members, current, _ = select_baskets(daily, load_config())
            mask = (members.SECID == 'A') & (members.comparison != 'issuer')
            members.loc[mask & members.member_SECID.eq('B'), 'weight'] = .8
            members.loc[mask & members.member_SECID.eq('C'), 'weight'] = .2
            rows = []
            for i, day in enumerate(dates):
                rows.extend([dict(bond('A', z=2.35 if i < 160 else 100), TRADEDATE=day),
                             dict(bond('B', issuer='Other', z=2.2), TRADEDATE=day),
                             dict(bond('C', issuer='Third', z=2.2), TRADEDATE=day)])
            # Missing 20% is allowed; missing 80% is not. Previous date target is missing.
            for row in rows:
                if row['SECID'] == 'C' and row['TRADEDATE'] == dates[-3]:
                    row.update(NUMTRADES=0, TRADE_STATUS='no_trades')
                if row['SECID'] == 'B' and row['TRADEDATE'] == dates[-4]:
                    row.update(NUMTRADES=0, TRADE_STATUS='no_trades')
                if row['SECID'] == 'A' and row['TRADEDATE'] == dates[-2]:
                    row.update(NUMTRADES=0, TRADE_STATUS='no_trades')
            rows.append(dict(bond('A', z=9999), TRADEDATE='2026-10-28'))
            pd.DataFrame(rows).to_csv(path, index=False)
            history, metrics = calculate_history(daily, members, current, HistoryReader(path), '2026-10-27', load_config())
            series = history[(history.SECID == 'A') & (history.comparison == 'peers')].set_index('observation_date')
            self.assertAlmostEqual(series.loc[dates[-3], 'coverage'], .8)
            self.assertAlmostEqual(series.loc[dates[-3], 'metric_bps'], 15)
            self.assertTrue(pd.isna(series.loc[dates[-4], 'metric_bps']))
            self.assertEqual(series.loc['2026-10-27', 'z_bps'], 270)  # daily is authoritative today
            self.assertNotIn('2026-10-28', series.index)
            stats = metrics[(metrics.SECID == 'A') & (metrics.comparison == 'peers')].set_index('window')
            self.assertEqual(stats.loc[90, 'n_obs'], 88)
            self.assertEqual(stats.loc[150, 'n_obs'], 148)
            self.assertAlmostEqual(stats.loc[90, 'excess_bps'], 35)
            self.assertTrue(pd.isna(stats.loc[90, 'change_1d_bps']))


class IntegrationTests(unittest.TestCase):
    def test_custom_primary_window_keyed_summary_and_contracts(self):
        from tests.test_spread_classification import metric
        config = load_config()
        config['classification'].update(primary_window=150, percentile_threshold=95, min_ordinary_observations=80)
        rows = [bond('B'), bond('A')]
        daily = prepare_daily(pd.DataFrame(rows), config)
        metrics = classify_metrics(pd.DataFrame([
            metric(kind, n=80, SECID=secid, window=window,
                   percentile=100 if window == 150 and kind == 'issuer' else 95)
            for secid in ['A', 'B'] for kind in ['self', 'issuer', 'peers', 'market']
            for window in [90, 150]]).sample(frac=1, random_state=2), config)
        candidates = aggregate_candidates(metrics, config).sample(frac=1, random_state=3)
        summary = build_summary(daily, metrics, candidates, config)
        self.assertEqual(summary.SECID.tolist(), ['B', 'A'])
        self.assertTrue(summary.candidate_status.eq('candidate').all())
        self.assertTrue(summary.candidate_basis.eq('issuer').all())
        self.assertTrue(summary['candidate__90__status'].eq('not_candidate').all())
        self.assertTrue(summary['candidate__150__status'].eq('candidate').all())
        self.assertTrue(summary['issuer__evidence_status'].eq('basket').all())
        self.assertTrue(summary['issuer__150__calendar_confirmed'].all())
        self.assertEqual(summary['issuer__150__anomaly_basis'].tolist(), ['percentile_only'] * 2)
        self.assertEqual(summary['issuer__90__robust_anomaly'].tolist(), [False] * 2)

    def test_versioned_publication_legacy_rows_booleans_and_input_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            daily, history, output = root / 'daily.xlsx', root / 'history.csv', root / 'analysis'
            config_path, manifest_path = root / 'config.json', root / 'run_manifest.json'
            config_path.write_text(json.dumps(dict(windows=[90, 150], mad_scale=2,
                classification=dict(primary_window=150, percentile_threshold=95, min_ordinary_observations=80))))
            rows = [bond('A', z=3), bond('B', z=2), bond('C', z=2),
                    bond('D', issuer='Other', z=2), bond('E', issuer='Third', z=2), bond('F', z=None)]
            dates = pd.bdate_range(end='2026-10-27', periods=151).strftime('%Y-%m-%d').tolist()
            pd.DataFrame([dict(r, TRADEDATE=day, ZSPREADATWAPRICE=2) for day in dates for r in rows]).to_csv(history, index=False)
            manifest_path.write_text(json.dumps(dict(mode='history', status='ok', trading_dates=dates)))
            write_daily(daily, rows)
            inputs = [daily, history, config_path, manifest_path, Path(__file__).resolve().parents[1] / 'config/spread_config.json']
            before = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
            first = run_analysis(daily, history, output, config_path)
            self.assertEqual(first['source_sha256'], before)
            self.assertEqual(before, {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
            self.assertEqual(first['classification_rule_version'], 'upper_tail_midrank_robust_v2')
            self.assertEqual(first['result_schema_version'], 3)
            self.assertIn('spread_classification.py', first['implementation_sha256'])
            summary = pd.read_csv(output / 'summary.csv').set_index('SECID')
            self.assertEqual(summary.loc['A', 'candidate_status'], 'candidate')
            self.assertEqual(summary.loc['A', 'candidate_basis'], 'issuer;peers')
            self.assertEqual(summary.loc['F', 'candidate_status'], 'not_fully_assessed')
            self.assertEqual(summary.loc['F', 'candidate_incomplete_directions'], 'issuer;peers')
            metrics = pd.read_csv(output / 'metrics.csv', dtype=str, keep_default_na=False)
            for field in ['calendar_confirmed', 'upper_tail_anomaly', 'candidate_qualified', 'robust_anomaly']:
                self.assertLessEqual(set(metrics[field]), {'true', 'false', ''})
            self.assertTrue(metrics.loc[metrics.SECID.eq('F'), 'candidate_qualified'].eq('').all())
            report = (output / 'snapshots/2026-10-27/report.md').read_text(encoding='utf-8')
            self.assertIn('P > 95', report)
            self.assertIn('80', report)
            self.assertIn('Основное окно: 150', report)
            self.assertIn('robust_score ≥ 3.5', report)
            self.assertIn('Основание аномальности', report)
            for name in ['summary', 'metrics', 'basket_members', 'comparison_history']:
                frame = pd.read_csv(output / (name + '.csv'))
                self.assertTrue(frame.analysis_run_id.eq(first['analysis_run_id']).all())
                self.assertTrue(frame.result_schema_version.eq(3).all())
            # Simulate pre-v2 root tables. Old snapshots must remain byte-identical.
            new_fields = ['analysis_run_id', 'result_schema_version', 'classification_rule_version',
                          'assessment_status', 'upper_tail_anomaly', 'candidate_qualified',
                          'evidence_status', 'calendar_confirmed', 'robust_anomaly', 'anomaly_basis']
            for name in ['metrics', 'basket_members', 'comparison_history']:
                target = output / (name + '.csv')
                pd.read_csv(target).drop(columns=new_fields, errors='ignore').to_csv(target, index=False)
            snapshot = output / 'snapshots/2026-10-27'
            snapshot_before = {p.name: p.read_bytes() for p in snapshot.iterdir()}
            write_daily(daily, [dict(r, TRADEDATE='2026-10-28') for r in rows])
            next_dates = dates + ['2026-10-28']
            manifest_path.write_text(json.dumps(dict(mode='history', status='ok', trading_dates=next_dates)))
            run_analysis(daily, history, output, config_path)
            second = run_analysis(daily, history, output, config_path)
            self.assertNotEqual(first['analysis_run_id'], second['analysis_run_id'])
            self.assertEqual(snapshot_before, {p.name: p.read_bytes() for p in snapshot.iterdir()})
            for name in ['metrics', 'basket_members', 'comparison_history']:
                saved = pd.read_csv(output / (name + '.csv'))
                self.assertTrue(saved.loc[saved.analysis_date.eq('2026-10-27'), 'analysis_run_id'].isna().all())
                self.assertTrue(saved.loc[saved.analysis_date.eq('2026-10-28'), 'analysis_run_id'].eq(second['analysis_run_id']).all())
                if name == 'metrics':
                    legacy = saved[saved.analysis_date.eq('2026-10-27')]
                    self.assertTrue(legacy[['assessment_status', 'upper_tail_anomaly', 'candidate_qualified',
                                            'robust_anomaly', 'anomaly_basis']].isna().all().all())
            self.assertEqual(set(pd.read_csv(output / 'summary.csv').analysis_date), {'2026-10-28'})
            self.assertFalse((output / 'candidates.csv').exists())

    def test_idempotent_snapshots_all_secids_and_input_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            daily, history, output = root / 'daily.xlsx', root / 'history.csv', root / 'analysis'
            rows = [bond('A'), bond('B', issuer='Other', z=2.2), bond('C', z=None)]
            write_daily(daily, rows)
            dates = pd.bdate_range(end='2026-10-27', periods=151).strftime('%Y-%m-%d')
            pd.DataFrame([dict(r, TRADEDATE=day) for day in dates for r in rows]).to_csv(history, index=False)
            digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
            before = [digest(daily), digest(history)]
            first = run_analysis(daily, history, output)
            run_analysis(daily, history, output)
            self.assertEqual(before, [digest(daily), digest(history)])
            summary = pd.read_csv(output / 'summary.csv')
            self.assertEqual(summary.SECID.tolist(), ['A', 'B', 'C'])
            for name, key in [('metrics', ['analysis_date','SECID','comparison','window']),
                              ('basket_members', ['analysis_date','SECID','comparison','member_SECID']),
                              ('comparison_history', ['analysis_date','observation_date','SECID','comparison'])]:
                frame = pd.read_csv(output / (name + '.csv'))
                self.assertFalse(frame.duplicated(key).any())
            self.assertLessEqual(first['history_read_passes'], 2)
            self.assertTrue((output / 'snapshots/2026-10-27/report.md').exists())
            # Changed basket removes stale members on a rerun of the same date.
            write_daily(daily, [rows[0], rows[2]])
            run_analysis(daily, history, output)
            self.assertEqual(len(pd.read_csv(output / 'basket_members.csv')), 0)
            # A new date retains the previous daily results.
            write_daily(daily, [dict(r, TRADEDATE='2026-10-28') for r in rows])
            run_analysis(daily, history, output)
            self.assertTrue((output / 'snapshots/2026-10-27/summary.csv').exists())
            self.assertEqual(set(pd.read_csv(output / 'metrics.csv').analysis_date), {'2026-10-27','2026-10-28'})
            self.assertEqual(len(pd.read_csv(output / 'summary.csv')), len(rows))
            self.assertEqual(set(pd.read_csv(output / 'summary.csv').analysis_date), {'2026-10-28'})
            with self.assertRaises(ValueError):
                run_analysis(daily, history, root)

    def test_config_rejects_typos_and_invalid_temperature(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            for value in [{'temperatur': 1}, {'temperature': 0}, {'windows': [90, 0]}]:
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_config(path)


class RecoveryIntegrationTests(unittest.TestCase):
    def test_recovery_exports_counts_provenance_and_partition_replacement(self):
        from tests.test_spread_recovery import recovery_fixture
        from moex_bonds.analysis.spread_recovery import EPISODE_KEYS, RECOVERY_RULE_VERSION
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            daily_frame, history_frame, dates = recovery_fixture()
            daily, history, output = root/'daily.xlsx', root/'history.csv', root/'analysis'
            manifest = root/'run_manifest.json'
            history_frame.to_csv(history, index=False)
            manifest.write_text(json.dumps(dict(mode='history', status='ok', trading_dates=dates)))
            row = dict(bond('A', z=1.5, MATDATE='2028-01-01'), TRADEDATE=dates[-1])
            write_daily(daily, [row])
            metadata = run_analysis(daily, history, output)
            summary = pd.read_csv(output/'summary.csv').iloc[0]
            episodes = pd.read_csv(output/'recovery_episodes.csv')
            self.assertEqual(summary['recovery__n_known'], 1)
            self.assertEqual(summary['recovery__n_recovered'], 1)
            self.assertEqual(metadata['recovery_rule_version'], RECOVERY_RULE_VERSION)
            self.assertIn('spread_recovery.py', metadata['implementation_sha256'])
            self.assertEqual(len(episodes), 1)
            self.assertEqual(metadata['recovery_coverage']['known_cases'], 1)
            self.assertTrue(episodes.analysis_run_id.eq(metadata['analysis_run_id']).all())
            first_day = dates[-1]
            old_snapshot = {p.name:p.read_bytes() for p in (output/'snapshots'/first_day).iterdir()}
            next_day = (pd.Timestamp(first_day)+pd.offsets.BDay()).date().isoformat()
            manifest.write_text(json.dumps(dict(mode='history', status='ok', trading_dates=dates+[next_day])))
            write_daily(daily, [dict(row, TRADEDATE=next_day)])
            run_analysis(daily, history, output)
            run_analysis(daily, history, output)
            episodes = pd.read_csv(output/'recovery_episodes.csv')
            self.assertEqual(set(episodes.analysis_date), {first_day, next_day})
            self.assertFalse(episodes.duplicated(EPISODE_KEYS).any())
            self.assertEqual(old_snapshot, {p.name:p.read_bytes() for p in (output/'snapshots'/first_day).iterdir()})
            self.assertEqual(set(pd.read_csv(output/'summary.csv').analysis_date), {next_day})

    def test_recovery_calendar_unavailable_is_not_a_failure_or_missing_file(self):
        from tests.test_spread_recovery import recovery_fixture
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, frame, dates = recovery_fixture()
            frame.to_csv(root/'history.csv', index=False)
            write_daily(root/'daily.xlsx', [dict(bond('A'), TRADEDATE=dates[-1])])
            run_analysis(root/'daily.xlsx', root/'history.csv', root/'analysis')
            summary = pd.read_csv(root/'analysis/summary.csv').iloc[0]
            self.assertEqual(summary['recovery__reason'], 'calendar_unconfirmed')
            self.assertTrue(pd.isna(summary['recovery__share']))
            self.assertTrue(pd.read_csv(root/'analysis/recovery_episodes.csv').empty)
            (root/'run_manifest.json').write_text(json.dumps(dict(mode='history', status='incomplete')))
            with self.assertRaisesRegex(ValueError, 'не завершена'):
                run_analysis(root/'daily.xlsx', root/'history.csv', root/'analysis')

    def test_recovery_config_defaults_override_and_validation(self):
        self.assertEqual(load_config()['recovery']['reference_window'], 90)
        self.assertEqual(load_config()['recovery']['horizon_months'], 3)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'override.json'
            path.write_text(json.dumps(dict(recovery=dict(recovery_fraction=.25))))
            self.assertEqual(load_config(path)['recovery']['recovery_fraction'], .25)
            for field, value in [('reference_window', True), ('min_observations', 91),
                                 ('entry_scale', 0), ('recovery_fraction', 1.1),
                                 ('horizon_months', 0)]:
                path.write_text(json.dumps(dict(recovery={field:value})))
                with self.assertRaises(ValueError):
                    load_config(path)


if __name__ == '__main__':
    unittest.main()
