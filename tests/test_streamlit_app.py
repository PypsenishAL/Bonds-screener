"""Read-only viewer contracts, using temporary independent snapshot fixtures."""
import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd
from openpyxl import Workbook


DAY = '2026-10-27'
IDENTITY = dict(analysis_run_id='run-a', result_schema_version=2,
                classification_rule_version='upper_tail_midrank_v1')


def write_fixture(root):
    folder = root / 'snapshots' / DAY
    folder.mkdir(parents=True)
    manifest = dict(status='ok', analysis_date=DAY, instruments=2, **IDENTITY,
                    config=dict(windows=[90, 150], classification=dict(primary_window=90),
                                issuer_aliases={'Alias': 'Alpha'}), source_sha256={})
    summary = pd.DataFrame([
        dict(analysis_date=DAY, SECID='A', BOARDID='TQCB', NAME='Same name', ISSUER='Alias',
             RATING='AAA', SECTOR='Banks', DURATION_YEARS=1., adv20=1000000., **IDENTITY),
        dict(analysis_date=DAY, SECID='B', BOARDID='TQCB', NAME='Same name', ISSUER='Other',
             RATING='AA', SECTOR='Transport', DURATION_YEARS=2., adv20=None, **IDENTITY)])
    summary['candidate__90__status'] = ['candidate', 'not_fully_assessed']
    summary['candidate__150__status'] = ['not_candidate', 'not_fully_assessed']
    metrics, history = [], []
    for secid in ('A', 'B'):
        for kind in ('self', 'peers', 'market', 'issuer'):
            for window in (90, 150):
                metrics.append(dict(analysis_date=DAY, SECID=secid, comparison=kind,
                    window=window, window_start='2026-10-26' if window == 90 else '2026-10-23',
                    window_end='2026-10-26', current_metric_bps=140. if kind == 'self' else 40.,
                    current_z_bps=140., hist_median_bps=120. if window == 90 else 110.,
                    excess_bps=20. if window == 90 else 30., n_obs=60,
                    assessment_status='evaluated', candidate_qualified='true' if secid == 'A' and window == 90 and kind == 'peers' else ('false' if secid == 'A' else ''),
                    upper_tail_anomaly='false', calendar_confirmed='true',
                    change_1d_bps=20., n_members=0 if kind == 'self' else 1,
                    n_issuers=0 if kind == 'self' else 1, n_eff=1., current_reason='', **IDENTITY))
            for date, own, premium in [('2026-10-23', 100., 20.), ('2026-10-26', 120., None), (DAY, 140., 40.)]:
                history.append(dict(analysis_date=DAY, observation_date=date, SECID=secid,
                    comparison=kind, z_bps=own, metric_bps=own if kind == 'self' else premium,
                    benchmark_bps=None if kind == 'self' or premium is None else own-premium,
                    coverage=1. if premium is not None or kind == 'self' else 0.,
                    n_available=1, reason='' if premium is not None or kind == 'self' else 'insufficient_basket_weight_coverage', **IDENTITY))
    members = pd.DataFrame([dict(analysis_date=DAY, SECID='A', comparison='peers',
        member_SECID='B', member_BOARDID='TQCB', weight=1., member_z_bps=100., **IDENTITY)])
    tables = dict(summary=summary, metrics=pd.DataFrame(metrics), comparison_history=pd.DataFrame(history), basket_members=members)
    for name, frame in tables.items():
        frame.to_csv(folder / (name+'.csv'), index=False, encoding='utf-8-sig')
    for path in (root / 'run_manifest.json', folder / 'run_manifest.json'):
        path.write_text(json.dumps(manifest), encoding='utf-8')
    return folder, manifest


def write_recovery_fixture(root, folder, manifest):
    from moex_bonds.analysis.spread_recovery import EPISODE_COLUMNS, RECOVERY_RULE_VERSION
    manifest['recovery_rule_version'] = RECOVERY_RULE_VERSION
    manifest['config']['recovery'] = dict(reference_window=90, min_observations=60,
        entry_scale=1., recovery_fraction=.5, horizon_months=3)
    case = dict(analysis_date=DAY, SECID='A', BOARDID='TQCB', entry_date='2026-05-08',
        reference_bps=150., scale_bps=15., reference_n_obs=90, entry_z_bps=210.,
        initial_excess_bps=60., initial_excess_scales=4., target_bps=180.,
        horizon_end='2026-08-08', end_observation_date='2026-08-07', end_z_bps=150.,
        first_recovery_date='2026-05-12', hit_timing='complete_path', days_to_recovery=4,
        reduction_bps=60., max_worsening_bps=40., n_expected=65, n_valid=65,
        outcome='recovered', reason='', recovery_rule_version=RECOVERY_RULE_VERSION, **IDENTITY)
    pd.DataFrame([case], columns=EPISODE_COLUMNS+list(IDENTITY)).to_csv(folder/'recovery_episodes.csv', index=False)
    summary = pd.read_csv(folder/'summary.csv')
    values = dict(status=['available','unavailable'], reason=['','insufficient_reference'],
        n_known=[1,0], n_recovered=[1,0], n_pending=[0,0], n_incomplete=[0,0],
        share=[1.,None], median_reduction_bps=[60.,None], median_days=[4.,None],
        median_worsening_bps=[40.,None])
    for field, value in values.items():
        summary['recovery__'+field] = value
    summary['recovery_rule_version'] = RECOVERY_RULE_VERSION
    summary.to_csv(folder/'summary.csv', index=False)
    for path in (root/'run_manifest.json', folder/'run_manifest.json'):
        path.write_text(json.dumps(manifest), encoding='utf-8')


class ViewerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = importlib.import_module('moex_bonds.ui.app')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'analysis'
        self.folder, self.manifest = write_fixture(self.root)

    def load(self):
        return self.app.load_snapshot(self.root, DAY)

    def upgrade_robust_fixture(self):
        self.manifest.update(result_schema_version=3,
                             classification_rule_version='upper_tail_midrank_robust_v2')
        self.manifest['config']['classification']['robust_score_threshold'] = 3.5
        for path in self.folder.glob('*.csv'):
            frame = pd.read_csv(path)
            frame['result_schema_version'] = 3
            frame['classification_rule_version'] = 'upper_tail_midrank_robust_v2'
            if path.stem == 'metrics':
                frame['robust_anomaly'] = 'false'
                frame['anomaly_basis'] = 'none'
                frame.loc[frame.window.eq(90), 'robust_anomaly'] = 'true'
                frame.loc[frame.window.eq(90), 'anomaly_basis'] = 'robust_only'
                partial = frame.SECID.eq('B') & frame.comparison.eq('self')
                frame.loc[partial, ['robust_anomaly', 'anomaly_basis']] = None
                frame['scale_reason'] = ''
                frame.loc[partial, 'scale_reason'] = 'zero_mad'
            if path.stem == 'summary':
                for kind in ('self', 'issuer', 'peers', 'market'):
                    for window in (90, 150):
                        frame[f'{kind}__{window}__robust_anomaly'] = 'true' if window == 90 else 'false'
                        frame[f'{kind}__{window}__anomaly_basis'] = 'robust_only' if window == 90 else 'none'
                        if kind == 'self':
                            frame.loc[frame.SECID.eq('B'), [f'{kind}__{window}__robust_anomaly',
                                                          f'{kind}__{window}__anomaly_basis']] = None
            frame.to_csv(path, index=False)
        for path in (self.root/'run_manifest.json', self.folder/'run_manifest.json'):
            path.write_text(json.dumps(self.manifest), encoding='utf-8')

    def test_robust_snapshot_nullable_flags_basis_and_window_display(self):
        old = self.app.assessment_details(self.load(), 'A', 90, 'peers')
        self.assertIn('Основание аномальности', old)
        self.assertTrue(old['Основание аномальности'].eq('Не сохранено').all())
        self.upgrade_robust_fixture()
        snap = self.load()
        self.assertEqual(str(snap.tables['metrics'].robust_anomaly.dtype), 'boolean')
        for window, label in [(90, 'Только robust score'), (150, 'Нет аномалии')]:
            details = self.app.assessment_details(snap, 'A', window, 'peers')
            self.assertTrue(details['Основание аномальности'].eq(label).all())
        details = self.app.assessment_details(snap, 'B', 90, 'peers')
        self.assertIn('Проверка неполная', details['Основание аномальности'].tolist())
        self.assertIn('MAD', details.to_string())

    def test_main_anomaly_column_uses_selected_window_and_preserves_existing_label(self):
        old = self.app.candidate_view(self.load(), 90)
        self.assertIn('anomaly_label', old)
        self.assertTrue(old.anomaly_label.eq('Не сохранено').all())
        self.upgrade_robust_fixture()
        snap = self.load()
        for window, label in [(90, 'Только robust score'), (150, 'Нет аномалии')]:
            view = self.app.candidate_view(snap, window).set_index('SECID')
            self.assertEqual(view.loc['A', 'anomaly_label'], label)
        partial = self.app.candidate_view(snap, 90).set_index('SECID').loc['B', 'anomaly_label']
        self.assertIn('К себе: Проверка неполная', partial)
        self.assertIn('Аналоги: Только robust score', partial)
        self.assertEqual(self.app.LABELS['basis_label'], 'Основание')

    def test_main_table_shows_anomaly_column_without_selecting_a_bond(self):
        from streamlit.testing.v1 import AppTest
        self.upgrade_robust_fixture()
        code = f"from pathlib import Path\nfrom app import render_app\nrender_app(Path({str(self.root)!r}), Path({str(Path(self.temp.name)/'daily_data')!r}))"
        at = AppTest.from_string(code).run(timeout=30)
        at.segmented_control(key='list_scope').set_value('Все выпуски').run(timeout=30)
        for mode in ('Торговля', 'Инвестирование'):
            at.segmented_control(key='mode').set_value(mode).run(timeout=30)
            for window, label in [(90, 'Только robust score'), (150, 'Нет аномалии')]:
                at.segmented_control(key='window').set_value(window).run(timeout=30)
                self.assertFalse(at.exception)
                table = at.dataframe[0].value
                self.assertIn('anomaly_label', table)
                self.assertEqual(table.anomaly_label.iloc[0], label)
                self.assertIn('basis_label', table)
                self.assertEqual(json.loads(at.dataframe[0].proto.columns)['anomaly_label']['label'],
                                 'Критерий аномальности')
                self.assertTrue(any('Выберите облигацию' in str(element.value) for element in at.info))

    def test_robust_snapshot_rejects_missing_invalid_and_inconsistent_fields(self):
        self.upgrade_robust_fixture()
        self.load()
        path = self.folder/'metrics.csv'
        original = path.read_bytes()
        for field, value in [('robust_anomaly', 'invalid'), ('anomaly_basis', 'unknown'),
                             ('anomaly_basis', 'both'), ('anomaly_basis', None)]:
            with self.subTest(field=field, value=value):
                frame = pd.read_csv(path)
                frame.loc[0, field] = value
                frame.to_csv(path, index=False)
                with self.assertRaises(self.app.DataUnavailable):
                    self.load()
                path.write_bytes(original)
        pd.read_csv(path).drop(columns=['robust_anomaly']).to_csv(path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()
        path.write_bytes(original)
        summary_path = self.folder/'summary.csv'
        frame = pd.read_csv(summary_path)
        frame.loc[0, 'peers__90__anomaly_basis'] = 'both'
        frame.to_csv(summary_path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_robust_snapshot_rejects_wrong_rule_pair(self):
        self.upgrade_robust_fixture()
        for path in (self.root/'run_manifest.json', self.folder/'run_manifest.json'):
            changed = dict(self.manifest, classification_rule_version='upper_tail_midrank_v1')
            path.write_text(json.dumps(changed), encoding='utf-8')
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_saved_recovery_view_and_old_snapshot_compatibility(self):
        legacy = self.app.recovery_view(self.load(), ('A','TQCB'))
        self.assertEqual(legacy['status'], 'not_saved')
        write_recovery_fixture(self.root, self.folder, self.manifest)
        snap = self.load()
        view = self.app.recovery_view(snap, ('A','TQCB'))
        self.assertIn('1 из 1', view['message'])
        self.assertIn('50%', view['criterion'])
        self.assertIn('90', view['criterion'])
        self.assertEqual(len(view['episodes']), 1)
        self.assertEqual(self.app.recovery_view(snap, ('B','TQCB'))['status'], 'unavailable')

    def test_declared_recovery_requires_consistent_saved_data(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        path = self.folder/'recovery_episodes.csv'
        original = path.read_bytes()
        path.unlink()
        with self.assertRaises(self.app.DataUnavailable):
            self.load()
        path.write_bytes(original)
        for field, value in [('analysis_run_id','wrong'), ('BOARDID','OTHER'),
                             ('outcome','unknown'), ('recovery_rule_version','wrong')]:
            frame = pd.read_csv(path)
            frame.loc[0, field] = value
            frame.to_csv(path, index=False)
            with self.assertRaises(self.app.DataUnavailable):
                self.load()
            path.write_bytes(original)
        summary_path = self.folder/'summary.csv'
        frame = pd.read_csv(summary_path)
        frame.loc[0,'recovery__share'] = .25
        frame.to_csv(summary_path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_recovery_wording_uses_snapshot_overrides_not_live_defaults(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        self.manifest['config']['recovery'].update(reference_window=120,
            recovery_fraction=.25, horizon_months=1)
        for path in (self.root/'run_manifest.json', self.folder/'run_manifest.json'):
            path.write_text(json.dumps(self.manifest), encoding='utf-8')
        # Update the saved case to this rule, not just its display label.
        path = self.folder/'recovery_episodes.csv'
        frame = pd.read_csv(path)
        frame.loc[0,'horizon_end'] = '2026-06-08'
        frame.loc[0,'end_observation_date'] = '2026-06-08'
        frame.loc[0,'target_bps'] = 195.
        frame.to_csv(path, index=False)
        view = self.app.recovery_view(self.load(), ('A','TQCB'))
        self.assertIn('25%', view['criterion'])
        self.assertIn('120', view['criterion'])
        self.assertIn('1 календар', view['criterion'])

    def test_declared_recovery_without_effective_config_is_rejected(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        del self.manifest['config']['recovery']
        for path in (self.root/'run_manifest.json', self.folder/'run_manifest.json'):
            path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_saved_cases_must_satisfy_snapshot_entry_threshold(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        self.manifest['config']['recovery']['entry_scale'] = 1000000.
        for path in (self.root/'run_manifest.json', self.folder/'run_manifest.json'):
            path.write_text(json.dumps(self.manifest), encoding='utf-8')
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_new_entry_on_analysis_date_is_a_valid_pending_case(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        path = self.folder/'recovery_episodes.csv'
        frame = pd.read_csv(path)
        frame.loc[0,'entry_date'] = DAY
        frame.loc[0,'horizon_end'] = '2027-01-27'
        for field in ('end_observation_date','end_z_bps','first_recovery_date','days_to_recovery','reduction_bps'):
            frame[field] = None
        frame.loc[0,'hit_timing'] = 'not_observed'
        frame.loc[0,'outcome'] = 'pending'
        frame['reason'] = 'awaiting_horizon'
        frame.loc[0,['n_expected','n_valid','max_worsening_bps']] = 0
        frame.to_csv(path, index=False)
        path = self.folder/'summary.csv'
        summary = pd.read_csv(path)
        summary.loc[0,'recovery__status'] = 'no_known_outcomes'
        summary.loc[0,['recovery__n_known','recovery__n_recovered']] = 0
        summary.loc[0,'recovery__n_pending'] = 1
        for field in ('share','median_reduction_bps','median_days','median_worsening_bps'):
            summary.loc[0,'recovery__'+field] = None
        summary.to_csv(path, index=False)
        self.assertEqual(self.app.recovery_view(self.load(), ('A','TQCB'))['status'], 'no_known_outcomes')

    def test_recovery_card_renders_saved_statistics_and_details(self):
        write_recovery_fixture(self.root, self.folder, self.manifest)
        st = MagicMock()
        self.app._render_recovery(st, self.load(), ('A','TQCB'))
        self.assertTrue(any('1 из 1' in str(call) for call in st.write.call_args_list))
        self.assertTrue(any('50%' in str(call) for call in st.caption.call_args_list))
        self.assertEqual(st.dataframe.call_args.args[0].data.shape[0], 1)

    def test_russian_numbers_and_numeric_table_storage(self):
        self.assertEqual(self.app.number(6880), '6\u2009880,0')
        self.assertEqual(self.app.number(-1400.5,signed=True), '−1\u2009400,5')
        self.assertEqual(self.app.number(0,signed=True), '0,0')
        frame = self.app.candidate_view(self.load(),90)
        styled = self.app.style_table(frame[['z_bps','change_1d_bps']])
        self.assertIn('+20,0',styled.to_html())
        self.assertEqual(styled.data.z_bps.iloc[0],140.)

    def test_sparkline_preserves_null_dates_and_uses_only_current_snapshot(self):
        snap = self.load()
        history = snap.tables['comparison_history']
        history.loc[history.comparison.eq('self') & history.observation_date.eq('2026-10-26'),'z_bps'] = float('nan')
        self.assertEqual(self.app.candidate_view(snap,90).iloc[0].z_history_30,[100.,None,140.])
        import base64
        from xml.etree import ElementTree as ET
        svg = ET.fromstring(base64.b64decode(self.app.spread_sparkline([100.,None,140.]).split(',')[1]))
        self.assertEqual(len(svg.findall('.//{http://www.w3.org/2000/svg}path')),2)
        self.assertIsNone(self.app.spread_sparkline([None,None]))

    @unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit runtime required')
    def test_negative_and_zero_metric_arrows_follow_numeric_delta(self):
        from streamlit.testing.v1 import AppTest
        from streamlit.proto.Metric_pb2 import Metric
        path = self.folder/'metrics.csv'
        metrics = pd.read_csv(path)
        code = f"from pathlib import Path\nfrom app import render_app\nrender_app(Path({str(self.root)!r}), Path({str(Path(self.temp.name)/'daily_data')!r}))"
        for value,direction in [(-20.,Metric.DOWN),(0.,Metric.NONE)]:
            metrics.loc[metrics.SECID.eq('A') & metrics.comparison.eq('peers') & metrics.window.eq(90),'excess_bps'] = value
            metrics.to_csv(path,index=False)
            at = AppTest.from_string(code).run(timeout=30)
            at.session_state['selected_bond'] = ('A','TQCB')
            at.run(timeout=30)
            self.assertFalse(at.exception)
            self.assertEqual(at.metric[3].proto.direction,direction)

    def test_scatter_quantile_view_retains_full_payload_and_has_legend(self):
        snap = self.load()
        view = self.app.candidate_view(snap,90)
        view = pd.concat([view.iloc[:1]]*101,ignore_index=True)
        view['SECID'] = [str(n) for n in range(101)]
        view['z_bps'] = list(range(100))+[6000.]
        spec,_,_ = self.app.scatter_spec(view,view,snap,None)
        self.assertEqual(len(spec['data']['values']),101)
        self.assertEqual(spec['encoding']['y']['scale']['domain'],[1.,99.])
        self.assertEqual(spec['encoding']['color']['legend']['orient'],'top')
        self.assertEqual(spec['data']['values'][-1]['spread_text'],'6\u2009000,0')
        all_spec,_,_ = self.app.scatter_spec(view,view,snap,None,show_all=True)
        self.assertNotIn('domain',all_spec['encoding']['y']['scale'])

    def test_history_annotations_and_bottom_only_dates(self):
        frame,own,premium = self.app.history_view(self.load(),'A','peers',150)
        layers = self.app.build_history_spec(frame,own,premium,DAY,'peers')['vconcat'][0]['layer']
        self.assertTrue(any(l['mark']['type']=='area' for l in layers))
        self.assertTrue(any(l['mark']['type']=='text' for l in layers))
        self.assertFalse(next(l for l in layers if l['mark']['type']=='line')['encoding']['x']['axis']['labels'])

    @unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit runtime not installed in this interpreter')
    def test_recovery_card_runtime_and_investing_do_not_share_the_block(self):
        from streamlit.testing.v1 import AppTest
        write_recovery_fixture(self.root, self.folder, self.manifest)
        code = f"from pathlib import Path\nfrom app import render_app\nrender_app(Path({str(self.root)!r}), Path({str(Path(self.temp.name)/'daily_data')!r}))"
        at = AppTest.from_string(code).run(timeout=30)
        at.session_state['selected_bond'] = ('A','TQCB')
        at.run(timeout=30)
        self.assertFalse(at.exception)
        self.assertTrue(any('1 из 1' in str(element.value) for element in at.markdown))
        at.segmented_control(key='mode').set_value('Инвестирование').run(timeout=30)
        self.assertFalse(at.exception)
        self.assertFalse(any('История восстановления собственного спреда' == element.value for element in at.subheader))

    def test_complete_snapshot_is_read_only_and_window_values_change(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        snap = self.load()
        a90 = self.app.candidate_view(snap, 90).set_index('SECID').loc['A']
        a150 = self.app.candidate_view(snap, 150).set_index('SECID').loc['A']
        self.assertEqual(a90['peers_delta'], 20.)
        self.assertEqual(a150['peers_delta'], 30.)
        self.assertEqual(a90['candidate_status'], 'candidate')
        self.assertEqual(a150['candidate_status'], 'not_candidate')
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_root_publishing_and_lock_block_dated_ok_snapshot(self):
        marker = self.root / 'run_manifest.json'
        marker.write_text(json.dumps(dict(self.manifest, status='publishing')), encoding='utf-8')
        with self.assertRaises(self.app.PublicationBusy):
            self.load()
        marker.write_text(json.dumps(self.manifest), encoding='utf-8')
        (self.root / '.daily_write.lock').write_text('123')
        with self.assertRaises(self.app.PublicationBusy):
            self.load()

    def test_malformed_manifest_is_a_data_state_not_an_unhandled_exception(self):
        (self.root / 'run_manifest.json').write_text('[]')
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_missing_view_columns_are_rejected_as_unavailable(self):
        path = self.folder / 'metrics.csv'
        pd.read_csv(path).drop(columns=['excess_bps']).to_csv(path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_invalid_exclusions_fail_locally_without_mixing_runs(self):
        path = self.folder / 'candidate_exclusions.csv'
        valid = dict(analysis_date=DAY, SECID='A', comparison='peers', candidate_SECID='B',
                     reason='rating_outside_limit', **IDENTITY)
        for row in (dict(other='bad columns'), dict(valid, analysis_run_id='old-run'),
                    dict(valid, analysis_date='2026-10-26')):
            with self.subTest(row=row):
                pd.DataFrame([row]).to_csv(path, index=False)
                snapshot = self.load()
                self.assertTrue(snapshot.tables['candidate_exclusions'].empty)
                self.assertTrue(snapshot.exclusions_reason)
                self.assertEqual(len(self.app.candidate_view(snapshot, 90)), 2)
        pd.DataFrame([valid]).to_csv(path, index=False)
        self.assertEqual(len(self.load().tables['candidate_exclusions']), 1)

    def test_mid_read_publication_does_not_return_partial_data(self):
        original = Path.read_bytes
        def read(path):
            value = original(path)
            if path == self.folder / 'summary.csv':
                (self.root / 'run_manifest.json').write_text(json.dumps(dict(self.manifest, status='publishing')))
            return value
        with patch.object(Path, 'read_bytes', read):
            with self.assertRaises(self.app.PublicationBusy):
                self.load()

    def test_old_date_can_have_different_root_run(self):
        (self.root / 'run_manifest.json').write_text(json.dumps(dict(self.manifest, analysis_date='2026-10-28', analysis_run_id='run-b')))
        self.assertEqual(self.load().manifest['analysis_run_id'], 'run-a')

    def test_mixed_run_duplicate_key_missing_table_and_future_schema_fail(self):
        path = self.folder / 'metrics.csv'
        good = path.read_bytes()
        frame = pd.read_csv(path)
        frame.loc[0, 'analysis_run_id'] = 'wrong'
        frame.to_csv(path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()
        path.write_bytes(good)
        frame = pd.read_csv(path)
        pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
        with self.assertRaises(self.app.DataUnavailable):
            self.load()
        path.write_bytes(good)
        path.unlink()
        with self.assertRaises(self.app.DataUnavailable):
            self.load()
        path.write_bytes(good)
        (self.folder / 'run_manifest.json').write_text(json.dumps(dict(self.manifest, result_schema_version=99)))
        with self.assertRaises(self.app.DataUnavailable):
            self.load()

    def test_legacy_is_available_without_inventing_classification(self):
        for path in (self.root / 'run_manifest.json', self.folder / 'run_manifest.json'):
            path.write_text(json.dumps({k:v for k,v in self.manifest.items() if k not in IDENTITY}))
        for path in self.folder.glob('*.csv'):
            frame = pd.read_csv(path).drop(columns=list(IDENTITY))
            frame.to_csv(path, index=False)
        view = self.app.candidate_view(self.load(), 90)
        self.assertEqual(set(view.candidate_status), {'not_saved'})

    def test_empty_baskets_and_nullable_false_remain_distinct(self):
        path = self.folder / 'basket_members.csv'
        pd.read_csv(path).iloc[:0].to_csv(path, index=False)
        snap = self.load()
        metrics = snap.tables['metrics']
        self.assertFalse(bool(metrics.upper_tail_anomaly.iloc[0]))
        self.assertTrue(pd.isna(metrics.candidate_qualified.iloc[-1]))
        self.assertTrue(snap.tables['basket_members'].empty)
        self.assertEqual(self.app.candidate_view(snap, 90).set_index('SECID').loc['B', 'candidate_status'], 'not_fully_assessed')

    def test_history_keeps_own_observations_when_premium_has_a_gap(self):
        snap = self.load()
        frame, own, premium = self.app.history_view(snap, 'A', 'peers', 90)
        self.assertEqual(frame.observation_date.tolist(), ['2026-10-26', DAY])
        self.assertEqual(frame.own_z.tolist(), [120., 140.])
        self.assertTrue(pd.isna(frame.premium.iloc[0]))
        self.assertEqual(frame.premium.iloc[1], 40.)
        self.assertEqual(own.hist_median_bps, 120.)
        self.assertEqual(premium.hist_median_bps, 120.)

    def test_filters_sort_and_selection_use_identity_not_name_or_position(self):
        snap = self.load()
        view = self.app.candidate_view(snap, 90)
        ordered = self.app.sort_view(view, 'adv20', False)
        self.assertEqual(ordered.SECID.tolist(), ['A','B'])
        self.assertEqual(self.app.selected_identity(ordered, [1]), ('B','TQCB'))
        only = self.app.filter_view(view, dict(status=['candidate'], min_liquidity=500000))
        self.assertEqual(only.SECID.tolist(), ['A'])
        self.assertEqual(len(snap.tables['basket_members']), 1)
        self.assertEqual(len(self.app.filter_view(view, dict(search='Other'))), 1)

    def test_issuer_aliases_use_snapshot_config(self):
        snap = self.load()
        values = self.app.issuer_keys(pd.Series([' Alias ', 'ALPHA']), snap.manifest)
        self.assertEqual(values.tolist(), ['alpha','alpha'])
        with self.assertRaises(self.app.DataUnavailable):
            self.app.issuer_keys(pd.Series(['Alias']), dict(config={}))
        with self.assertRaises(self.app.DataUnavailable):
            self.app.issuer_keys(pd.Series(['Alias']), dict(config=dict(issuer_aliases={'Alias':'Alpha','Alpha':'Alias'})))

    def test_enrichment_accepts_only_the_checked_workbook_bytes(self):
        snap = self.load()
        daily = Path(self.temp.name) / 'daily_data'
        (daily / 'daily').mkdir(parents=True)
        path = daily / 'daily' / 'bonds_daily.xlsx'
        wb = Workbook()
        sheet = wb.active
        sheet.title = 'Итог'
        columns = ['SECID','ISSUER','RATING','SECTOR','TRADEDATE','BOARDID','ZSPREADATWAPRICE','DURATION_YEARS','NUMTRADES','TRADE_STATUS','YTM_WAPRICE','MATDATE']
        sheet.append(columns)
        sheet.append(['A','Alias','AAA','Banks',DAY,'TQCB',1.4,1,1,'traded',17.,'2027-10-27'])
        wb.create_sheet('Поля').append(['SECID [SECID]'])
        wb.save(path)
        snap.manifest['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        enriched, reason = self.app.load_enrichment(snap, daily)
        self.assertFalse(reason)
        self.assertEqual(enriched.iloc[0].YTM_WAPRICE, 17.)
        original = Path.read_bytes
        checked = path.read_bytes()
        def replace_after_capture(target):
            raw = original(target)
            if target == path:
                target.write_bytes(b'replaced after capture')
            return raw
        with patch.object(Path,'read_bytes',replace_after_capture):
            enriched, reason = self.app.load_enrichment(snap,daily)
        self.assertFalse(reason)
        self.assertEqual(enriched.iloc[0].YTM_WAPRICE,17.)
        path.write_bytes(checked)
        path.write_bytes(path.read_bytes()+b'changed revision')
        enriched, reason = self.app.load_enrichment(snap, daily)
        self.assertTrue(enriched.empty)
        self.assertTrue(reason)
        path.write_bytes(b'not an Excel workbook')
        snap.manifest['source_sha256'][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
        enriched, reason = self.app.load_enrichment(snap, daily)
        self.assertTrue(enriched.empty)
        self.assertTrue(reason)

    def test_curve_is_optional_and_date_bound(self):
        daily = Path(self.temp.name) / 'daily_data'
        (daily / '_support').mkdir(parents=True)
        status = dict(status='ok', trade_date=DAY)
        (daily / 'run_manifest.json').write_text(json.dumps(dict(collection_status='ok', trade_date=DAY, cbr_zcyc=status)))
        (daily / '_support' / 'cbr_zcyc_status.json').write_text(json.dumps(status))
        frame = pd.DataFrame({'Дата данных':[DAY]*12, 'Срок до погашения, лет':[.25,.5,.75,1,2,3,5,7,10,15,20,30], 'Доходность, % годовых':[13.]*12})
        frame.to_csv(daily/'cbr_zcyc.csv', index=False)
        data, reason = self.app.load_curve(DAY, daily)
        self.assertFalse(reason)
        self.assertEqual(len(data),12)
        original = Path.read_bytes
        status_path = daily / '_support' / 'cbr_zcyc_status.json'
        def replace_during_capture(target):
            raw = original(target)
            if target == daily/'cbr_zcyc.csv':
                status_path.write_text(json.dumps(dict(status, status='updating')))
            return raw
        with patch.object(Path,'read_bytes',replace_during_capture):
            data, reason = self.app.load_curve(DAY,daily)
        self.assertTrue(data.empty)
        self.assertTrue(reason)
        status_path.write_text(json.dumps(status))
        data, reason = self.app.load_curve('2026-10-26', daily)
        self.assertTrue(data.empty)
        self.assertTrue(reason)
        (daily / 'run_manifest.json').write_text(json.dumps(dict(collection_status='ok', trade_date=DAY, cbr_zcyc=[])))
        data, reason = self.app.load_curve(DAY, daily)
        self.assertTrue(data.empty)
        self.assertTrue(reason)

    def test_assessment_details_explain_both_directions_and_daily_change(self):
        snap = self.load()
        metrics = snap.tables['metrics']
        metrics.loc[metrics.comparison.eq('self'), 'change_1d_reason'] = 'previous_trading_date_unavailable'
        metrics.loc[metrics.comparison.eq('peers'), 'quality_flags'] = 'basket_composition_differs_from_current'
        details = self.app.assessment_details(snap, 'A', 90, 'peers')
        text = details.to_string(index=False)
        self.assertIn('Нет подтверждённого значения за предыдущую торговую дату', text)
        self.assertIn('В отдельные даты доступна только часть текущего состава корзины', text)

    def test_zoom_is_restored_only_within_the_same_bond_window_context(self):
        state = {}
        full = [1792713600000., 1793059200000.]
        self.app.history_domain(state, 'A:90', full)
        self.app.capture_zoom(state, 'A:90', {'zoom_x': {'observation_date':[1792800000000.,1792972800000.]}}, full)
        self.assertEqual(self.app.history_domain(state, 'A:90', full), [1792800000000.,1792972800000.])
        self.app.capture_zoom(state, 'old-context', {'zoom_x': {}}, full)
        self.assertEqual(state['history_domain'], [1792800000000.,1792972800000.])
        self.assertEqual(self.app.history_domain(state, 'B:150', full), full)

    def test_history_chart_carries_null_premium_without_erasing_own_spread(self):
        frame, own, premium = self.app.history_view(self.load(), 'A', 'peers', 150)
        spec = self.app.build_history_spec(frame, own, premium, DAY, 'peers', None)
        data = spec['data']['values']
        self.assertEqual([r['own_z'] for r in data], [100.,120.,140.])
        self.assertIsNone(data[1]['premium'])
        json.dumps(spec, allow_nan=False)

    def test_rebuilt_chart_restores_scale_domain_without_filtering_history(self):
        frame, own, premium = self.app.history_view(self.load(), 'A', 'peers', 150)
        domain = [1792800000000.,1792972800000.]
        spec = self.app.build_history_spec(frame, own, premium, DAY, 'peers', domain)
        self.assertEqual(spec['vconcat'][0]['layer'][0]['encoding']['x']['scale'].get('domain'),
                         ['2026-10-24T00:00:00+00:00','2026-10-26T00:00:00+00:00'])
        self.assertEqual(len(spec['data']['values']),3)

    def test_shortlist_keeps_numeric_issuer_only_values_and_full_scatter(self):
        snap = self.load()
        snap.tables['summary']['candidate__90__basis'] = ['issuer','peers']
        snap.tables['summary']['candidate__90__status'] = ['candidate','candidate']
        snap.tables['metrics'].loc[snap.tables['metrics'].SECID.eq('B'), 'excess_bps'] = 50.
        full = self.app.candidate_view(snap,90)
        background, table = self.app.browsing_views(full,{},True)
        self.assertEqual(self.app.sort_view(table,'peers_delta',False).SECID.tolist(),['B','A'])
        self.assertIn('эмитенту',full.iloc[0].basis_label)
        full.loc[full.SECID.eq('A'),'peers_delta'] = float('nan')
        self.assertEqual(self.app.sort_view(full,'peers_delta',False).SECID.tolist(),['B','A'])
        full.loc[full.SECID.eq('B'),'candidate_status'] = 'not_candidate'
        background, table = self.app.browsing_views(full,{'status':['not_candidate']},True)
        self.assertEqual(background.SECID.tolist(),['A','B'])
        self.assertEqual(table.SECID.tolist(),['A'])
        spec,_,_ = self.app.scatter_spec(background,full,snap,None)
        self.assertEqual({r['SECID'] for r in spec['data']['values']},{'A','B'})
        snap.tables['summary']['candidate__90__incomplete_directions'] = ['peers','']
        self.assertIn('Нет полного вывода по аналогам',self.app.candidate_view(snap,90).iloc[0].basis_label)
        _, table = self.app.browsing_views(full,{'status':['not_candidate']},False)
        self.assertEqual(table.SECID.tolist(),['B'])
        self.assertEqual(self.app.active_filter_count({'status':['not_candidate']},None,True),0)
        self.assertEqual(self.app.active_filter_count({'status':['not_candidate']},None,False),1)

    def test_shortlist_empty_states_and_partial_classification(self):
        self.assertEqual(self.app.active_filter_count({'search':None},None),0)
        full = self.app.candidate_view(self.load(),90)
        empty = full.iloc[:0]
        self.assertIn('по выбранным фильтрам',self.app.shortlist_notice(full,empty))
        full['candidate_status'] = ['not_candidate','not_saved']
        self.assertIn('Среди бумаг',self.app.shortlist_notice(full,empty))
        full['candidate_status'] = 'not_saved'
        self.assertIn('не сохранён',self.app.shortlist_notice(full,empty))

    def test_headline_values_use_distinct_saved_fields_by_window_and_basket(self):
        snap = self.load()
        metrics = snap.tables['metrics']
        own = metrics.SECID.eq('A') & metrics.window.eq(90) & metrics.comparison.eq('self')
        peers = metrics.SECID.eq('A') & metrics.window.eq(90) & metrics.comparison.eq('peers')
        metrics.loc[own,'current_metric_bps'] = 180.
        metrics.loc[peers,['benchmark_bps','current_metric_bps','excess_bps']] = [130.,50.,20.]
        self.assertEqual(self.app.headline_values(snap,'A','peers',90),(180.,130.,50.,20.))
        market = metrics.SECID.eq('A') & metrics.window.eq(150) & metrics.comparison.eq('market')
        metrics.loc[market,['benchmark_bps','current_metric_bps','excess_bps']] = [140.,0.,float('nan')]
        values = self.app.headline_values(snap,'A','market',150)
        self.assertEqual(values[:3],(140.,140.,0.))
        self.assertTrue(pd.isna(values[3]))
        metrics.drop(columns=['benchmark_bps'],inplace=True)
        self.assertIn('Спред корзины не сохранён',self.app.assessment_details(snap,'A',90,'peers').to_string(index=False))

    @unittest.skipUnless(importlib.util.find_spec('streamlit'), 'Streamlit runtime not installed in this interpreter')
    def test_interface_opens_without_autoselection_and_investment_columns_are_visible(self):
        from streamlit.testing.v1 import AppTest
        code = f"from pathlib import Path\nfrom app import render_app\nrender_app(Path({str(self.root)!r}), Path({str(Path(self.temp.name)/'daily_data')!r}))"
        at = AppTest.from_string(code).run(timeout=30)
        self.assertFalse(at.exception)
        self.assertTrue(any('Выберите облигацию' in str(element.value) for element in at.info))
        self.assertEqual(len(at.dataframe[0].value),1)
        at.multiselect(key='filter_issuer').set_value(['Other']).run(timeout=30)
        self.assertEqual(len(at.dataframe[0].value), 0)
        at.multiselect(key='filter_issuer').set_value([]).run(timeout=30)
        self.assertEqual(len(at.dataframe[0].value), 1)
        at.segmented_control(key='list_scope').set_value('Все выпуски').run(timeout=30)
        self.assertEqual(len(at.dataframe[0].value), 2)
        at.multiselect(key='filter_status').set_value(['not_fully_assessed']).run(timeout=30)
        at.segmented_control(key='list_scope').set_value('Выделены анализом').run(timeout=30)
        self.assertEqual(at.multiselect(key='filter_status').value,['not_fully_assessed'])
        self.assertTrue(at.multiselect(key='filter_status').disabled)
        self.assertEqual(len(at.dataframe[0].value),1)
        at.segmented_control(key='list_scope').set_value('Все выпуски').run(timeout=30)
        self.assertEqual(len(at.dataframe[0].value),1)
        at.button(key='reset_filters').click().run(timeout=30)
        at.session_state['selected_bond'] = ('A','TQCB')
        at.run(timeout=30)
        self.assertFalse(at.exception)
        self.assertIn('Same name', [element.value for element in at.subheader])
        full_domain = [self.app._milliseconds('2026-10-26'),self.app._milliseconds(DAY)]
        at.session_state['history_domain'] = [full_domain[0]+1000,full_domain[1]-1000]
        at.run(timeout=30)
        at.button(key='reset_history_zoom').click().run(timeout=30)
        self.assertEqual(at.session_state['history_domain'],full_domain)
        first_table = at.dataframe[0].proto.id
        at.session_state['selected_bond'] = ('B','TQCB')
        at.run(timeout=30)
        self.assertNotEqual(first_table, at.dataframe[0].proto.id)
        at.selectbox(key='sort_field').set_value('adv20').run(timeout=30)
        self.assertFalse(at.session_state['sort_ascending'])
        at.segmented_control(key='mode').set_value('Инвестирование').run(timeout=30)
        self.assertFalse(at.exception)
        self.assertIn('YTM_WAPRICE', at.dataframe[0].value.columns)
        self.assertEqual(at.session_state['sort_field'],'adv20')


if __name__ == '__main__':
    unittest.main()
