"""Overwrite only the current daily outputs; never accumulate run folders."""
import argparse
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

DAY = date(2026, 10, 26)


class DailyStorageTests(unittest.TestCase):
    def test_default_main_directory_is_fixed_daily_data(self):
        from moex_bonds.collection.main import parse_args
        with patch('sys.argv', ['main.py']):
            output = Path(parse_args().output)
        self.assertEqual(output, Path(__file__).resolve().parents[1] / 'data/daily_data')

    def test_two_runs_replace_every_daily_source_and_clear_old_support(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0, timeout=1,
                                      daily=True, csv_only=False)

            def collect(client, path, section, day):
                self.assertEqual(client.output, root)
                return (pd.DataFrame([dict(SECID='TEST', BOARDID='TQCB', TRADEDATE=day)])
                        if section == 'history' else pd.DataFrame())

            def cbr(filename):
                def fetch(day, output, **kwargs):
                    self.assertEqual(output, root)
                    (output / filename).write_text(day.isoformat(), encoding='utf-8')
                    return dict(status='ok', trade_date=day.isoformat())
                return fetch

            def daily(output, **kwargs):
                self.assertEqual(output, root)
                dest = output / 'daily'
                dest.mkdir(exist_ok=True)
                (dest / 'bonds_daily.csv').write_text(args.date.isoformat(), encoding='utf-8')
                if kwargs['xlsx']:
                    (dest / 'bonds_daily.xlsx').write_bytes(b'first workbook')
                return {'metadata': {'trade_date': args.date.isoformat()}}

            with (patch('moex_bonds.collection.main.complete_day', side_effect=collect), patch('moex_bonds.collection.main.MoexClient.get', return_value=None),
                  patch('moex_bonds.collection.main.run_zcyc', side_effect=cbr('cbr_zcyc.csv')),
                  patch('moex_bonds.collection.main.run_ruonia', side_effect=cbr('cbr_ruonia.csv')),
                  patch('moex_bonds.collection.daily.run_daily', side_effect=daily)):
                self.assertEqual(run(args), 0)
                (root / 'raw_responses/old_only.json').write_text('old', encoding='utf-8')
                (root / 'daily/_support/smartlab_raw').mkdir(parents=True)
                (root / 'daily/_support/smartlab_raw/old.txt').write_text('old', encoding='utf-8')
                (root / '_support').mkdir(exist_ok=True)
                (root / '_support/old_cbr.html').write_text('old', encoding='utf-8')
                (root / 'errors.jsonl').write_text('old error', encoding='utf-8')
                args.date, args.csv_only = date(2026, 10, 27), True
                self.assertEqual(run(args), 0)
            self.assertFalse([p for p in root.glob('run_*') if p.is_dir()])
            self.assertFalse((root / 'raw_responses/old_only.json').exists())
            self.assertFalse((root / 'daily/_support/smartlab_raw/old.txt').exists())
            self.assertFalse((root / '_support/old_cbr.html').exists())
            self.assertEqual((root / 'errors.jsonl').read_text(encoding='utf-8'), '')
            self.assertFalse((root / 'daily/bonds_daily.xlsx').exists())
            self.assertEqual((root / 'daily/bonds_daily.csv').read_text(encoding='utf-8'), '2026-10-27')
            for name in ('cbr_zcyc.csv', 'cbr_ruonia.csv'):
                self.assertEqual((root / name).read_text(encoding='utf-8'), '2026-10-27')
            m = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(m['trade_date'], '2026-10-27')
            self.assertEqual(m['cbr_zcyc']['trade_date'], m['cbr_ruonia']['trade_date'])

    def test_failed_new_run_removes_all_old_tables_and_records_current_failure(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'daily').mkdir()
            for name in ('daily/bonds_daily.csv', 'daily/bonds_daily.xlsx', 'cbr_zcyc.csv',
                         'cbr_ruonia.csv', 'endpoints_status.csv', 'run_manifest.json'):
                (root / name).write_text('old', encoding='utf-8')
            args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0, timeout=1, daily=False)

            def fail_date(*args, **kwargs):
                manifest = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['collection_status'], 'incomplete')
                raise ValueError('network failure')

            with (patch('moex_bonds.collection.main.latest_trade_date', side_effect=fail_date),
                  patch('moex_bonds.collection.main.run_zcyc') as curve, patch('moex_bonds.collection.main.run_ruonia') as ruonia):
                self.assertEqual(run(args), 2)
                curve.assert_not_called()
                ruonia.assert_not_called()
            self.assertFalse(list(root.glob('run_*/*')))
            for name in ('daily/bonds_daily.csv', 'daily/bonds_daily.xlsx', 'cbr_zcyc.csv', 'cbr_ruonia.csv'):
                self.assertFalse((root / name).exists(), name)
            m = json.loads((root / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(m['collection_status'], 'incomplete')
            self.assertEqual(m['error'], 'network failure')

    def test_cleanup_preserves_unrelated_files_archives_and_explicit_history(self):
        from moex_bonds.storage.daily_storage import reset_daily_output
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'raw_responses').mkdir()
            (root / 'raw_responses/old.json').write_text('old')
            (root / 'cbr_ruonia.csv').write_text('old')
            for directory in ('history_180d', 'run_old_archive'):
                (root / directory).mkdir()
                (root / directory / 'keep.txt').write_text('keep')
            (root / 'notes.txt').write_text('keep')
            reset_daily_output(root)
            self.assertFalse((root / 'raw_responses').exists())
            self.assertFalse((root / 'cbr_ruonia.csv').exists())
            for name in ('notes.txt', 'history_180d/keep.txt', 'run_old_archive/keep.txt'):
                self.assertEqual((root / name).read_text(), 'keep')

    def test_cleanup_preflight_rejects_links_before_deleting_anything(self):
        from moex_bonds.storage.daily_storage import reset_daily_output
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bonds_universe.csv').write_text('keep on rejection')
            (root / 'raw_responses').mkdir()
            with patch.object(Path, 'is_symlink', lambda path: path.name == 'raw_responses'):
                with self.assertRaises(ValueError):
                    reset_daily_output(root)
            self.assertEqual((root / 'bonds_universe.csv').read_text(), 'keep on rejection')

    def test_locked_csv_cannot_leave_old_success_manifest_after_partial_cleanup(self):
        from moex_bonds.storage.daily_storage import reset_daily_output
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bonds_universe.csv').write_text('old universe')
            (root / 'day_quotes.csv').write_text('old quotes')
            status = root / 'run_manifest.json'
            status.write_text(json.dumps({'mode': 'day', 'collection_status': 'ok'}), encoding='utf-8')
            unlink = Path.unlink

            def locked(path, *args, **kwargs):
                if path == root / 'day_quotes.csv':
                    raise PermissionError('CSV is open in Excel')
                return unlink(path, *args, **kwargs)

            with patch.object(Path, 'unlink', locked):
                with self.assertRaises(PermissionError):
                    reset_daily_output(root)
            saved = json.loads(status.read_text(encoding='utf-8'))
            self.assertEqual(saved['collection_status'], 'incomplete')
            self.assertEqual(saved['cleanup_status'], 'error')
            self.assertIn('PermissionError', saved['error'])

    def test_lock_rejects_second_writer_and_releases_after_failure(self):
        from moex_bonds.storage.daily_storage import output_lock
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                with output_lock(root):
                    with self.assertRaisesRegex(RuntimeError, 'занята'):
                        with output_lock(root):
                            self.fail('second writer acquired lock')
                    raise RuntimeError('fixture failure')
            with output_lock(root):
                self.assertTrue((root / '.daily_write.lock').exists())
            self.assertFalse((root / '.daily_write.lock').exists())

    def test_project_and_drive_root_cannot_be_cleanup_targets(self):
        from moex_bonds.storage.daily_storage import reset_daily_output
        for target in (Path(__file__).resolve().parents[1], Path(Path(__file__).anchor)):
            with self.subTest(target=target), self.assertRaises(ValueError):
                reset_daily_output(target)

    def test_both_cbr_clis_default_to_same_folder_and_refresh_matching_manifest(self):
        import moex_bonds.sources.cbr_zcyc as cbr_zcyc
        import moex_bonds.sources.cbr_ruonia as cbr_ruonia
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = dict(mode='day', collection_status='ok', trade_date=DAY.isoformat())
            path = root / 'run_manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            for module, function, key in ((cbr_zcyc, 'run_zcyc', 'cbr_zcyc'), (cbr_ruonia, 'run_ruonia', 'cbr_ruonia')):
                result = dict(status='ok', trade_date=DAY.isoformat(), date_from='2026-10-19')
                with (patch.object(module, 'DEFAULT_DAILY_OUTPUT', root),
                      patch.object(module, function, return_value=result) as fetch,
                      patch('sys.argv', [module.__name__ + '.py', '--date', DAY.isoformat()])):
                    self.assertEqual(module.main(), 0)
                    self.assertEqual(fetch.call_args.args[:2], (DAY, root))
                saved = json.loads(path.read_text(encoding='utf-8'))
                self.assertEqual(saved[key]['status'], 'ok')
            self.assertEqual(set(saved) - set(manifest), {'cbr_zcyc', 'cbr_ruonia'})

    def test_separate_cbr_cli_rejects_date_mismatch_without_deleting_table(self):
        import moex_bonds.sources.cbr_zcyc as cbr_zcyc
        import moex_bonds.sources.cbr_ruonia as cbr_ruonia
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'run_manifest.json').write_text(json.dumps(dict(mode='day', collection_status='ok',
                trade_date=DAY.isoformat())), encoding='utf-8')
            for module, function, filename in ((cbr_zcyc, 'run_zcyc', 'cbr_zcyc.csv'),
                                               (cbr_ruonia, 'run_ruonia', 'cbr_ruonia.csv')):
                (root / filename).write_text('keep')
                with (patch.object(module, function) as fetch,
                      patch('sys.argv', [module.__name__ + '.py', '--date', '2026-10-27', '--output', folder])):
                    with self.assertRaises(SystemExit):
                        module.main()
                    fetch.assert_not_called()
                self.assertEqual((root / filename).read_text(), 'keep')


if __name__ == '__main__':
    unittest.main()
