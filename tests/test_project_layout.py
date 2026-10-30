"""Contracts for relocation: fixed defaults and historical input provenance."""
from pathlib import Path
import unittest


class LayoutTests(unittest.TestCase):
    def test_defaults_do_not_depend_on_current_directory(self):
        from moex_bonds.paths import PROJECT_DIR, DAILY_DIR, HISTORY_DIR, ANALYSIS_DIR, CONFIG_FILE
        from moex_bonds.collection.main import parse_args
        from unittest.mock import patch
        root = Path(__file__).resolve().parent
        if root.name == 'tests':
            root = root.parent
        self.assertEqual(PROJECT_DIR, root)
        self.assertEqual(DAILY_DIR, root / 'data/daily_data')
        self.assertEqual(HISTORY_DIR, root / 'data/history_data')
        self.assertEqual(ANALYSIS_DIR, root / 'data/spread_analysis')
        self.assertTrue(CONFIG_FILE.is_file())
        with patch('sys.argv', ['main.py']):
            self.assertEqual(Path(parse_args().output), DAILY_DIR)

    def test_legacy_daily_provenance_is_exact_and_custom_paths_stay_isolated(self):
        from moex_bonds.paths import PROJECT_DIR, DAILY_DIR, recorded_source_hash
        old = PROJECT_DIR / 'daily_data/daily/bonds_daily.xlsx'
        current = DAILY_DIR / 'daily/bonds_daily.xlsx'
        hashes = {str(old): 'old-hash'}
        self.assertEqual(recorded_source_hash(hashes, current), 'old-hash')
        self.assertIsNone(recorded_source_hash(hashes, PROJECT_DIR / 'other/daily/bonds_daily.xlsx'))
        self.assertEqual(recorded_source_hash({**hashes, str(current): 'new-hash'}, current), 'new-hash')


if __name__ == '__main__':
    unittest.main()
