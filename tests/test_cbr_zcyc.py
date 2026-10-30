"""Offline checks for CBR parsing, transport and synchronization with MOEX."""
import argparse
from datetime import date
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd
import requests

import moex_bonds.sources.cbr_zcyc as cbr

DAY = date(2026, 10, 27)
TERMS = ['0,25', '0,5', '0,75', '1', '2', '3', '5', '7', '10', '15', '20', '30']
RATES = ['13,09', '13,43', '13,73', '14,01', '14,87', '15,43',
         '16,03', '16,28', '16,43', '16,52', '16,56', '16,59']


def html(day='27.10.2026', rates=None, terms=None):
    header = ''.join(f'<th>{t}</th>' for t in (terms or TERMS))
    values = ''.join(f'<td>{v}</td>' for v in (rates if rates is not None else RATES))
    return ('<html><meta charset="utf-8"><table class="data spaced">'
            '<tr><th rowspan="2">Дата</th><th colspan="12">Срок до погашения, лет</th></tr>'
            f'<tr>{header}</tr><tr><td><a>{day}</a></td>{values}</tr></table></html>')


def response(text=None, status=200):
    r = requests.Response()
    r.status_code = status
    r.url = cbr.URL
    r._content = (text if text is not None else html()).encode('utf-8')
    return r


class ParseTests(unittest.TestCase):
    def test_exact_date_and_percentage_points(self):
        rows = cbr.parse_zcyc(html(), DAY)
        self.assertEqual(len(rows), 12)
        self.assertEqual(rows[0], dict(trade_date='2026-10-27', maturity_years=.25, yield_pct=13.09))
        self.assertEqual(rows[-1]['maturity_years'], 30)
        self.assertEqual(rows[-1]['yield_pct'], 16.59)

    def test_header_order_drives_mapping(self):
        rows = cbr.parse_zcyc(html(terms=list(reversed(TERMS)), rates=list(reversed(RATES))), DAY)
        self.assertEqual(rows[0]['maturity_years'], .25)
        self.assertEqual(rows[0]['yield_pct'], 13.09)

    def test_other_date_is_not_substituted(self):
        with self.assertRaises(cbr.NoCurveForDate):
            cbr.parse_zcyc(html(day='26.10.2026'), DAY)

    def test_empty_valid_table_has_unavailable_status(self):
        document = html().replace('<td><a>27.10.2026</a></td>', '<td>26.10.2026</td>')
        with self.assertRaises(cbr.NoCurveForDate):
            cbr.parse_zcyc(document, DAY)

    def test_unexpected_page_is_an_error(self):
        with self.assertRaises(ValueError):
            cbr.parse_zcyc('<html>Access denied</html>', DAY)

    def test_invalid_rates_are_not_zero_or_nan(self):
        for rate in ('—', '', 'NaN', 'inf', 'abc'):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                cbr.parse_zcyc(html(rates=[rate] + RATES[1:]), DAY)

    def test_missing_cell_or_duplicate_term_rejected(self):
        for document in (html(rates=RATES[:-1]), html(terms=TERMS[:-1] + ['20'])):
            with self.subTest(document=document), self.assertRaises(ValueError):
                cbr.parse_zcyc(document, DAY)

    def test_duplicate_date_rejected(self):
        row = '<tr><td>27.10.2026</td>' + ''.join(f'<td>{v}</td>' for v in RATES) + '</tr>'
        with self.assertRaises(ValueError):
            cbr.parse_zcyc(html().replace('</table>', row + '</table>'), DAY)


class FetchTests(unittest.TestCase):
    def test_exact_range_and_separate_table_and_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('requests.Session.get', return_value=response()) as get:
                result = cbr.run_zcyc(DAY, folder, timeout=2)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['trade_date'], DAY.isoformat())
            self.assertEqual(get.call_args.kwargs['params'], {
                'UniDbQuery.Posted': 'True', 'UniDbQuery.From': '27.10.2026', 'UniDbQuery.To': '27.10.2026'})
            self.assertEqual(get.call_args.kwargs['timeout'], 2)
            frame = pd.read_csv(Path(folder) / 'cbr_zcyc.csv')
            self.assertEqual(len(frame), 12)
            self.assertEqual(frame.iloc[0].tolist(), ['2026-10-27', .25, 13.09])
            self.assertTrue((Path(folder) / '_support/cbr_zcyc.html').exists())
            saved = json.loads((Path(folder) / '_support/cbr_zcyc_status.json').read_text(encoding='utf-8'))
            self.assertEqual(saved['status'], 'ok')

    def test_retry_on_timeout_and_503(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch('requests.Session.get', side_effect=[requests.ReadTimeout('test'), response(status=503), response()]) as get,
                  patch('moex_bonds.sources.cbr_zcyc.time.sleep')):
                result = cbr.run_zcyc(DAY, folder)
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(get.call_count, 3)
            self.assertEqual(len(result['attempts']), 3)

    def test_no_retry_for_403_and_stale_table_removed(self):
        with tempfile.TemporaryDirectory() as folder:
            stale = Path(folder) / 'cbr_zcyc.csv'
            stale.write_text('old result', encoding='utf-8')
            with patch('requests.Session.get', return_value=response(status=403)) as get:
                result = cbr.run_zcyc(DAY, folder)
            self.assertEqual(result['status'], 'error')
            self.assertIn('403', result['error'])
            self.assertEqual(get.call_count, 1)
            self.assertFalse(stale.exists())

    def test_missing_date_recorded_without_output(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('requests.Session.get', return_value=response(html(day='26.10.2026'))):
                result = cbr.run_zcyc(DAY, folder)
            self.assertEqual(result['status'], 'unavailable')
            self.assertFalse((Path(folder) / 'cbr_zcyc.csv').exists())

    def test_network_failure_clears_stale_html(self):
        with tempfile.TemporaryDirectory() as folder:
            support = Path(folder) / '_support'
            support.mkdir()
            (support / 'cbr_zcyc.html').write_text('old response', encoding='utf-8')
            with patch('requests.Session.get', side_effect=requests.ConnectionError('test')):
                result = cbr.run_zcyc(DAY, folder, retries=0)
            self.assertEqual(result['status'], 'error')
            self.assertIn('ConnectionError', result['error'])
            self.assertFalse((support / 'cbr_zcyc.html').exists())

    def test_input_cli_reuses_saved_date_and_refreshes_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run_manifest.json'
            original = dict(mode='day', collection_status='ok', trade_date='2026-10-27',
                            daily_export={'status': 'ok'}, cbr_zcyc={'status': 'error'})
            path.write_text(json.dumps(original), encoding='utf-8')
            with (patch('sys.argv', ['cbr_zcyc.py', '--input', folder]),
                  patch('requests.Session.get', return_value=response())):
                self.assertEqual(cbr.main(), 0)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['trade_date'], '2026-10-27')
            self.assertEqual(saved['cbr_zcyc']['status'], 'ok')
            self.assertEqual(saved['daily_export'], original['daily_export'])
            self.assertEqual(len(pd.read_csv(Path(folder) / 'cbr_zcyc.csv')), 12)

    def test_input_cli_with_separate_output_preserves_source_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run_manifest.json'
            original = json.dumps(dict(mode='day', collection_status='ok', trade_date='2026-10-27'))
            path.write_text(original, encoding='utf-8')
            destination = str(Path(folder) / 'curve')
            with (patch('sys.argv', ['cbr_zcyc.py', '--input', folder, '--output', destination]),
                  patch('requests.Session.get', return_value=response())):
                self.assertEqual(cbr.main(), 0)
            self.assertEqual(path.read_text(encoding='utf-8'), original)
            self.assertTrue((Path(destination) / 'cbr_zcyc.csv').exists())


class SyncTests(unittest.TestCase):
    def run_main(self, folder, downloader, during_yields=None):
        from moex_bonds.collection.main import run
        history = pd.DataFrame([dict(SECID='TEST', BOARDID='TQCB', TRADEDATE=DAY.isoformat())])

        def collect(client, path, section, day):
            self.assertEqual(day, DAY.isoformat())
            if section == 'history':
                return history
            if during_yields:
                during_yields()
            return pd.DataFrame()

        args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0,
                                  timeout=1, daily=True, csv_only=True)
        with (patch('moex_bonds.collection.main.complete_day', side_effect=collect),
              patch('moex_bonds.collection.main.MoexClient.get', return_value=None),
              patch('moex_bonds.collection.main.run_zcyc', side_effect=downloader) as cbr_call,
              patch('moex_bonds.collection.main.run_ruonia', return_value={'status': 'ok'}),
              patch('moex_bonds.collection.daily.run_daily', return_value={'metadata': {'trade_date': DAY.isoformat()}}) as daily):
            result = run(args)
        return result, cbr_call, daily

    def test_common_date_and_parallel_collection(self):
        started, collecting = threading.Event(), threading.Event()

        def download(day, output, **kwargs):
            self.assertEqual(day, DAY)
            started.set()
            self.assertTrue(collecting.wait(2), 'CBR request must overlap MOEX collection')
            return dict(status='ok', trade_date=day.isoformat(), rows=12)

        def yields():
            self.assertTrue(started.wait(2))
            collecting.set()

        with tempfile.TemporaryDirectory() as folder:
            code, cbr_call, daily = self.run_main(folder, download, yields)
            self.assertEqual(code, 0)
            cbr_call.assert_called_once()
            daily.assert_called_once()
            manifest = json.loads((Path(folder) / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['cbr_zcyc']['trade_date'], manifest['trade_date'])
            self.assertEqual(manifest['daily_export']['trade_date'], manifest['trade_date'])

    def test_cbr_failure_does_not_stop_daily_export(self):
        for status in ('error', 'unavailable'):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                code, _, daily = self.run_main(folder, lambda *a, **kw: dict(status=status, error='test'))
                self.assertEqual(code, 4)
                daily.assert_called_once()
                manifest = json.loads((Path(folder) / 'run_manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['collection_status'], 'ok')
                self.assertEqual(manifest['daily_export']['status'], 'ok')
                self.assertEqual(manifest['cbr_zcyc']['status'], status)

    def test_failed_moex_does_not_request_cbr(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0,
                                      timeout=1, daily=True, csv_only=True)
            with (patch('moex_bonds.collection.main.complete_day', return_value=pd.DataFrame()),
                  patch('moex_bonds.collection.main.run_zcyc') as fetch):
                self.assertEqual(run(args), 2)
                fetch.assert_not_called()


if __name__ == '__main__':
    unittest.main()
