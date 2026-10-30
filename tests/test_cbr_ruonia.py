"""Offline RUONIA schema, weekly-window and daily synchronization checks."""
import argparse
from datetime import date
import importlib
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import pandas as pd
import requests

DAY = date(2026, 10, 26)
START = date(2026, 10, 19)
HEADERS = ['Дата ставки', 'Ставка RUONIA, % годовых', 'Объем сделок RUONIA, млрд руб.',
           'Количество сделок, ед.',
           'Количество участников RUONIA, совершавших сделки в данный день, ед.',
           'Минимальная процентная ставка, % годовых',
           '25-й процентиль по процентным ставкам, % годовых',
           '75-й процентиль по процентным ставкам, % годовых',
           'Максимальная процентная ставка, % годовых', 'Статус расчета', 'Дата публикации']
VALUES = ['26.10.2026', '14,04', '810,27', '69', '17', '13,85', '14,00', '14,10', '14,25',
          'Стандартный', '27.10.2026']


def html(rows=None, order=None):
    order = order if order is not None else list(range(11))
    head = ''.join(f'<th>{HEADERS[i]}</th>' for i in order)
    body = ''.join('<tr>' + ''.join(f'<td>{row[i]}</td>' for i in order) + '</tr>'
                   for row in (rows if rows is not None else [VALUES]))
    return f'<html><meta charset="utf-8"><table class="data"><tr>{head}</tr>{body}</table></html>'


def response(text=None, status=200):
    r = requests.Response()
    r.status_code = status
    r.url = 'https://www.cbr.ru/hd_base/ruonia/dynamics/'
    r._content = (text if text is not None else html()).encode('utf-8')
    return r


class RuoniaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Report the absent feature as an assertion before implementation.
        try:
            cls.cbr = importlib.import_module('moex_bonds.sources.cbr_ruonia')
        except ModuleNotFoundError:
            raise AssertionError('Нужен отдельный загрузчик cbr_ruonia.py')

    def test_source_fields_types_and_separate_publication_date(self):
        rows = self.cbr.parse_ruonia(html(), START, DAY)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['rate_date'], '2026-10-26')
        self.assertEqual(rows[0]['publication_date'], '2026-10-27')
        self.assertEqual(rows[0]['ruonia_pct'], 14.04)
        self.assertEqual(rows[0]['volume_bln_rub'], 810.27)
        self.assertEqual(rows[0]['trades_count'], 69)
        self.assertIsInstance(rows[0]['participants_count'], int)
        self.assertEqual(rows[0]['calculation_status'], 'Стандартный')

    def test_header_order_does_not_change_mapping_and_dates_ascending(self):
        earlier = ['19.10.2026'] + VALUES[1:]
        rows = self.cbr.parse_ruonia(html([VALUES, earlier], list(reversed(range(11)))), START, DAY)
        self.assertEqual([r['rate_date'] for r in rows], ['2026-10-19', '2026-10-26'])
        self.assertEqual(rows[0]['ruonia_pct'], 14.04)

    def test_duplicate_or_outside_window_rejected(self):
        for rows in ([VALUES, VALUES], [['18.10.2026'] + VALUES[1:]], [['27.10.2026'] + VALUES[1:]]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.cbr.parse_ruonia(html(rows), START, DAY)

    def test_missing_numeric_values_remain_missing(self):
        values = VALUES.copy()
        values[1], values[2] = '—', ''
        row = self.cbr.parse_ruonia(html([values]), START, DAY)[0]
        self.assertIsNone(row['ruonia_pct'])
        self.assertIsNone(row['volume_bln_rub'])
        self.assertIn('ruonia_pct', row['data_gaps'])

    def test_invalid_numbers_counts_and_dates_rejected(self):
        for index, value in [(1, 'nan'), (1, 'inf'), (2, 'abc'), (3, '1,5'), (4, '-1'),
                             (0, '31.02.2026'), (10, 'yesterday')]:
            values = VALUES.copy()
            values[index] = value
            with self.subTest(index=index, value=value), self.assertRaises(ValueError):
                self.cbr.parse_ruonia(html([values]), START, DAY)

    def test_no_publications_distinct_from_schema_error(self):
        for document in ('<p>За выбранный вами период нет информации.</p>', html([])):
            with self.subTest(document=document), self.assertRaises(self.cbr.NoRuoniaForPeriod):
                self.cbr.parse_ruonia(document, START, DAY)
        with self.assertRaises(ValueError):
            self.cbr.parse_ruonia('<p>Access denied</p>', START, DAY)
        with self.assertRaises(ValueError):
            self.cbr.parse_ruonia(html(order=list(range(10))), START, DAY)

    def test_exact_week_csv_schema_and_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('requests.Session.get', return_value=response()) as get:
                status = self.cbr.run_ruonia(DAY, folder, timeout=2)
            self.assertEqual(status['status'], 'ok')
            self.assertEqual(get.call_args.kwargs['params'], {
                'UniDbQuery.Posted': 'True', 'UniDbQuery.From': '19.10.2026', 'UniDbQuery.To': '26.10.2026'})
            self.assertEqual(get.call_args.kwargs['timeout'], 2)
            frame = pd.read_csv(Path(folder) / 'cbr_ruonia.csv')
            self.assertEqual(frame.iloc[0]['trade_date'], '2026-10-26')
            self.assertEqual(frame.iloc[0]['ruonia_pct'], 14.04)
            self.assertEqual(status['latest_rate_date'], '2026-10-26')
            self.assertEqual(status['date_from'], '2026-10-19')
            self.assertEqual(status['date_to'], '2026-10-26')
            self.assertTrue((Path(folder) / '_support/cbr_ruonia.html').exists())
            saved = json.loads((Path(folder) / '_support/cbr_ruonia_status.json').read_text(encoding='utf-8'))
            self.assertEqual([c['key'] for c in saved['columns']], list(frame.columns))

    def test_publication_lag_is_explicit_and_no_forward_fill(self):
        values = ['23.10.2026'] + VALUES[1:]
        with tempfile.TemporaryDirectory() as folder:
            with patch('requests.Session.get', return_value=response(html([values]))):
                status = self.cbr.run_ruonia(DAY, folder)
            frame = pd.read_csv(Path(folder) / 'cbr_ruonia.csv')
            self.assertEqual(status['status'], 'ok')
            self.assertEqual(status['lag_days'], 3)
            self.assertFalse(status['has_rate_on_trade_date'])
            self.assertEqual(frame.rate_date.tolist(), ['2026-10-23'])

    def test_retry_and_error_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            with (patch('requests.Session.get', side_effect=[requests.ReadTimeout('test'), response(status=503), response()]) as get,
                  patch('moex_bonds.sources.cbr_ruonia.time.sleep')):
                self.assertEqual(self.cbr.run_ruonia(DAY, folder)['status'], 'ok')
            self.assertEqual(get.call_count, 3)
            with patch('requests.Session.get', return_value=response(status=403)) as get:
                self.assertEqual(self.cbr.run_ruonia(DAY, folder)['status'], 'error')
            self.assertEqual(get.call_count, 1)
            self.assertFalse((Path(folder) / 'cbr_ruonia.csv').exists())
            with patch('requests.Session.get', side_effect=requests.ConnectionError('test')):
                status = self.cbr.run_ruonia(DAY, folder, retries=0)
            self.assertFalse((Path(folder) / '_support/cbr_ruonia.html').exists())
            self.assertIn('ConnectionError', status['error'])

    def test_unavailable_does_not_publish_old_table(self):
        with tempfile.TemporaryDirectory() as folder:
            stale = Path(folder) / 'cbr_ruonia.csv'
            stale.write_text('old', encoding='utf-8')
            with patch('requests.Session.get', return_value=response('<p>Нет информации</p>')):
                self.assertEqual(self.cbr.run_ruonia(DAY, folder)['status'], 'unavailable')
            self.assertFalse(stale.exists())

    def test_input_cli_updates_only_ruonia_in_same_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'run_manifest.json'
            original = dict(mode='day', collection_status='ok', trade_date='2026-10-26',
                            cbr_zcyc={'status': 'ok'}, daily_export={'status': 'ok'})
            path.write_text(json.dumps(original), encoding='utf-8')
            with (patch('sys.argv', ['cbr_ruonia.py', '--input', folder]),
                  patch('requests.Session.get', return_value=response())):
                self.assertEqual(self.cbr.main(), 0)
            saved = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(saved['cbr_ruonia']['trade_date'], original['trade_date'])
            self.assertEqual(saved['cbr_zcyc'], original['cbr_zcyc'])
            self.assertEqual(saved['daily_export'], original['daily_export'])
            with (patch('sys.argv', ['cbr_ruonia.py', '--input', folder, '--output', str(Path(folder) / 'other')]),
                  patch('requests.Session.get', return_value=response())):
                self.assertEqual(self.cbr.main(), 0)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), saved)


class RuoniaSyncTests(unittest.TestCase):
    def test_both_cbr_jobs_overlap_moex_and_share_date(self):
        from moex_bonds.collection.main import run
        ready = [threading.Event(), threading.Event()]
        collecting = threading.Event()

        def download(index):
            def fetch(day, output, **kwargs):
                self.assertEqual(day, DAY)
                ready[index].set()
                self.assertTrue(collecting.wait(3))
                return dict(status='ok', trade_date=day.isoformat())
            return fetch

        def collect(client, path, section, day):
            if section == 'history':
                return pd.DataFrame([dict(SECID='TEST', BOARDID='TQCB', TRADEDATE=day)])
            self.assertTrue(all(e.wait(3) for e in ready), 'Both CBR requests must start during MOEX collection')
            collecting.set()
            return pd.DataFrame()

        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0, timeout=1, daily=False)
            with (patch('moex_bonds.collection.main.complete_day', side_effect=collect), patch('moex_bonds.collection.main.MoexClient.get', return_value=None),
                  patch('moex_bonds.collection.main.run_zcyc', side_effect=download(0)),
                  patch('moex_bonds.collection.main.run_ruonia', side_effect=download(1))):
                self.assertEqual(run(args), 0)
            m = json.loads((Path(folder) / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(m['trade_date'], m['cbr_ruonia']['trade_date'])
            self.assertEqual(m['trade_date'], m['cbr_zcyc']['trade_date'])

    def test_ruonia_failures_do_not_block_curve_or_daily_and_export_failure_has_priority(self):
        from moex_bonds.collection.main import run
        for curve_status, ruonia_result, export_fails in [
                ('ok', {'status': 'error'}, False), ('ok', {'status': 'unavailable'}, False),
                ('ok', OSError('disk error'), False), ('error', {'status': 'ok'}, False),
                ('ok', {'status': 'error'}, True)]:
            with self.subTest(curve=curve_status, ruonia=ruonia_result, export_fails=export_fails), tempfile.TemporaryDirectory() as folder:
                args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0, timeout=1, daily=True, csv_only=True)
                hist = pd.DataFrame([dict(SECID='TEST', BOARDID='TQCB', TRADEDATE=DAY.isoformat())])
                def ruonia(*args, **kwargs):
                    if isinstance(ruonia_result, Exception):
                        raise ruonia_result
                    return ruonia_result
                with (patch('moex_bonds.collection.main.complete_day', side_effect=[hist, pd.DataFrame()]),
                      patch('moex_bonds.collection.main.MoexClient.get', return_value=None),
                      patch('moex_bonds.collection.main.run_zcyc', return_value={'status': curve_status}) as curve,
                      patch('moex_bonds.collection.main.run_ruonia', side_effect=ruonia),
                      patch('moex_bonds.collection.daily.run_daily', side_effect=ValueError('export test') if export_fails else None,
                            return_value={'metadata': {'trade_date': DAY.isoformat()}}) as daily):
                    self.assertEqual(run(args), 3 if export_fails else 4)
                curve.assert_called_once()
                daily.assert_called_once()

    def test_no_ruonia_request_when_moex_has_no_confirmed_day(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=DAY, secids='', pause=0, timeout=1, daily=False)
            with (patch('moex_bonds.collection.main.complete_day', return_value=pd.DataFrame()),
                  patch('moex_bonds.collection.main.run_ruonia') as ruonia, patch('moex_bonds.collection.main.run_zcyc') as curve):
                self.assertEqual(run(args), 2)
                ruonia.assert_not_called()
                curve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
