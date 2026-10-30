"""Offline checks only; fixtures never enter results/."""
import tempfile
import unittest
import argparse
import json
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

import requests


class ProbeTests(unittest.TestCase):
    def test_dynamic_columns_and_empty_schema(self):
        from moex_bonds.sources.moex_client import tables
        result = tables({'a': {'columns': ['LAST', 'SECID'], 'data': [[99, 'TEST']]},
                         'b': {'columns': ['SECID'], 'data': []}})
        self.assertEqual(result['a'].iloc[0].to_dict(), {'LAST': 99, 'SECID': 'TEST'})
        self.assertEqual(list(result['b'].columns), ['SECID'])

    def test_cursor_pages_and_terminal_empty(self):
        from moex_bonds.sources.moex_client import MoexClient
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0)
            client.get = Mock(side_effect=[
                {'history': {'columns': ['SECID'], 'data': [['A'], ['B']]},
                 'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'], 'data': [[0, 3, 2]]}},
                {'history': {'columns': ['SECID'], 'data': [['C']]},
                 'history.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'], 'data': [[2, 3, 2]]}},
            ])
            frame, status = client.collect('x', 'history')
            self.assertEqual(frame.SECID.tolist(), ['A', 'B', 'C'])
            self.assertEqual(status, 'cursor_complete')
            self.assertEqual(client.get.call_args_list[1].args[1]['start'], 2)

    def test_ignored_start_does_not_duplicate(self):
        from moex_bonds.sources.moex_client import MoexClient
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0)
            client.get = Mock(return_value={'securities': {'columns': ['SECID'], 'data': [['A']]}})
            frame, status = client.collect('x', 'securities')
            self.assertEqual(len(frame), 1)
            self.assertEqual(status, 'repeated_page_completeness_unconfirmed')

    def test_network_failure_has_no_http_status(self):
        from moex_bonds.sources.moex_client import MoexClient
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0, retries=1)
            with patch.object(client.session, 'get', side_effect=requests.ConnectionError('offline')):
                with patch('moex_bonds.sources.moex_client.time.sleep'):
                    self.assertIsNone(client.get('x'))
            self.assertEqual(len(client.records), 2)
            self.assertIsNone(client.records[-1]['http_status'])
            self.assertEqual(client.records[-1]['outcome'], 'network_error')
            self.assertTrue((Path(folder) / 'errors.jsonl').exists())

    def test_access_denied_is_not_empty_success(self):
        from moex_bonds.sources.moex_client import MoexClient
        response = Mock(status_code=403, headers={}, content=b'Forbidden', text='Forbidden',
                        url='https://iss.moex.com/iss/x')
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0)
            with patch.object(client.session, 'get', return_value=response):
                self.assertIsNone(client.get('x'))
            self.assertEqual(client.records[-1]['outcome'], 'access_restricted')

    def test_no_date_is_invented_for_trade_time(self):
        from moex_bonds.collection.main import age_seconds
        self.assertIsNone(age_seconds('12:00:00', '2026-10-27T13:00:00+03:00'))
        self.assertEqual(age_seconds('2026-10-27 12:00:00', '2026-10-27T13:00:00+03:00'), 3600)

    def test_zero_is_present_but_not_nonzero(self):
        from moex_bonds.collection.main import availability
        row = availability('Z', 'ZSPREADBP', 'test', [{'SECID': 'A', 'ZSPREADBP': 0},
                                                   {'SECID': 'B', 'ZSPREADBP': None}], 2)
        self.assertEqual(row['nonempty_count'], 1)
        self.assertEqual(row['nonzero_count'], 0)
        self.assertEqual(row['status'], 'Частично доступно')

    def test_other_board_cannot_overwrite_selected_quote(self):
        from moex_bonds.collection.main import observe
        sample = {'BOARDID': 'TQCB'}
        observations = []
        observe({'marketdata': {'columns': ['SECID', 'BOARDID', 'LAST'],
                  'data': [['A', 'TQCB', 99], ['A', 'OTHER', 10]]}},
                'A', 'market', observations, sample, [], '2026-10-27T13:00:00+03:00')
        self.assertEqual(sample['market.marketdata.LAST'], 99)
        self.assertEqual(len(observations), 1)

    def test_complete_flow_exports_only_observed_fields(self):
        """Exercise daily raw/CSV export with isolated fixtures outside results/."""
        from moex_bonds.collection.main import run
        import pandas as pd

        def block(columns, data):
            return {'columns': columns, 'data': data}

        def respond(url, params=None, **kwargs):
            query = params or {}
            if 'reference' in url or 'columns.json' in url:
                payload = {}
            elif '/history/' in url:
                section = 'history_yields' if '/yields/' in url or url.endswith('/yields.json') else 'history'
                payload = {section: block(['SECID', 'BOARDID', 'TRADEDATE', 'ZSPREADBP', 'VALUE'],
                                         [['FIXTURE', 'TQCB', '2026-10-26', 125, 1000]]),
                           section + '.cursor': block(['INDEX', 'TOTAL', 'PAGESIZE'], [[0, 1, 100]])}
            elif 'bondization' in url:
                payload = {name: block(['value'], []) for name in ('coupons', 'amortizations', 'offers')}
            elif 'orderbook' in url:
                payload = {'orderbook': block(['BUYSELL', 'PRICE'], [['B', 99]])}
            elif '/engines/' not in url:
                payload = {'description': block(['name', 'value'], [['ISIN', 'FIXTURE_ISIN']])}
            else:
                payload = {'securities': block(['SECID', 'BOARDID', 'SHORTNAME'],
                            [] if query.get('start', 0) else [['FIXTURE', 'TQCB', 'Fixture']]),
                           'marketdata': block(['SECID', 'BOARDID', 'ZSPREADBP'],
                            [] if query.get('start', 0) else [['FIXTURE', 'TQCB', 125]])}
            response = requests.Response()
            response.status_code = 200
            response.url = url
            response._content = json.dumps(payload).encode()
            return response

        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=date(2026, 10, 26), daily=False,
                                      secids='', pause=0, timeout=1, csv_only=True)
            with (patch('requests.Session.get', side_effect=respond),
                  patch('moex_bonds.collection.main.run_ruonia', return_value={'status': 'ok'}),
                  patch('moex_bonds.collection.main.run_zcyc', return_value={'status': 'ok', 'trade_date': '2026-10-26'})):
                result = run(args)
            self.assertEqual(result, 0)
            sample = pd.read_csv(Path(folder) / 'bonds_universe.csv')
            self.assertEqual(sample.SECID.tolist(), ['FIXTURE'])
            day = pd.read_csv(Path(folder) / 'day_quotes.csv')
            self.assertEqual(set(day.TRADEDATE), {'2026-10-26'})
            self.assertFalse((Path(folder) / 'historical_sample.csv').exists())
            self.assertTrue(list((Path(folder) / 'raw_responses').glob('*.json')))


if __name__ == '__main__':
    unittest.main()
