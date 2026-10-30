"""Network-free mode checks; all outputs live in temporary directories."""
import argparse
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

FIXED = 'Облигация с фиксированным (известным) купоном'


def response(payload, url):
    result = requests.Response()
    result.status_code = 200
    result.url = url
    result._content = json.dumps(payload).encode()
    return result


def block(columns, data):
    return {'columns': columns, 'data': data}


class CollectionModeTests(unittest.TestCase):
    def test_date_timeout_reports_network_failure_not_missing_date(self):
        from moex_bonds.collection.main import latest_trade_date
        from moex_bonds.sources.moex_client import MoexClient
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0, timeout=15, retries=0)
            with patch('requests.Session.get', side_effect=requests.ReadTimeout('read timeout=15')):
                with self.assertRaises(ValueError) as caught:
                    latest_trade_date(client)
            message = str(caught.exception)
            self.assertIn('ReadTimeout', message)
            self.assertIn('HTTP-ответ не получен', message)
            self.assertIn('/dates.json', message)
            self.assertNotIn('MOEX не вернула последнюю опубликованную торговую дату', message)

    def test_date_http_error_preserves_status(self):
        from moex_bonds.collection.main import latest_trade_date
        from moex_bonds.sources.moex_client import MoexClient
        denied = response({'error': 'Forbidden'}, 'https://iss.moex.com/iss/dates.json')
        denied.status_code = 403
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0, retries=0)
            with patch('requests.Session.get', return_value=denied):
                with self.assertRaises(ValueError) as caught:
                    latest_trade_date(client)
            self.assertIn('HTTP 403', str(caught.exception))
            self.assertNotIn('HTTP-ответ не получен', str(caught.exception))

    def test_successful_date_response_with_wrong_schema_is_distinct(self):
        from moex_bonds.collection.main import latest_trade_date
        from moex_bonds.sources.moex_client import MoexClient
        with tempfile.TemporaryDirectory() as folder:
            client = MoexClient(Path(folder), pause=0, retries=0)
            with patch('requests.Session.get', return_value=response({'unexpected': block(['foo'], [[1]])}, 'https://iss.moex.com/iss/dates.json')):
                with self.assertRaises(ValueError) as caught:
                    latest_trade_date(client)
            self.assertIn('HTTP 200', str(caught.exception))
            self.assertIn('unexpected', str(caught.exception))

    def test_day_export_uses_collected_date_without_previous_day_shift(self):
        from moex_bonds.collection.daily import run_daily
        from tests.test_daily import inputs, SECID
        samples, histories, events, smart = inputs()
        samples[0]['securities.PREVDATE'] = '2026-10-23'
        for row in histories:
            row['endpoint'] = row['endpoint'].replace('/{SECID}.json', '.json').replace('{BOARDID}', 'TQCB')
            row.pop('window_days', None)
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)
            (source/'run_manifest.json').write_text(json.dumps(dict(mode='day', trade_date='2026-10-26',
                as_of_date='2026-10-27', collection_status='ok')), encoding='utf-8')
            pd.DataFrame(samples).to_csv(source/'bonds_universe.csv', index=False)
            pd.DataFrame(histories).to_csv(source/'day_quotes.csv', index=False)
            pd.DataFrame(events).to_csv(source/'bond_events.csv', index=False)
            with patch('moex_bonds.collection.daily.SmartLabClient.fetch', return_value=smart[SECID]):
                dataset = run_daily(source, as_of=date(2026,10,28), pause=0, xlsx=False,
                                    history_dir=source.parent / (source.name + '-test-history'))
            self.assertEqual(dataset['metadata']['trade_date'], '2026-10-26')
            self.assertEqual(dataset['metadata']['final_count'], 1)
            self.assertIsNone(dataset['rows'][0]['ADV20_RUB'])
            self.assertEqual(dataset['metadata']['history_update']['status'], 'not_initialized')

    def test_incomplete_day_page_cannot_be_published_as_success(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=date(2026,10,26), secids='', pause=0,
                                      timeout=1, daily=True, csv_only=True)
            with (patch('moex_bonds.collection.main.MoexClient.collect', return_value=(pd.DataFrame(), 'request_error_incomplete')),
                  patch('moex_bonds.collection.daily.run_daily') as export):
                self.assertEqual(run(args), 2)
                export.assert_not_called()
            manifest=json.loads((Path(folder)/'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['collection_status'], 'incomplete')

    def test_daily_collects_more_than_twenty_and_no_range_history(self):
        from moex_bonds.collection.main import run
        calls = []
        ids = [f'FIXTURE{i:02}' for i in range(25)]

        def get(url, params=None, **kwargs):
            params = params or {}
            calls.append((url, params))
            self.assertNotIn('from', params)
            self.assertNotIn('till', params)
            if url.endswith('/dates.json'):
                payload = {'dates': block(['from', 'till'], [['2020-01-01', '2026-10-26']])}
            elif '/history/' in url:
                self.assertEqual(params['date'], '2026-10-26')
                start = params.get('start', 0)
                section = 'history_yields' if url.endswith('/yields.json') else 'history'
                if section == 'history':
                    rows = [[s, 'TQCB', '2026-10-26', 3, FIXED, 'RUB', 'SUR', '2028-01-01', 100] for s in ids[start:start+10]]
                    cols = ['SECID', 'BOARDID', 'TRADEDATE', 'TRADINGSESSION', 'BONDTYPE', 'FACEUNIT', 'CURRENCYID', 'MATDATE', 'VALUE']
                else:
                    rows = [[s, 'TQCB', '2026-10-26', 50] for s in ids[start:start+10]]
                    cols = ['SECID', 'BOARDID', 'TRADEDATE', 'ZSPREADBP']
                payload = {section: block(cols, rows)}
            elif url.endswith('/bondization.json'):
                payload = {s: block(['value'], []) for s in ('coupons', 'amortizations', 'offers')}
            elif '/boards/TQCB/securities.json' in url:
                payload = {'securities': block(['SECID', 'BOARDID', 'ISIN', 'BONDTYPE', 'FACEUNIT', 'CURRENCYID'],
                             [[s, 'TQCB', s, FIXED, 'SUR', 'SUR'] for s in ids])}
            else:
                payload = {'description': block(['name', 'value'], [['TYPE', 'exchange_bond']])}
            return response(payload, url)

        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=None, secids='', pause=0, timeout=1, daily=False, csv_only=True)
            with (patch('requests.Session.get', side_effect=get),
                  patch('moex_bonds.collection.main.run_ruonia', return_value={'status': 'ok'}),
                  patch('moex_bonds.collection.main.run_zcyc', return_value={'status': 'ok', 'trade_date': '2026-10-26'}) as cbr):
                self.assertEqual(run(args), 0)
            self.assertEqual(cbr.call_args.args[0], date(2026, 10, 26))
            universe = pd.read_csv(Path(folder)/'bonds_universe.csv')
            quotes = pd.read_csv(Path(folder)/'day_quotes.csv')
            self.assertEqual(len(universe), 25)
            self.assertEqual(set(quotes.TRADEDATE), {'2026-10-26'})
            self.assertFalse((Path(folder)/'historical_sample.csv').exists())
            manifest = json.loads((Path(folder)/'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['trade_date'], '2026-10-26')
            self.assertEqual(manifest['mode'], 'day')
            self.assertFalse(any('/orderbook' in url for url, _ in calls))

    def test_explicit_missing_date_does_not_fall_back(self):
        from moex_bonds.collection.main import run
        with tempfile.TemporaryDirectory() as folder:
            args = argparse.Namespace(output=folder, date=date(2026, 10, 25), secids='', pause=0,
                                      timeout=1, daily=False, csv_only=True)
            with patch('requests.Session.get', side_effect=lambda url, **kw: response({'history': block(['SECID'], [])}, url)):
                self.assertEqual(run(args), 2)
            manifest = json.loads((Path(folder)/'run_manifest.json').read_text(encoding='utf-8'))
            self.assertNotEqual(manifest.get('collection_status'), 'ok')

    def test_separate_history_uses_saved_date_and_no_daily_enrichment(self):
        from moex_bonds.collection.history import run_history
        from tests.test_daily import inputs, SECID
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder)/'day'
            source.mkdir()
            (source/'run_manifest.json').write_text(json.dumps({'mode':'day','trade_date':'2026-10-26','collection_status':'ok'}), encoding='utf-8')
            samples, quotes, events, smart = inputs()
            pd.DataFrame(samples).to_csv(source/'bonds_universe.csv', index=False)
            pd.DataFrame(quotes).to_csv(source/'day_quotes.csv', index=False)
            pd.DataFrame(events).to_csv(source/'bond_events.csv', index=False)
            support = source/'daily/_support'
            support.mkdir(parents=True)
            (support/'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
            calls=[]
            def get(url, params=None, **kwargs):
                calls.append((url,params))
                self.assertIn('/history/',url)
                self.assertEqual(params['till'],'2026-10-26')
                self.assertEqual(params['from'],'2026-10-13')
                sect='history_yields' if '/yields/' in url else 'history'
                row = quotes[1] if sect == 'history_yields' else quotes[0]
                payload={sect:block(list(row),[list(row.values())]),
                         sect+'.cursor':block(['INDEX','TOTAL','PAGESIZE'],[[0,1,100]])}
                return response(payload,url)
            args=argparse.Namespace(input=str(source),output=str(Path(folder)/'history'),history_days=10,
                                    secids='',pause=0,timeout=1)
            dates = pd.bdate_range('2026-10-13', '2026-10-26').strftime('%Y-%m-%d').tolist()
            with (patch('requests.Session.get',side_effect=get), patch('moex_bonds.collection.daily.run_daily') as daily,
                  patch('moex_bonds.collection.history.trading_dates', return_value=dates)):
                self.assertEqual(run_history(args),0)
                daily.assert_not_called()
            self.assertEqual(len(calls),2)
            frame=pd.read_csv(Path(args.output)/'_support/historical_sample.csv')
            self.assertEqual(len(frame),2)
            self.assertEqual(set(frame.SECID), {SECID})


if __name__ == '__main__':
    unittest.main()

class EmptyDatesRetryTests(unittest.TestCase):
 def test_transient_empty_dates_retried(self):
  from unittest.mock import Mock
  from moex_bonds.collection.main import latest_trade_date
  client=Mock(); client.get.side_effect=[{'dates':block(['from','till'],[])},{'dates':block(['from','till'],[['2020-01-01','2026-10-30']])}]
  with patch('moex_bonds.collection.main.time.sleep') as sleep:
   self.assertEqual(latest_trade_date(client),'2026-10-30')
  self.assertEqual(client.get.call_count,2); sleep.assert_called_once_with(6)
 def test_persistent_empty_fails_without_date_guess(self):
  from unittest.mock import Mock
  from moex_bonds.collection.main import latest_trade_date
  client=Mock(); client.get.return_value={'dates':block(['from','till'],[])}; client.records=[]
  with patch('moex_bonds.collection.main.time.sleep'), self.assertRaisesRegex(ValueError,'dates.till'):
   latest_trade_date(client)
  self.assertEqual(client.get.call_count,3)
