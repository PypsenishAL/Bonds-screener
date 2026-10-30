"""Gap recovery must never overwrite the latest-only daily contract."""
from datetime import date
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from tests.test_daily import inputs, DAY, SECID
from moex_bonds.collection.daily import build_daily
from moex_bonds.storage.history_storage import append_daily, publish_history


class SyncTests(unittest.TestCase):
    def prepare(self, folder, *, dates=None, subset=None):
        output = Path(folder) / 'history'
        s, h, e, smart = inputs()
        today = '2026-10-29'
        # Current daily source is Thursday; the base stopped on Monday.
        h = [{**r, 'TRADEDATE': today, 'TRADEMOMENT': today} for r in h]
        s[0]['securities.MATDATE'] = h[0]['MATDATE']
        rows, _ = build_daily(s, h, e, smart, date.fromisoformat(today))
        base = {**rows[0], 'TRADEDATE': DAY.isoformat()}
        metadata = dict(status='ok', selection_date=DAY.isoformat(),
                        trading_dates=dates or [DAY.isoformat()])
        publish_history(output, [base], metadata)
        dataset = dict(rows=rows, metadata=dict(trade_date=today, replace_secids=subset))
        context = dict(samples=s, events=e, smartlab=smart)
        return output, dataset, context, h

    def response(self, payload, url):
        result = requests.Response()
        result.status_code, result.url = 200, url
        result._content = json.dumps(payload).encode()
        return result

    def server(self, templates, *, fail_day=None, closed=(), zero_day=None):
        calls = []
        def get(url, params=None, **kwargs):
            day = params['date']
            calls.append((day, url, params))
            if day == fail_day:
                raise requests.ReadTimeout('VPN')
            section = 'history_yields' if url.endswith('/yields.json') else 'history'
            row = {**templates[section == 'history_yields'], 'TRADEDATE': day, 'TRADEMOMENT': day}
            if day == zero_day:
                row['NUMTRADES'] = 0
            raw = [] if day in closed else [row]
            if params.get('start', 0):
                raw = []
            return self.response({section: {'columns': list(row), 'data': [list(r.values()) for r in raw]},
                section+'.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'], 'data': [[0, len(raw), 100]]}}, url)
        return get, calls

    def test_skipped_dates_are_loaded_once_and_latest_quote_not_refetched(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            get, calls = self.server(h, zero_day='2026-10-28')
            with patch('requests.Session.get', side_effect=get):
                result = append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(list(frame.TRADEDATE), ['2026-10-26', '2026-10-27', '2026-10-28', '2026-10-29'])
            self.assertEqual(result['recovered_dates'], ['2026-10-27', '2026-10-28'])
            self.assertEqual(len(calls), 4)
            self.assertFalse(frame.duplicated(['SECID', 'BOARDID', 'TRADEDATE']).any())
            self.assertTrue(frame.loc[frame.TRADEDATE.eq('2026-10-28'), 'WAPRICE'].isna().all())

    def test_failed_missing_day_preserves_base_and_does_not_skip_it_on_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            before = {p.name: p.read_bytes() for p in output.glob('*') if p.is_file()}
            get, _ = self.server(h, fail_day='2026-10-28')
            with patch('requests.Session.get', side_effect=get), patch('time.sleep'), self.assertRaises(ValueError):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            self.assertEqual(before, {p.name: p.read_bytes() for p in output.glob('*') if p.is_file()})
            get, _ = self.server(h)
            with patch('requests.Session.get', side_effect=get):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            self.assertEqual(len(pd.read_csv(output / 'bonds_history.csv')), 4)

    def test_legacy_interior_gaps_since_initial_build_are_repaired(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder, dates=['2026-10-26', '2026-10-28'])
            get, calls = self.server(h)
            with patch('requests.Session.get', side_effect=get):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            self.assertIn('2026-10-27', pd.read_csv(output / 'bonds_history.csv').TRADEDATE.tolist())

    def test_subset_update_does_not_advance_full_sync_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder, subset=[SECID])
            get, _ = self.server(h)
            with patch('requests.Session.get', side_effect=get):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            meta = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(meta['synchronized_through'], DAY.isoformat())

    def test_weekend_closure_and_special_sunday_session_are_source_driven(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            old = {**dataset['rows'][0], 'TRADEDATE': '2026-10-23'}
            publish_history(output, [old], dict(status='ok', selection_date='2026-10-23', trading_dates=['2026-10-23']))
            dataset['rows'][0]['TRADEDATE'] = '2026-10-26'
            dataset['metadata']['trade_date'] = '2026-10-26'
            get, calls = self.server(h, closed=['2026-10-24'])
            with patch('requests.Session.get', side_effect=get):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            self.assertEqual(pd.read_csv(output / 'bonds_history.csv').TRADEDATE.tolist(),
                             ['2026-10-23', '2026-10-25', '2026-10-26'])

    def test_new_issue_is_not_backfilled_before_its_first_observation(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            new = 'RU000NEWISSUE'
            dataset['rows'].append({**dataset['rows'][0], 'SECID': new})
            context['samples'].append({**context['samples'][0], 'SECID': new, 'securities.ISIN': new})
            context['smartlab'][new] = {**context['smartlab'][SECID], 'SECID': new}
            get, _ = self.server(h)
            with patch('requests.Session.get', side_effect=get):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(frame.loc[frame.SECID.eq(new), 'TRADEDATE'].tolist(), ['2026-10-29'])

    def test_past_only_issue_is_recovered_with_reference_loaded_once(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            new = 'RU000PASTONLY'
            get, _ = self.server(h)
            def extra(url, params=None, **kwargs):
                payload = get(url, params, **kwargs).json()
                section = 'history_yields' if url.endswith('/yields.json') else 'history'
                block = payload[section]
                if section == 'history' and 'BONDTYPE' not in block['columns']:
                    block['columns'].append('BONDTYPE')
                    for row in block['data']:
                        row.append(context['samples'][0]['securities.BONDTYPE'])
                row = list(block['data'][0])
                row[block['columns'].index('SECID')] = new
                block['data'].append(row)
                payload[section+'.cursor']['data'][0][1] = 2
                return self.response(payload, url)
            def reference(client, sample, **kwargs):
                for key, value in context['samples'][0].items():
                    if key not in ('SECID', 'BOARDID'):
                        sample.setdefault(key, value)
                sample['securities.ISIN'] = new
                return [{**context['events'][0], 'probe_secid': new}]
            with (patch('requests.Session.get', side_effect=extra),
                  patch('moex_bonds.collection.history_sync.collect_reference', side_effect=reference) as ref,
                  patch('moex_bonds.collection.history_sync.SmartLabClient.fetch', return_value={**context['smartlab'][SECID], 'SECID': new}) as smart):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(frame.loc[frame.SECID.eq(new), 'TRADEDATE'].tolist(), ['2026-10-27', '2026-10-28'])
            self.assertEqual(ref.call_count, 1)
            self.assertEqual(smart.call_count, 1)

    def test_failed_reference_request_cannot_masquerade_as_ineligible_issue(self):
        from moex_bonds.collection.main import collect_reference
        class Client:
            records = []
            def get(self, *args, **kwargs):
                return None
        with self.assertRaises(ValueError):
            collect_reference(Client(), inputs()[0][0], strict=True)

    def test_raw_day_tables_are_processed_in_bounded_batches(self):
        from moex_bonds.collection.history import RequestPool
        original = RequestPool.map
        def bounded(pool, function, items):
            items = list(items)
            if function.__name__ == 'fetch_day':
                self.assertLessEqual(len(items), 1)
            return original(pool, function, items)
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            get, _ = self.server(h)
            with patch('requests.Session.get', side_effect=get), patch.object(RequestPool, 'map', bounded):
                append_daily(output, dataset, sync_context=context, workers=1, pause=0, timeout=.01)

    def test_incomplete_current_reference_cannot_advance_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            context['samples'][0]['calendar.offers.pagination'] = 'request_error_incomplete'
            context['events'] = []
            dataset['rows'] = []
            original = (output / 'bonds_history.csv').read_bytes()
            get, _ = self.server(h)
            with patch('requests.Session.get', side_effect=get), self.assertRaises(ValueError):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            self.assertEqual((output / 'bonds_history.csv').read_bytes(), original)

    def test_missing_record_is_explicit_and_not_a_fresh_price(self):
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            get, calls = self.server(h)
            def mutate(url, params=None, **kwargs):
                response = get(url, params, **kwargs)
                payload = response.json()
                if params['date'] == '2026-10-28':
                    for name, block in payload.items():
                        if name.endswith('.cursor'):
                            continue
                        if name == 'history_yields':
                            block['data'] = []
                        else:
                            block['data'][0][block['columns'].index('SECID')] = 'OTHER'
                            block['data'][0][block['columns'].index('FACEUNIT')] = 'USD'
                return self.response(payload, url)
            with patch('requests.Session.get', side_effect=mutate):
                append_daily(output, dataset, sync_context=context, pause=0, timeout=.01)
            frame = pd.read_csv(output / 'bonds_history.csv')
            row = frame[frame.TRADEDATE.eq('2026-10-28')].iloc[0]
            self.assertEqual(row.TRADE_STATUS, 'no_history_record')
            self.assertTrue(pd.isna(row.WAPRICE))
            self.assertTrue(pd.isna(row.NUMTRADES))

    def test_successful_daily_export_keeps_latest_only_and_recovers_past(self):
        from moex_bonds.collection.daily import run_daily
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            source = Path(folder) / 'daily'
            source.mkdir()
            for name, rows in [('bonds_universe.csv', context['samples']), ('day_quotes.csv', h), ('bond_events.csv', context['events'])]:
                pd.DataFrame(rows).to_csv(source / name, index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(context['smartlab']), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', collection_status='ok',
                trade_date=dataset['metadata']['trade_date'])), encoding='utf-8')
            get, calls = self.server(h)
            with patch('requests.Session.get', side_effect=get):
                result = run_daily(source, reuse_smartlab=True, xlsx=False, history_dir=output, pause=0, timeout=.01)
            self.assertEqual(result['metadata']['history_update']['recovered_dates'], ['2026-10-27', '2026-10-28'])
            self.assertEqual(pd.read_csv(source / 'daily/bonds_daily.csv')['Дата данных'].unique().tolist(), ['2026-10-29'])
            self.assertEqual(pd.read_csv(source / 'day_quotes.csv').TRADEDATE.unique().tolist(), ['2026-10-29'])
            self.assertEqual(len(pd.read_csv(output / 'bonds_history.csv')), 4)
            self.assertEqual(sorted(p.name for p in source.iterdir() if p.is_dir()), ['daily'])

    def test_daily_export_stays_on_latest_day_even_if_sync_fails(self):
        from moex_bonds.collection.daily import run_daily
        with tempfile.TemporaryDirectory() as folder:
            output, dataset, context, h = self.prepare(folder)
            source = Path(folder) / 'daily'
            source.mkdir()
            for name, rows in [('bonds_universe.csv', context['samples']), ('day_quotes.csv', h), ('bond_events.csv', context['events'])]:
                pd.DataFrame(rows).to_csv(source / name, index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(context['smartlab']), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', collection_status='ok',
                trade_date=dataset['metadata']['trade_date'])), encoding='utf-8')
            get, _ = self.server(h, fail_day='2026-10-28')
            with patch('requests.Session.get', side_effect=get), patch('time.sleep'):
                result = run_daily(source, reuse_smartlab=True, xlsx=False, history_dir=output, pause=0, timeout=.01)
            self.assertEqual(result['metadata']['history_update']['status'], 'error')
            self.assertEqual(pd.read_csv(source / 'daily/bonds_daily.csv')['Дата данных'].unique().tolist(), ['2026-10-29'])
            self.assertEqual(pd.read_csv(source / 'day_quotes.csv').TRADEDATE.unique().tolist(), ['2026-10-29'])


if __name__ == '__main__':
    unittest.main()
