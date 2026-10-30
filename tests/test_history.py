"""Offline checks for the history/daily contract, including absence of trades."""
import argparse
from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import requests

from tests.test_daily import inputs, DAY, SECID
from moex_bonds.collection.daily import build_daily


class HistoryTests(unittest.TestCase):
    def test_full_180_date_pipeline_with_pagination_and_three_month_issue(self):
        from moex_bonds.collection.history import run_history, BOARD_HISTORY
        s, h, e, smart = inputs()
        second = 'RU000A10TEST'
        s += [{**s[0], 'SECID': second, 'securities.ISIN': second}]
        smart[second] = {**smart[SECID], 'SECID': second}
        e += [{**e[0], 'probe_secid': second}]
        quotes = h + [{**r, 'SECID': second} for r in h]
        calendar = sorted((DAY - timedelta(days=n)).isoformat() for n in range(400)
                          if (DAY - timedelta(days=n)).weekday() < 5 or n == 1)[-180:]
        short_dates = calendar[-60:]
        def get(url, params=None, **kwargs):
            section = 'history_yields' if '/yields/' in url else 'history'
            if url.endswith(BOARD_HISTORY):
                day = params['date']
                data = [[day, 'TQCB', 3]] if day in calendar else []
                payload = {'history': {'columns': ['TRADEDATE', 'BOARDID', 'TRADINGSESSION'], 'data': data}}
            else:
                secid = second if second in url else SECID
                days = short_dates if secid == second else calendar
                # A valid missing record is distinct from an explicit zero-trade day.
                days = [d for d in days if not (section == 'history' and d == calendar[-3])]
                raw = [{**h[section == 'history_yields'], 'SECID': secid, 'TRADEDATE': d,
                        'TRADEMOMENT': d, 'NUMTRADES': 0 if d == calendar[-2] else 12} for d in days]
                start = params.get('start', 0)
                page = raw[start:start+100]
                columns = list(raw[0])
                payload = {section: {'columns': columns, 'data': [[r[c] for c in columns] for r in page]},
                           section+'.cursor': {'columns': ['INDEX', 'TOTAL', 'PAGESIZE'], 'data': [[start, len(raw), 100]]}}
            response = requests.Response()
            response.status_code, response.url = 200, url
            response._content = json.dumps(payload).encode()
            return response
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'day', Path(folder) / 'history'
            source.mkdir()
            for name, values in [('bonds_universe.csv', s), ('day_quotes.csv', quotes), ('bond_events.csv', e)]:
                pd.DataFrame(values).to_csv(source / name, index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', trade_date=DAY.isoformat(), collection_status='ok')), encoding='utf-8')
            args = argparse.Namespace(input=str(source), output=str(output), history_days=180, workers=4,
                                      secids='', pause=0, timeout=1)
            with patch('requests.Session.get', side_effect=get):
                self.assertEqual(run_history(args), 0)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(len(frame), 240)
            self.assertEqual(frame.groupby('SECID').size().to_dict(), {SECID: 180, second: 60})
            self.assertFalse(frame.duplicated(['SECID', 'BOARDID', 'TRADEDATE']).any())
            self.assertTrue(frame.loc[frame.TRADEDATE.eq(calendar[-2]), 'WAPRICE'].isna().all())
            self.assertEqual(set(frame.loc[frame.TRADEDATE.eq(calendar[-3]), 'TRADE_STATUS']), {'no_history_record'})
            metadata = json.loads((output / 'run_manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(metadata['trading_dates'], calendar)
            self.assertTrue(list((output / '_support/moex').glob('worker_*/endpoints_status.csv')))

    def test_zero_trades_never_exports_calculated_prices_as_fresh(self):
        samples, quotes, events, smart = inputs()
        quotes[0]['NUMTRADES'] = 0
        rows, _ = build_daily(samples, quotes, events, smart, DAY)
        self.assertIsNone(rows[0]['WAPRICE'])
        self.assertIsNone(rows[0]['ZSPREAD'])
        self.assertIsNone(rows[0]['YTM_WAPRICE'])
        self.assertEqual(rows[0]['TRADE_STATUS'], 'no_trades')

    def test_history_short_issue_missing_day_and_same_daily_fields(self):
        from moex_bonds.collection.history import build_history
        s, h, e, smart = inputs()
        first = '2026-10-23'
        h = [{**r, 'TRADEDATE': first, 'TRADEMOMENT': first} for r in h]
        dates = ['2026-10-22', first, '2026-10-24', DAY.isoformat()]
        rows, coverage = build_history(s, h, e, smart, dates)
        self.assertEqual([r['TRADEDATE'] for r in rows], dates[1:])
        self.assertEqual(rows[1]['TRADE_STATUS'], 'no_history_record')
        self.assertIsNone(rows[1]['WAPRICE'])
        self.assertIsNone(rows[1]['NUMTRADES'])
        self.assertEqual(rows[0]['ACCINT'], h[0]['ACCINT'])
        self.assertEqual(rows[0]['ZSPREADATWAPRICE'], h[0]['ZSPREADATWAPRICE'])
        self.assertEqual(coverage[0]['first_observation'], first)

    def test_calendar_uses_published_records_including_weekends(self):
        from moex_bonds.collection.history import trading_dates
        class Client:
            def get(self, path, params, purpose):
                # Sunday is a published session; Monday has no board records.
                day = params['date']
                return {'history': {'columns': ['TRADEDATE', 'BOARDID'],
                                    'data': [] if day == '2026-10-26' else [[day, 'TQCB']]}}
        dates = trading_dates(Client(), date(2026, 10, 27), 3)
        self.assertEqual(dates, ['2026-10-24', '2026-10-25', '2026-10-27'])

    def test_calendar_network_error_is_not_a_holiday(self):
        from moex_bonds.collection.history import trading_dates
        class Client:
            def get(self, *args, **kwargs):
                return None
        with self.assertRaises(ValueError):
            trading_dates(Client(), DAY, 2)

    def test_storage_upsert_replaces_full_date_and_preserves_older(self):
        from moex_bonds.storage.history_storage import publish_history, append_daily
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            old = dict(SECID=SECID, BOARDID='TQCB', TRADEDATE='2026-10-23', WAPRICE=99)
            today = {**old, 'TRADEDATE': DAY.isoformat(), 'WAPRICE': 100}
            publish_history(output, [old, today], dict(status='ok', trading_dates=['2026-10-23', DAY.isoformat()]))
            dataset = dict(rows=[{**today, 'WAPRICE': 101}], metadata=dict(trade_date=DAY.isoformat()))
            append_daily(output, dataset)
            append_daily(output, dataset)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.iloc[-1].WAPRICE, 101)
            append_daily(output, dict(rows=[], metadata=dict(trade_date=DAY.isoformat())))
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(list(frame.TRADEDATE), ['2026-10-23'])

    def test_daily_without_base_does_not_bootstrap_history(self):
        from moex_bonds.storage.history_storage import append_daily
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'new'
            state = append_daily(output, dict(rows=[], metadata=dict(trade_date=DAY.isoformat())))
            self.assertEqual(state['status'], 'not_initialized')
            self.assertFalse(output.exists())

    def test_explicit_daily_subset_preserves_other_bonds_on_same_date(self):
        from moex_bonds.storage.history_storage import publish_history, append_daily
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            row = dict(SECID=SECID, BOARDID='TQCB', TRADEDATE=DAY.isoformat(), WAPRICE=99)
            publish_history(output, [row, {**row, 'SECID': 'OTHER'}],
                            dict(status='ok', trading_dates=[DAY.isoformat()]))
            append_daily(output, dict(rows=[], metadata=dict(trade_date=DAY.isoformat(), replace_secids=[SECID])))
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(list(frame.SECID), ['OTHER'])

    def test_append_failure_never_marks_new_csv_as_old_success(self):
        from moex_bonds.storage.history_storage import publish_history, append_daily
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            row = dict(SECID=SECID, BOARDID='TQCB', TRADEDATE=DAY.isoformat(), WAPRICE=99)
            publish_history(output, [row], dict(status='ok', trading_dates=[DAY.isoformat()]))
            original = (output / 'bonds_history.csv').read_bytes()
            replace = Path.replace
            def failed_replace(path, target):
                if path.name == 'bonds_history.csv.tmp':
                    raise PermissionError('file open in Excel')
                return replace(path, target)
            with patch.object(Path, 'replace', failed_replace), self.assertRaises(PermissionError):
                append_daily(output, dict(rows=[{**row, 'WAPRICE': 101}], metadata=dict(trade_date=DAY.isoformat())))
            self.assertEqual((output / 'bonds_history.csv').read_bytes(), original)
            self.assertNotEqual(json.loads((output / 'run_manifest.json').read_text())['status'], 'ok')

    def test_day_pipeline_appends_custom_base_even_when_xlsx_fails(self):
        from moex_bonds.collection.daily import run_daily
        from moex_bonds.storage.history_storage import publish_history
        s, h, e, smart = inputs()
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / 'day', Path(folder) / 'history'
            source.mkdir()
            output.mkdir()
            old = dict(SECID=SECID, BOARDID='TQCB', TRADEDATE='2026-10-23', WAPRICE=99)
            publish_history(output, [old], dict(status='ok', trading_dates=['2026-10-23']))
            for name, values in [('bonds_universe.csv', s), ('day_quotes.csv', h), ('bond_events.csv', e)]:
                pd.DataFrame(values).to_csv(source / name, index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', trade_date=DAY.isoformat(), collection_status='ok')), encoding='utf-8')
            empty = pd.DataFrame(columns=['SECID', 'BOARDID', 'TRADEDATE'])
            with (patch('moex_bonds.collection.history_sync.fetch_day', side_effect=lambda client, day: (day, empty, empty)),
                  patch('moex_bonds.collection.daily.export_xlsx', side_effect=RuntimeError('export failed')), self.assertRaises(RuntimeError)):
                run_daily(source, reuse_smartlab=True, history_dir=output)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(list(frame.TRADEDATE), ['2026-10-23', DAY.isoformat()])
            self.assertTrue((source / 'daily/bonds_daily.csv').exists())
            with patch('moex_bonds.collection.history_sync.fetch_day', side_effect=lambda client, day: (day, empty, empty)):
                run_daily(source, reuse_smartlab=True, history_dir=output, xlsx=False)
            self.assertEqual(len(pd.read_csv(output / 'bonds_history.csv')), 2)

    def test_collection_rejects_incomplete_pagination_and_wrong_dates(self):
        from moex_bonds.collection.history import collect_instrument
        s, h, _, _ = inputs()
        class Client:
            def collect(self, *args, **kwargs):
                return pd.DataFrame([h[0]]), 'request_error_incomplete'
        with self.assertRaisesRegex(ValueError, 'Неполная история'):
            collect_instrument(Client(), (s[0], '2026-10-23', DAY.isoformat(), {DAY.isoformat()}))

    def test_history_cannot_omit_the_known_daily_record(self):
        from moex_bonds.collection.history import collect_instrument
        s, h, _, _ = inputs()
        class Client:
            def collect(self, *args, **kwargs):
                return pd.DataFrame([{**h[0], 'TRADEDATE': '2026-10-23'}]), 'cursor_complete'
        with self.assertRaisesRegex(ValueError, 'дневного запуска'):
            collect_instrument(Client(), (s[0], '2026-10-23', DAY.isoformat(), {'2026-10-23', DAY.isoformat()}))

    def test_failed_base_update_preserves_daily_export(self):
        from moex_bonds.collection.daily import run_daily
        s, h, e, smart = inputs()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'day'
            source.mkdir()
            for name, values in [('bonds_universe.csv', s), ('day_quotes.csv', h), ('bond_events.csv', e)]:
                pd.DataFrame(values).to_csv(source / name, index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', trade_date=DAY.isoformat(), collection_status='ok')), encoding='utf-8')
            with patch('moex_bonds.storage.history_storage.append_daily', side_effect=PermissionError('Excel lock')):
                dataset = run_daily(source, reuse_smartlab=True, xlsx=False)
            self.assertTrue((source / 'daily/bonds_daily.csv').exists())
            self.assertEqual(dataset['metadata']['history_update']['status'], 'error')

    def test_rebuild_path_cannot_be_nested_in_daily(self):
        from moex_bonds.storage.history_storage import history_output
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'daily'
            for output in (source, source / 'history', source.parent):
                with self.assertRaises(ValueError):
                    history_output(source, output)

    def test_unknown_trade_count_does_not_export_price(self):
        s, h, e, smart = inputs()
        h[0].pop('NUMTRADES')
        rows, _ = build_daily(s, h, e, smart, DAY)
        self.assertEqual(rows[0]['TRADE_STATUS'], 'unknown_trade_count')
        self.assertIsNone(rows[0]['WAPRICE'])

    def test_history_rebuild_uses_final_filters_and_overwrites(self):
        from moex_bonds.collection.history import run_history
        s, h, e, smart = inputs()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'daily'
            source.mkdir()
            pd.DataFrame(s + [{**s[0], 'SECID': 'EXCLUDED'}]).to_csv(source / 'bonds_universe.csv', index=False)
            pd.DataFrame(h).to_csv(source / 'day_quotes.csv', index=False)
            pd.DataFrame(e).to_csv(source / 'bond_events.csv', index=False)
            support = source / 'daily/_support'
            support.mkdir(parents=True)
            (support / 'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
            (source / 'run_manifest.json').write_text(json.dumps(dict(mode='day', trade_date=DAY.isoformat(), collection_status='ok')), encoding='utf-8')
            output = Path(folder) / 'history'
            args = argparse.Namespace(input=str(source), output=str(output), history_days=2, secids='', pause=0, timeout=1, workers=2)
            def collect(client, path, section, params, **kwargs):
                self.assertIn(SECID, path)
                self.assertEqual(params['from'], '2026-10-23')
                return pd.DataFrame([h[1] if section == 'history_yields' else h[0]]), 'cursor_complete'
            with patch('moex_bonds.collection.history.trading_dates', return_value=['2026-10-23', DAY.isoformat()]), patch('moex_bonds.sources.moex_client.MoexClient.collect', autospec=True, side_effect=collect):
                self.assertEqual(run_history(args), 0)
                (output / 'notes.txt').write_text('keep', encoding='utf-8')
                self.assertEqual(run_history(args), 0)
            frame = pd.read_csv(output / 'bonds_history.csv')
            self.assertEqual(set(frame.SECID), {SECID})
            self.assertEqual(len(frame), 1)
            self.assertFalse(list(output.glob('run_*/*')))
            self.assertTrue((output / 'notes.txt').exists())


if __name__ == '__main__':
    unittest.main()
