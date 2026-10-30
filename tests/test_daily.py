from datetime import date
import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from moex_bonds.collection.daily import (rating_grade, parse_smartlab, previous_trade_date,
                   unique_history, build_daily, HISTORY_ENDPOINT, YIELDS_ENDPOINT)


SECID = 'RU000A10EF52'
ASOF = date(2026, 10, 27)
DAY = date(2026, 10, 26)


def inputs():
    sample = {'SECID': SECID, 'BOARDID': 'TQCB', 'securities.PREVDATE': '2026-10-26',
              'securities.SHORTNAME': 'Сбер Sb52R', 'securities.FACEUNIT': 'SUR',
              'securities.CURRENCYID': 'SUR', 'securities.BONDTYPE': 'Облигация с фиксированным (известным) купоном',
              'securities.BONDSUBTYPE': 'До погашения', 'securities.ISIN': SECID,
              'securities/{SECID}.json.description.TYPE': 'exchange_bond',
              'securities/{SECID}.json.description.ISSUENAME': 'Облигации процентные неконвертируемые',
              'calendar.offers.rows': 0, 'calendar.offers.pagination': 'empty_terminal',
              'calendar.coupons.pagination': 'empty_terminal'}
    history = {'SECID': SECID, 'BOARDID': 'TQCB', 'TRADEDATE': DAY.isoformat(),
               'endpoint': HISTORY_ENDPOINT, 'source_section': 'history', 'window_days': 180,
               'TRADINGSESSION': 3, 'WAPRICE': 100.01, 'CLOSE': 100, 'ACCINT': 6.61,
               'VALUE': 66789544.3, 'NUMTRADES': 1270, 'VOLUME': 66785,
               'MATDATE': '2028-08-29', 'FACEUNIT': 'RUB', 'CURRENCYID': 'SUR',
               'FACEVALUE': 1000, 'COUPONPERCENT': 14.2, 'COUPONVALUE': 11.67,
               'ZSPREADATWAPRICE': .76, 'ZSPREAD': .76}
    yields = {'SECID': SECID, 'BOARDID': 'TQCB', 'TRADEDATE': DAY.isoformat(),
              'endpoint': YIELDS_ENDPOINT, 'source_section': 'history_yields', 'window_days': 180,
              'YIELDDATETYPE': 'MATDATE', 'YIELDDATE': '2028-08-29', 'WAPRICE': 100.01,
              'EFFECTIVEYIELDWAPRICE': 15.1532, 'DURATIONWAPRICE': 591,
              'DURATION': 591, 'GSPREADBP': 47, 'TRADEMOMENT': DAY.isoformat()}
    smart = {SECID: {'SECID': SECID, 'status': 'ok', 'rating': 'AAA', 'sector': 'Банки',
                    'issuer': 'Сбербанк ПАО', 'title': 'Облигация Сбербанк ПАО 001Р-SBER52',
                    'offer_type': 'Оферты отсутствуют', 'offer_date': '—',
                    'bond_type': sample['securities.BONDTYPE'], 'retrieved_at_msk': '2026-10-27T22:00:00+03:00',
                    'url': f'https://smart-lab.ru/q/bonds/{SECID}/'}}
    events = [{'probe_secid': SECID, 'section': 'coupons', 'coupondate': '2026-11-08'}]
    return [sample], [history, yields], events, smart


class DailyTests(unittest.TestCase):
    def test_rating_boundary_and_scales(self):
        self.assertGreaterEqual(rating_grade('BB'), rating_grade('BB'))
        self.assertLess(rating_grade('BB-'), rating_grade('BB'))
        self.assertEqual(rating_grade('ruBBB+'), rating_grade('BBB+(RU)'))
        self.assertEqual(rating_grade('AAA / BB-'), rating_grade('BB-'))
        self.assertIsNone(rating_grade('нет рейтинга'))
        self.assertIsNone(rating_grade('AAA / неизвестная шкала'))
        self.assertIsNone(rating_grade('AAA / рейтинг отозван'))

    def test_parse_identity_fields_and_series(self):
        html = f'''<h1>Облигация ГПБ (АО) БО 004Р-20 с фиксированным купоном ({SECID}) Доходность, Цена</h1>
        <div class="quotes-simple-table__row"><div class="quotes-simple-table__item">Кредитный рейтинг</div><div class="quotes-simple-table__item"><span>AA+</span></div></div>
        <div class="quotes-simple-table__row"><div class="quotes-simple-table__item">Сектор</div><div class="quotes-simple-table__item">Банки</div></div>'''
        row = parse_smartlab(html, SECID)
        self.assertEqual(row['issuer'], 'ГПБ (АО)')
        self.assertEqual(row['rating'], 'AA+')
        self.assertEqual(row['sector'], 'Банки')
        with self.assertRaises(ValueError):
            parse_smartlab(html, 'RU000A10F8T1')

    def test_previous_day_requires_matching_run(self):
        s, *_ = inputs()
        self.assertEqual(previous_trade_date(s, ASOF, ASOF), DAY)
        with self.assertRaises(ValueError):
            previous_trade_date(s, date(2026, 10, 28), ASOF)

    def test_overlapping_windows_deduplicated_but_conflicts_rejected(self):
        _, h, *_ = inputs()
        second = {**h[0], 'window_days': 30}
        self.assertEqual(len(unique_history(pd.DataFrame([h[0], second]), HISTORY_ENDPOINT)), 1)
        second['ZSPREAD'] = 99
        with self.assertRaises(ValueError):
            unique_history(pd.DataFrame([h[0], second]), HISTORY_ENDPOINT)

    def test_daily_uses_historical_values_and_maturity_yield(self):
        s, h, e, smart = inputs()
        s[0]['securities.ACCRUEDINT'] = 999
        s[0]['marketdata.ZSPREAD'] = 999
        rows, excluded = build_daily(s, h, e, smart, DAY)
        self.assertEqual(excluded, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['ACCINT'], 6.61)
        self.assertEqual(rows[0]['ZSPREAD'], .76)
        self.assertEqual(rows[0]['YTM_WAPRICE'], 15.1532)
        self.assertEqual(rows[0]['NEXTCOUPON'], '2026-11-08')
        self.assertIsNone(rows[0]['ADV20_RUB'])

    def test_filters_fail_closed(self):
        cases = [({'securities.FACEUNIT': 'USD'}, None),
                 ({'securities.BONDTYPE': 'Облигация с плавающим купоном'}, None),
                 ({'securities.OFFERDATE': '2026-10-25'}, None),
                 ({'securities.PUTOPTIONDATE': '2028-01-01'}, None),
                 ({'calendar.offers.rows': 1}, None),
                 ({'calendar.offers.pagination': 'request_error_incomplete'}, None),
                 ({'securities/{SECID}.json.description.ISSUENAME': 'Субординированные облигации'}, None),
                 ({'securities/{SECID}.json.description.TYPE': 'ofz_bond'}, None),
                 ({}, {'rating': 'BB-'}), ({}, {'status': 'http_error'}),
                 ({}, {'sector': None}), ({}, {'offer_type': 'Call'})]
        for changes, smart_changes in cases:
            with self.subTest(changes=changes, smart=smart_changes):
                s, h, e, smart = inputs()
                s[0].update(changes)
                smart[SECID].update(smart_changes or {})
                rows, excluded = build_daily(s, h, e, smart, DAY)
                self.assertEqual(rows, [])
                self.assertTrue(excluded[0]['reasons'])

    def test_three_calendar_months_inclusive(self):
        for maturity, keep in [('2027-01-25', False), ('2027-01-26', True)]:
            s, h, e, smart = inputs()
            h[0]['MATDATE'] = maturity
            h[1]['YIELDDATE'] = maturity
            rows, _ = build_daily(s, h, e, smart, DAY)
            self.assertEqual(bool(rows), keep)

    def test_offer_yield_not_labelled_ytm_and_stale_yields_not_carried(self):
        s, h, e, smart = inputs()
        h[1]['YIELDDATETYPE'] = 'OFFER'
        self.assertEqual(build_daily(s, h, e, smart, DAY)[0], [])
        s, h, e, smart = inputs()
        h[1]['TRADEMOMENT'] = '2026-10-23'
        rows, _ = build_daily(s, h, e, smart, DAY)
        self.assertIsNone(rows[0]['YTM_WAPRICE'])
        self.assertIsNone(rows[0]['GSPREADBP'])
        self.assertIn('устарев', rows[0]['DATA_GAPS'])

    def test_no_previous_instrument_day_fallback(self):
        s, h, e, smart = inputs()
        h[0]['TRADEDATE'] = '2026-10-23'
        self.assertEqual(build_daily(s, h, e, smart, DAY)[0], [])

    def test_structured_subordination_flag_excluded(self):
        s, h, e, smart = inputs()
        s[0]['securities/{SECID}.json.description.ISSUBORDINATED'] = 1
        self.assertEqual(build_daily(s, h, e, smart, DAY)[0], [])

    def test_past_offer_in_earlier_history_is_excluded(self):
        s, h, e, smart = inputs()
        h.append({**h[0], 'TRADEDATE': '2026-10-23', 'OFFERDATE': '2026-10-24'})
        self.assertEqual(build_daily(s, h, e, smart, DAY)[0], [])

    def test_adv20_excludes_future_and_deduplicates_overlapping_windows(self):
        s, h, e, smart = inputs()
        old = [{**h[0], 'TRADEDATE': (pd.Timestamp(DAY) - pd.Timedelta(days=n)).date().isoformat(),
                'VALUE': 100.0} for n in range(20)]
        future = {**h[0], 'TRADEDATE': '2026-10-27', 'VALUE': 999999999.0}
        history = old + [{**r, 'window_days': 30} for r in old] + [future, h[1]]
        rows, _ = build_daily(s, history, e, smart, DAY)
        self.assertEqual(rows[0]['ADV20_RUB'], 100.0)

    def test_daily_pipeline_integration_is_default_and_failure_visible(self):
        from moex_bonds.collection.main import finish_daily
        import argparse
        args = argparse.Namespace(daily=True, pause=0, timeout=1)
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            (target / 'run_manifest.json').write_text('{}', encoding='utf-8')
            with patch('moex_bonds.collection.daily.run_daily', return_value={'metadata': {'final_count': 4}}) as call:
                self.assertEqual(finish_daily(args, target), 0)
                self.assertEqual(call.call_count, 1)
            with patch('moex_bonds.collection.daily.run_daily', side_effect=ValueError('missing PREVDATE')):
                self.assertEqual(finish_daily(args, target), 3)
                self.assertIn('missing PREVDATE', (target / 'run_manifest.json').read_text(encoding='utf-8'))
            with patch('moex_bonds.collection.daily.run_daily') as call:
                self.assertEqual(finish_daily(argparse.Namespace(daily=False), target), 0)
                call.assert_not_called()

    def test_failed_export_or_csv_only_cannot_leave_old_final_xlsx(self):
        from moex_bonds.collection.daily import run_daily
        for xlsx in (True, False):
            with self.subTest(xlsx=xlsx), tempfile.TemporaryDirectory() as folder:
                source = Path(folder)
                output = source / 'daily'
                support = output / '_support'
                support.mkdir(parents=True)
                s, h, e, smart = inputs()
                pd.DataFrame(s).to_csv(source / 'bonds_sample.csv', index=False)
                pd.DataFrame(h).to_csv(source / 'historical_sample.csv', index=False)
                pd.DataFrame(e).to_csv(source / 'bond_events.csv', index=False)
                (source / 'run_manifest.json').write_text(json.dumps({'as_of_date': ASOF.isoformat()}), encoding='utf-8')
                (support / 'smartlab_enrichment.json').write_text(json.dumps(smart), encoding='utf-8')
                (output / 'bonds_daily.xlsx').write_bytes(b'old workbook')
                (support / 'previous_bonds_daily.xlsx').write_bytes(b'older backup')
                with patch('moex_bonds.collection.daily.export_xlsx', side_effect=RuntimeError('export failed')):
                    if xlsx:
                        with self.assertRaisesRegex(RuntimeError, 'export failed'):
                            run_daily(source, as_of=ASOF, reuse_smartlab=True)
                    else:
                        run_daily(source, as_of=ASOF, reuse_smartlab=True, xlsx=False)
                self.assertFalse((output / 'bonds_daily.xlsx').exists())
                self.assertFalse((support / 'previous_bonds_daily.xlsx').exists())


if __name__ == '__main__':
    unittest.main()
