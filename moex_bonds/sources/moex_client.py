"""Small synchronous, auditable MOEX ISS client; no authentication."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import hashlib
import json
import time

import pandas as pd
import requests

MSK = ZoneInfo('Europe/Moscow')
BASE = 'https://iss.moex.com/iss/'


def tables(payload):
    """Read each ISS block using its own dynamic columns (including empty blocks)."""
    if not isinstance(payload, dict):
        return {}
    return {name: pd.DataFrame(block['data'], columns=block['columns'])
            for name, block in payload.items()
            if isinstance(block, dict) and isinstance(block.get('columns'), list)
            and isinstance(block.get('data'), list)}


class MoexClient:
    def __init__(self, output: Path, pause=0.25, timeout=15, retries=1):
        self.output = Path(output)
        self.raw = self.output / 'raw_responses'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'moex-public-data-probe/1.0'
        self.pause, self.timeout, self.retries = pause, timeout, retries
        self.records, self.schemas, self.pages = [], [], []

    def get(self, path, params=None, purpose=''):
        url = BASE + path.lstrip('/')
        params = dict(params or {})
        for attempt in range(self.retries + 1):
            if self.records:
                time.sleep(self.pause)
            started = datetime.now(MSK)
            clock = time.monotonic()
            prepared = requests.Request('GET', url, params=params).prepare().url
            record = dict(request_id=len(self.records) + 1, url=prepared, purpose=purpose,
                          attempt=attempt + 1, requested_at_msk=started.isoformat(),
                          http_status=None, outcome='', error='', rows=0,
                          sections='{}', raw_file='', response_date='', server='',
                          cache_control='', age_header='', authentication='none')
            payload = None
            retry = False
            try:
                response = self.session.get(url, params=params, timeout=(self.timeout, self.timeout))
                record.update(http_status=response.status_code, url=response.url,
                              response_date=response.headers.get('Date', ''),
                              server=response.headers.get('Server', ''),
                              cache_control=response.headers.get('Cache-Control', ''),
                              age_header=response.headers.get('Age', ''))
                stem = f"{record['request_id']:05d}_{hashlib.sha256(prepared.encode()).hexdigest()[:12]}"
                try:
                    parsed = response.json()
                except ValueError:
                    parsed = None
                raw_path = self.raw / (stem + ('.json' if parsed is not None else '.txt'))
                raw_path.write_bytes(response.content)
                record['raw_file'] = str(raw_path.relative_to(self.output))
                if response.status_code == 200:
                    if isinstance(parsed, dict):
                        payload = parsed
                        frames = tables(payload)
                        counts = {name: len(frame) for name, frame in frames.items()}
                        record['sections'] = json.dumps(counts, ensure_ascii=False)
                        record['rows'] = sum(n for k, n in counts.items() if not k.endswith('.cursor'))
                        record['outcome'] = 'ok' if record['rows'] else 'http_200_empty'
                        for name, frame in frames.items():
                            self.schemas.append(dict(request_id=record['request_id'], url=response.url,
                                section=name, rows=len(frame), columns=list(frame.columns),
                                sample=frame.head(3).where(pd.notna(frame), None).to_dict('records'),
                                metadata=payload[name].get('metadata', {})))
                    else:
                        record.update(outcome='invalid_json', error='HTTP 200 without ISS JSON object')
                else:
                    record.update(outcome='access_restricted' if response.status_code in (401, 403)
                                  else 'http_error', error=response.text[:2000])
                    retry = response.status_code == 429 or response.status_code >= 500
            except requests.RequestException as error:
                record.update(outcome='network_error', error=f'{type(error).__name__}: {error}')
                retry = True
            except (ValueError, TypeError) as error:
                record.update(outcome='schema_error', error=str(error))
            record.update(received_at_msk=datetime.now(MSK).isoformat(),
                          elapsed_seconds=round(time.monotonic() - clock, 4))
            self.records.append(record)
            if record['error']:
                with (self.output / 'errors.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if retry and attempt < self.retries:
                # A bounded pause; do not hammer rate-limited or unavailable servers.
                time.sleep(min(2 ** (attempt + 1), 30))
                continue
            return payload
        return None

    def collect(self, path, section, params=None, purpose='', first=None, start_key='start', max_pages=1000):
        """Cursor first; otherwise request until empty. Detect servers ignoring start."""
        params = dict(params or {})
        offset, seen, chunks, status = 0, set(), [], 'max_pages_incomplete'
        for page in range(max_pages):
            query = {**params, start_key: offset}
            payload = first if page == 0 and first is not None else self.get(path, query, purpose)
            if payload is None:
                status = 'request_error_incomplete'
                break
            frame = tables(payload).get(section)
            if frame is None:
                status = 'section_missing'
                break
            if frame.empty:
                if not chunks:
                    chunks.append(frame)
                status = 'empty_terminal'
                break
            fingerprint = hashlib.sha256(frame.to_json(orient='split').encode()).hexdigest()
            if fingerprint in seen:
                status = 'repeated_page_completeness_unconfirmed'
                break
            seen.add(fingerprint)
            chunks.append(frame)
            cursor = tables(payload).get(section + '.cursor')
            if cursor is not None and not cursor.empty:
                current = cursor.iloc[0].to_dict()
                total = current.get('TOTAL')
                if total is not None and offset + len(frame) >= int(total):
                    status = 'cursor_complete'
                    break
            offset += len(frame)
        result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        self.pages.append(dict(path=path, section=section, params=params, rows=len(result),
                               pages=len(seen), status=status))
        return result, status

    def save(self):
        pd.DataFrame(self.records).to_csv(self.output / 'endpoints_status.csv', index=False, encoding='utf-8-sig')
        (self.output / 'schemas_and_samples.json').write_text(
            json.dumps(self.schemas, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        (self.output / 'pagination.json').write_text(
            json.dumps(self.pages, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        (self.output / 'errors.jsonl').touch(exist_ok=True)
