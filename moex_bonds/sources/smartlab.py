"""Read-only public Smart-Lab search and bond-card enrichment, keyed by SECID."""
from datetime import datetime
import json
from pathlib import Path
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
import requests

from moex_bonds.sources.moex_client import MSK

BASE = 'https://smart-lab.ru'


def clean(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def parse_smartlab(html, secid):
    soup = BeautifulSoup(html, 'html.parser')
    heading = soup.select_one('h1')
    title = clean(heading.get_text(' ', strip=True)) if heading else ''
    if not re.search(r'\(' + re.escape(secid) + r'\)', title, re.I):
        raise ValueError('SECID карточки не совпадает с запросом')
    if not title.startswith('Облигация '):
        raise ValueError('Не распознано название карточки облигации')
    fields = {}
    for row in soup.select('.quotes-simple-table__row'):
        cells = row.select('.quotes-simple-table__item')
        if len(cells) >= 2:
            fields[clean(cells[0].get_text(' ', strip=True))] = clean(cells[1].get_text(' ', strip=True))
    name = title[len('Облигация '):].split('(' + secid + ')')[0].strip()
    name = re.split(r'\s+с (?:фиксированным|плавающим|переменным|индексируемым) купоном', name, flags=re.I)[0]
    # Keep digits in a company name; strip a recognisable issue-series suffix only.
    series = re.search(r'\s+(?:(?:БО|серия|серии)\s+)?(?:\d{2,}[\w.\-]*|\d[РP][\w.\-]*)(?:\s.*)?$', name, re.I)
    issuer = name[:series.start()].strip() if series else None
    def field(prefix):
        return next((v for k, v in fields.items() if k.startswith(prefix)), None)
    return dict(SECID=secid, title=title, issuer=issuer,
                rating=field('Кредитный рейтинг'), sector=field('Сектор'),
                bond_type=field('Тип облигации'), offer_date=field('Дата оферты'),
                offer_type=field('Тип оферты'))


class SmartLabClient:
    def __init__(self, output, pause=.5, timeout=15):
        self.output = Path(output)
        self.raw = self.output / 'smartlab_raw'
        self.raw.mkdir(parents=True, exist_ok=True)
        self.pause, self.timeout = pause, timeout
        self.session = requests.Session()
        self.session.headers['User-Agent'] = 'bond-daily-research/1.0 (public pages)'
        self.log = []

    def request(self, method, path, secid, **kwargs):
        if self.log:
            time.sleep(self.pause)
        url = urljoin(BASE, path)
        parsed = urlparse(url)
        if parsed.scheme != 'https' or parsed.netloc != 'smart-lab.ru':
            raise ValueError('Неожиданный домен Smart-Lab')
        entry = dict(SECID=secid, method=method, url=url, received_at_msk='',
                     http_status=None, error='', raw_file='')
        response = None
        try:
            response = self.session.request(method, url, timeout=(self.timeout, self.timeout), **kwargs)
            entry.update(http_status=response.status_code, url=response.url)
            raw = self.raw / f'{len(self.log)+1:04d}_{secid}_{method}.txt'
            raw.write_bytes(response.content)
            entry['raw_file'] = str(raw.relative_to(self.output))
            response.raise_for_status()
            if urlparse(response.url).netloc != 'smart-lab.ru':
                raise ValueError('Неожиданный redirect Smart-Lab')
        except (requests.RequestException, ValueError) as error:
            entry['error'] = f'{type(error).__name__}: {error}'
        entry['received_at_msk'] = datetime.now(MSK).isoformat()
        self.log.append(entry)
        # Persist even if a later instrument fails or the process is interrupted.
        (self.output / 'smartlab_requests.json').write_text(
            json.dumps(self.log, ensure_ascii=False, indent=2), encoding='utf-8')
        if entry['error']:
            raise ValueError(entry['error'])
        return response

    def fetch(self, secid):
        if not re.fullmatch(r'[A-Za-z0-9]{1,20}', secid):
            raise ValueError('Некорректный SECID')
        result = dict(SECID=secid, status='error', url='', rating=None, sector=None, issuer=None)
        try:
            # Same read-only autocomplete request as the website's search field.
            response = self.request('POST', '/bonds/ajaxsearch/', secid,
                                    data={'value': secid, 'json': 1, 'forums': 1})
            hits = response.json().get('results', [])
            expected = f'/q/bonds/{secid}/'
            match = next((hit.get('data') for hit in hits
                          if hit.get('data', '').rstrip('/') == expected.rstrip('/')), None)
            if not match:
                raise ValueError('Точный SECID не найден поиском Smart-Lab')
            result['url'] = BASE + match
            response = self.request('GET', match, secid)
            result.update(parse_smartlab(response.content, secid), status='ok')
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            result['error'] = str(error)
        result['retrieved_at_msk'] = datetime.now(MSK).isoformat()
        return result
