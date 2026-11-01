"""Portable, bounded saved-data bundle. No network or UI imports."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from moex_bonds.paths import PROJECT_DIR, recorded_source_hash
from moex_bonds.analysis.spread_results import file_hash

TABLES = ('summary', 'metrics', 'comparison_history', 'basket_members', 'recovery_episodes', 'candidate_exclusions')

def read_json(path):
    value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(value, dict):
        raise ValueError('Expected JSON object: ' + str(path))
    return value

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temp, path)

def data_mode(environ=None):
    mode = (os.environ if environ is None else environ).get('BONDS_DATA_MODE', 'local')
    if mode not in ('local', 'published'):
        raise ValueError('BONDS_DATA_MODE must be local or published')
    return mode

def allowed(day, curve, schema=1):
    from datetime import date
    if date.fromisoformat(day).isoformat() != day:
        raise ValueError('Invalid publication date')
    names = {'daily_data/daily/bonds_daily.xlsx', 'daily_data/run_manifest.json',
             'daily_data/_support/cbr_zcyc_status.json', 'spread_analysis/run_manifest.json',
             f'spread_analysis/snapshots/{day}/run_manifest.json'}
    names.update(f'spread_analysis/snapshots/{day}/{name}.csv' for name in TABLES)
    if schema == 2:
        names.add(f'spread_analysis/snapshots/{day}/recovery_history.csv')
    elif schema != 1:
        raise ValueError('Unsupported publication schema')
    if curve:
        names.add('daily_data/cbr_zcyc.csv')
    return names

def safe_file(root, name):
    pure = PurePosixPath(name)
    if pure.is_absolute() or '..' in pure.parts or '\\' in name or ':' in name or str(pure) != name:
        raise ValueError('Unsafe bundle path: ' + name)
    root = Path(root).resolve()
    target = root.joinpath(*pure.parts)
    for item in (target, *target.parents):
        if item == root:
            break
        if item.is_symlink() or getattr(item, 'is_junction', lambda: False)():
            raise ValueError('Links are not allowed')
    if not target.resolve().is_relative_to(root):
        raise ValueError('Path outside bundle')
    return target

def validate_bundle(root):
    root = Path(root)
    m = read_json(root/'publication_manifest.json')
    schema = m.get('publication_schema_version')
    if schema not in (1, 2) or m.get('status') != 'ok':
        raise ValueError('Incomplete publication')
    if not {'analysis_date','source_statuses','source_bindings','files','analysis_run_id','total_size_bytes'}.issubset(m):
        raise ValueError('Missing publication metadata')
    day = m['analysis_date']
    status = m['source_statuses']['cbr_zcyc']
    if status['status'] not in ('ok', 'unavailable', 'error') or status['trade_date'] != day:
        raise ValueError('Invalid CBR status')
    expected = allowed(day, status['status'] == 'ok', schema)
    if set(m['files']) != expected:
        raise ValueError('Unexpected bundle file list')
    actual = {str(p.relative_to(root)).replace('\\','/') for p in root.rglob('*') if p.is_file()}
    if actual != expected | {'publication_manifest.json'}:
        raise ValueError('Extra or missing bundle files')
    total = 0
    for name, info in m['files'].items():
        p = safe_file(root, name)
        if p.stat().st_size != info['size_bytes'] or file_hash(p) != info['sha256']:
            raise ValueError('Bundle checksum mismatch: '+name)
        total += info['size_bytes']
    if total != m['total_size_bytes']:
        raise ValueError('Bundle size mismatch')
    analysis = read_json(root/f'spread_analysis/snapshots/{day}/run_manifest.json')
    recovery_version = analysis.get('recovery_rule_version')
    if ((schema == 1 and recovery_version == 'own_spread_recovery_v2')
            or (schema == 2 and recovery_version in (None, 'own_spread_recovery_v1'))):
        raise ValueError('Publication schema and recovery version differ')
    for marker in (analysis, read_json(root/'spread_analysis/run_manifest.json')):
        if marker.get('status') != 'ok' or marker.get('analysis_date') != day or marker.get('analysis_run_id') != m['analysis_run_id']:
            raise ValueError('Mixed analysis publications')
        if marker.get('recovery_rule_version') != recovery_version:
            raise ValueError('Mixed recovery versions in publication markers')
        if marker.get('investing_rule_version') != analysis.get('investing_rule_version'):
            raise ValueError('Mixed investing versions in publication markers')
    binding = m['source_bindings']['daily_workbook']
    if binding['path'] != 'daily_data/daily/bonds_daily.xlsx' or analysis['source_sha256'].get(binding['source_key']) != binding['sha256'] or m['files'][binding['path']]['sha256'] != binding['sha256']:
        raise ValueError('Invalid workbook binding')
    daily = read_json(root/'daily_data/run_manifest.json')
    curve = read_json(root/'daily_data/_support/cbr_zcyc_status.json')
    if daily.get('collection_status') != 'ok' or daily['trade_date'] != day or curve != status or daily.get('cbr_zcyc') != status:
        raise ValueError('Mixed daily/CBR data')
    import pandas as pd
    from moex_bonds.analysis.spread_investing import validate_saved
    validate_saved(pd.read_csv(root/f'spread_analysis/snapshots/{day}/summary.csv'),analysis)
    return m

def build_bundle(daily, analysis, destination):
    daily, analysis, destination = map(Path, (daily, analysis, destination))
    marker = read_json(analysis/'run_manifest.json')
    day = marker['analysis_date']
    snapshot = analysis/'snapshots'/day
    source = read_json(snapshot/'run_manifest.json')
    book = daily/'daily/bonds_daily.xlsx'
    hashes = source['source_sha256']
    digest = file_hash(book)
    if recorded_source_hash(hashes, book) != digest:
        raise ValueError('Workbook is not an input of this analysis')
    # The exact local/legacy key is resolved before crossing OS boundaries.
    direct = str(book.resolve())
    key = direct
    if hashes.get(direct) != digest:
        key = str(PROJECT_DIR / book.resolve().relative_to(PROJECT_DIR/'data'))
    status = read_json(daily/'_support/cbr_zcyc_status.json')
    schema = 2 if source.get('recovery_rule_version') == 'own_spread_recovery_v2' else 1
    names = allowed(day, status['status'] == 'ok', schema)
    destination.mkdir(parents=True, exist_ok=False)
    files = {}
    for name in sorted(names):
        relative = PurePosixPath(name)
        src = (daily if relative.parts[0] == 'daily_data' else analysis).joinpath(*relative.parts[1:])
        dst = safe_file(destination, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        before = file_hash(src)
        shutil.copyfile(src, dst)
        if before != file_hash(dst) or before != file_hash(src):
            raise ValueError('Source changed while copying')
        files[name] = {'sha256': before, 'size_bytes': dst.stat().st_size}
    m = dict(publication_schema_version=schema, status='ok', analysis_date=day,
             analysis_run_id=source['analysis_run_id'], files=files,
             source_bindings={'daily_workbook':dict(path='daily_data/daily/bonds_daily.xlsx', source_key=key, sha256=digest)},
             source_statuses={'cbr_zcyc':status}, total_size_bytes=sum(v['size_bytes'] for v in files.values()))
    save_json(destination/'publication_manifest.json', m)
    return validate_bundle(destination)
