"""Shared location and bounded overwrite policy for the daily collectors."""
from contextlib import contextmanager
from datetime import date
import json
import os
from pathlib import Path
import shutil

from moex_bonds.paths import PROJECT_DIR, DAILY_DIR

DEFAULT_DAILY_OUTPUT = DAILY_DIR
DAILY_FILES = (
    'bonds_universe.csv', 'day_quotes.csv', 'bond_events.csv', 'fields_availability.csv',
    'run_manifest.json', 'report.md', 'endpoints_status.csv', 'errors.jsonl',
    'pagination.json', 'schemas_and_samples.json', 'cbr_zcyc.csv', 'cbr_ruonia.csv',
)
DAILY_DIRECTORIES = ('raw_responses', '_support', 'daily')


def checked_output(output):
    base = Path(output).resolve()
    if base == Path(base.anchor) or base == PROJECT_DIR or base in PROJECT_DIR.parents:
        raise ValueError('Для выгрузки нужна отдельная папка, например ' + str(DEFAULT_DAILY_OUTPUT))
    return base


def checked_child(base, name):
    path = base / name
    # Resolve every final target before recursive deletion. Never follow a
    # symlink or Windows junction outside the explicitly selected output.
    if (path.is_symlink() or getattr(path, 'is_junction', lambda: False)()
            or path.resolve() != base / name):
        raise ValueError('Нельзя очищать ссылку или внешний путь: ' + str(path))
    return path


def reset_daily_output(output):
    """Remove only owned daily artifacts; preflight all paths before any delete."""
    base = checked_output(output)
    files = [checked_child(base, name) for name in DAILY_FILES]
    directories = [checked_child(base, name) for name in DAILY_DIRECTORIES]
    if any(p.exists() and not p.is_file() for p in files):
        raise ValueError('На месте дневного файла обнаружен каталог; очистка отменена')
    if any(p.exists() and not p.is_dir() for p in directories):
        raise ValueError('На месте служебного каталога обнаружен файл; очистка отменена')
    status_file = base / 'run_manifest.json'
    marker = dict(mode='day', collection_status='incomplete', cleanup_status='in_progress',
                  storage_mode='overwrite', output_dir=str(base))

    def save_marker():
        status_file.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding='utf-8')

    # Invalidate success before any delete. On Windows an open CSV can stop
    # cleanup halfway; the remaining old files must never retain an ok marker.
    save_marker()
    try:
        for path in files:
            if path != status_file:
                path.unlink(missing_ok=True)
        for path in directories:
            if path.exists():
                shutil.rmtree(checked_child(base, path.name))
    except OSError as error:
        marker.update(cleanup_status='error', error=f'{type(error).__name__}: {error}')
        save_marker()
        raise
    marker['cleanup_status'] = 'ok'
    save_marker()


@contextmanager
def output_lock(output):
    """One CLI writer per directory, including the whole main + CBR job group."""
    base = checked_output(output)
    base.mkdir(parents=True, exist_ok=True)
    lock = base / '.daily_write.lock'
    try:
        stream = lock.open('x', encoding='utf-8')
    except FileExistsError:
        raise RuntimeError('Папка выгрузки занята другим запуском: ' + str(base)) from None
    try:
        with stream:
            stream.write(str(os.getpid()))
        yield base
    finally:
        lock.unlink(missing_ok=True)


def matching_daily_manifest(output, day: date):
    """Keep a separate CBR retry aligned with any existing daily collection."""
    path = Path(output) / 'run_manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('mode') != 'day' or manifest.get('collection_status') != 'ok':
        raise ValueError('В этой папке нет завершённого дневного сбора; сначала запустите main.py')
    if manifest.get('trade_date') != day.isoformat():
        raise ValueError('Дата ЦБ не совпадает с датой торгов в папке; используйте --input или обновите main.py --date')
    return manifest
