"""Project locations, independent of the shell's working directory."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / 'data'
DAILY_DIR = DATA_DIR / 'daily_data'
HISTORY_DIR = DATA_DIR / 'history_data'
ANALYSIS_DIR = DATA_DIR / 'spread_analysis'
CONFIG_FILE = PROJECT_DIR / 'config/spread_config.json'


def recorded_source_hash(hashes, path):
    """Find an input digest without rewriting historical manifests.

    Only the three project data folders were relocated. Custom input paths
    must match exactly; a matching filename alone never establishes identity.
    The caller still verifies the actual bytes against the returned digest.
    """
    path = Path(path).resolve()
    direct = hashes.get(str(path))
    if direct is not None:
        return direct
    try:
        relative = path.relative_to(DATA_DIR)
    except ValueError:
        return None
    if relative.parts and relative.parts[0] in {'daily_data', 'history_data', 'spread_analysis'}:
        return hashes.get(str(PROJECT_DIR / relative))
    return None

PUBLISHED_DIR = DATA_DIR / "published"
PUBLICATION_CONFIG = PROJECT_DIR / "config/github_publish.json"
PUBLICATION_WORK = PROJECT_DIR / ".work/publication"
