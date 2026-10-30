"""Daily publication orchestration. Existing producers remain authoritative."""
from contextlib import ExitStack
import os
from pathlib import Path
import shutil
import sys
from moex_bonds.paths import PROJECT_DIR, DAILY_DIR, HISTORY_DIR, ANALYSIS_DIR
from moex_bonds.publication.bundle import read_json, save_json, build_bundle, validate_bundle
from moex_bonds.publication.git_publish import GitPublisher
from moex_bonds.analysis.spread_results import file_hash
from moex_bonds.storage.daily_storage import output_lock, checked_child

WORK = PROJECT_DIR/'.work/publication'
PUBLISHED = PROJECT_DIR/'data/published'
STATE = WORK/'state.json'

def check_args(args):
    if getattr(args,'secids','') or not getattr(args,'daily',True) or getattr(args,'csv_only',False):
        raise ValueError('Publication requires a full daily XLSX run')
    if Path(getattr(args,'output',DAILY_DIR)).resolve() != DAILY_DIR or getattr(args,'history_output',None):
        raise ValueError('Publication requires standard data paths')

def validate_ui(folder, manifest):
    from moex_bonds.ui.app import load_snapshot, load_enrichment, load_curve
    snap = load_snapshot(folder/'spread_analysis', manifest['analysis_date'])
    frame, reason = load_enrichment(snap, folder/'daily_data', publication=manifest)
    if frame.empty or reason:
        raise ValueError('Published workbook unavailable: '+reason)
    if manifest['source_statuses']['cbr_zcyc']['status'] == 'ok':
        curve, reason = load_curve(manifest['analysis_date'], folder/'daily_data')
        if curve.empty or reason:
            raise ValueError('Published CBR curve unavailable: '+reason)

def deliver(git, state):
    manifest = validate_bundle(PUBLISHED)
    if file_hash(PUBLISHED/'publication_manifest.json') != state['manifest_sha256']:
        raise ValueError('Pending bundle changed')
    local, remote = git.preflight(pending=state)
    sha = state.get('commit_sha')
    if not sha and state.get('tree'):
        if (git.git('rev-parse','HEAD^') == state['base_sha'] and
            git.git('rev-parse','HEAD^{tree}') == state['tree'] and
            git.git('show','-s','--format=%s','HEAD') == state['label']):
            sha = local
            state['commit_sha'] = sha
    if sha:
        if local != sha:
            raise RuntimeError('HEAD is not the pending commit')
        if remote != sha:
            if git.git('merge-base','--is-ancestor',sha,remote,check=False).returncode == 0:
                print('Publication delivered previously; remote has newer commits')
            elif remote == state['base_sha']:
                git.push(sha)
            else:
                raise RuntimeError('Remote diverged; manual synchronization required')
    else:
        if local != state['base_sha'] or remote != state['base_sha']:
            raise RuntimeError('Branch changed before commit')
        if not git.stage():
            state.update(stage='published',commit_sha=local)
            save_json(STATE,state)
            return
        state['tree'] = git.git('write-tree')
        save_json(STATE,state)
        sha = git.commit(state['label'])
        state.update(stage='committed',commit_sha=sha)
        save_json(STATE,state)
        git.push(sha)
    state['stage'] = 'published'
    save_json(STATE,state)
    save_json(WORK/'last_result.json',dict(status='ok',remote_sha=state['commit_sha'],
        total_size_bytes=manifest['total_size_bytes'], largest_file_bytes=max(v['size_bytes'] for v in manifest['files'].values()),
        git_objects=git.git('count-objects','-v')))

def run(args):
    stage, code = 'preflight', 20
    try:
        check_args(args)
        config = read_json(PROJECT_DIR/'config/github_publish.json')
        url = config['remote_url']
        if url != 'https://github.com/PypsenishAL/Bonds-screener.git':
            raise ValueError('Unexpected publication repository')
        git = GitPublisher(PROJECT_DIR,url,config.get('branch'),config.get('timeout_seconds',120))
        WORK.mkdir(parents=True,exist_ok=True)
        try:
            lock = (WORK/'update.lock').open('x',encoding='utf-8')
        except FileExistsError:
            print('Publication locked; verify PID in .work/publication/update.lock',file=sys.stderr)
            return 24
        with lock:
            lock.write(str(os.getpid())); lock.flush()
            try:
                state = read_json(STATE) if STATE.exists() else None
                if getattr(args,'publish_retry',False):
                    if not state:
                        raise ValueError('No pending publication')
                    git.branch = state['branch']
                    if state['stage'] == 'installing':
                        staging = checked_child(WORK,'staging')
                        backup = checked_child(WORK,'previous')
                        if staging.exists():
                            validate_bundle(staging)
                            if file_hash(staging/'publication_manifest.json') != state['manifest_sha256']:
                                raise ValueError('Installation intent mismatch')
                            if PUBLISHED.exists():
                                if backup.exists():
                                    raise ValueError('Ambiguous installation; preserve directories for inspection')
                                checked_child(PROJECT_DIR/'data','published').rename(backup)
                            staging.rename(PUBLISHED)
                        state['stage'] = 'prepared'
                        save_json(STATE,state)
                    code, stage = 23, 'retry'
                    deliver(git,state)
                    return 0
                if state and state['stage'] != 'published':
                    raise ValueError('Pending delivery exists; use --publish-retry')
                base,_ = git.preflight()
                history = read_json(HISTORY_DIR/'run_manifest.json')
                if history.get('status') != 'ok' or not history.get('trading_dates'):
                    raise ValueError('Initialize history.py first')
                code,stage = 21,'collection'
                from moex_bonds.collection.main import run as collect
                result = collect(args)
                daily = read_json(DAILY_DIR/'run_manifest.json')
                export = daily.get('daily_export',{})
                if result not in (0,4) or daily.get('collection_status') != 'ok' or export.get('status') != 'ok' or export.get('xlsx_status') != 'ok' or export.get('history_update',{}).get('status') != 'ok':
                    raise ValueError('Required daily/history stage failed')
                day = daily['trade_date']
                history = read_json(HISTORY_DIR/'run_manifest.json')
                if history.get('status') != 'ok' or history.get('synchronized_through') != day:
                    raise ValueError('History is not synchronized through daily date')
                code,stage = 22,'analysis'
                from moex_bonds.analysis.analyze_spreads import run_analysis
                run_analysis(DAILY_DIR/'daily/bonds_daily.xlsx',HISTORY_DIR/'bonds_history.csv',ANALYSIS_DIR,analysis_date=day)
                stage='bundle'
                staging = checked_child(WORK,'staging')
                if staging.exists(): shutil.rmtree(staging)
                with ExitStack() as locks:
                    for source in (DAILY_DIR,HISTORY_DIR,ANALYSIS_DIR): locks.enter_context(output_lock(source))
                    manifest = build_bundle(DAILY_DIR,ANALYSIS_DIR,staging)
                    validate_ui(staging,manifest)
                state = dict(stage='installing',branch=git.branch,remote_url=url,base_sha=base,
                    analysis_run_id=manifest['analysis_run_id'],manifest_sha256=file_hash(staging/'publication_manifest.json'),
                    label=f"Publish {day} {manifest['analysis_run_id']}")
                save_json(STATE,state)
                backup = checked_child(WORK,'previous')
                if backup.exists(): shutil.rmtree(backup)
                if PUBLISHED.exists():
                    checked_child(PROJECT_DIR/'data','published').rename(backup)
                try: staging.rename(PUBLISHED)
                except OSError:
                    if backup.exists(): backup.rename(PUBLISHED)
                    raise
                state = dict(stage='prepared',branch=git.branch,remote_url=url,base_sha=base,
                    analysis_run_id=manifest['analysis_run_id'],manifest_sha256=file_hash(PUBLISHED/'publication_manifest.json'),
                    label=f"Publish {day} {manifest['analysis_run_id']}")
                save_json(STATE,state)
                code,stage = 23,'delivery'
                deliver(git,state)
                if backup.exists(): shutil.rmtree(checked_child(WORK,'previous'))
                return 0
            finally:
                lock.close()
                (WORK/'update.lock').unlink(missing_ok=True)
    except Exception as error:
        save_json(WORK/'last_result.json',dict(status='error',stage=stage,error_type=type(error).__name__))
        print(f'Publication failed at {stage}: {error}',file=sys.stderr)
        return code
