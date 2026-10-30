"""Bounded noninteractive Git operations. Never stages source code."""
import os
from pathlib import Path
import subprocess
import shutil


def git_candidates():
    """Installed Windows Git locations when an IDE inherited an old PATH."""
    if os.name != 'nt':
        return []
    return [Path(os.environ.get('ProgramFiles', 'C:/Program Files'))/'Git/cmd/git.exe',
            Path(os.environ.get('ProgramFiles(x86)', 'C:/Program Files (x86)'))/'Git/cmd/git.exe',
            Path.home()/'AppData/Local/Programs/Git/cmd/git.exe',
            Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe']


def resolve_git():
    executable = shutil.which('git')
    if executable:
        return executable
    for candidate in git_candidates():
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError('Git executable not found. Install Git for Windows, add it to PATH and restart VS Code.')

class GitPublisher:
    def __init__(self, root, remote, branch=None, timeout=120):
        self.root = Path(root).resolve()
        self.remote, self.branch, self.timeout = remote, branch, timeout

    def git(self, *args, check=True):
        env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GCM_INTERACTIVE='never')
        result = subprocess.run([resolve_git(), '-C', str(self.root), *args], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', env=env, timeout=self.timeout)
        if check and result.returncode:
            # Do not echo arbitrary credential-bearing remote/hook output.
            raise RuntimeError('Git operation failed: '+args[0]+'; verify branch, network and saved HTTPS login')
        return result.stdout.strip() if check else result

    def preflight(self, pending=None):
        if Path(self.git('rev-parse','--show-toplevel')).resolve() != self.root:
            raise RuntimeError('Wrong Git root')
        if self.git('remote','get-url','origin') != self.remote:
            raise RuntimeError('origin does not match configured repository')
        if self.git('remote','get-url','--push','--all','origin').splitlines() != [self.remote]:
            raise RuntimeError('Push URL differs from configured repository')
        current = self.git('branch','--show-current')
        if not current:
            raise RuntimeError('Detached HEAD')
        self.branch = self.branch or current
        if current != self.branch:
            raise RuntimeError('Wrong publication branch')
        self.git('var','GIT_AUTHOR_IDENT')
        staged = self.git('diff','--cached','--name-only').splitlines()
        if staged:
            if not pending or any(not p.startswith('data/published/') for p in staged):
                raise RuntimeError('Index is not empty')
            if self.git('diff','--name-only','--','data/published'):
                raise RuntimeError('Index differs from pending bundle')
            if pending.get('tree') and self.git('write-tree') != pending['tree']:
                raise RuntimeError('Index tree differs from pending publication')
        changes = self.git('diff','--name-only').splitlines()
        if any(not p.startswith('data/published/') for p in changes):
            raise RuntimeError('Tracked code has uncommitted changes')
        self.git('fetch','origin',self.branch)
        local = self.git('rev-parse','HEAD')
        remote = self.remote_sha()
        if pending:
            return local, remote
        if local != remote:
            raise RuntimeError('Local and remote branches differ; synchronize manually')
        return local, remote

    def remote_sha(self):
        return self.git('rev-parse',f'refs/remotes/origin/{self.branch}')

    def stage(self):
        self.git('add','--all','--','data/published')
        paths = self.git('diff','--cached','--name-only').splitlines()
        if any(not p.startswith('data/published/') for p in paths):
            raise RuntimeError('Unexpected staged path')
        return bool(paths)

    def commit(self, label):
        self.git('commit','-m',label)
        return self.git('rev-parse','HEAD')

    def push(self, sha):
        if self.git('rev-parse','HEAD') != sha:
            raise RuntimeError('HEAD differs from pending publication')
        self.git('push','origin',f'HEAD:refs/heads/{self.branch}')
        rows = self.git('ls-remote','origin',f'refs/heads/{self.branch}').split()
        if not rows or rows[0] != sha:
            raise RuntimeError('Remote publication not confirmed')
