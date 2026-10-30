"""Temporary Git remotes; no network or production data."""
import tempfile, subprocess, unittest
from pathlib import Path
from moex_bonds.publication.git_publish import GitPublisher
class GitTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name)/'work'; self.remote=Path(self.temp.name)/'remote.git'
  self.call('init','--bare',str(self.remote),cwd=Path(self.temp.name))
  self.call('init','-b','main',str(self.root),cwd=Path(self.temp.name))
  self.call('config','user.email','test@example.invalid'); self.call('config','user.name','Test')
  (self.root/'app.py').write_text('# fixture')
  self.call('add','app.py'); self.call('commit','-m','initial')
  self.call('remote','add','origin',str(self.remote)); self.call('push','-u','origin','main')
  self.g=GitPublisher(self.root,str(self.remote),'main')
 def call(self,*args,cwd=None):
  return subprocess.run(['git','-C',str(cwd or self.root),*args],check=True,capture_output=True,text=True).stdout.strip()
 def test_remote_guard(self):
  with self.assertRaises(RuntimeError): GitPublisher(self.root,'https://wrong.invalid').preflight()
 def test_dirty_code(self):
  (self.root/'app.py').write_text('changed')
  with self.assertRaises(RuntimeError): self.g.preflight()
 def test_foreign_index(self):
  (self.root/'extra').write_text('x'); self.call('add','extra')
  with self.assertRaises(RuntimeError): self.g.preflight()
 def test_delivery_and_no_change(self):
  self.g.preflight(); p=self.root/'data/published'; p.mkdir(parents=True); (p/'test').write_text('x')
  self.assertTrue(self.g.stage()); sha=self.g.commit('publication'); self.g.push(sha)
  self.assertEqual(self.g.preflight(),(sha,sha)); self.assertFalse(self.g.stage())
 def test_unpublished_commit(self):
  (self.root/'app.py').write_text('new'); self.call('add','app.py'); self.call('commit','-m','other')
  with self.assertRaises(RuntimeError): self.g.preflight()

 def test_pushurl_guard(self):
  self.call('config','remote.origin.pushurl','https://wrong.invalid/repo.git')
  with self.assertRaises(RuntimeError): self.g.preflight()
 def test_pending_index_recovery(self):
  base,_=self.g.preflight(); p=self.root/'data/published'; p.mkdir(parents=True); (p/'test').write_text('x')
  self.g.stage(); tree=self.g.git('write-tree')
  self.assertEqual(self.g.preflight(pending={'tree':tree}),(base,base))
  (p/'test').write_text('changed')
  with self.assertRaises(RuntimeError): self.g.preflight(pending={'tree':tree})

class GitDiscoveryTests(unittest.TestCase):
 def test_resolves_existing_fallback_without_path(self):
  from unittest.mock import patch
  from moex_bonds.publication.git_publish import resolve_git
  with tempfile.TemporaryDirectory() as d:
   executable=Path(d)/'git.exe'; executable.write_bytes(b'fixture')
   with patch('moex_bonds.publication.git_publish.shutil.which',return_value=None), patch('moex_bonds.publication.git_publish.git_candidates',return_value=[executable]):
    self.assertEqual(resolve_git(),str(executable))
 def test_missing_git_actionable(self):
  from unittest.mock import patch
  from moex_bonds.publication.git_publish import resolve_git
  with patch('moex_bonds.publication.git_publish.shutil.which',return_value=None), patch('moex_bonds.publication.git_publish.git_candidates',return_value=[]):
   with self.assertRaisesRegex(RuntimeError,'Git.*PATH'): resolve_git()
