import unittest, tempfile, json
from pathlib import Path
class BundleTests(unittest.TestCase):
 def test_tamper_and_traversal(self):
  from moex_bonds.publication.bundle import validate_bundle
  with tempfile.TemporaryDirectory() as d:
   p=Path(d); (p/'publication_manifest.json').write_text(json.dumps({'publication_schema_version':1,'status':'ok','files':{'../escape':{'sha256':'x','size_bytes':0}}}))
   with self.assertRaises(ValueError): validate_bundle(p)
 def test_mode(self):
  from moex_bonds.publication.bundle import data_mode
  self.assertEqual(data_mode({'BONDS_DATA_MODE':'published'}), 'published')
  with self.assertRaises(ValueError): data_mode({'BONDS_DATA_MODE':'oops'})
if __name__=='__main__': unittest.main()

class IntegrityTests(unittest.TestCase):
 def fixture(self, root):
  from moex_bonds.publication.bundle import allowed,save_json
  from moex_bonds.analysis.spread_results import file_hash
  day='2026-10-27'; status=dict(status='unavailable',trade_date=day,error='missing')
  files={}
  for name in allowed(day,False):
   p=root/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'fixture')
  book='daily_data/daily/bonds_daily.xlsx'; digest=file_hash(root/book)
  marker=dict(status='ok',analysis_date=day,analysis_run_id='test',source_sha256={'C:\\old\\daily.xlsx':digest})
  for n in ('spread_analysis/run_manifest.json',f'spread_analysis/snapshots/{day}/run_manifest.json'): save_json(root/n,marker)
  save_json(root/'daily_data/run_manifest.json',dict(collection_status='ok',trade_date=day,cbr_zcyc=status))
  save_json(root/'daily_data/_support/cbr_zcyc_status.json',status)
  for name in allowed(day,False): files[name]=dict(sha256=file_hash(root/name),size_bytes=(root/name).stat().st_size)
  m=dict(publication_schema_version=1,status='ok',analysis_date=day,analysis_run_id='test',files=files,source_statuses={'cbr_zcyc':status},source_bindings={'daily_workbook':dict(path=book,source_key='C:\\old\\daily.xlsx',sha256=digest)},total_size_bytes=sum(v['size_bytes'] for v in files.values()))
  save_json(root/'publication_manifest.json',m)
 def test_integrity_and_relocation(self):
  import shutil
  from moex_bonds.publication.bundle import validate_bundle
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a'; a.mkdir(); self.fixture(a); validate_bundle(a)
   b=Path(d)/'b'; shutil.copytree(a,b); validate_bundle(b)
   (b/'daily_data/daily/bonds_daily.xlsx').write_bytes(b'tamper')
   with self.assertRaises(ValueError): validate_bundle(b)
 def test_extra_file(self):
  from moex_bonds.publication.bundle import validate_bundle
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); self.fixture(root); (root/'secret').write_text('x')
   with self.assertRaises(ValueError): validate_bundle(root)
 def test_source_binding(self):
  from moex_bonds.publication.bundle import validate_bundle,read_json,save_json
  with tempfile.TemporaryDirectory() as d:
   root=Path(d); self.fixture(root); m=read_json(root/'publication_manifest.json'); m['source_bindings']['daily_workbook']['source_key']='daily.xlsx'; save_json(root/'publication_manifest.json',m)
   with self.assertRaises(ValueError): validate_bundle(root)
