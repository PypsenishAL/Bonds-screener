import unittest
from argparse import Namespace
class WorkflowTests(unittest.TestCase):
 def test_reject_subset_before_collection(self):
  from moex_bonds.publication.workflow import check_args
  with self.assertRaises(ValueError): check_args(Namespace(secids='X', daily=True, csv_only=False))
