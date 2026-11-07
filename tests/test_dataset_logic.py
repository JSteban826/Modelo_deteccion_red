import unittest
from app.graph_service import build_graph
from app.model.dataset import features,label
class DatasetTests(unittest.TestCase):
 def test_features(self): self.assertEqual(len(features(build_graph(42),'PC01')),16)
 def test_label(self): self.assertIn(label(build_graph(42),'PC01'),[0,1])
