import unittest
from app.graph_service import build_graph, edge_weight, shortest_route
from app.automata import run, sequence_for_path

class CoreTests(unittest.TestCase):
    def test_graph_has_20_devices(self):
        self.assertEqual(build_graph(42).number_of_nodes(), 20)
    def test_weight_prefers_low_latency(self):
        self.assertLess(edge_weight(10,50,5), edge_weight(100,50,5))
    def test_route(self):
        g=build_graph(42); path,metrics=shortest_route(g,"PC01","SRV02")
        self.assertEqual(path[0],"PC01"); self.assertEqual(path[-1],"SRV02"); self.assertGreater(metrics["saltos"],0)
    def test_automata_accepts_valid_path(self):
        sequence=sequence_for_path(["A","B","C"]); self.assertTrue(run(sequence)["accepted"])
    def test_automata_rejects_invalid(self):
        self.assertFalse(run("ctf")["accepted"])

if __name__ == "__main__": unittest.main()
