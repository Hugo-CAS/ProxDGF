import unittest

import torch

from proxdgf import build_current_graph, build_industry_prior, build_market_context


class GraphConstructionTests(unittest.TestCase):
    def test_current_graph_constraints_and_missing_stock(self) -> None:
        torch.manual_seed(1)
        returns = torch.randn(2, 60, 8)
        returns[0, 4, 3] = float("nan")
        graph = build_current_graph(returns, top_k=3)
        self.assertEqual(graph.shape, (2, 8, 8))
        self.assertTrue(torch.allclose(graph, graph.transpose(-1, -2)))
        self.assertGreaterEqual(float(graph.min()), 0.0)
        self.assertEqual(float(graph.diagonal(dim1=-2, dim2=-1).abs().max()), 0.0)
        self.assertEqual(float(graph[0, 3].abs().max()), 0.0)

    def test_binary_industry_prior(self) -> None:
        labels = torch.tensor([0, 0, 1, -1])
        prior = build_industry_prior(labels)
        expected = torch.tensor(
            [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=torch.float32,
        )
        self.assertTrue(torch.equal(prior, expected))

    def test_context_shape_and_finiteness(self) -> None:
        returns = torch.randn(3, 5, 60, 9) * 1e-3
        minute = torch.arange(5) * 15 + 59
        context = build_market_context(returns, minute, session_minutes=240)
        self.assertEqual(context.shape, (3, 5, 5))
        self.assertTrue(torch.isfinite(context).all())


if __name__ == "__main__":
    unittest.main()
