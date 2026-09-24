import unittest

import torch

from proxdgf import (
    ProxDGF,
    ProxDGFConfig,
    build_current_graph,
    build_industry_prior,
    build_market_context,
    make_temporal_features,
    persistence_variance,
    variance_qlike,
)
from proxdgf.model import CausalTCN


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(9)
        self.returns = torch.randn(2, 4, 60, 10) * 1e-3
        self.features = make_temporal_features(self.returns)
        self.current = build_current_graph(self.returns, top_k=4)
        self.context = build_market_context(
            self.returns, torch.tensor([59.0, 74.0, 89.0, 104.0]), session_minutes=240
        )
        self.prior = build_industry_prior(torch.arange(10) // 2)
        self.baseline = persistence_variance(self.returns)

    def test_forward_constraints_and_gradients(self) -> None:
        model = ProxDGF(ProxDGFConfig(tau=0.005))
        output = model(
            self.features, self.current, self.context, self.prior, self.baseline
        )
        self.assertEqual(output.variance.shape, (2, 4, 10, 3))
        self.assertTrue((output.variance > 0).all())
        self.assertTrue(
            torch.allclose(
                output.fusion_weights.sum(dim=-1),
                torch.ones_like(output.innovations),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(output.fused_graphs, output.fused_graphs.transpose(-1, -2))
        )
        self.assertGreaterEqual(float(output.fused_graphs.min()), 0.0)
        self.assertEqual(
            float(output.fused_graphs.diagonal(dim1=-2, dim2=-1).abs().max()), 0.0
        )

        target = self.baseline * 1.1
        loss = variance_qlike(target, output.variance)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.isfinite(g).all() for g in gradients))

    def test_retention_decreases_with_innovation_at_fixed_context(self) -> None:
        model = ProxDGF()
        fusion = model.graph_fusion
        context = torch.zeros(1, 2, 5)
        base = torch.ones(1, 2, 5, 5) - torch.eye(5).view(1, 1, 5, 5)
        changed = base.clone()
        changed[:, 1] = 0.0
        prior = torch.zeros(5, 5)
        output = fusion(changed, context, prior)
        self.assertGreater(float(output.innovations[0, 1]), 0.0)
        history_weights = output.weights[0, :, 0]
        self.assertLess(float(history_weights[1]), float(history_weights[0]))

    def test_tau_removes_additional_edges(self) -> None:
        model = ProxDGF(ProxDGFConfig(tau=0.0))
        no_threshold = model.graph_fusion(self.current, self.context, self.prior).graphs
        model.graph_fusion.set_tau(0.05)
        thresholded = model.graph_fusion(self.current, self.context, self.prior).graphs
        self.assertLessEqual(
            int((thresholded > 0).sum()), int((no_threshold > 0).sum())
        )

    def test_temporal_encoder_is_causal(self) -> None:
        encoder = CausalTCN(3, 16, 3, (1, 2, 4))
        prefix = torch.randn(2, 3, 20)
        suffix = torch.randn(2, 3, 7)
        prefix_output = encoder.forward_sequence(prefix)
        full_output = encoder.forward_sequence(torch.cat((prefix, suffix), dim=-1))
        self.assertTrue(torch.allclose(prefix_output, full_output[..., :20], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
