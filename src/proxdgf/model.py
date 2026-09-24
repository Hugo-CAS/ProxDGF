"""Proximal Dynamic Graph Fusion and its volatility forecaster."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import ProxDGFConfig


@dataclass
class GraphFusionOutput:
    graphs: Tensor
    weights: Tensor
    innovations: Tensor


@dataclass
class ProxDGFOutput:
    """Predictions and graph diagnostics returned by :class:`ProxDGF`."""

    variance: Tensor
    volatility: Tensor
    log_correction: Tensor
    fused_graphs: Tensor
    fusion_weights: Tensor
    innovations: Tensor


class CausalTCN(nn.Module):
    """Shared causal temporal encoder with left-padded dilated convolutions."""

    def __init__(
        self,
        input_channels: int,
        hidden_dim: int,
        kernel_size: int,
        dilations: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers = []
        for index, dilation in enumerate(dilations):
            in_channels = input_channels if index == 0 else hidden_dim
            layers.append(
                nn.Conv1d(
                    in_channels,
                    hidden_dim,
                    kernel_size,
                    dilation=dilation,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.kernel_size = kernel_size

    def forward_sequence(self, inputs: Tensor) -> Tensor:
        hidden = inputs
        for layer in self.layers:
            padding = (self.kernel_size - 1) * layer.dilation[0]
            update = F.silu(layer(F.pad(hidden, (padding, 0))))
            hidden = update + hidden if update.shape == hidden.shape else update
        return hidden

    def forward(self, inputs: Tensor) -> Tensor:
        return self.forward_sequence(inputs)[..., -1]


class DynamicGraphFusion(nn.Module):
    """Fuse graph history, a structural prior, and current interactions."""

    WEIGHT_ORDER = ("history", "prior", "current")

    def __init__(self, config: ProxDGFConfig) -> None:
        super().__init__()
        self.retention_gate = nn.Linear(config.context_dim, 1)
        self.source_gate = nn.Linear(config.context_dim, 1)
        self.innovation_slope = nn.Parameter(torch.tensor(config.innovation_slope_raw))
        self.register_buffer("tau", torch.tensor(float(config.tau)))
        self.eps = config.eps
        nn.init.zeros_(self.retention_gate.weight)
        nn.init.zeros_(self.source_gate.weight)
        nn.init.constant_(self.retention_gate.bias, config.retention_bias)
        nn.init.constant_(self.source_gate.bias, config.source_bias)

    def set_tau(self, tau: float) -> None:
        """Set a validation-selected nonnegative proximal threshold."""

        if tau < 0:
            raise ValueError("tau must be nonnegative")
        self.tau.fill_(tau)

    @staticmethod
    def _constrain_graph(graph: Tensor) -> Tensor:
        graph = 0.5 * (graph + graph.transpose(-1, -2))
        graph = graph.clamp_min(0.0)
        n = graph.shape[-1]
        eye = torch.eye(n, dtype=torch.bool, device=graph.device)
        return graph.masked_fill(eye, 0.0)

    @staticmethod
    def _expand_prior(prior: Tensor, batch: int, origins: int, stocks: int) -> Tensor:
        if prior.shape[-2:] != (stocks, stocks):
            raise ValueError("structural_prior has incompatible stock dimensions")
        if prior.ndim == 2:
            return prior.view(1, 1, stocks, stocks).expand(batch, origins, -1, -1)
        if prior.ndim == 3:
            if prior.shape[0] not in (1, batch):
                raise ValueError("structural_prior batch dimension is incompatible")
            return prior.unsqueeze(1).expand(batch, origins, -1, -1)
        if prior.ndim == 4:
            if prior.shape[0] not in (1, batch) or prior.shape[1] not in (1, origins):
                raise ValueError("structural_prior batch or origin dimension is incompatible")
            return prior.expand(batch, origins, -1, -1)
        raise ValueError("structural_prior must have 2, 3, or 4 dimensions")

    def forward(self, current_graphs: Tensor, context: Tensor, structural_prior: Tensor) -> GraphFusionOutput:
        if current_graphs.ndim != 4 or current_graphs.shape[-1] != current_graphs.shape[-2]:
            raise ValueError("current_graphs must have shape [batch, origins, stocks, stocks]")
        batch, origins, stocks, _ = current_graphs.shape
        if context.shape[:2] != (batch, origins):
            raise ValueError("context must share batch and origin dimensions")

        current_graphs = self._constrain_graph(current_graphs)
        prior = self._constrain_graph(
            self._expand_prior(structural_prior, batch, origins, stocks)
        )
        previous = current_graphs[:, 0]
        graphs = []
        weights = []
        innovations = []

        for origin in range(origins):
            current = current_graphs[:, origin]
            innovation = torch.linalg.vector_norm(
                (current - previous).flatten(1), dim=1
            ) / torch.linalg.vector_norm(previous.flatten(1), dim=1).clamp_min(self.eps)
            retention_logit = self.retention_gate(context[:, origin]).squeeze(-1)
            source_logit = self.source_gate(context[:, origin]).squeeze(-1)
            retention = torch.sigmoid(
                retention_logit - F.softplus(self.innovation_slope) * innovation
            )
            source = torch.sigmoid(source_logit)

            history_weight = retention
            prior_weight = (1.0 - retention) * source
            current_weight = (1.0 - retention) * (1.0 - source)
            origin_weights = torch.stack(
                (history_weight, prior_weight, current_weight), dim=-1
            )
            fused = (
                history_weight[:, None, None] * previous
                + prior_weight[:, None, None] * prior[:, origin]
                + current_weight[:, None, None] * current
            )
            updated = self._constrain_graph(F.relu(fused - self.tau))

            graphs.append(updated)
            weights.append(origin_weights)
            innovations.append(innovation)
            previous = updated

        return GraphFusionOutput(
            graphs=torch.stack(graphs, dim=1),
            weights=torch.stack(weights, dim=1),
            innovations=torch.stack(innovations, dim=1),
        )


class ChebyshevGraphFilter(nn.Module):
    """Degree-two graph filter for local, one-hop, and two-hop signals."""

    def __init__(self, hidden_dim: int, horizons: int, eps: float) -> None:
        super().__init__()
        self.local = nn.Linear(hidden_dim, horizons)
        self.one_hop = nn.Linear(hidden_dim, horizons, bias=False)
        self.two_hop = nn.Linear(hidden_dim, horizons, bias=False)
        self.eps = eps
        nn.init.zeros_(self.local.weight)
        nn.init.zeros_(self.local.bias)
        nn.init.zeros_(self.one_hop.weight)
        nn.init.zeros_(self.two_hop.weight)

    def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
        degree = adjacency.sum(dim=-1)
        inverse_sqrt = torch.where(
            degree > 0,
            degree.clamp_min(self.eps).rsqrt(),
            torch.zeros_like(degree),
        )
        normalized = inverse_sqrt.unsqueeze(-1) * adjacency * inverse_sqrt.unsqueeze(-2)
        one_hop = torch.matmul(normalized, features)
        two_hop = 2.0 * torch.matmul(normalized, one_hop) - features
        return self.local(features) - self.one_hop(one_hop) + self.two_hop(two_hop)


class ProxDGF(nn.Module):
    """End-to-end Proximal Dynamic Graph Fusion forecaster."""

    def __init__(self, config: ProxDGFConfig | None = None) -> None:
        super().__init__()
        self.config = config or ProxDGFConfig()
        self.temporal_encoder = CausalTCN(
            input_channels=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            kernel_size=self.config.kernel_size,
            dilations=self.config.dilations,
        )
        self.graph_fusion = DynamicGraphFusion(self.config)
        self.graph_filter = ChebyshevGraphFilter(
            self.config.hidden_dim, len(self.config.horizons), self.config.eps
        )

    def forward(
        self,
        features: Tensor,
        current_graphs: Tensor,
        context: Tensor,
        structural_prior: Tensor,
        baseline_variance: Tensor,
    ) -> ProxDGFOutput:
        """Forecast multi-horizon realized variance and volatility."""

        if features.ndim != 5:
            raise ValueError(
                "features must have shape [batch, origins, stocks, channels, lookback]"
            )
        batch, origins, stocks, channels, lookback = features.shape
        if channels != self.config.input_channels:
            raise ValueError("features have an unexpected channel count")
        expected_baseline = (batch, origins, stocks, len(self.config.horizons))
        if baseline_variance.shape != expected_baseline:
            raise ValueError(f"baseline_variance must have shape {expected_baseline}")

        encoded = self.temporal_encoder(
            features.reshape(batch * origins * stocks, channels, lookback)
        ).reshape(batch, origins, stocks, self.config.hidden_dim)
        fusion = self.graph_fusion(current_graphs, context, structural_prior)
        correction = self.graph_filter(encoded, fusion.graphs)
        clipped = correction.clamp(
            -self.config.correction_clip, self.config.correction_clip
        )
        variance = baseline_variance.clamp_min(self.config.eps) * torch.exp(clipped)
        return ProxDGFOutput(
            variance=variance,
            volatility=variance.sqrt(),
            log_correction=clipped,
            fused_graphs=fusion.graphs,
            fusion_weights=fusion.weights,
            innovations=fusion.innovations,
        )
