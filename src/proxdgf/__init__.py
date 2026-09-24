"""Public API for Proximal Dynamic Graph Fusion."""

from .config import ProxDGFConfig
from .features import make_temporal_features, persistence_variance
from .graphs import build_current_graph, build_industry_prior, build_market_context
from .losses import variance_qlike, volatility_ratio_loss
from .model import ProxDGF, ProxDGFOutput

__all__ = [
    "ProxDGF",
    "ProxDGFConfig",
    "ProxDGFOutput",
    "build_current_graph",
    "build_industry_prior",
    "build_market_context",
    "make_temporal_features",
    "persistence_variance",
    "variance_qlike",
    "volatility_ratio_loss",
]

__version__ = "0.1.0"
