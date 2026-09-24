"""Construction of current graphs, structural priors, and market context."""

import math

import torch
from torch import Tensor


def _zero_diagonal(matrix: Tensor) -> Tensor:
    n = matrix.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=matrix.device)
    return matrix.masked_fill(eye, 0.0)


def build_current_graph(
    window_returns: Tensor,
    *,
    top_k: int = 20,
    eps: float = 1e-12,
) -> Tensor:
    """Build a symmetric current graph from positive residual correlations.

    The input shape is ``[..., lookback, stocks]``. A stock participates only
    when its complete lookback window is finite. The function removes an
    equal-weight local market component, retains each row's largest affinities,
    and averages the directed top-k graph with its transpose.
    """

    if window_returns.ndim < 2:
        raise ValueError("window_returns must have lookback and stock dimensions")
    n = window_returns.shape[-1]
    if n < 2:
        raise ValueError("at least two stocks are required")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if eps <= 0:
        raise ValueError("eps must be positive")

    valid = torch.isfinite(window_returns).all(dim=-2)
    x = torch.nan_to_num(window_returns)
    x = x - x.mean(dim=-2, keepdim=True)

    valid_f = valid.to(x.dtype)
    count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    market = (x * valid_f.unsqueeze(-2)).sum(dim=-1) / count
    denominator = market.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    beta = (x * market.unsqueeze(-1)).sum(dim=-2) / denominator
    residual = x - market.unsqueeze(-1) * beta.unsqueeze(-2)
    residual = residual / residual.square().sum(dim=-2, keepdim=True).clamp_min(eps).sqrt()

    affinity = torch.einsum("...li,...lj->...ij", residual, residual).clamp_min(0.0)
    pair_valid = valid.unsqueeze(-1) & valid.unsqueeze(-2)
    affinity = _zero_diagonal(affinity.masked_fill(~pair_valid, 0.0))

    k = min(top_k, n - 1)
    values, indices = affinity.topk(k, dim=-1)
    directed = torch.zeros_like(affinity).scatter(-1, indices, values)
    graph = 0.5 * (directed + directed.transpose(-1, -2))
    return _zero_diagonal(graph)


def build_industry_prior(
    labels: Tensor,
    *,
    unknown_value: int | float = -1,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Build a binary same-industry structural prior.

    ``labels`` may have shape ``[stocks]`` or ``[..., stocks]``. Entries equal
    to ``unknown_value`` receive no prior edges.
    """

    if labels.ndim < 1:
        raise ValueError("labels must contain a stock dimension")
    known = labels != unknown_value
    same = labels.unsqueeze(-1) == labels.unsqueeze(-2)
    prior = same & known.unsqueeze(-1) & known.unsqueeze(-2)
    n = labels.shape[-1]
    eye = torch.eye(n, dtype=torch.bool, device=labels.device)
    return prior.masked_fill(eye, False).to(dtype=dtype)


def build_market_context(
    window_returns: Tensor,
    minute_index: Tensor | float,
    *,
    session_minutes: int,
    eps: float = 1e-8,
) -> Tensor:
    """Construct the five market-context features used by ProxDGF.

    The returned features are unnormalized. Fit their means and scales on the
    training split and apply that transformation before calling the model.
    """

    if window_returns.ndim < 2:
        raise ValueError("window_returns must have lookback and stock dimensions")
    if session_minutes <= 0 or eps <= 0:
        raise ValueError("session_minutes and eps must be positive")

    available = torch.isfinite(window_returns)
    x = torch.nan_to_num(window_returns)
    counts = available.sum(dim=-1).clamp_min(1)
    market_return = x.sum(dim=-1) / counts
    market_variance = market_return.square().sum(dim=-1)

    cross_mean = x.sum(dim=-1) / counts
    centered = (x - cross_mean.unsqueeze(-1)) * available
    cross_variance = centered.square().sum(dim=-1) / counts
    dispersion = cross_variance.sqrt().mean(dim=-1)

    stock_available = available.any(dim=-2)
    positive = x.sum(dim=-2) > 0
    breadth = (positive & stock_available).sum(dim=-1) / stock_available.sum(
        dim=-1
    ).clamp_min(1)

    minute = torch.as_tensor(minute_index, dtype=x.dtype, device=x.device)
    target_shape = market_variance.shape
    minute = torch.broadcast_to(minute, target_shape)
    angle = 2.0 * math.pi * minute / float(session_minutes)
    return torch.stack(
        (
            torch.log(market_variance + eps),
            torch.log(dispersion + eps),
            breadth.to(x.dtype),
            torch.sin(angle),
            torch.cos(angle),
        ),
        dim=-1,
    )
