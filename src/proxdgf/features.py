"""Causal temporal features and persistence forecasts."""

from collections.abc import Sequence

import torch
from torch import Tensor


def make_temporal_features(
    window_returns: Tensor,
    *,
    return_scale: float = 1000.0,
    return_clip: float | None = 20.0,
) -> Tensor:
    """Create return, log-squared-return, and availability channels.

    Args:
        window_returns: Tensor shaped ``[..., lookback, stocks]``. Non-finite
            entries represent unavailable returns.
        return_scale: Multiplicative scale applied before constructing channels.
        return_clip: Optional symmetric clipping level for scaled returns.

    Returns:
        Tensor shaped ``[..., stocks, 3, lookback]``.
    """

    if window_returns.ndim < 2:
        raise ValueError("window_returns must have lookback and stock dimensions")
    available = torch.isfinite(window_returns)
    scaled = torch.nan_to_num(window_returns) * return_scale
    if return_clip is not None:
        if return_clip <= 0:
            raise ValueError("return_clip must be positive or None")
        scaled = scaled.clamp(-return_clip, return_clip)
    channels = torch.stack(
        (scaled, torch.log1p(scaled.square()), available.to(scaled.dtype)), dim=-2
    )
    return channels.transpose(-1, -3).contiguous()


def persistence_variance(
    window_returns: Tensor,
    horizons: Sequence[int] = (15, 30, 60),
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Scale trailing realized variance to each forecast horizon.

    Args:
        window_returns: Tensor shaped ``[..., lookback, stocks]``.
        horizons: Forecast horizons in minutes.
        eps: Positive numerical floor.

    Returns:
        Tensor shaped ``[..., stocks, len(horizons)]`` on the variance scale.
    """

    if window_returns.ndim < 2:
        raise ValueError("window_returns must have lookback and stock dimensions")
    if not horizons or any(h <= 0 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if eps <= 0:
        raise ValueError("eps must be positive")
    lookback = window_returns.shape[-2]
    if lookback < 1:
        raise ValueError("lookback must be positive")
    trailing = torch.nan_to_num(window_returns).square().sum(dim=-2)
    scale = torch.as_tensor(horizons, dtype=trailing.dtype, device=trailing.device)
    baseline = trailing.unsqueeze(-1) * (scale / float(lookback))
    return baseline.clamp_min(eps)
