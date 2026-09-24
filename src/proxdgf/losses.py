"""Losses used to train and evaluate ProxDGF."""

import torch
from torch import Tensor


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    if mask.shape != values.shape:
        mask = torch.broadcast_to(mask, values.shape)
    selected = values.masked_select(mask.bool())
    if selected.numel() == 0:
        raise ValueError("mask selects no observations")
    return selected.mean()


def variance_qlike(
    target_variance: Tensor,
    forecast_variance: Tensor,
    mask: Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Mean QLIKE loss on the realized-variance scale."""

    target = target_variance.clamp_min(eps)
    forecast = forecast_variance.clamp_min(eps)
    ratio = target / forecast
    return _masked_mean(ratio - torch.log(ratio) - 1.0, mask)


def volatility_ratio_loss(
    target_volatility: Tensor,
    forecast_volatility: Tensor,
    mask: Tensor | None = None,
    *,
    eps: float = 1e-8,
) -> Tensor:
    """Mean volatility ratio loss used for reported VRL scores."""

    target = target_volatility.clamp_min(eps)
    forecast = forecast_volatility.clamp_min(eps)
    ratio = target / forecast
    return _masked_mean(ratio - torch.log(ratio) - 1.0, mask)
