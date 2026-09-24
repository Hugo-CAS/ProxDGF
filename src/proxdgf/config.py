"""Configuration for the ProxDGF model."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ProxDGFConfig:
    """Architecture and graph-fusion settings.

    ``tau`` controls proximal sparsification and is intentionally fixed during
    fitting. Select it on validation data and keep it fixed for evaluation.
    """

    input_channels: int = 3
    hidden_dim: int = 16
    context_dim: int = 5
    horizons: tuple[int, ...] = (15, 30, 60)
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16)
    tau: float = 0.005
    correction_clip: float = 8.0
    retention_bias: float = 2.2
    source_bias: float = -1.4
    innovation_slope_raw: float = -1.0
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.input_channels < 1:
            raise ValueError("input_channels must be positive")
        if self.hidden_dim < 1 or self.context_dim < 1:
            raise ValueError("hidden_dim and context_dim must be positive")
        if not self.horizons or any(h <= 0 for h in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not self.dilations or any(d <= 0 for d in self.dilations):
            raise ValueError("dilations must contain positive integers")
        if self.tau < 0:
            raise ValueError("tau must be nonnegative")
        if self.correction_clip <= 0 or self.eps <= 0:
            raise ValueError("correction_clip and eps must be positive")
