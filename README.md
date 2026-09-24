# ProxDGF

Official model implementation of **Proximal Dynamic Graph Fusion (ProxDGF)** for
intraday realized-volatility forecasting.

ProxDGF combines three sources of cross-stock structure at every forecast origin:

1. a current interaction graph estimated from recent market-adjusted returns;
2. the recurrent fused graph from the previous intraday origin; and
3. a structural prior, such as a binary industry graph.

Market context and graph innovation determine the three fusion weights. A
nonnegative proximal update then shrinks weak edges before a graph encoder propagates
causal temporal features for multi-horizon forecasting.

This repository contains the main model only. It intentionally excludes proprietary
data, dataset-specific preprocessing, baseline implementations, ablation runners,
hyperparameter-search results, and plotting code.

## Installation

Python 3.10 or later and PyTorch 2.2 or later are required.

```bash
git clone https://github.com/Hugo-CAS/ProxDGF.git
cd ProxDGF
python -m pip install -e .
```

## Expected tensors

| Input | Shape | Meaning |
|---|---:|---|
| `features` | `[batch, origins, stocks, channels, lookback]` | Per-stock temporal channels |
| `current_graphs` | `[batch, origins, stocks, stocks]` | Current interaction graphs |
| `context` | `[batch, origins, context_dim]` | Normalized market context |
| `structural_prior` | `[stocks, stocks]`, `[batch, stocks, stocks]`, or `[batch, origins, stocks, stocks]` | Structural prior |
| `baseline_variance` | `[batch, origins, stocks, horizons]` | Persistence forecast on the variance scale |

The default temporal channels are returns, log squared-return magnitude, and an
availability indicator. The default context contains log trailing market variance,
log cross-sectional dispersion, positive-return breadth, and sine/cosine intraday
time. Context means and scales must be fitted on the training split only.

## Minimal use

```python
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

# Recent return windows: [batch, origins, lookback, stocks].
returns = torch.randn(2, 4, 60, 20) * 1e-3
minute_index = torch.tensor([59.0, 74.0, 89.0, 104.0])
labels = torch.arange(20) // 4

features = make_temporal_features(returns)
current_graphs = build_current_graph(returns, top_k=8)
context = build_market_context(returns, minute_index, session_minutes=240)
prior = build_industry_prior(labels)
baseline = persistence_variance(returns, horizons=(15, 30, 60))

model = ProxDGF(ProxDGFConfig(tau=0.005))
output = model(features, current_graphs, context, prior, baseline)

target_variance = torch.rand_like(output.variance) + 1e-5
loss = variance_qlike(target_variance, output.variance)
loss.backward()

print(output.variance.shape)       # [2, 4, 20, 3]
print(output.fused_graphs.shape)   # [2, 4, 20, 20]
print(output.fusion_weights.shape) # [2, 4, 3]: history, prior, current
```

The complete runnable version is in [`examples/minimal_usage.py`](examples/minimal_usage.py).

## Model components

### Current interaction graph

`build_current_graph` removes an equal-weight local market component, computes
positive residual correlations, retains the largest `K` affinities per row, and
symmetrizes the result. Missing stocks with incomplete lookback windows receive no
current edges.

### Adaptive proximal graph fusion

Let `B_t`, `A_(t-1)`, and `I_d` denote the current graph, recurrent graph state, and
structural prior. ProxDGF computes graph innovation from the normalized Frobenius
distance between `B_t` and `A_(t-1)`. Two gates allocate simplex weights to the three
sources. For their convex combination `Q_t`, the constrained proximal update is

```text
A_t = relu(Q_t - tau).
```

The implementation preserves symmetry, nonnegativity, and a zero diagonal. `tau`
is a fixed model hyperparameter and should be selected using validation data.

### Volatility forecaster

A shared causal temporal convolutional network encodes each stock independently. A
degree-two Chebyshev graph filter then combines stock-specific, one-hop, and two-hop
signals. Its output adjusts a trailing realized-variance persistence forecast on the
log scale. The model returns both variance and unannualized volatility forecasts.

## Training and evaluation

Use `variance_qlike` to optimize the variance forecast. Use
`volatility_ratio_loss` when reporting the volatility ratio loss described in the
paper. Both functions accept a Boolean mask for unavailable targets.

For a new market:

1. create strictly causal return windows and future realized-variance targets;
2. fit all normalizers on the training split only;
3. select `tau` and other hyperparameters on a validation split;
4. reset the recurrent graph state at the start of each trading day; and
5. keep target masks identical across compared models.

## Tests

The test suite uses generated tensors and does not require market data.

```bash
python -m unittest discover -s tests -v
python examples/minimal_usage.py
```

The tests cover current-graph construction, structural priors, causal temporal
encoding, simplex fusion weights, graph constraints, innovation-dependent retention,
positive forecasts, and end-to-end gradients.

## Repository scope

```text
src/proxdgf/
  config.py      Model configuration
  features.py    Causal feature and persistence-baseline builders
  graphs.py      Current graph, structural prior, and context construction
  model.py       ProxDGF, graph fusion, TCN, and graph filter
  losses.py      Training and evaluation losses
examples/
  minimal_usage.py
tests/
```

## Citation

The paper citation will be added after publication. Until then, cite this repository
using [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE).
