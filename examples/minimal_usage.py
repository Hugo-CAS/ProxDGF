"""Run one generated ProxDGF optimization step without external data."""

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


def main() -> None:
    torch.manual_seed(7)
    batch, origins, lookback, stocks = 2, 4, 60, 20
    returns = torch.randn(batch, origins, lookback, stocks) * 1e-3
    minute_index = torch.tensor([59.0, 74.0, 89.0, 104.0])
    industry_labels = torch.arange(stocks) // 4

    features = make_temporal_features(returns)
    current_graphs = build_current_graph(returns, top_k=8)
    context = build_market_context(returns, minute_index, session_minutes=240)
    context = (context - context.mean(dim=(0, 1), keepdim=True)) / context.std(
        dim=(0, 1), keepdim=True
    ).clamp_min(1e-2)
    structural_prior = build_industry_prior(industry_labels)
    baseline = persistence_variance(returns)

    model = ProxDGF(ProxDGFConfig(tau=0.005))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    target_variance = baseline * torch.exp(torch.randn_like(baseline) * 0.1)

    optimizer.zero_grad(set_to_none=True)
    output = model(features, current_graphs, context, structural_prior, baseline)
    loss = variance_qlike(target_variance, output.variance)
    loss.backward()
    optimizer.step()

    print(f"loss={loss.item():.6f}")
    print("variance", tuple(output.variance.shape))
    print("fused_graphs", tuple(output.fused_graphs.shape))
    print("fusion_weights", tuple(output.fusion_weights.shape))


if __name__ == "__main__":
    main()
