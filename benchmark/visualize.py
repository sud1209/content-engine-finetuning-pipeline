"""
Benchmark card visualization for tweet-scorer fine-tuning evaluation.

Generates a four-chart Plotly figure comparing finetuned vs base model
against Haiku ground-truth scores.
"""

import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dataset.schemas import DIMENSIONS


def generate_benchmark_card(metrics: dict, output_path: str) -> None:
    """
    Render a four-panel benchmark card and save as both .html and .png.

    Parameters
    ----------
    metrics : dict
        Output of benchmark.metrics.compute_metrics — nested dict with keys
        "finetuned" and "base", each containing per-dimension dicts and an
        "aggregate" sub-dict.
    output_path : str
        Path prefix (without extension). Both ``<output_path>.html`` and
        ``<output_path>.png`` will be written.
    """
    ft = metrics["finetuned"]
    base = metrics["base"]

    # ------------------------------------------------------------------ #
    # Chart 1 — Radar: MAE per dimension (lower = better)
    # ------------------------------------------------------------------ #
    ft_mae = [ft[d]["mae"] for d in DIMENSIONS]
    base_mae = [base[d]["mae"] for d in DIMENSIONS]
    dim_labels = DIMENSIONS + [DIMENSIONS[0]]  # close the polygon
    ft_mae_closed = ft_mae + [ft_mae[0]]
    base_mae_closed = base_mae + [base_mae[0]]

    radar_ft = go.Scatterpolar(
        r=ft_mae_closed,
        theta=dim_labels,
        fill="toself",
        name="Finetuned",
        line=dict(color="#1f77b4"),
        fillcolor="rgba(31, 119, 180, 0.15)",
    )
    radar_base = go.Scatterpolar(
        r=base_mae_closed,
        theta=dim_labels,
        fill="toself",
        name="Base",
        line=dict(color="#d62728"),
        fillcolor="rgba(214, 39, 40, 0.15)",
    )

    # ------------------------------------------------------------------ #
    # Chart 2 — Bar: Within-1 agreement per dimension
    # ------------------------------------------------------------------ #
    ft_w1 = [ft[d]["within_1_agreement"] for d in DIMENSIONS]
    base_w1 = [base[d]["within_1_agreement"] for d in DIMENSIONS]

    bar_ft_w1 = go.Bar(
        x=DIMENSIONS,
        y=ft_w1,
        name="Finetuned",
        marker_color="#1f77b4",
        showlegend=False,
    )
    bar_base_w1 = go.Bar(
        x=DIMENSIONS,
        y=base_w1,
        name="Base",
        marker_color="#d62728",
        showlegend=False,
    )

    # ------------------------------------------------------------------ #
    # Chart 3 — Bar: Aggregate metrics
    # ------------------------------------------------------------------ #
    agg_labels = ["Composite MAE", "Tier Accuracy", "Never-list F1", "Mean Latency (ms)"]
    agg_keys = ["composite_mae", "tier_accuracy", "never_list_f1", "mean_latency_ms"]

    ft_agg = [ft["aggregate"][k] for k in agg_keys]
    base_agg = [base["aggregate"][k] for k in agg_keys]

    bar_ft_agg = go.Bar(
        x=agg_labels,
        y=ft_agg,
        name="Finetuned",
        marker_color="#1f77b4",
        showlegend=False,
    )
    bar_base_agg = go.Bar(
        x=agg_labels,
        y=base_agg,
        name="Base",
        marker_color="#d62728",
        showlegend=False,
    )

    # ------------------------------------------------------------------ #
    # Chart 4 — Table: Cost comparison
    # ------------------------------------------------------------------ #
    cost_table = go.Table(
        header=dict(
            values=["<b>Option</b>", "<b>Cost / 1K calls</b>", "<b>Approx latency</b>"],
            fill_color="#2c3e50",
            font=dict(color="white", size=12),
            align="left",
        ),
        cells=dict(
            values=[
                ["Haiku API (claude-haiku-3)", "Self-hosted (A10G / vLLM)"],
                ["$0.50", "~$0.00"],
                ["~800 ms", "~400 ms"],
            ],
            fill_color=[["#ecf0f1", "#dfe6e9"]],
            align="left",
            font=dict(size=12),
        ),
    )

    # ------------------------------------------------------------------ #
    # Assemble figure
    # ------------------------------------------------------------------ #
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "MAE per Dimension (lower = better)",
            "Within-1 Agreement per Dimension",
            "Aggregate Metrics",
            "Cost Comparison",
        ),
        specs=[
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # Row 1, Col 1 — Radar
    fig.add_trace(radar_ft, row=1, col=1)
    fig.add_trace(radar_base, row=1, col=1)

    # Row 1, Col 2 — Within-1 bar
    fig.add_trace(bar_ft_w1, row=1, col=2)
    fig.add_trace(bar_base_w1, row=1, col=2)

    # Row 2, Col 1 — Aggregate bar
    fig.add_trace(bar_ft_agg, row=2, col=1)
    fig.add_trace(bar_base_agg, row=2, col=1)

    # Row 2, Col 2 — Cost table
    fig.add_trace(cost_table, row=2, col=2)

    # ------------------------------------------------------------------ #
    # Layout polish
    # ------------------------------------------------------------------ #
    fig.update_layout(
        title=dict(
            text="Tweet-Scorer Fine-Tune Benchmark Card",
            font=dict(size=20),
            x=0.5,
            xanchor="center",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        barmode="group",
        height=900,
        width=1400,
        template="plotly_white",
        margin=dict(t=100, b=60, l=60, r=60),
    )

    # Polar axis: invert so lower MAE is visually "better" (closer to center)
    fig.update_polars(radialaxis=dict(autorange="reversed"))

    # Within-1 bar y-axis: 0–1 scale
    fig.update_yaxes(range=[0, 1], row=1, col=2, title_text="Agreement Rate")
    fig.update_xaxes(tickangle=-30, row=1, col=2)

    # Aggregate bar axis labels
    fig.update_xaxes(tickangle=-20, row=2, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)

    # ------------------------------------------------------------------ #
    # Save outputs
    # ------------------------------------------------------------------ #
    html_path = output_path + ".html"
    png_path = output_path + ".png"

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    fig.write_html(html_path)
    fig.write_image(png_path)  # requires kaleido: pip install kaleido
