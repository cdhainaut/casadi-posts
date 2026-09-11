"""Figures for the second post: what sharing the design block does to the Hessian.

Reads ``results.json`` and ``patterns.npz`` (written by ``example.py``) and writes

    figures/hessian_patterns.png   Lagrangian Hessian, shared vs per-condition design
    figures/cost.png               block density and graph size

Run:  python figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"

COLORS = {
    "design per condition": "#2e8b57",
    "design shared": "#1f4e79",
    "design frozen": "#8ab6e0",
}


def block_edges(label: str, data) -> list[int]:
    design = int(data[f"{label}|design_block"])
    condition = int(data[f"{label}|condition_block"])
    total = int(data[f"{label}|hessian_shape"][0])
    edges = [design]
    position = design
    while position < total - condition:
        position += condition
        edges.append(position)
    return edges[:-1]


def hessian_figure(data) -> None:
    labels = ["design shared", "design per condition"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, label in zip(axes, labels):
        rows = np.array(data[f"{label}|hessian_rows"])
        cols = np.array(data[f"{label}|hessian_cols"])
        shape = data[f"{label}|hessian_shape"]
        nvars = int(shape[0])
        density = len(rows) / (nvars * (nvars + 1) / 2)
        ax.plot(cols, rows, ".", color=COLORS[label], markersize=2.2)
        ax.set_xlim(-1, shape[1])
        ax.set_ylim(shape[0], -1)
        ax.set_aspect("equal")
        ax.set_title(
            f"{label}\n{nvars} x {nvars}, {len(rows)} nonzeros ({density:.0%})",
            fontsize=11,
        )
        ax.set_xlabel("variables", fontsize=9.5)
        ax.set_ylabel("variables", fontsize=9.5)
        for edge in block_edges(label, data):
            ax.axvline(edge - 0.5, color="#e08214", linewidth=0.8, alpha=0.85)
            ax.axhline(edge - 0.5, color="#e08214", linewidth=0.8, alpha=0.85)

    fig.suptitle(
        "One design block shared by every condition (left) against one design block "
        "per condition (right)",
        fontsize=12,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(FIGURES / "hessian_patterns.png", dpi=170)
    plt.close(fig)
    print("wrote figures/hessian_patterns.png")


def cost_figure(cases: list[dict], data) -> None:
    labels = [case["label"] for case in cases]
    nodes = np.array([case["nodes_hessian"] for case in cases], dtype=float)
    times = np.array([case["hessian_ms"] for case in cases], dtype=float)
    density = np.array(
        [
            case["nnz_hessian"]
            / (case["variables"] * (case["variables"] + 1) / 2)
            for case in cases
        ]
    )
    colors = [COLORS[label] for label in labels]
    y = np.arange(len(cases))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
    panels = (
        (axes[0], density, "Hessian density", "nonzeros / triangle", "{:.0%}"),
        (axes[1], nodes, "Hessian graph size", "nodes (log scale)", None),
        (axes[2], times, "one Hessian evaluation", "milliseconds (log scale)", None),
    )
    for ax, values, title, xlabel, formatter in panels:
        ax.barh(y, values, color=colors, height=0.55)
        if title != "Hessian density":
            ax.set_xscale("log")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(right=float(np.max(values)) * 1.35)
        for position, value in zip(y, values):
            text = (
                formatter.format(value)
                if formatter
                else (f"{value / 1000:.0f} k" if value >= 1000 else f"{value:.1f} ms")
            )
            ax.text(value * 1.05, position, text, va="center", fontsize=9.5)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10.5)
    fig.suptitle(
        "Same conditions, same controls: sharing the design block keeps the graph "
        "large and couples every condition",
        fontsize=12,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(FIGURES / "cost.png", dpi=170)
    plt.close(fig)
    print("wrote figures/cost.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cases = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    data = np.load(HERE / "patterns.npz")
    hessian_figure(data)
    cost_figure(cases, data)


if __name__ == "__main__":
    main()
