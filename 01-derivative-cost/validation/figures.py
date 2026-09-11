"""Figures for the first post: derivative cost and sparsity patterns.

Reads ``results.json`` and ``patterns.npz`` (written by ``example.py``) and writes

    figures/cost.png       graph size and evaluation time, five writings
    figures/patterns.png   closure Jacobian and Lagrangian Hessian

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
SAND = "SAND (closure constraints)"

COLORS = {
    "elimination": "#1f4e79",
    "elimination, A design-frozen": "#8ab6e0",
    "elimination, g linear": "#2e8b57",
    "SAND (closure constraints)": "#e08214",
    "rootfinder (implicit)": "#c0392b",
}


def cost_figure(cases: list[dict]) -> None:
    labels = [case["label"] for case in cases]
    nodes = np.array([case["nodes_hessian"] for case in cases], dtype=float)
    times = np.array([case["hessian_ms"] for case in cases], dtype=float)
    colors = [COLORS[label] for label in labels]
    y = np.arange(len(cases))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, values, title, xlabel in (
        (axes[0], nodes, "Hessian graph size", "nodes in the graph (log scale)"),
        (axes[1], times, "one Hessian evaluation", "milliseconds (log scale)"),
    ):
        ax.barh(y, values, color=colors, height=0.6)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_xlabel(xlabel, fontsize=9.5)
        ax.grid(axis="x", alpha=0.22, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(right=float(np.max(values)) * 9.0)
        for position, value in zip(y, values):
            text = (
                f"{value / 1000:.0f} k"
                if title.startswith("Hessian graph")
                else f"{value:.1f} ms"
            )
            ax.text(value * 1.3, position, text, va="center", fontsize=9.5)

    notes_nodes = ["reference", f"\u00f7{nodes[0] / nodes[1]:.0f}",
                   f"\u00f7{nodes[0] / nodes[2]:.1f}", "same", "same"]
    notes_times = ["reference", f"\u00f7{times[0] / times[1]:.0f}",
                   f"\u00f7{times[0] / times[2]:.1f}", "same",
                   f"\u00d7{times[4] / times[0]:.1f}"]
    for ax, values, notes in ((axes[0], nodes, notes_nodes), (axes[1], times, notes_times)):
        for position, value, note in zip(y, values, notes):
            ax.text(float(np.max(values)) * 3.6, position, note, va="center",
                    fontsize=8.5, color="0.35")
        ax.axhline(3.5, color="0.8", linewidth=0.8)
        ax.axhline(1.5, color="0.8", linewidth=0.8)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=10.5)
    fig.suptitle("Five equivalent writings of the same dense system", fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.035,
        "The three solve formulations cost the same. What drives the derivative cost is the "
        f"dependence of A on the design parameters (\u00f7{nodes[0] / nodes[1]:.0f}) "
        f"and the non-linearity of g (\u00f7{nodes[0] / nodes[2]:.1f}).",
        ha="center",
        fontsize=9.5,
        color="0.25",
    )
    fig.tight_layout(rect=(0, 0.075, 1, 0.94))
    fig.savefig(FIGURES / "cost.png", dpi=170)
    plt.close(fig)
    print("wrote figures/cost.png")


def patterns_figure(data, cases: list[dict]) -> None:
    sand = next(case for case in cases if case["label"] == SAND)
    design, controls, state, response = (
        int(value) for value in data[f"{SAND}|blocks"]
    )
    boundaries = np.cumsum([design, controls, state, response])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, key, title, ylabel in (
        (axes[0], "closure", "Closure Jacobian   A y - b", "closure rows"),
        (axes[1], "hessian", "Lagrangian Hessian", "design | controls | state | response"),
    ):
        rows = np.array(data[f"{SAND}|{key}_rows"])
        cols = np.array(data[f"{SAND}|{key}_cols"])
        shape = data[f"{SAND}|{key}_shape"]
        ax.plot(cols, rows, ".", color="#1f4e79", markersize=2.6)
        ax.set_xlim(-1, shape[1])
        ax.set_ylim(shape[0], -1)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n{shape[0]} x {shape[1]}, {len(rows)} nonzeros", fontsize=11)
        ax.set_xlabel("design | controls | state | response", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        for boundary in boundaries[:-1]:
            ax.axvline(boundary - 0.5, color="#e08214", linewidth=0.8, alpha=0.8)
        if key == "hessian":
            for boundary in boundaries[:-1]:
                ax.axhline(boundary - 0.5, color="#e08214", linewidth=0.8, alpha=0.8)

    fig.suptitle(
        "Small patterns, huge graph \u2014 "
        f"evaluating that Hessian walks {sand['nodes_hessian'] / 1000:.0f} k nodes",
        fontsize=12,
        y=0.97,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    fig.savefig(FIGURES / "patterns.png", dpi=170)
    plt.close(fig)
    print("wrote figures/patterns.png")


def scaling_figure() -> None:
    rows = json.loads((HERE / "scaling.json").read_text(encoding="utf-8"))
    sizes = np.array([row["N"] for row in rows], dtype=float)

    graphs = (
        ("gradient", "nodes_gradient", "#2e8b57", "o"),
        ("Hessian-vector product", "nodes_hessian_vector", "#7fb3d5", "v"),
        ("exact Hessian", "nodes_hessian", "#1f4e79", "s"),
        ("exact Hessian, design frozen", "nodes_hessian_frozen", "#b8cfe8", "^"),
        ("exact Hessian, coefficients constant", "nodes_hessian_state_frozen", "#8e44ad", "P"),
        ("synthetic dense cubic", "nodes_cubic", "0.55", "D"),
    )
    times = (
        ("gradient", "gradient_ms", "#2e8b57", "o"),
        ("Hessian-vector product", "hessian_vector_ms", "#7fb3d5", "v"),
        ("exact Hessian", "hessian_ms", "#1f4e79", "s"),
        ("exact Hessian, design frozen", "hessian_frozen_ms", "#b8cfe8", "^"),
        ("exact Hessian, coefficients constant", "hessian_state_frozen_ms", "#8e44ad", "P"),
        ("synthetic dense cubic", "cubic_ms", "0.55", "D"),
        ("dense constant matrix", "constant_ms", "0.2", "x"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.9))
    for ax, spec, title, ylabel in (
        (axes[0], graphs, "Size of the derivative graph", "nodes in the graph"),
        (axes[1], times, "Cost of one evaluation", "milliseconds"),
    ):
        for label, key, color, marker in spec:
            values = np.array([row[key] for row in rows], dtype=float)
            ax.loglog(sizes, values, marker=marker, color=color, label=label,
                      linewidth=1.4, markersize=5)
        ax.set_title(title, fontsize=11.5, pad=10)
        ax.set_xlabel("dense system size N", fontsize=9.5)
        ax.set_ylabel(ylabel, fontsize=9.5)
        ax.grid(which="both", alpha=0.2, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")

    last = rows[-1]
    fig.suptitle("Second-order differentiation of the model is the cost, not dense algebra",
                 fontsize=12.5, y=0.98)
    fig.text(
        0.5,
        0.035,
        f"At N = {last['N']} the exact Hessian costs "
        f"{last['hessian_ms'] / last['hessian_state_frozen_ms']:.0f}\u00d7 the same model with a "
        f"constant matrix, {last['hessian_ms'] / last['cubic_ms']:.0f}\u00d7 a synthetic dense "
        f"cubic of the same shape and {last['hessian_ms'] / last['hessian_vector_ms']:.0f}\u00d7 "
        f"its own Hessian-vector product.",
        ha="center",
        fontsize=9,
        color="0.25",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(FIGURES / "scaling.png", dpi=170)
    plt.close(fig)
    print("wrote figures/scaling.png")


def main() -> None:
    FIGURES.mkdir(exist_ok=True)
    cases = json.loads((HERE / "results.json").read_text(encoding="utf-8"))
    data = np.load(HERE / "patterns.npz")
    cost_figure(cases)
    patterns_figure(data, cases)
    scaling_figure()


if __name__ == "__main__":
    main()
