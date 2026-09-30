"""Plot mean Hessian-call cost and whole-solve time from published records."""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WRITINGS = (("sparse_lifted", "sparse, lifted", "o-"),
            ("dense_lifted", "dense, lifted", "s--"))


def main() -> None:
    rows = json.loads((ROOT / "validation/results.json").read_text())["homotopy"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for key, label, style in WRITINGS:
        cells = sorted(((row["field_ratio"], row) for name, row in rows.items()
                        if name.endswith(key)), key=lambda item: item[0])
        for axis, metric in zip(axes, ("hessian", "solve")):
            values = [row["derivatives"]["hessian_mean_seconds"] * 1e3
                      if metric == "hessian" else row["solve_seconds"] for _, row in cells]
            axis.plot([z for z, _ in cells], values, style, label=label,
                      linewidth=1.8, markersize=5)
            changed = [(z, value) for (z, row), value in zip(cells, values)
                       if row.get("finding") == "different_stationary_point"]
            if changed:
                axis.scatter(*zip(*changed), marker="x", s=70, color="crimson",
                             label="different stationary point", zorder=4)
    for axis, title, ylabel in zip(axes, ("Hessian callback", "Complete solve"),
                                  ("mean time per call (ms)", "opti.solve() time (s)")):
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("field unknowns per response unknown, z")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True, which="both", linewidth=0.4, alpha=0.5)
        axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(ROOT / "crossover.png", dpi=160)
    plt.close(figure)
    print("wrote crossover.png")


if __name__ == "__main__":
    main()
