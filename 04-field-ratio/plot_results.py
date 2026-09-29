"""Plot the crossover: Hessian evaluation cost of the two writings against field ratio.

Reads validation/results.json and writes crossover.png. The ratio z is the
number of field unknowns per response unknown; the sparse writing pays for the
field, the dense writing does not.

    python plot_results.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
WRITINGS = (("sparse_lifted", "sparse, lifted", "o-"),
            ("dense_lifted", "dense, lifted", "s--"))


def main() -> None:
    rows = json.loads((ROOT / "validation" / "results.json").read_text())["homotopy"]
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    for key, label, style in WRITINGS:
        cells = sorted(
            ((int(name.split("_z")[1].split("_")[0]), value) for name, value in rows.items()
             if name.endswith(key)),
            key=lambda item: item[0],
        )
        axis.plot([z for z, _ in cells], [value["hessian_mean_ms"] for _, value in cells],
                  style, label=label, linewidth=1.8, markersize=5)
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("field unknowns per response unknown  z")
    axis.set_ylabel("exact Hessian, one evaluation (ms)")
    axis.set_title("Lifting pays until the field grows past the response")
    axis.grid(True, which="both", linewidth=0.4, alpha=0.5)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(ROOT / "crossover.png", dpi=160)
    print(f"wrote {ROOT / 'crossover.png'}")


if __name__ == "__main__":
    main()
