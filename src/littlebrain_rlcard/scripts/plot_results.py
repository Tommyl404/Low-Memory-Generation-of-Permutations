"""
plot_results.py — Build plots from saved CSV result files.

Usage::

    python -m littlebrain_rlcard.scripts.plot_results path/to/results.csv

CSV format expected: columns ``m_bits,random_avg,myopic_avg``
"""

from __future__ import annotations

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path: str) -> dict[int, dict[str, float]]:
    """Read a CSV with columns m_bits, random_avg, myopic_avg."""
    results: dict[int, dict[str, float]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            m = int(row["m_bits"])
            results[m] = {
                "random": float(row["random_avg"]),
                "myopic": float(row["myopic_avg"]),
            }
    return results


def plot(results: dict[int, dict[str, float]], out_path: str) -> None:
    """Create and save the plot."""
    m_vals = sorted(results.keys())
    random_avgs = [results[m]["random"] for m in m_vals]
    myopic_avgs = [results[m]["myopic"] for m in m_vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m_vals, random_avgs, "o-", label="RandomAgent")
    ax.plot(m_vals, myopic_avgs, "s--", label="MyopicGuesser")
    ax.set_xlabel("m_bits")
    ax.set_ylabel("Avg correct guesses")
    ax.set_title("Shuffle-Guess score vs m_bits")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m littlebrain_rlcard.scripts.plot_results <csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "artifacts", "plots",
    )
    out_dir = os.path.normpath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results_from_csv.png")

    results = load_csv(csv_path)
    plot(results, out_path)


if __name__ == "__main__":
    main()
