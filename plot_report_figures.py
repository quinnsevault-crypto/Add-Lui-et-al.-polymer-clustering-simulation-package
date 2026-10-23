"""Build a compact, report-ready set from the long-core ensemble summary."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
RESULTS = HERE / "long_core_results"
SUMMARY = RESULTS / "ensemble_summary.csv"
OUT = RESULTS / "report_figures"
POLYMERS = ("P1", "P2", "P3")
COLORS = {"P1": "#2878B5", "P2": "#E07A1F", "P3": "#3A923A", "CONTROL": "#555555"}


def _curve(ax, frame: pd.DataFrame, *, mean: str, sd: str, label: str, color: str) -> None:
    frame = frame.sort_values("time")
    x = frame["time"].to_numpy()
    y = frame[mean].to_numpy()
    err = frame[sd].to_numpy()
    ax.plot(x, y, label=label, color=color, linewidth=2.2)
    ax.fill_between(x, y - err, y + err, color=color, alpha=0.16, linewidth=0)


def plot_strain(df: pd.DataFrame, strain: str) -> None:
    data = df[(df.strain == strain) & (df.concentration.isin([0, 7000]))]
    fig, (ax_ratio, ax_log) = plt.subplots(2, 1, figsize=(9.5, 8), sharex=True,
                                          gridspec_kw={"height_ratios": [1, 1.1]})
    for polymer in POLYMERS:
        group = data[(data.polymer == polymer) & (data.concentration == 7000)]
        _curve(ax_ratio, group, mean="fold_change_mean", sd="fold_change_sd",
               label=f"{polymer} (7,000 a.u.)", color=COLORS[polymer])
    ax_ratio.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2,
                     label="No-polymer reference")
    ax_ratio.set_ylabel("Luminescence / paired control")
    ax_ratio.set_title(f"{strain}: polymer effects over time", loc="left", weight="bold")
    ax_ratio.legend(frameon=False, ncol=2, loc="best")
    ax_ratio.grid(alpha=0.2)

    for polymer, concentration in (("CONTROL", 0), ("P1", 7000), ("P2", 7000), ("P3", 7000)):
        group = data[(data.polymer == polymer) & (data.concentration == concentration)]
        label = "No polymer" if polymer == "CONTROL" else f"{polymer} (7,000 a.u.)"
        _curve(ax_log, group, mean="log10_lum_mean", sd="log10_lum_sd",
               label=label, color=COLORS[polymer])
    ax_log.set_ylabel("Mean log₁₀ population luminescence")
    ax_log.set_xlabel("Simulation time (a.u.)")
    ax_log.grid(alpha=0.2)
    ax_log.legend(frameon=False, ncol=2, loc="best")
    fig.suptitle(f"{strain} core ensemble — mean ± SD across 10 seeds", fontsize=14, y=0.99)
    fig.text(0.5, 0.005,
             "Tau-leap approximation (100-a.u. step); ratios are paired to same-seed, same-time no-polymer controls.",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(OUT / f"{strain}_core_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_endpoint(df: pd.DataFrame) -> None:
    end = df[(df.time == 30000) & (df.polymer.isin(POLYMERS))].copy()
    order = [(strain, polymer) for strain in ("MM32", "BB170") for polymer in POLYMERS]
    end["label"] = end.apply(lambda r: f"{r.strain}  {r.polymer}", axis=1)
    end["order"] = end.apply(lambda r: order.index((r.strain, r.polymer)), axis=1)
    end = end.sort_values("order")
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = list(range(len(end)))
    colors = [COLORS[p] for p in end.polymer]
    ax.errorbar(end.fold_change_mean, y, xerr=end.fold_change_sd, fmt="none",
                ecolor="#555555", elinewidth=1.2, capsize=3, zorder=1)
    ax.scatter(end.fold_change_mean, y, c=colors, s=55, zorder=2, edgecolor="white", linewidth=0.5)
    ax.axvline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_yticks(y, end.label)
    ax.invert_yaxis()
    ax.set_xlabel("Fold change vs. paired no-polymer control at 30,000 a.u. (mean ± SD)")
    ax.set_title("Endpoint comparison across strains and polymers", loc="left", weight="bold")
    ax.grid(axis="x", alpha=0.25)
    fig.text(0.5, 0.015, "10 seeds per condition; tau-leap approximation (100-a.u. step).",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(OUT / "endpoint_fold_change_comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY)
    for strain in ("MM32", "BB170"):
        plot_strain(df, strain)
    plot_endpoint(df)
    print(f"Wrote 3 report figures to {OUT}")


if __name__ == "__main__":
    main()
