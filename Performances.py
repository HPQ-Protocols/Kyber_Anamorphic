#!/usr/bin/env python3
"""Generate paper figures and compact tables from measured CSV files.

Run Microbenchmark_ProtocolModelingFinal.py first.  This script contains no
hard-coded experimental result and never substitutes analytical guesses for
missing measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "blue": "#2878B5",
    "orange": "#F28E2B",
    "green": "#2CA02C",
    "red": "#D62728",
    "purple": "#9467BD",
    "gray": "#7F7F7F",
}


def require_csv(input_dir: Path, filename: str) -> pd.DataFrame:
    path = input_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required measurement file not found: {path}. "
            "Run Microbenchmark_ProtocolModelingFinal.py first."
        )
    return pd.read_csv(path)


def save_figure(fig: plt.Figure, output_dir: Path, filename: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def short_operation_name(name: str) -> str:
    replacements = {
        "Complete authenticated AKE": "Complete AKE",
        "Ordinary protected record": "Ordinary record",
        "ML-KEM ": "ML-KEM\n",
        "HMAC-SHA256 tag": "HMAC-SHA256",
    }
    result = str(name)
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result


def plot_component_latency(components: pd.DataFrame, output_dir: Path) -> None:
    selected = components[~components["operation"].str.startswith("AnoDec")].copy()
    selected["label"] = selected["operation"].map(short_operation_name)
    x = np.arange(len(selected))

    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    bars = ax.bar(
        x,
        selected["median_ms"],
        yerr=selected["ci95_ms"],
        capsize=4,
        color=[COLORS["blue"] if "AnoEnc" not in item else COLORS["purple"] for item in selected["operation"]],
        alpha=0.88,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(selected["label"], rotation=38, ha="right")
    ax.set_ylabel("Median CPU time (ms)")
    ax.set_title("Measured cryptographic component latency")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    if (selected["median_ms"] > 0).all() and selected["median_ms"].max() / selected["median_ms"].min() > 50:
        ax.set_yscale("log")
    for bar, value in zip(bars, selected["median_ms"]):
        ax.annotate(
            f"{value:.4g}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8,
        )
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_component_latency.png")


def plot_computation_vs_communication(protocols: pd.DataFrame, output_dir: Path) -> None:
    data = protocols.copy()
    labels = [str(item).replace(" protected record", "\nrecord").replace("Authenticated AKE ", "AKE ") for item in data["configuration"]]
    x = np.arange(len(data))

    fig, ax_bytes = plt.subplots(figsize=(11.5, 6.0))
    bars = ax_bytes.bar(x, data["communication_bytes"], color=COLORS["blue"], alpha=0.78, width=0.62)
    ax_bytes.set_ylabel("Serialized communication (bytes)", color=COLORS["blue"])
    ax_bytes.tick_params(axis="y", labelcolor=COLORS["blue"])
    ax_bytes.set_xticks(x)
    ax_bytes.set_xticklabels(labels, rotation=28, ha="right")
    ax_bytes.grid(axis="y", linestyle="--", alpha=0.25)

    ax_cpu = ax_bytes.twinx()
    ax_cpu.errorbar(
        x, data["median_ms"], yerr=data["ci95_ms"],
        color=COLORS["red"], marker="o", linewidth=2.2, capsize=4,
        label="CPU time",
    )
    ax_cpu.set_ylabel("Median CPU time (ms)", color=COLORS["red"])
    ax_cpu.tick_params(axis="y", labelcolor=COLORS["red"])

    for bar, value in zip(bars, data["communication_bytes"]):
        ax_bytes.annotate(
            f"{int(value)} B", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, -15), textcoords="offset points", ha="center", fontsize=8,
            color="white", fontweight="bold",
        )
    for index, value in enumerate(data["median_ms"]):
        ax_cpu.annotate(
            f"{value:.4g} ms", (index, value), xytext=(0, 9),
            textcoords="offset points", ha="center", fontsize=8, color=COLORS["red"],
        )

    ax_bytes.set_title("Computation versus communication for the proposed protocol")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_computation_vs_communication.png")


def plot_anamorphic_tradeoff(anamorphic: pd.DataFrame, output_dir: Path) -> None:
    data = anamorphic.sort_values("ell")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].errorbar(
        data["ell"], data["anoenc_median_ms"], yerr=data["anoenc_ci95_ms"],
        color=COLORS["purple"], marker="o", linewidth=2, capsize=4,
    )
    axes[0].set_ylabel("AnoEnc median time (ms)")
    axes[0].set_title("Encoding cost")

    axes[1].plot(data["ell"], data["mean_trials"], marker="o", linewidth=2, label="Measured")
    axes[1].plot(data["ell"], data["theoretical_trials"], linestyle="--", marker="s", label=r"Theory: $2^\ell$")
    axes[1].set_ylabel("Trials per accepted nonce")
    axes[1].set_title("Rejection-sampling work")
    axes[1].legend(fontsize=8)

    axes[2].plot(
        data["ell"], data["covert_throughput_bits_s"],
        color=COLORS["green"], marker="o", linewidth=2,
    )
    axes[2].set_ylabel("Covert throughput (bit/s)")
    axes[2].set_title("Hidden-channel throughput")

    for ax in axes:
        ax.set_xlabel(r"Hidden bits per record, $\ell$")
        ax.set_xticks(data["ell"])
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Anamorphic throughput--computation trade-off", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_anamorphic_tradeoff.png")


def plot_detectability(security: pd.DataFrame, output_dir: Path) -> None:
    data = security.sort_values("ell")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))

    axes[0].plot(data["ell"], data["normal_shannon_entropy_bits_per_byte"], marker="o", label="Normal")
    axes[0].plot(data["ell"], data["anamorphic_shannon_entropy_bits_per_byte"], marker="s", label="Anamorphic")
    axes[0].set_ylabel("Shannon entropy (bit/byte)")
    axes[0].set_title("Entropy")
    axes[0].legend(fontsize=8)

    axes[1].plot(data["ell"], data["tv_distance"], marker="o", label="TV distance")
    axes[1].plot(data["ell"], data["ks_statistic"], marker="s", label="KS statistic")
    axes[1].set_ylabel("Distance/statistic")
    axes[1].set_title("Distributional distance")
    axes[1].legend(fontsize=8)

    axes[2].plot(data["ell"], data["classifier_accuracy"], marker="o", label="Accuracy")
    axes[2].plot(data["ell"], data["classifier_roc_auc"], marker="s", label="ROC-AUC")
    axes[2].axhline(0.5, color=COLORS["gray"], linestyle="--", linewidth=1, label="Random guessing")
    axes[2].set_ylim(0.35, 0.65)
    axes[2].set_ylabel("Score")
    axes[2].set_title("Classifier distinguishability")
    axes[2].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel(r"Hidden bits per record, $\ell$")
        ax.set_xticks(data["ell"])
        ax.grid(True, linestyle="--", alpha=0.3)
    fig.suptitle("Normal versus anamorphic nonce detectability", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_anamorphic_detectability.png")


def export_paper_tables(
    protocols: pd.DataFrame,
    anamorphic: pd.DataFrame,
    security: pd.DataFrame | None,
    output_dir: Path,
) -> None:
    protocol_table = protocols[[
        "configuration", "median_ms", "ci95_ms", "throughput_ops_s", "communication_bytes", "ell"
    ]].copy()
    protocol_table.to_csv(output_dir / "paper_table_protocol_performance.csv", index=False)

    ano_table = anamorphic[[
        "ell", "anoenc_median_ms", "anoenc_ci95_ms", "mean_trials",
        "theoretical_trials", "covert_throughput_bits_s",
    ]].copy()
    ano_table.to_csv(output_dir / "paper_table_anamorphic_performance.csv", index=False)

    if security is not None:
        detectability_table = security[[
            "ell",
            "normal_shannon_entropy_bits_per_byte",
            "anamorphic_shannon_entropy_bits_per_byte",
            "normal_min_entropy_bits_per_byte",
            "anamorphic_min_entropy_bits_per_byte",
            "normal_chi2_p_value",
            "anamorphic_chi2_p_value",
            "ks_statistic",
            "ks_p_value",
            "tv_distance",
            "classifier_accuracy",
            "classifier_balanced_accuracy",
            "classifier_roc_auc",
        ]].copy()
        detectability_table.to_csv(output_dir / "paper_table_detectability.csv", index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("experimental_results"))
    parser.add_argument("--output-dir", type=Path, default=Path("experimental_figures"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    components = require_csv(args.input_dir, "component_benchmarks.csv")
    protocols = require_csv(args.input_dir, "protocol_summary.csv")
    anamorphic = require_csv(args.input_dir, "anamorphic_by_ell.csv")
    security_path = args.input_dir / "security_metrics.csv"
    security = pd.read_csv(security_path) if security_path.exists() else None

    plot_component_latency(components, args.output_dir)
    plot_computation_vs_communication(protocols, args.output_dir)
    plot_anamorphic_tradeoff(anamorphic, args.output_dir)
    if security is not None:
        plot_detectability(security, args.output_dir)
    else:
        print("security_metrics.csv not found; detectability figure was skipped")
    export_paper_tables(protocols, anamorphic, security, args.output_dir)
    print(f"all figures and paper tables are in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()