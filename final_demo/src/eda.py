"""Exploratory data analysis outputs for THUCNews."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocessing import LABEL_ORDER

sns.set_theme(style="whitegrid")


def ensure_dirs(output_dir: Path) -> Path:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def save_label_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    counts = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    total = int(counts.sum())

    plt.figure(figsize=(11, 5))
    axis = sns.barplot(x=counts.index, y=counts.values)
    plt.title("Distribution of News Categories")
    plt.xlabel("News Category")
    plt.ylabel("Number of Articles")

    for index, count in enumerate(counts.values):
        percent = count / total * 100 if total else 0
        axis.text(
            index,
            count,
            f"{count:,}\n{percent:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(figures_dir / "label_distribution.png", dpi=180)
    plt.close()


def save_text_length_distribution(df: pd.DataFrame, figures_dir: Path) -> None:
    text_p99 = max(1, int(df["text_length"].quantile(0.99)))
    text_median = float(df["text_length"].median())
    text_p90 = float(df["text_length"].quantile(0.90))

    plt.figure(figsize=(10, 5))
    sns.histplot(df.loc[df["text_length"] <= text_p99, "text_length"], bins=80)
    plt.axvline(text_median, color="black", linestyle="--", linewidth=1.0)
    plt.axvline(text_p90, color="darkred", linestyle="--", linewidth=1.0)
    plt.text(text_median, plt.ylim()[1] * 0.92, "Median", rotation=90, va="top")
    plt.text(text_p90, plt.ylim()[1] * 0.92, "P90", rotation=90, va="top")
    plt.title("News Text Length Distribution (capped at P99)")
    plt.xlabel("Text Length")
    plt.ylabel("Number of Articles")
    plt.tight_layout()
    plt.savefig(figures_dir / "text_length_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df, x="label_name", y="text_length", order=LABEL_ORDER)
    plt.yscale("log")
    plt.title("Text Length by News Category")
    plt.xlabel("News Category")
    plt.ylabel("Text Length (log scale)")
    plt.tight_layout()
    plt.savefig(figures_dir / "text_length_by_label.png", dpi=180)
    plt.close()


def build_eda_summary(df: pd.DataFrame) -> dict[str, object]:
    label_counts = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    return {
        "rows": int(len(df)),
        "text_length": {
            "mean": float(df["text_length"].mean()),
            "median": float(df["text_length"].median()),
            "p90": float(df["text_length"].quantile(0.90)),
            "p99": float(df["text_length"].quantile(0.99)),
        },
        "label_counts": {key: int(value) for key, value in label_counts.items()},
    }


def run_eda(df: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    """Create all required EDA figures and summary files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = ensure_dirs(output_dir)

    save_label_distribution(df, figures_dir)
    save_text_length_distribution(df, figures_dir)

    label_counts = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    label_counts.rename_axis("label_name").reset_index(name="count").to_csv(
        output_dir / "label_counts.csv", index=False
    )

    summary = build_eda_summary(df)
    with (output_dir / "eda_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    return summary
