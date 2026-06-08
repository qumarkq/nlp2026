"""Evaluation helpers for news classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from preprocessing import LABEL_ORDER

plt.rcParams["font.sans-serif"] = [
    "PingFang TC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def set_font_from_path(font_path: Path | None) -> None:
    if font_path is None:
        return
    font_manager.fontManager.addfont(str(font_path))
    font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]


def compute_metric_dict(labels, predictions) -> dict[str, object]:
    accuracy = accuracy_score(labels, predictions)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(LABEL_ORDER))),
        target_names=LABEL_ORDER,
        output_dict=True,
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "classification_report": report,
    }


def compute_metrics_for_trainer(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metrics = compute_metric_dict(labels, predictions)
    return {
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def save_confusion_matrix(labels, predictions, output_path: Path) -> None:
    matrix = confusion_matrix(
        labels, predictions, labels=list(range(len(LABEL_ORDER)))
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABEL_ORDER,
        yticklabels=LABEL_ORDER,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def save_metrics(metrics: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)


def save_per_class_metrics(metrics_path: Path, output_path: Path) -> None:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    report = metrics["classification_report"]
    rows = [
        {
            "label": label,
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1-score": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in LABEL_ORDER
    ]
    metrics_df = pd.DataFrame(rows)

    plot_df = metrics_df.melt(
        id_vars=["label", "support"],
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(13, 6))
    axis = sns.barplot(data=plot_df, x="label", y="score", hue="metric")
    axis.set_ylim(0, 1.02)
    plt.title("Per-class Classification Metrics")
    plt.xlabel("News Category")
    plt.ylabel("Score")
    plt.legend(title="Metric", loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Redraw evaluation figures")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--predictions-csv", type=Path)
    input_group.add_argument("--metrics-json", type=Path)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--font-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_font_from_path(args.font_path)
    if args.predictions_csv is not None:
        predictions_df = pd.read_csv(args.predictions_csv)
        save_confusion_matrix(
            predictions_df["label"].to_numpy(),
            predictions_df["prediction"].to_numpy(),
            args.output_path,
        )
    else:
        save_per_class_metrics(args.metrics_json, args.output_path)


if __name__ == "__main__":
    main()
