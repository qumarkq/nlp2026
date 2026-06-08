"""Preprocessing for THUCNews category classification."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

LABEL_ORDER = [
    "财经",
    "彩票",
    "房产",
    "股票",
    "家居",
    "教育",
    "科技",
    "社会",
    "时尚",
    "时政",
    "体育",
    "星座",
    "游戏",
    "娱乐",
]
LABEL_TO_ID = {name: idx for idx, name in enumerate(LABEL_ORDER)}
ID_TO_LABEL = {idx: name for name, idx in LABEL_TO_ID.items()}


def clean_text(value: Any) -> str:
    """Normalize text while preserving Chinese content for MacBERT."""
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def split_news_text(value: Any) -> tuple[str, str]:
    """Split THUCNews rows formatted as 'category\\tcontent'."""
    if not isinstance(value, str) or "\t" not in value:
        return "", ""
    label_name, content = value.split("\t", 1)
    return clean_text(label_name), clean_text(content)


def prepare_dataframe(dataset: list[dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert the dataset into a cleaned DataFrame and summary stats."""
    raw_count = len(dataset)
    rows: list[dict[str, Any]] = []

    for item in dataset:
        label_name, content = split_news_text(item.get("text"))
        if label_name not in LABEL_TO_ID or content == "":
            continue
        rows.append(
            {
                "text": content,
                "label_name": label_name,
                "label": LABEL_TO_ID[label_name],
                "text_length": len(content),
            }
        )

    df = pd.DataFrame(rows, columns=["text", "label_name", "label", "text_length"])

    label_counts = df["label_name"].value_counts().reindex(LABEL_ORDER, fill_value=0)
    summary = {
        "raw_rows": int(raw_count),
        "rows_before_drop": int(raw_count),
        "clean_rows": int(len(df)),
        "dropped_rows": int(raw_count - len(df)),
        "label_counts": {key: int(value) for key, value in label_counts.items()},
    }
    return df, summary
