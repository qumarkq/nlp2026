"""Dataset loading helpers for THUCNews."""

from __future__ import annotations

import random
from pathlib import Path

from preprocessing import LABEL_ORDER

DEFAULT_DATA_PATH = Path("THUCNews")
TEXT_SUFFIX = ".txt"


def has_category_folders(data_path: Path) -> bool:
    """Return whether a local path looks like the original THUCNews folder."""
    if not data_path.is_dir():
        return False
    return any((data_path / label).is_dir() for label in LABEL_ORDER)


def collect_text_paths(data_path: Path) -> list[tuple[str, Path]]:
    """Collect original THUCNews text files with their category names."""
    rows: list[tuple[str, Path]] = []
    for label in LABEL_ORDER:
        label_dir = data_path / label
        if not label_dir.is_dir():
            continue
        for text_path in sorted(label_dir.rglob(f"*{TEXT_SUFFIX}")):
            if text_path.name.startswith("._"):
                continue
            rows.append((label, text_path))
    return rows


def load_category_folder_dataset(
    data_path: Path,
    dev_sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Load the original THUCNews folder layout: category/*.txt."""
    text_paths = collect_text_paths(data_path)
    if dev_sample is not None:
        sample_size = min(dev_sample, len(text_paths))
        text_paths = random.Random(seed).sample(text_paths, sample_size)

    rows: list[dict[str, str]] = []
    for label, text_path in text_paths:
        content = text_path.read_text(encoding="utf-8", errors="ignore")
        rows.append({"text": f"{label}\t{content}"})
    return rows


def load_local_dataset(
    data_path: Path,
    dev_sample: int | None = None,
    seed: int = 42,
) -> list[dict[str, str]]:
    """Load a local original THUCNews category folder."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {data_path}")

    if not has_category_folders(data_path):
        raise ValueError(
            "Local data must be the original THUCNews folder layout, for example "
            "THUCNews/体育/*.txt and THUCNews/财经/*.txt."
        )

    return load_category_folder_dataset(data_path, dev_sample=dev_sample, seed=seed)


def load_news_dataset(
    dev_sample: int | None = None,
    seed: int = 42,
    data_path: Path = DEFAULT_DATA_PATH,
) -> list[dict[str, str]]:
    """Load the local THUCNews dataset."""
    return load_local_dataset(data_path, dev_sample=dev_sample, seed=seed)
