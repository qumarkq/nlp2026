"""Lazy dataset for MacBERT fine-tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class NewsTextDataset(Dataset):
    """Tokenize news articles on demand to avoid full RAM tokenization."""

    def __init__(self, df: pd.DataFrame, indices: np.ndarray, tokenizer, max_length: int):
        self.texts = df["text"].to_numpy()
        self.labels = df["label"].to_numpy()
        self.indices = indices
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, object]:
        row_index = int(self.indices[index])
        encoded = self.tokenizer(
            str(self.texts[row_index]),
            truncation=True,
            max_length=self.max_length,
        )
        encoded["label"] = int(self.labels[row_index])
        return encoded
