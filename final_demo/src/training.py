"""MacBERT fine-tuning pipeline."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Callable
import math
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from evaluation import compute_metrics_for_trainer
from news_dataset import NewsTextDataset
from preprocessing import ID_TO_LABEL, LABEL_ORDER, LABEL_TO_ID

MODEL_NAME = "hfl/chinese-macbert-base"


class WeightedLossTrainer(Trainer):
    """Trainer that applies class-weighted cross entropy."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels", inputs.pop("label", None))
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device)
            if self.class_weights is not None
            else None
        )
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def stratify_or_none(labels: pd.Series, test_size: float):
    counts = labels.value_counts()
    test_count = math.ceil(len(labels) * test_size)
    train_count = len(labels) - test_count
    can_stratify = (
        len(counts) == len(LABEL_ORDER)
        and counts.min() >= 2
        and test_count >= len(LABEL_ORDER)
        and train_count >= len(LABEL_ORDER)
    )
    return labels if can_stratify else None


def split_indices(
    df: pd.DataFrame, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices into train/validation/test using 80/10/10."""
    indices = np.arange(len(df))
    train_indices, temp_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=seed,
        stratify=stratify_or_none(df["label"], test_size=0.2),
    )
    temp_labels = df["label"].iloc[temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,
        random_state=seed,
        stratify=stratify_or_none(temp_labels, test_size=0.5),
    )
    return train_indices, val_indices, test_indices


def compute_class_weights(df: pd.DataFrame, train_indices: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split only."""
    labels = df["label"].iloc[train_indices].astype(int)
    counts = labels.value_counts().reindex(range(len(LABEL_ORDER)), fill_value=0)
    total = counts.sum()
    weights = total / (len(LABEL_ORDER) * counts.clip(lower=1))
    return torch.tensor(weights.to_numpy(dtype=np.float32))


def find_latest_checkpoint(model_dir: Path) -> str | None:
    checkpoints = sorted(
        model_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1])
        if path.name.split("-")[-1].isdigit()
        else -1,
    )
    return str(checkpoints[-1]) if checkpoints else None


def train_model(
    df: pd.DataFrame,
    model_dir: Path,
    max_length: int = 256,
    batch_size: int = 32,
    dataloader_num_workers: int = 2,
    epochs: int = 5,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    gradient_accumulation_steps: int = 1,
    logging_dir: Path | None = None,
    seed: int = 42,
    resume_from_checkpoint: str | None = None,
    use_class_weights: bool = False,
    log_fn: Callable[[str], None] | None = None,
) -> tuple[Trainer, NewsTextDataset, dict[str, int]]:
    """Fine-tune MacBERT and return the trainer plus tokenized test set."""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for training. Run this on the RTX 5070 machine "
            "with a CUDA-enabled PyTorch installation."
        )

    if log_fn is not None:
        log_fn("Splitting train/validation/test data.")
    train_indices, val_indices, test_indices = split_indices(df, seed=seed)
    if log_fn is not None:
        log_fn(
            "Split sizes: "
            f"train={len(train_indices)}, "
            f"validation={len(val_indices)}, "
            f"test={len(test_indices)}."
        )

    if log_fn is not None:
        log_fn("Loading tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if log_fn is not None:
        log_fn("Building lazy tokenization datasets.")
    train_dataset = NewsTextDataset(df, train_indices, tokenizer, max_length)
    val_dataset = NewsTextDataset(df, val_indices, tokenizer, max_length)
    test_dataset = NewsTextDataset(df, test_indices, tokenizer, max_length)

    if log_fn is not None:
        log_fn("Loading MacBERT sequence classification model.")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_ORDER),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    class_weights = compute_class_weights(df, train_indices) if use_class_weights else None
    if log_fn is not None:
        if class_weights is None:
            log_fn("Class-weighted loss disabled.")
        else:
            weights_text = {
                label: round(float(class_weights[index]), 4)
                for index, label in enumerate(LABEL_ORDER)
            }
            log_fn(f"Class-weighted loss enabled: {weights_text}")

    model_dir.mkdir(parents=True, exist_ok=True)
    if logging_dir is not None:
        logging_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)
        if log_fn is not None:
            log_fn(f"TensorBoard logging directory: {logging_dir}")

    args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=500,
        save_total_limit=3,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        fp16=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=dataloader_num_workers,
        report_to=["tensorboard"],
        seed=seed,
        load_best_model_at_end=False,
        metric_for_best_model="macro_f1",
    )

    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "processing_class": tokenizer,
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics_for_trainer,
    }
    if use_class_weights:
        trainer = WeightedLossTrainer(**trainer_kwargs, class_weights=class_weights)
    else:
        trainer = Trainer(**trainer_kwargs)

    if resume_from_checkpoint == "auto":
        resume_from_checkpoint = find_latest_checkpoint(model_dir)

    if log_fn is not None:
        log_fn("Running Trainer.train().")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    if log_fn is not None:
        log_fn("Saving trained model and trainer state.")
    trainer.save_state()
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    split_counts = {
        "train_rows": len(train_indices),
        "validation_rows": len(val_indices),
        "test_rows": len(test_indices),
    }
    return trainer, test_dataset, split_counts
