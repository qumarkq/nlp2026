"""Command-line entrypoint for the THUCNews classification project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from dataset_loader import DEFAULT_DATA_PATH, load_news_dataset
from preprocessing import LABEL_ORDER, prepare_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="THUCNews Chinese news classification")
    parser.add_argument("--dev-sample", type=int, default=None)
    parser.add_argument("--skip-eda", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Local THUCNews folder. Defaults to ./THUCNews.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dataloader-num-workers", type=int, default=2)
    parser.add_argument("--use-class-weights", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--logging-dir", type=Path, default=Path("outputs/tensorboard")
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models/macbert_news_classifier")
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        default=None,
        help="Checkpoint path, or 'auto' to use the latest checkpoint in model-dir.",
    )
    return parser.parse_args()


def get_cuda_info() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    return {
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0)
        if cuda_available
        else None,
    }


def write_training_summary(output_dir: Path, summary: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def log_message(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    line = f"[{timestamp}] {message}"
    print(line, flush=True)
    with (output_dir / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    log_message(args.output_dir, "Starting THUCNews classification pipeline.")
    cuda_info = get_cuda_info()
    log_message(args.output_dir, f"CUDA status: {cuda_info}")

    if not args.skip_train and not cuda_info["cuda_available"]:
        message = (
            "CUDA is required for training, but CUDA is not available. "
            "Use --skip-train for EDA-only runs, or run training on the RTX 5070 "
            "CUDA machine."
        )
        log_message(args.output_dir, message)
        raise SystemExit(1)

    data_source = str(args.data_path)
    log_message(args.output_dir, f"Loading dataset from {data_source}.")
    dataset = load_news_dataset(
        dev_sample=args.dev_sample,
        seed=args.seed,
        data_path=args.data_path,
    )
    log_message(args.output_dir, f"Loaded rows: {len(dataset)}")

    log_message(args.output_dir, "Preprocessing dataset.")
    df, preprocessing_summary = prepare_dataframe(dataset)
    log_message(args.output_dir, f"Clean rows: {len(df)}")

    summary: dict[str, object] = {
        **preprocessing_summary,
        "mode": "development" if args.dev_sample is not None else "full",
        "data_source": data_source,
        "dev_sample": args.dev_sample,
        "dataset_name": "local THUCNews",
        "model_name": "hfl/chinese-macbert-base",
        "num_labels": len(LABEL_ORDER),
        "epochs_requested": args.epochs,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "dataloader_num_workers": args.dataloader_num_workers,
        "use_class_weights": args.use_class_weights,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tensorboard_logging_dir": str(args.logging_dir),
        "cuda": cuda_info,
    }
    write_training_summary(args.output_dir, summary)
    log_message(args.output_dir, "Wrote training_summary.json.")

    if not args.skip_eda:
        from eda import run_eda

        log_message(args.output_dir, "Running EDA.")
        summary["eda_summary"] = run_eda(df, args.output_dir)
        write_training_summary(args.output_dir, summary)
        log_message(args.output_dir, "Finished EDA.")

    if args.skip_train:
        log_message(args.output_dir, "Skipping training by request.")
        return

    from evaluation import compute_metric_dict, save_confusion_matrix, save_metrics
    from training import train_model

    log_message(args.output_dir, "Starting MacBERT training.")
    trainer, test_dataset, split_counts = train_model(
        df=df,
        model_dir=args.model_dir,
        max_length=args.max_length,
        batch_size=args.batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_dir=args.logging_dir,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint,
        use_class_weights=args.use_class_weights,
        log_fn=lambda message: log_message(args.output_dir, message),
    )
    log_message(args.output_dir, "Finished MacBERT training.")

    log_message(args.output_dir, "Evaluating test split.")
    predictions_output = trainer.predict(test_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids

    metrics = compute_metric_dict(labels, predictions)
    save_metrics(metrics, args.output_dir / "metrics.json")
    save_confusion_matrix(
        labels, predictions, args.output_dir / "figures" / "confusion_matrix.png"
    )

    pd.DataFrame({"label": labels, "prediction": predictions}).to_csv(
        args.output_dir / "predictions.csv", index=False
    )

    summary.update(
        {
            **split_counts,
            "epochs_completed": args.epochs,
            "final_checkpoint": str(args.model_dir),
            "metrics": {
                key: metrics[key]
                for key in [
                    "accuracy",
                    "macro_precision",
                    "macro_recall",
                    "macro_f1",
                    "weighted_f1",
                ]
            },
        }
    )
    write_training_summary(args.output_dir, summary)
    log_message(args.output_dir, "Pipeline completed successfully.")


if __name__ == "__main__":
    main()
