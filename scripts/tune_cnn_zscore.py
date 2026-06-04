"""Tune the existing CNN with z-score normalization on the validation split.

This script runs a small, practical configuration grid through
``run_normalization_training.py``, ranks runs by validation macro F1, and
writes the best configuration without touching the test split by default.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAINING_SCRIPT = PROJECT_ROOT / "scripts" / "run_normalization_training.py"

DEFAULT_CONFIGS = [
    {
        "name": "baseline",
        "image_size": 128,
        "batch_size": 64,
        "epochs": 60,
        "patience": 8,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "dropout": 0.0,
        "label_smoothing": 0.0,
        "lr_factor": 1.0,
        "lr_patience": 2,
    },
    {
        "name": "dropout03_wd1e4",
        "image_size": 128,
        "batch_size": 64,
        "epochs": 60,
        "patience": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.0,
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
    {
        "name": "dropout05_wd1e4",
        "image_size": 128,
        "batch_size": 64,
        "epochs": 60,
        "patience": 8,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "dropout": 0.5,
        "label_smoothing": 0.0,
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
    {
        "name": "lr5e4_drop03_smooth05",
        "image_size": 128,
        "batch_size": 64,
        "epochs": 60,
        "patience": 10,
        "learning_rate": 5e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.05,
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
    {
        "name": "lr3e4_drop03_smooth05",
        "image_size": 128,
        "batch_size": 64,
        "epochs": 60,
        "patience": 10,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "dropout": 0.3,
        "label_smoothing": 0.05,
        "lr_factor": 0.5,
        "lr_patience": 2,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune CNN + zscore using validation macro F1, then optionally run one final test."
    )
    parser.add_argument("--output-dir", default="reports/cnn_zscore_regularization_tuning")
    parser.add_argument("--max-total-samples", type=int, default=0)
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run configurations again even when their summary.csv already exists.",
    )
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        help="After tuning, run the best configuration once on the test split.",
    )
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_command(config, output_dir, args, eval_split):
    command = [
        sys.executable,
        str(TRAINING_SCRIPT),
        "--eval-split",
        eval_split,
        "--image-size",
        str(config["image_size"]),
        "--max-total-samples",
        str(args.max_total_samples),
        "--normalizers",
        "zscore",
        "--models",
        "cnn",
        "--cnn-epochs",
        str(config["epochs"]),
        "--cnn-batch-size",
        str(config["batch_size"]),
        "--cnn-patience",
        str(config["patience"]),
        "--cnn-learning-rate",
        str(config["learning_rate"]),
        "--cnn-weight-decay",
        str(config["weight_decay"]),
        "--cnn-dropout",
        str(config["dropout"]),
        "--cnn-label-smoothing",
        str(config["label_smoothing"]),
        "--cnn-lr-factor",
        str(config["lr_factor"]),
        "--cnn-lr-patience",
        str(config["lr_patience"]),
        "--seed",
        str(args.seed),
        "--output-dir",
        str(output_dir),
    ]
    if args.torch_num_threads is not None:
        command.extend(["--torch-num-threads", str(args.torch_num_threads)])
    return command


def run_command(command):
    print("\n" + "=" * 92)
    print("Running:")
    print(" ".join(command))
    print("=" * 92)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def read_result(summary_path, config):
    summary = pd.read_csv(summary_path)
    if summary.empty:
        raise ValueError(f"No result rows found in {summary_path}")

    row = summary.iloc[0].to_dict()
    return {
        "config_name": config["name"],
        "image_size": config["image_size"],
        "cnn_batch_size": config["batch_size"],
        "cnn_epochs": config["epochs"],
        "cnn_patience": config["patience"],
        "cnn_learning_rate": config["learning_rate"],
        "cnn_weight_decay": config["weight_decay"],
        "cnn_dropout": config["dropout"],
        "cnn_label_smoothing": config["label_smoothing"],
        "cnn_lr_factor": config["lr_factor"],
        "cnn_lr_patience": config["lr_patience"],
        **row,
    }


def save_best_config(best_row, output_dir):
    config = {
        "selection_metric": "validation_f1_macro",
        "normalizer": "zscore",
        "model": "cnn",
        "image_size": int(best_row["image_size"]),
        "cnn_batch_size": int(best_row["cnn_batch_size"]),
        "cnn_epochs": int(best_row["cnn_epochs"]),
        "cnn_patience": int(best_row["cnn_patience"]),
        "cnn_learning_rate": float(best_row["cnn_learning_rate"]),
        "cnn_weight_decay": float(best_row["cnn_weight_decay"]),
        "cnn_dropout": float(best_row["cnn_dropout"]),
        "cnn_label_smoothing": float(best_row["cnn_label_smoothing"]),
        "cnn_lr_factor": float(best_row["cnn_lr_factor"]),
        "cnn_lr_patience": int(best_row["cnn_lr_patience"]),
        "validation_accuracy": float(best_row["accuracy"]),
        "validation_f1_macro": float(best_row["f1_macro"]),
    }
    path = output_dir / "best_config.json"
    path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config, path


def print_test_command(config, args, output_dir):
    test_dir = output_dir / "final_test"
    command = build_command(
        {
            "image_size": config["image_size"],
            "batch_size": config["cnn_batch_size"],
            "epochs": config["cnn_epochs"],
            "patience": config["cnn_patience"],
            "learning_rate": config["cnn_learning_rate"],
            "weight_decay": config["cnn_weight_decay"],
            "dropout": config["cnn_dropout"],
            "label_smoothing": config["cnn_label_smoothing"],
            "lr_factor": config["cnn_lr_factor"],
            "lr_patience": config["cnn_lr_patience"],
        },
        test_dir,
        args,
        eval_split="test",
    )
    print("\nFinal test command (run only once after model selection):")
    print(" ".join(command))
    return command


def main():
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for config in DEFAULT_CONFIGS:
        run_dir = runs_dir / config["name"]
        summary_path = run_dir / "summary.csv"
        if args.force or not summary_path.exists():
            command = build_command(config, run_dir, args, eval_split="val")
            run_command(command)
        else:
            print(f"\nSkipping completed configuration: {config['name']}")
        results.append(read_result(summary_path, config))

    ranking = pd.DataFrame(results).sort_values(
        ["f1_macro", "accuracy"],
        ascending=[False, False],
    )
    ranking_path = output_dir / "tuning_summary.csv"
    ranking.to_csv(ranking_path, index=False)

    best_row = ranking.iloc[0]
    best_config, best_path = save_best_config(best_row, output_dir)

    print("\n" + "=" * 92)
    print("CNN + zscore validation ranking")
    print("=" * 92)
    columns = [
        "config_name",
        "cnn_learning_rate",
        "cnn_weight_decay",
        "cnn_dropout",
        "cnn_label_smoothing",
        "accuracy",
        "f1_macro",
    ]
    print(ranking[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved ranking: {ranking_path}")
    print(f"Saved best configuration: {best_path}")

    test_command = print_test_command(best_config, args, output_dir)
    if args.run_final_test:
        print("\nWARNING: Running the test split. Do not use this result for further tuning.")
        run_command(test_command)


if __name__ == "__main__":
    main()
