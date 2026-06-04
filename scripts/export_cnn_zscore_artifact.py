import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simple_cnn import load_simple_cnn_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Export a reusable CNN + zscore model artifact.")
    parser.add_argument(
        "--checkpoint",
        default="reports/cnn_zscore_regularization_tuning/learned_tsne_model/best_zscore_cnn.pt",
    )
    parser.add_argument("--train-csv", default="data/splits/clean/train.csv")
    parser.add_argument(
        "--summary-csv",
        default="reports/cnn_zscore_regularization_tuning/learned_tsne_model/summary.csv",
    )
    parser.add_argument("--output-dir", default="models/cnn_zscore")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_image(image_path, image_size):
    with Image.open(resolve_path(image_path)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def compute_zscore_params(train_df, image_size):
    pixel_sum = np.zeros((image_size, image_size, 3), dtype=np.float64)
    pixel_square_sum = np.zeros_like(pixel_sum)
    count = 0
    for image_path in train_df["image_path"]:
        image = load_image(image_path, image_size)
        pixel_sum += image
        pixel_square_sum += image * image
        count += 1
    mean = pixel_sum / count
    variance = np.maximum(pixel_square_sum / count - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale == 0.0] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def main():
    args = parse_args()
    source_checkpoint = resolve_path(args.checkpoint)
    _, checkpoint = load_simple_cnn_checkpoint(source_checkpoint)
    if checkpoint["normalizer"] != "zscore":
        raise ValueError("This exporter only supports CNN + zscore checkpoints.")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint = output_dir / "best_model.pt"

    if not checkpoint.get("normalization_params"):
        train_df = pd.read_csv(resolve_path(args.train_csv))
        mean, scale = compute_zscore_params(train_df, int(checkpoint["image_size"]))
        checkpoint["normalization_params"] = {
            "mean": torch.from_numpy(mean),
            "scale": torch.from_numpy(scale),
        }
    torch.save(checkpoint, output_checkpoint)

    summary_path = resolve_path(args.summary_csv)
    metrics = json.loads(pd.read_csv(summary_path).iloc[0].to_json()) if summary_path.exists() else {}
    manifest = {
        "artifact_type": "teavision_simple_cnn",
        "checkpoint": output_checkpoint.name,
        "architecture": "SimpleCNN",
        "normalizer": "zscore",
        "image_size": int(checkpoint["image_size"]),
        "input_format": "RGB float32 in [0, 1], resized with LANCZOS",
        "class_names": checkpoint["class_names"],
        "feature_dimension": 128,
        "metrics": metrics,
        "source_checkpoint": str(source_checkpoint.relative_to(PROJECT_ROOT)),
    }
    manifest_path = output_dir / "model_artifact.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Saved checkpoint: {output_checkpoint}")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
