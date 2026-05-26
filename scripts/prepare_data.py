import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import (
    create_class_mapping,
    create_kfold_split,
    create_stratified_split,
    duplicate_report,
    scan_dataset,
)
from src.data.preprocessing import compute_mean_std_from_csv, save_preprocessing_params


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare tea disease image dataset.")
    parser.add_argument("--dataset-dir", default="dataset", help="Class-folder image dataset directory.")
    parser.add_argument("--splits-dir", default="data/splits", help="Output directory for split CSV files.")
    parser.add_argument("--config-dir", default="config", help="Output directory for mapping files.")
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--kfold", action="store_true", help="Also create stratified k-fold split files.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--skip-hashes", action="store_true", help="Skip exact/perceptual duplicate hashes.")
    return parser.parse_args()


def main():
    args = parse_args()
    splits_dir = Path(args.splits_dir)
    config_dir = Path(args.config_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    mapping = create_class_mapping(args.dataset_dir, config_dir / "class_mapping.json")
    metadata = scan_dataset(args.dataset_dir, include_hashes=not args.skip_hashes)
    metadata_path = splits_dir / "metadata.csv"
    metadata.to_csv(metadata_path, index=False)

    invalid = metadata[~metadata["is_valid"].astype(bool)]
    if not invalid.empty:
        invalid.to_csv(splits_dir / "invalid_images.csv", index=False)

    reports = duplicate_report(metadata)
    duplicate_counts = {}
    for name, report in reports.items():
        duplicate_counts[name] = len(report)
        if not report.empty:
            report.to_csv(splits_dir / f"{name}.csv", index=False)

    train_df, val_df, test_df = create_stratified_split(
        metadata,
        splits_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    if args.kfold:
        create_kfold_split(metadata, splits_dir / "folds", n_splits=args.n_splits, seed=args.seed)

    mean, std = compute_mean_std_from_csv(splits_dir / "train.csv", image_size=args.image_size, root_dir=PROJECT_ROOT)
    save_preprocessing_params(config_dir / "preprocessing_params.json", args.image_size, mean, std)

    summary = {
        "dataset_dir": args.dataset_dir,
        "num_classes": len(mapping),
        "num_images": len(metadata),
        "num_valid_images": int(metadata["is_valid"].sum()),
        "num_invalid_images": int((~metadata["is_valid"].astype(bool)).sum()),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "image_size": args.image_size,
        "mean": mean,
        "std": std,
        "duplicates": duplicate_counts,
    }
    (splits_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
