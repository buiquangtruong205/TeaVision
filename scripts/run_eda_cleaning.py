import argparse
import json
import shutil
import sys
from pathlib import Path

import imagehash
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splitter import IMAGE_EXTENSIONS, create_class_mapping, create_stratified_split, exact_file_hash


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA and cleaning checks for image dataset.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--min-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--move-corrupted", action="store_true")
    parser.add_argument("--corrupted-dir", default="data/corrupted")
    return parser.parse_args()


def relative_path(path):
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def scan_images(dataset_dir, min_size):
    dataset_path = Path(dataset_dir)
    class_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    class_to_id = {path.name: index for index, path in enumerate(class_dirs)}
    rows = []

    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            row = {
                "image_path": relative_path(image_path),
                "label": class_to_id[class_dir.name],
                "class_name": class_dir.name,
                "width": None,
                "height": None,
                "file_size": image_path.stat().st_size,
                "is_valid": False,
                "is_small": False,
                "is_exact_duplicate": False,
                "is_phash_duplicate": False,
                "error": "",
                "file_hash": "",
                "phash": "",
            }

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image = image.convert("RGB")
                    row["width"], row["height"] = image.size
                    row["is_valid"] = True
                    row["is_small"] = image.width < min_size or image.height < min_size
                    row["file_hash"] = exact_file_hash(image_path)
                    row["phash"] = str(imagehash.phash(image))
            except (OSError, UnidentifiedImageError) as exc:
                row["error"] = str(exc)

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        valid = df["is_valid"].astype(bool)
        df.loc[valid, "is_exact_duplicate"] = df.loc[valid].duplicated("file_hash", keep="first")
        df.loc[valid, "is_phash_duplicate"] = df.loc[valid].duplicated("phash", keep="first")
    return df


def move_corrupted_images(corrupted_df, corrupted_dir):
    output_dir = Path(corrupted_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    moved = []
    for _, row in corrupted_df.iterrows():
        src = PROJECT_ROOT / row["image_path"]
        dst = output_dir / row["class_name"] / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.exists():
            shutil.move(str(src), str(dst))
            moved.append({"source": row["image_path"], "destination": relative_path(dst)})
    return moved


def plot_class_distribution(df, output_path):
    counts = df[df["is_valid"].astype(bool)]["class_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    counts.plot(kind="bar", ax=ax, color="#3b82f6")
    ax.set_title("Class Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return counts


def plot_sample_images(df, output_path, samples_per_class):
    valid_df = df[df["is_valid"].astype(bool) & ~df["is_small"].astype(bool)]
    class_names = sorted(valid_df["class_name"].unique())
    if not class_names:
        return

    rows = len(class_names)
    cols = samples_per_class
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = [axes]

    for row_idx, class_name in enumerate(class_names):
        samples = valid_df[valid_df["class_name"] == class_name].head(samples_per_class)
        for col_idx in range(cols):
            ax = axes[row_idx][col_idx]
            ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_name, fontsize=9)
            if col_idx >= len(samples):
                continue
            image_path = PROJECT_ROOT / samples.iloc[col_idx]["image_path"]
            with Image.open(image_path) as image:
                ax.imshow(image.convert("RGB"))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_report(report_path, summary, class_counts):
    lines = [
        "# EDA Report",
        "",
        "## Summary",
        "",
        f"- Dataset directory: `{summary['dataset_dir']}`",
        f"- Total images: {summary['total_images']}",
        f"- Valid images: {summary['valid_images']}",
        f"- Corrupted images: {summary['corrupted_images']}",
        f"- Small images (< {summary['min_size']}x{summary['min_size']}): {summary['small_images']}",
        f"- Exact duplicate images removed from clean list: {summary['exact_duplicates']}",
        f"- Perceptual duplicate images removed from clean list: {summary['phash_duplicates']}",
        f"- Clean images: {summary['clean_images']}",
        "",
        "## Class Distribution",
        "",
    ]
    for class_name, count in class_counts.items():
        lines.append(f"- {class_name}: {count}")
    lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `data/splits/eda_metadata.csv`",
            "- `data/splits/cleaned_metadata.csv`",
            "- `data/splits/corrupted_images.csv`",
            "- `data/splits/small_images.csv`",
            "- `data/splits/exact_duplicates.csv`",
            "- `data/splits/phash_duplicates.csv`",
            "- `reports/figures/class_distribution.png`",
            "- `reports/figures/sample_images.png`",
            "- `data/splits/clean/train.csv`",
            "- `data/splits/clean/val.csv`",
            "- `data/splits/clean/test.csv`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(args):
    splits_dir = Path(args.splits_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    splits_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    create_class_mapping(args.dataset_dir, Path(args.config_dir) / "class_mapping.json")
    df = scan_images(args.dataset_dir, args.min_size)

    corrupted_df = df[~df["is_valid"].astype(bool)].copy()
    small_df = df[df["is_valid"].astype(bool) & df["is_small"].astype(bool)].copy()
    exact_dups = df[df["is_valid"].astype(bool) & df["is_exact_duplicate"].astype(bool)].copy()
    phash_dups = df[df["is_valid"].astype(bool) & df["is_phash_duplicate"].astype(bool)].copy()

    clean_df = df[
        df["is_valid"].astype(bool)
        & ~df["is_small"].astype(bool)
        & ~df["is_exact_duplicate"].astype(bool)
        & ~df["is_phash_duplicate"].astype(bool)
    ].copy()

    df.to_csv(splits_dir / "eda_metadata.csv", index=False)
    clean_df.to_csv(splits_dir / "cleaned_metadata.csv", index=False)
    corrupted_df.to_csv(splits_dir / "corrupted_images.csv", index=False)
    small_df.to_csv(splits_dir / "small_images.csv", index=False)
    exact_dups.to_csv(splits_dir / "exact_duplicates.csv", index=False)
    phash_dups.to_csv(splits_dir / "phash_duplicates.csv", index=False)

    moved = []
    if args.move_corrupted and not corrupted_df.empty:
        moved = move_corrupted_images(corrupted_df, args.corrupted_dir)

    clean_split_dir = splits_dir / "clean"
    clean_train, clean_val, clean_test = create_stratified_split(
        clean_df,
        clean_split_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    class_counts = plot_class_distribution(df, figures_dir / "class_distribution.png")
    plot_sample_images(clean_df, figures_dir / "sample_images.png", args.samples_per_class)

    summary = {
        "dataset_dir": args.dataset_dir,
        "total_images": len(df),
        "valid_images": int(df["is_valid"].sum()),
        "corrupted_images": len(corrupted_df),
        "small_images": len(small_df),
        "exact_duplicates": len(exact_dups),
        "phash_duplicates": len(phash_dups),
        "clean_images": len(clean_df),
        "clean_train_size": len(clean_train),
        "clean_val_size": len(clean_val),
        "clean_test_size": len(clean_test),
        "min_size": args.min_size,
        "moved_corrupted": moved,
    }
    (splits_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(reports_dir / "eda_report.md", summary, class_counts)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
