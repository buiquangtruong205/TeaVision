import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import imagehash
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

try:
    import yaml
except ImportError:
    yaml = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data.splitter import IMAGE_EXTENSIONS, create_class_mapping, create_stratified_split, exact_file_hash
from src.data.preprocessing import compute_mean_std_from_csv, save_preprocessing_params


def parse_args():
    parser = argparse.ArgumentParser(description="Run EDA and cleaning checks for image dataset.")
    parser.add_argument("--dataset-dir", default="data/dataset")
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--min-size", type=int, default=None)
    parser.add_argument("--blur-threshold", type=float, default=None)
    parser.add_argument("--dark-threshold", type=float, default=None)
    parser.add_argument("--bright-threshold", type=float, default=None)
    parser.add_argument("--contrast-threshold", type=float, default=None)
    parser.add_argument("--val-size", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=6)
    parser.add_argument("--move-corrupted", action="store_true")
    parser.add_argument("--corrupted-dir", default="data/corrupted")
    return parser.parse_args()


def load_config(config_dir):
    config_path = Path(config_dir) / "config.yaml"
    if not config_path.exists():
        return {}
    if yaml is None:
        return {}
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def config_get(config, section, key, default):
    return config.get(section, {}).get(key, default)


def has_images(dataset_dir):
    path = Path(dataset_dir)
    if not path.exists():
        return False
    return any(item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS for item in path.rglob("*"))


def resolve_dataset_dir(dataset_dir):
    requested = Path(dataset_dir)
    fallback = Path("dataset")
    if has_images(requested):
        return requested
    if requested.as_posix() == "data/dataset" and has_images(fallback):
        print(f"Không tìm thấy ảnh trong {requested}; dùng fallback {fallback}.")
        return fallback
    return requested


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
                "blur_score": None,
                "brightness_mean": None,
                "contrast_std": None,
                "is_blurry": False,
                "is_too_dark": False,
                "is_too_bright": False,
                "is_low_contrast": False,
                "is_low_quality": False,
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
                    array = np.asarray(image)
                    gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
                    row["blur_score"] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    row["brightness_mean"] = float(gray.mean())
                    row["contrast_std"] = float(gray.std())
            except (OSError, UnidentifiedImageError) as exc:
                row["error"] = str(exc)

            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        valid = df["is_valid"].astype(bool)
        df.loc[valid, "is_exact_duplicate"] = df.loc[valid].duplicated("file_hash", keep="first")
        df.loc[valid, "is_phash_duplicate"] = df.loc[valid].duplicated("phash", keep="first")
    return df


def mark_quality_flags(df, blur_threshold, dark_threshold, bright_threshold, contrast_threshold):
    if df.empty:
        return df

    valid = df["is_valid"].astype(bool)
    df.loc[valid, "is_blurry"] = df.loc[valid, "blur_score"] < blur_threshold
    df.loc[valid, "is_too_dark"] = df.loc[valid, "brightness_mean"] < dark_threshold
    df.loc[valid, "is_too_bright"] = df.loc[valid, "brightness_mean"] > bright_threshold
    df.loc[valid, "is_low_contrast"] = df.loc[valid, "contrast_std"] < contrast_threshold
    df.loc[valid, "is_low_quality"] = (
        df.loc[valid, "is_blurry"].astype(bool)
        | df.loc[valid, "is_too_dark"].astype(bool)
        | df.loc[valid, "is_too_bright"].astype(bool)
        | df.loc[valid, "is_low_contrast"].astype(bool)
    )
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


def plot_quality_issues_by_class(summary_df, output_path):
    plot_df = summary_df.set_index("class_name")[
        ["blurry", "too_bright", "too_dark", "low_contrast", "phash_duplicate"]
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", stacked=True, ax=ax)
    ax.set_title("Image Issues by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(
        ["Blurry", "Too bright", "Too dark", "Low contrast", "Perceptual duplicate"],
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_usable_by_class(summary_df, output_path):
    plot_df = summary_df.set_index("class_name")[["usable", "not_used"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", stacked=True, ax=ax, color=["#16a34a", "#dc2626"])
    ax.set_title("Usable vs Removed Images by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(["Usable", "Removed"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_initial_vs_clean_by_class(summary_df, output_path):
    plot_df = summary_df.set_index("class_name")[["total", "usable"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_df.plot(kind="bar", ax=ax, color=["#64748b", "#2563eb"])
    ax.set_title("Initial vs Clean Images by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Images")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(["Initial", "Clean"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_initial_tsne(df, output_path, image_size, max_images=1200, seed=42):
    valid_df = df[df["is_valid"].astype(bool)].copy()
    if len(valid_df) < 3:
        return False
    if len(valid_df) > max_images:
        valid_df = valid_df.sample(max_images, random_state=seed)

    features = []
    labels = []
    for _, row in valid_df.iterrows():
        image_path = PROJECT_ROOT / row["image_path"]
        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
                features.append(np.asarray(image, dtype=np.float32).reshape(-1) / 255.0)
                labels.append(row["class_name"])
        except (OSError, UnidentifiedImageError):
            continue

    if len(features) < 3:
        return False

    features = np.vstack(features)
    features = StandardScaler().fit_transform(features)
    pca_components = min(50, features.shape[0] - 1, features.shape[1])
    if pca_components >= 2:
        features = PCA(n_components=pca_components, random_state=seed).fit_transform(features)

    perplexity = min(30, max(1, (features.shape[0] - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca" if features.shape[1] >= 2 else "random",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    plot_df = pd.DataFrame({"tsne_1": embedding[:, 0], "tsne_2": embedding[:, 1], "class_name": labels})
    class_names = sorted(plot_df["class_name"].unique())
    cmap = plt.get_cmap("tab20", max(len(class_names), 1))

    fig, ax = plt.subplots(figsize=(10, 7))
    for index, class_name in enumerate(class_names):
        class_df = plot_df[plot_df["class_name"] == class_name]
        ax.scatter(
            class_df["tsne_1"],
            class_df["tsne_2"],
            s=18,
            alpha=0.75,
            color=cmap(index),
            label=class_name,
            edgecolors="none",
        )
    ax.set_title("Initial Dataset t-SNE")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def markdown_table(rows, headers):
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    return lines


def write_report(report_path, summary, class_counts, quality_summary):
    report_df = quality_summary.rename(
        columns={
            "stt": "STT",
            "class_name": "Nhãn",
            "total": "Tổng ảnh",
            "blurry": "Ảnh mờ",
            "too_bright": "Quá sáng",
            "too_dark": "Quá tối",
            "low_contrast": "Tương phản thấp",
            "low_quality": "Kém chất lượng",
            "phash_duplicate": "Trùng phash",
            "not_used": "Không dùng được",
            "usable": "Còn dùng được",
        }
    )
    report_total = {"STT": "", "Nhãn": "TỔNG"}
    for column in [
        "Tổng ảnh",
        "Ảnh mờ",
        "Quá sáng",
        "Quá tối",
        "Tương phản thấp",
        "Kém chất lượng",
        "Trùng phash",
        "Không dùng được",
        "Còn dùng được",
    ]:
        report_total[column] = int(report_df[column].sum())
    report_df = pd.concat([report_df, pd.DataFrame([report_total])], ignore_index=True)

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
        f"- Blurry images removed from clean list: {summary['blurry_images']}",
        f"- Too dark images removed from clean list: {summary['too_dark_images']}",
        f"- Too bright images removed from clean list: {summary['too_bright_images']}",
        f"- Low contrast images removed from clean list: {summary['low_contrast_images']}",
        f"- Low quality images removed from clean list: {summary['low_quality_images']}",
        f"- Clean images: {summary['clean_images']}",
        f"- Image size for preprocessing: {summary['image_size']}",
        f"- Train mean: {summary['mean']}",
        f"- Train std: {summary['std']}",
        "",
        "## Quality Thresholds",
        "",
        f"- Blur score threshold: {summary['blur_threshold']}",
        f"- Dark threshold: {summary['dark_threshold']}",
        f"- Bright threshold: {summary['bright_threshold']}",
        f"- Contrast threshold: {summary['contrast_threshold']}",
        "",
        "## Class Distribution",
        "",
    ]
    for class_name, count in class_counts.items():
        lines.append(f"- {class_name}: {count}")
    lines.extend(
        [
            "",
            "## Cleaning Summary by Class",
            "",
        ]
    )
    headers = [
        "STT",
        "Nhãn",
        "Tổng ảnh",
        "Ảnh mờ",
        "Quá sáng",
        "Quá tối",
        "Tương phản thấp",
        "Kém chất lượng",
        "Trùng phash",
        "Không dùng được",
        "Còn dùng được",
    ]
    lines.extend(markdown_table(report_df[headers].to_dict("records"), headers))
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "![Class distribution](figures/class_distribution.png)",
            "",
            "![Image issues by class](figures/quality_issues_by_class.png)",
            "",
            "![Usable vs removed images by class](figures/usable_vs_removed_by_class.png)",
            "",
            "![Initial vs clean images by class](figures/initial_vs_clean_by_class.png)",
            "",
            "![Initial dataset t-SNE](figures/initial_tsne.png)",
            "",
            "![Sample images](figures/sample_images.png)",
        ]
    )
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
            "- `data/splits/blurry_images.csv`",
            "- `data/splits/too_dark_images.csv`",
            "- `data/splits/too_bright_images.csv`",
            "- `data/splits/low_contrast_images.csv`",
            "- `data/splits/low_quality_images.csv`",
            "- `config/preprocessing_params.json`",
            "- `reports/figures/class_distribution.png`",
            "- `reports/figures/quality_issues_by_class.png`",
            "- `reports/figures/usable_vs_removed_by_class.png`",
            "- `reports/figures/initial_vs_clean_by_class.png`",
            "- `reports/figures/initial_tsne.png`",
            "- `reports/figures/sample_images.png`",
            "- `data/splits/clean/train.csv`",
            "- `data/splits/clean/val.csv`",
            "- `data/splits/clean/test.csv`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_quality_summary(df):
    flags = [
        "is_blurry",
        "is_too_dark",
        "is_too_bright",
        "is_low_contrast",
        "is_low_quality",
        "is_phash_duplicate",
        "is_small",
        "is_exact_duplicate",
    ]
    for column in flags:
        df[column] = df[column].astype(bool)

    df["not_used"] = (
        ~df["is_valid"].astype(bool)
        | df["is_small"]
        | df["is_exact_duplicate"]
        | df["is_phash_duplicate"]
        | df["is_low_quality"]
    )

    summary = (
        df.groupby("class_name")
        .agg(
            total=("image_path", "count"),
            blurry=("is_blurry", "sum"),
            too_bright=("is_too_bright", "sum"),
            too_dark=("is_too_dark", "sum"),
            low_contrast=("is_low_contrast", "sum"),
            low_quality=("is_low_quality", "sum"),
            phash_duplicate=("is_phash_duplicate", "sum"),
            not_used=("not_used", "sum"),
            usable=("not_used", lambda values: (~values).sum()),
        )
        .sort_index()
        .reset_index()
    )
    summary.insert(0, "stt", range(1, len(summary) + 1))
    return summary


def print_quality_summary(summary_df):
    initial_df = summary_df[["stt", "class_name", "total"]].rename(
        columns={
            "stt": "STT",
            "class_name": "Nhãn",
            "total": "Tổng ảnh ban đầu",
        }
    )
    initial_total = pd.DataFrame(
        [
            {
                "STT": "",
                "Nhãn": "TỔNG",
                "Tổng ảnh ban đầu": int(initial_df["Tổng ảnh ban đầu"].sum()),
            }
        ]
    )
    initial_df = pd.concat([initial_df, initial_total], ignore_index=True)
    print("\nBẢNG DỮ LIỆU BAN ĐẦU THEO NHÃN")
    print(initial_df.to_string(index=False))

    display_df = summary_df.rename(
        columns={
            "stt": "STT",
            "class_name": "Nhãn",
            "total": "Tổng ảnh",
            "blurry": "Ảnh mờ",
            "too_bright": "Quá sáng",
            "too_dark": "Quá tối",
            "low_contrast": "Tương phản thấp",
            "low_quality": "Kém chất lượng",
            "phash_duplicate": "Trùng phash",
            "not_used": "Không dùng được",
            "usable": "Còn dùng được",
        }
    )
    numeric_columns = [
        "Tổng ảnh",
        "Ảnh mờ",
        "Quá sáng",
        "Quá tối",
        "Tương phản thấp",
        "Kém chất lượng",
        "Trùng phash",
        "Không dùng được",
        "Còn dùng được",
    ]
    quality_total = {"STT": "", "Nhãn": "TỔNG"}
    for column in numeric_columns:
        quality_total[column] = int(display_df[column].sum())
    display_df = pd.concat([display_df, pd.DataFrame([quality_total])], ignore_index=True)
    print("\nBẢNG TỔNG HỢP CHẤT LƯỢNG ẢNH THEO NHÃN")
    print(display_df.to_string(index=False))


def print_output_paths(paths):
    print("\nĐƯỜNG DẪN OUTPUT")
    for label, path in paths:
        print(f"- {label}: {path}")


def main(args):
    config = load_config(args.config_dir)
    args.dataset_dir = resolve_dataset_dir(args.dataset_dir)
    args.image_size = args.image_size or config_get(config, "data", "image_size", 224)
    args.min_size = args.min_size or config_get(config, "preprocessing", "min_size", 32)
    args.blur_threshold = args.blur_threshold or config_get(config, "preprocessing", "blur_threshold", 80.0)
    args.dark_threshold = args.dark_threshold or config_get(config, "preprocessing", "dark_threshold", 40.0)
    args.bright_threshold = args.bright_threshold or config_get(config, "preprocessing", "bright_threshold", 220.0)
    args.contrast_threshold = args.contrast_threshold or config_get(config, "preprocessing", "contrast_threshold", 25.0)
    args.val_size = args.val_size or config_get(config, "data", "val_size", 0.15)
    args.test_size = args.test_size or config_get(config, "data", "test_size", 0.15)
    args.seed = args.seed or config_get(config, "project", "seed", 42)

    splits_dir = Path(args.splits_dir)
    reports_dir = Path(args.reports_dir)
    figures_dir = reports_dir / "figures"
    splits_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    create_class_mapping(args.dataset_dir, Path(args.config_dir) / "class_mapping.json")
    df = scan_images(args.dataset_dir, args.min_size)
    df = mark_quality_flags(
        df,
        blur_threshold=args.blur_threshold,
        dark_threshold=args.dark_threshold,
        bright_threshold=args.bright_threshold,
        contrast_threshold=args.contrast_threshold,
    )

    corrupted_df = df[~df["is_valid"].astype(bool)].copy()
    small_df = df[df["is_valid"].astype(bool) & df["is_small"].astype(bool)].copy()
    exact_dups = df[df["is_valid"].astype(bool) & df["is_exact_duplicate"].astype(bool)].copy()
    phash_dups = df[df["is_valid"].astype(bool) & df["is_phash_duplicate"].astype(bool)].copy()
    blurry_df = df[df["is_valid"].astype(bool) & df["is_blurry"].astype(bool)].copy()
    too_dark_df = df[df["is_valid"].astype(bool) & df["is_too_dark"].astype(bool)].copy()
    too_bright_df = df[df["is_valid"].astype(bool) & df["is_too_bright"].astype(bool)].copy()
    low_contrast_df = df[df["is_valid"].astype(bool) & df["is_low_contrast"].astype(bool)].copy()
    low_quality_df = df[df["is_valid"].astype(bool) & df["is_low_quality"].astype(bool)].copy()

    clean_df = df[
        df["is_valid"].astype(bool)
        & ~df["is_small"].astype(bool)
        & ~df["is_exact_duplicate"].astype(bool)
        & ~df["is_phash_duplicate"].astype(bool)
        & ~df["is_low_quality"].astype(bool)
    ].copy()

    df.to_csv(splits_dir / "eda_metadata.csv", index=False)
    clean_df.to_csv(splits_dir / "cleaned_metadata.csv", index=False)
    corrupted_df.to_csv(splits_dir / "corrupted_images.csv", index=False)
    small_df.to_csv(splits_dir / "small_images.csv", index=False)
    exact_dups.to_csv(splits_dir / "exact_duplicates.csv", index=False)
    phash_dups.to_csv(splits_dir / "phash_duplicates.csv", index=False)
    blurry_df.to_csv(splits_dir / "blurry_images.csv", index=False)
    too_dark_df.to_csv(splits_dir / "too_dark_images.csv", index=False)
    too_bright_df.to_csv(splits_dir / "too_bright_images.csv", index=False)
    low_contrast_df.to_csv(splits_dir / "low_contrast_images.csv", index=False)
    low_quality_df.to_csv(splits_dir / "low_quality_images.csv", index=False)

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
    mean, std = compute_mean_std_from_csv(
        clean_split_dir / "train.csv",
        image_size=args.image_size,
        root_dir=PROJECT_ROOT,
    )
    preprocessing_params = save_preprocessing_params(
        Path(args.config_dir) / "preprocessing_params.json",
        image_size=args.image_size,
        mean=mean,
        std=std,
    )

    class_counts = plot_class_distribution(df, figures_dir / "class_distribution.png")
    plot_sample_images(clean_df, figures_dir / "sample_images.png", args.samples_per_class)
    quality_summary = build_quality_summary(df.copy())
    quality_summary.to_csv(splits_dir / "quality_summary_by_class.csv", index=False)
    plot_quality_issues_by_class(quality_summary, figures_dir / "quality_issues_by_class.png")
    plot_usable_by_class(quality_summary, figures_dir / "usable_vs_removed_by_class.png")
    plot_initial_vs_clean_by_class(quality_summary, figures_dir / "initial_vs_clean_by_class.png")
    has_initial_tsne = plot_initial_tsne(
        df,
        figures_dir / "initial_tsne.png",
        image_size=min(args.image_size, 64),
        seed=args.seed,
    )

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "total_images": len(df),
        "valid_images": int(df["is_valid"].sum()),
        "corrupted_images": len(corrupted_df),
        "small_images": len(small_df),
        "exact_duplicates": len(exact_dups),
        "phash_duplicates": len(phash_dups),
        "blurry_images": len(blurry_df),
        "too_dark_images": len(too_dark_df),
        "too_bright_images": len(too_bright_df),
        "low_contrast_images": len(low_contrast_df),
        "low_quality_images": len(low_quality_df),
        "clean_images": len(clean_df),
        "clean_train_size": len(clean_train),
        "clean_val_size": len(clean_val),
        "clean_test_size": len(clean_test),
        "image_size": args.image_size,
        "mean": mean,
        "std": std,
        "min_size": args.min_size,
        "blur_threshold": args.blur_threshold,
        "dark_threshold": args.dark_threshold,
        "bright_threshold": args.bright_threshold,
        "contrast_threshold": args.contrast_threshold,
        "moved_corrupted": moved,
        "initial_tsne": str(figures_dir / "initial_tsne.png") if has_initial_tsne else None,
    }
    (splits_dir / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(reports_dir / "eda_report.md", summary, class_counts, quality_summary)
    print_quality_summary(quality_summary)
    print_output_paths(
        [
            ("Báo cáo EDA", reports_dir / "eda_report.md"),
            ("Biểu đồ số ảnh mỗi lớp", figures_dir / "class_distribution.png"),
            ("Biểu đồ lỗi theo nhãn", figures_dir / "quality_issues_by_class.png"),
            ("Biểu đồ dùng được/không dùng được", figures_dir / "usable_vs_removed_by_class.png"),
            ("Biểu đồ trước/sau làm sạch", figures_dir / "initial_vs_clean_by_class.png"),
            ("Biểu đồ t-SNE dữ liệu ban đầu", figures_dir / "initial_tsne.png"),
            ("Ảnh mẫu mỗi lớp", figures_dir / "sample_images.png"),
            ("Metadata đầy đủ", splits_dir / "eda_metadata.csv"),
            ("Metadata sạch", splits_dir / "cleaned_metadata.csv"),
            ("Tổng hợp theo nhãn", splits_dir / "quality_summary_by_class.csv"),
            ("Tham số preprocessing", Path(args.config_dir) / "preprocessing_params.json"),
            ("Ảnh kém chất lượng", splits_dir / "low_quality_images.csv"),
            ("Ảnh mờ", splits_dir / "blurry_images.csv"),
            ("Ảnh quá sáng", splits_dir / "too_bright_images.csv"),
            ("Ảnh trùng phash", splits_dir / "phash_duplicates.csv"),
            ("Train sạch", clean_split_dir / "train.csv"),
            ("Val sạch", clean_split_dir / "val.csv"),
            ("Test sạch", clean_split_dir / "test.csv"),
        ]
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args)
