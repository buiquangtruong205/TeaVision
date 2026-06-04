import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageOps, UnidentifiedImageError
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.simple_cnn import load_simple_cnn_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate figures that describe the cleaned dataset immediately before training."
    )
    parser.add_argument("--metadata-csv", default="data/splits/eda_metadata.csv")
    parser.add_argument("--train-csv", default="data/splits/clean/train.csv")
    parser.add_argument("--val-csv", default="data/splits/clean/val.csv")
    parser.add_argument("--test-csv", default="data/splits/clean/test.csv")
    parser.add_argument("--output-dir", default="reports/figures/before_train")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--tsne-max-images", type=int, default=1200)
    parser.add_argument(
        "--tsne-checkpoint",
        default="models/cnn_zscore/best_model.pt",
        help="CNN checkpoint produced by run_normalization_training.py.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_csv(path):
    csv_path = resolve_path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing required CSV: {csv_path}")
    return pd.read_csv(csv_path)


def save_figure(fig, output_path):
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def add_bar_labels(ax, fmt="{:.0f}"):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=2, fontsize=8)


def plot_class_distribution(pretrain_df, output_path):
    counts = pretrain_df["class_name"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#2563eb")
    ax.set_title("Class Distribution Before Training")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=30)
    add_bar_labels(ax)
    save_figure(fig, output_path)


def plot_split_distribution(split_df, output_path):
    counts = (
        split_df.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["train", "val", "test"], fill_value=0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(11, 6))
    counts.plot(kind="bar", ax=ax, color=["#2563eb", "#f59e0b", "#16a34a"])
    ax.set_title("Train / Validation / Test Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Split")
    add_bar_labels(ax)
    save_figure(fig, output_path)


def plot_histogram(pretrain_df, column, title, xlabel, output_path):
    values = pd.to_numeric(pretrain_df[column], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.histplot(values, bins=40, kde=True, ax=ax, color="#2563eb")
    ax.axvline(values.mean(), color="#dc2626", linestyle="--", label=f"Mean: {values.mean():.2f}")
    ax.axvline(values.median(), color="#16a34a", linestyle=":", label=f"Median: {values.median():.2f}")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Number of Images")
    ax.legend()
    save_figure(fig, output_path)


def removal_mask(metadata_df):
    bool_columns = [
        "is_valid",
        "is_small",
        "is_exact_duplicate",
        "is_phash_duplicate",
        "is_low_quality",
    ]
    missing = [column for column in bool_columns if column not in metadata_df.columns]
    if missing:
        raise ValueError(f"Metadata CSV is missing removal columns: {', '.join(missing)}")
    flags = metadata_df[bool_columns].copy()
    for column in bool_columns:
        flags[column] = flags[column].astype(bool)
    return (
        ~flags["is_valid"]
        | flags["is_small"]
        | flags["is_exact_duplicate"]
        | flags["is_phash_duplicate"]
        | flags["is_low_quality"]
    )


def plot_removal_rate(metadata_df, output_path):
    plot_df = metadata_df[["class_name"]].copy()
    plot_df["status"] = np.where(removal_mask(metadata_df), "Removed", "Kept")
    counts = plot_df.groupby(["class_name", "status"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=["Kept", "Removed"], fill_value=0).sort_index()
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100.0

    fig, ax = plt.subplots(figsize=(11, 6))
    percentages.plot(kind="bar", stacked=True, ax=ax, color=["#16a34a", "#dc2626"])
    ax.set_title("Image Removal Rate by Class")
    ax.set_xlabel("Class")
    ax.set_ylabel("Percentage of Images (%)")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Status", loc="upper left", bbox_to_anchor=(1.01, 1))
    for container in ax.containers:
        labels = [f"{value:.1f}%" if value >= 4 else "" for value in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", color="white", fontsize=8)
    save_figure(fig, output_path)


def open_rgb(image_path):
    with Image.open(resolve_path(image_path)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def plot_sample_images(pretrain_df, output_path, samples_per_class, seed):
    class_names = sorted(pretrain_df["class_name"].unique())
    fig, axes = plt.subplots(
        len(class_names),
        samples_per_class,
        figsize=(samples_per_class * 2.4, len(class_names) * 2.4),
        squeeze=False,
    )
    for row_index, class_name in enumerate(class_names):
        class_df = pretrain_df[pretrain_df["class_name"] == class_name]
        samples = class_df.sample(min(samples_per_class, len(class_df)), random_state=seed)
        for column_index, ax in enumerate(axes[row_index]):
            ax.axis("off")
            if column_index == 0:
                ax.text(
                    -0.08,
                    0.5,
                    class_name,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="right",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                )
            if column_index >= len(samples):
                continue
            try:
                ax.imshow(open_rgb(samples.iloc[column_index]["image_path"]))
            except (OSError, UnidentifiedImageError):
                ax.text(0.5, 0.5, "Cannot open", ha="center", va="center")
    fig.suptitle("Sample Images from Each Class Before Training", fontsize=14, y=1.01)
    save_figure(fig, output_path)


def balanced_sample(df, max_images, seed):
    if len(df) <= max_images:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    per_class = max(1, max_images // df["class_name"].nunique())
    sampled = [
        group.sample(min(len(group), per_class), random_state=seed)
        for _, group in df.groupby("class_name")
    ]
    return pd.concat(sampled, ignore_index=True).sample(frac=1.0, random_state=seed).head(max_images)


def load_resized_array(image_path, image_size):
    image = open_rgb(image_path).resize((image_size, image_size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def compute_zscore_params(train_df, image_size):
    pixel_sum = np.zeros((image_size, image_size, 3), dtype=np.float64)
    pixel_square_sum = np.zeros_like(pixel_sum)
    count = 0
    for image_path in train_df["image_path"]:
        try:
            image = load_resized_array(image_path, image_size)
        except (OSError, UnidentifiedImageError):
            continue
        pixel_sum += image
        pixel_square_sum += image * image
        count += 1
    if count == 0:
        raise ValueError("No readable train images were found for z-score normalization.")
    mean = pixel_sum / count
    variance = np.maximum(pixel_square_sum / count - mean * mean, 0.0)
    std = np.sqrt(variance)
    std[std < 1e-7] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def extract_cnn_features(sampled_df, train_df, checkpoint_path):
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise RuntimeError(f"PyTorch is required for learned-feature t-SNE: {exc}") from exc

    model, checkpoint = load_simple_cnn_checkpoint(resolve_path(checkpoint_path))
    image_size = int(checkpoint["image_size"])

    normalizer = checkpoint["normalizer"]
    if normalizer != "zscore":
        raise ValueError(f"Learned-feature t-SNE currently supports zscore checkpoints, got: {normalizer}")
    normalization_params = checkpoint.get("normalization_params")
    if normalization_params:
        mean = normalization_params["mean"].numpy()
        std = normalization_params["scale"].numpy()
    else:
        mean, std = compute_zscore_params(train_df, image_size)

    features = []
    labels = []
    batch_images = []
    batch_labels = []

    def flush_batch():
        if not batch_images:
            return
        images = np.stack(batch_images)
        images = (images - mean) / std
        tensor = torch.from_numpy(images.transpose(0, 3, 1, 2)).float()
        with torch.no_grad():
            features.extend(model.forward_features(tensor).numpy())
        labels.extend(batch_labels)
        batch_images.clear()
        batch_labels.clear()

    for _, row in sampled_df.iterrows():
        try:
            batch_images.append(load_resized_array(row["image_path"], image_size))
        except (OSError, UnidentifiedImageError):
            continue
        batch_labels.append(row["class_name"])
        if len(batch_images) >= 64:
            flush_batch()
    flush_batch()
    return np.asarray(features), labels


def plot_tsne(pretrain_df, train_df, output_path, checkpoint_path, max_images, seed):
    sampled_df = balanced_sample(pretrain_df, max_images, seed)
    features, labels = extract_cnn_features(sampled_df, train_df, checkpoint_path)

    if len(features) < 3:
        raise ValueError("At least 3 readable images are required for t-SNE.")

    features = StandardScaler().fit_transform(features)
    n_components = min(50, features.shape[0] - 1, features.shape[1])
    if n_components >= 2:
        features = PCA(n_components=n_components, random_state=seed).fit_transform(features)
    perplexity = min(30, max(2, (len(features) - 1) // 3))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(features)

    plot_df = pd.DataFrame({"t-SNE 1": embedding[:, 0], "t-SNE 2": embedding[:, 1], "class_name": labels})
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(
        data=plot_df,
        x="t-SNE 1",
        y="t-SNE 2",
        hue="class_name",
        palette="tab10",
        s=28,
        alpha=0.8,
        linewidth=0,
        ax=ax,
    )
    ax.set_title(f"t-SNE of CNN Learned Features After Training (n={len(plot_df)})")
    ax.legend(title="Class", loc="upper left", bbox_to_anchor=(1.01, 1))
    save_figure(fig, output_path)


def validate_columns(df, required, name):
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {', '.join(missing)}")


def main():
    args = parse_args()
    if args.samples_per_class < 1:
        raise ValueError("--samples-per-class must be at least 1.")
    if args.tsne_max_images < 3:
        raise ValueError("--tsne-max-images must be at least 3.")

    sns.set_theme(style="whitegrid", context="notebook")
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_df = read_csv(args.metadata_csv)
    split_frames = []
    for split_name, csv_path in [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]:
        split_df = read_csv(csv_path)
        split_df["split"] = split_name
        split_frames.append(split_df)
    pretrain_df = pd.concat(split_frames, ignore_index=True)

    required_columns = ["image_path", "class_name", "width", "height", "brightness_mean", "blur_score"]
    validate_columns(pretrain_df, required_columns, "Clean split CSVs")
    validate_columns(metadata_df, ["class_name"], "EDA metadata CSV")

    plots = [
        ("01_class_distribution.png", lambda path: plot_class_distribution(pretrain_df, path)),
        ("02_train_val_test_distribution.png", lambda path: plot_split_distribution(pretrain_df, path)),
        ("03_histogram_width.png", lambda path: plot_histogram(pretrain_df, "width", "Histogram of Image Width", "Width (pixels)", path)),
        ("04_histogram_height.png", lambda path: plot_histogram(pretrain_df, "height", "Histogram of Image Height", "Height (pixels)", path)),
        ("05_histogram_brightness.png", lambda path: plot_histogram(pretrain_df, "brightness_mean", "Histogram of Image Brightness", "Mean Grayscale Brightness", path)),
        ("06_histogram_sharpness.png", lambda path: plot_histogram(pretrain_df, "blur_score", "Histogram of Image Sharpness", "Laplacian Variance (higher is sharper)", path)),
        ("07_image_removal_rate.png", lambda path: plot_removal_rate(metadata_df, path)),
        ("08_sample_images_by_class.png", lambda path: plot_sample_images(pretrain_df, path, args.samples_per_class, args.seed)),
        (
            "09_tsne.png",
            lambda path: plot_tsne(
                pretrain_df,
                split_frames[0],
                path,
                args.tsne_checkpoint,
                args.tsne_max_images,
                args.seed,
            ),
        ),
    ]
    for filename, plot_function in plots:
        plot_function(output_dir / filename)

    print(f"\nGenerated {len(plots)} figures in: {output_dir}")
    print(f"Clean images before training: {len(pretrain_df)}")
    print(f"Classes: {pretrain_df['class_name'].nunique()}")


if __name__ == "__main__":
    main()
