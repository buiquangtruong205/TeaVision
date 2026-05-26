import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import StratifiedKFold, train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _relative_path(path, base_dir):
    try:
        return path.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def exact_file_hash(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_dhash(image, hash_size=8):
    image = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(image.getdata())
    bits = []
    for row in range(hash_size):
        start = row * (hash_size + 1)
        for col in range(hash_size):
            bits.append(pixels[start + col] > pixels[start + col + 1])
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return f"{value:0{hash_size * hash_size // 4}x}"


def scan_dataset(dataset_dir, include_hashes=True, relative_to=None):
    """Scan a class-folder dataset and return image metadata.

    Expected layout:
        dataset_dir/class_name/image.jpg
    """

    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")

    base_dir = Path(relative_to) if relative_to else Path.cwd()
    class_dirs = sorted(path for path in dataset_path.iterdir() if path.is_dir())
    class_to_id = {path.name: index for index, path in enumerate(class_dirs)}
    rows = []

    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            row = {
                "image_path": _relative_path(image_path, base_dir),
                "label": class_to_id[class_dir.name],
                "class_name": class_dir.name,
                "width": None,
                "height": None,
                "file_size": image_path.stat().st_size,
                "is_valid": False,
                "error": "",
            }

            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    row["width"], row["height"] = image.size
                    row["is_valid"] = True
                    if include_hashes:
                        row["file_hash"] = exact_file_hash(image_path)
                        row["perceptual_hash"] = perceptual_dhash(image)
            except (OSError, UnidentifiedImageError) as exc:
                row["error"] = str(exc)

            rows.append(row)

    return pd.DataFrame(rows)


def create_class_mapping(dataset_dir, output_path):
    class_names = sorted(path.name for path in Path(dataset_dir).iterdir() if path.is_dir())
    mapping = {str(index): class_name for index, class_name in enumerate(class_names)}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    return mapping


def duplicate_report(df):
    reports = {}
    if "file_hash" in df.columns:
        exact = df[df.duplicated("file_hash", keep=False) & df["file_hash"].notna()]
        reports["exact_duplicates"] = exact.sort_values("file_hash")
    if "perceptual_hash" in df.columns:
        near = df[df.duplicated("perceptual_hash", keep=False) & df["perceptual_hash"].notna()]
        reports["near_duplicates"] = near.sort_values("perceptual_hash")
    return reports


def create_stratified_split(df, output_dir, val_size=0.15, test_size=0.15, seed=42):
    if not 0 < val_size < 1 or not 0 < test_size < 1 or val_size + test_size >= 1:
        raise ValueError("val_size and test_size must be positive and sum to less than 1")

    valid_df = df[df["is_valid"].astype(bool)].copy() if "is_valid" in df.columns else df.copy()
    if valid_df.empty:
        raise ValueError("No valid images available for splitting")

    train_val_df, test_df = train_test_split(
        valid_df,
        test_size=test_size,
        stratify=valid_df["label"],
        random_state=seed,
    )
    relative_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df["label"],
        random_state=seed,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_path / "train.csv", index=False)
    val_df.to_csv(output_path / "val.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    return train_df, val_df, test_df


def create_kfold_split(df, output_dir, n_splits=5, seed=42):
    valid_df = df[df["is_valid"].astype(bool)].reset_index(drop=True) if "is_valid" in df.columns else df.reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    folds = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(valid_df, valid_df["label"])):
        train_df = valid_df.iloc[train_idx]
        val_df = valid_df.iloc[val_idx]
        train_df.to_csv(output_path / f"train_fold_{fold}.csv", index=False)
        val_df.to_csv(output_path / f"val_fold_{fold}.csv", index=False)
        folds.append((train_df, val_df))
    return folds


def stratified_split(samples_csv, output_dir, val_size=0.15, test_size=0.15, seed=42):
    df = pd.read_csv(samples_csv)
    return create_stratified_split(df, output_dir, val_size=val_size, test_size=test_size, seed=seed)
