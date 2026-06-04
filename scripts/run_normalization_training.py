import argparse
import copy
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.simple_cnn import SimpleCNN


IMAGE_NET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_NET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
ALL_NORMALIZERS = ["minmax", "zscore", "robust", "imagenet"]
ALL_MODELS = ["knn", "svm", "random_forest", "xgboost", "cnn"]


def progress(iterable=None, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs) if iterable is not None else tqdm(**kwargs)


@dataclass
class ExperimentResult:
    normalizer: str
    model_name: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float
    parameters: int
    gflops: float | None
    inference_time_ms: float
    confusion: list
    per_class: pd.DataFrame


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare normalization methods and training models on cleaned TeaVision splits."
    )
    parser.add_argument("--train-csv", default="data/splits/clean/train.csv")
    parser.add_argument("--val-csv", default="data/splits/clean/val.csv")
    parser.add_argument("--test-csv", default="data/splits/clean/test.csv")
    parser.add_argument("--class-mapping", default="config/class_mapping.json")
    parser.add_argument("--eval-split", choices=["val", "test"], default="val")
    parser.add_argument("--output-dir", default="reports/normalization_training")
    parser.add_argument("--image-size", type=int, default=128, help="Feature image size for KNN/SVM/CNN experiments.")
    parser.add_argument("--normalizers", nargs="+", choices=ALL_NORMALIZERS, default=ALL_NORMALIZERS)
    parser.add_argument("--preprocessing-params", default="config/preprocessing_params.json")
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Run every default normalizer/model pair, overriding --normalizers and --models.",
    )
    parser.add_argument("--knn-neighbors", type=int, default=5)
    parser.add_argument("--svm-c", type=float, default=1.0)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--rf-max-depth", type=int, default=None)
    parser.add_argument("--xgb-estimators", type=int, default=300)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.1)
    parser.add_argument("--cnn-epochs", type=int, default=30)
    parser.add_argument("--cnn-batch-size", type=int, default=64)
    parser.add_argument("--cnn-patience", type=int, default=3)
    parser.add_argument("--cnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--cnn-weight-decay", type=float, default=0.0)
    parser.add_argument("--cnn-dropout", type=float, default=0.0)
    parser.add_argument("--cnn-label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--cnn-lr-factor",
        type=float,
        default=1.0,
        help="ReduceLROnPlateau factor. Set below 1.0 to enable learning-rate reduction.",
    )
    parser.add_argument("--cnn-lr-patience", type=int, default=2)
    parser.add_argument(
        "--torch-num-threads",
        type=int,
        default=os.cpu_count() or 1,
        help="CPU threads for PyTorch CNN training. Use a smaller value if your machine becomes unresponsive.",
    )
    parser.add_argument(
        "--max-total-samples",
        type=int,
        default=0,
        help="Maximum total samples across train/val/test while preserving split ratios. Set 0 to use all.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Override train sample limit. By default this is derived from --max-total-samples.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Override validation/test sample limit. By default this is derived from --max-total-samples.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_class_names(mapping_path):
    mapping = json.loads(resolve_path(mapping_path).read_text(encoding="utf-8"))
    return [mapping[str(index)] for index in sorted(map(int, mapping.keys()))]


def load_preprocessing_params(params_path):
    path = resolve_path(params_path)
    if not path.exists():
        return None, None

    params = json.loads(path.read_text(encoding="utf-8"))
    mean = np.array(params["mean"], dtype=np.float32)
    std = np.array(params["std"], dtype=np.float32)
    return mean, std


def read_split(csv_path, max_samples=None, seed=42):
    df = pd.read_csv(resolve_path(csv_path))
    if max_samples == 0:
        max_samples = None
    if max_samples is not None and len(df) > max_samples:
        samples_per_class = max(1, max_samples // df["label"].nunique())
        sampled_groups = []
        for _, group in df.groupby("label"):
            sampled_groups.append(group.sample(min(len(group), samples_per_class), random_state=seed))
        df = pd.concat(sampled_groups, ignore_index=True)
        df = df.sample(frac=1.0, random_state=seed).head(max_samples)
    return df.reset_index(drop=True)


def split_sample_limits(train_csv, val_csv, test_csv, max_total_samples):
    if max_total_samples is None or max_total_samples == 0:
        return None, None, None

    split_sizes = [
        len(pd.read_csv(resolve_path(train_csv))),
        len(pd.read_csv(resolve_path(val_csv))),
        len(pd.read_csv(resolve_path(test_csv))),
    ]
    total_size = sum(split_sizes)
    if total_size == 0 or total_size <= max_total_samples:
        return None, None, None

    ratio = max_total_samples / total_size
    limits = [max(1, int(round(size * ratio))) if size else 0 for size in split_sizes]

    while sum(limits) > max_total_samples:
        index = max(range(len(limits)), key=lambda item: limits[item])
        limits[index] -= 1
    while sum(limits) < max_total_samples:
        index = max(range(len(limits)), key=lambda item: split_sizes[item] - limits[item])
        if limits[index] >= split_sizes[index]:
            break
        limits[index] += 1

    return tuple(limits)


def load_images(df, image_size, split_name="images"):
    images = np.empty((len(df), image_size, image_size, 3), dtype=np.float32)
    labels = df["label"].to_numpy(dtype=np.int64)
    image_paths = progress(
        enumerate(df["image_path"]),
        total=len(df),
        desc=f"Loading {split_name}",
        unit="img",
        leave=False,
    )
    for index, image_path in image_paths:
        path = resolve_path(image_path)
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.LANCZOS)
            images[index] = np.asarray(image, dtype=np.float32) / 255.0
    return images, labels


def flatten(images):
    return images.reshape(images.shape[0], -1)


def normalize_features(normalizer, train_images, eval_images, dataset_mean=None, dataset_std=None):
    print(f"[{normalizer}] Preparing feature tensors...")
    if normalizer == "minmax":
        return flatten(train_images), flatten(eval_images)
    if normalizer == "dataset":
        if dataset_mean is None or dataset_std is None:
            raise ValueError("Normalizer 'dataset' requires --preprocessing-params with mean/std.")
        train = (train_images - dataset_mean) / dataset_std
        eval_data = (eval_images - dataset_mean) / dataset_std
        return flatten(train), flatten(eval_data)
    if normalizer == "imagenet":
        train = (train_images - IMAGE_NET_MEAN) / IMAGE_NET_STD
        eval_data = (eval_images - IMAGE_NET_MEAN) / IMAGE_NET_STD
        return flatten(train), flatten(eval_data)

    scaler_cls = {"zscore": StandardScaler, "robust": RobustScaler}.get(normalizer)
    if scaler_cls is None:
        raise ValueError(f"Unsupported normalizer: {normalizer}")

    scaler = scaler_cls()
    train = scaler.fit_transform(flatten(train_images))
    eval_data = scaler.transform(flatten(eval_images))
    return train, eval_data


def normalize_images_for_cnn(normalizer, train_images, val_images, eval_images, dataset_mean=None, dataset_std=None):
    if normalizer == "minmax":
        return train_images, val_images, eval_images
    if normalizer == "dataset":
        if dataset_mean is None or dataset_std is None:
            raise ValueError("Normalizer 'dataset' requires --preprocessing-params with mean/std.")
        return (
            (train_images - dataset_mean) / dataset_std,
            (val_images - dataset_mean) / dataset_std,
            (eval_images - dataset_mean) / dataset_std,
        )
    if normalizer == "imagenet":
        return (
            (train_images - IMAGE_NET_MEAN) / IMAGE_NET_STD,
            (val_images - IMAGE_NET_MEAN) / IMAGE_NET_STD,
            (eval_images - IMAGE_NET_MEAN) / IMAGE_NET_STD,
        )

    train_flat, val_flat = normalize_features(normalizer, train_images, val_images, dataset_mean, dataset_std)
    _, eval_flat = normalize_features(normalizer, train_images, eval_images, dataset_mean, dataset_std)
    shape = train_images.shape[1:]
    return train_flat.reshape((-1, *shape)), val_flat.reshape((-1, *shape)), eval_flat.reshape((-1, *shape))


def build_cnn_normalization_params(normalizer, train_images, dataset_mean=None, dataset_std=None):
    import torch

    if normalizer == "zscore":
        mean = train_images.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = train_images.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[scale == 0.0] = 1.0
        return {"mean": torch.from_numpy(mean), "scale": torch.from_numpy(scale)}
    if normalizer == "dataset":
        return {
            "mean": torch.from_numpy(np.asarray(dataset_mean, dtype=np.float32)),
            "scale": torch.from_numpy(np.asarray(dataset_std, dtype=np.float32)),
        }
    if normalizer == "imagenet":
        return {
            "mean": torch.from_numpy(IMAGE_NET_MEAN.copy()),
            "scale": torch.from_numpy(IMAGE_NET_STD.copy()),
        }
    return {}


def evaluate_predictions(normalizer, model_name, y_true, y_pred, class_names, parameters, gflops, inference_time_ms):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": per_class_precision,
            "recall": per_class_recall,
            "f1_score": per_class_f1,
            "support": per_class_support,
        }
    )
    return ExperimentResult(
        normalizer=normalizer,
        model_name=model_name,
        accuracy=accuracy_score(y_true, y_pred),
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        f1_macro=f1_macro,
        precision_weighted=precision_weighted,
        recall_weighted=recall_weighted,
        f1_weighted=f1_weighted,
        parameters=int(parameters),
        gflops=None if gflops is None else float(gflops),
        inference_time_ms=float(inference_time_ms),
        confusion=confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        per_class=per_class,
    )


def train_knn(normalizer, x_train, y_train, x_test, y_test, class_names, neighbors):
    print(f"[{normalizer} + knn] Fitting KNN...")
    model = KNeighborsClassifier(n_neighbors=neighbors)
    model.fit(x_train, y_train)
    print(f"[{normalizer} + knn] Predicting...")
    start = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(x_test))

    parameters = x_train.size
    gflops = None
    return evaluate_predictions(normalizer, "knn", y_test, y_pred, class_names, parameters, gflops, inference_ms)


def train_svm(normalizer, x_train, y_train, x_test, y_test, class_names, c_value):
    print(f"[{normalizer} + svm] Fitting LinearSVC...")
    model = LinearSVC(C=c_value, class_weight="balanced", max_iter=5000, random_state=42)
    model.fit(x_train, y_train)
    print(f"[{normalizer} + svm] Predicting...")
    start = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(x_test))

    parameters = model.coef_.size + model.intercept_.size
    gflops = None
    return evaluate_predictions(normalizer, "svm", y_test, y_pred, class_names, parameters, gflops, inference_ms)


def train_random_forest(normalizer, x_train, y_train, x_test, y_test, class_names, args):
    print(f"[{normalizer} + random_forest] Fitting RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        max_depth=args.rf_max_depth,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(x_train, y_train)
    print(f"[{normalizer} + random_forest] Predicting...")
    start = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(x_test))

    parameters = sum(tree.tree_.node_count for tree in model.estimators_)
    gflops = None
    return evaluate_predictions(
        normalizer, "random_forest", y_test, y_pred, class_names, parameters, gflops, inference_ms
    )


def train_xgboost(normalizer, x_train, y_train, x_test, y_test, class_names, args):
    try:
        from xgboost import XGBClassifier
    except (ImportError, OSError) as exc:
        print(f"\n[SKIP] xgboost + {normalizer}: cannot import XGBoost ({exc})")
        return None

    print(f"[{normalizer} + xgboost] Fitting XGBClassifier...")
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(class_names),
        n_estimators=args.xgb_estimators,
        max_depth=args.xgb_max_depth,
        learning_rate=args.xgb_learning_rate,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=args.seed,
    )
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, sample_weight=sample_weights)
    print(f"[{normalizer} + xgboost] Predicting...")
    start = time.perf_counter()
    y_pred = model.predict(x_test)
    inference_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(x_test))

    parameters = len(model.get_booster().get_dump())
    gflops = None
    return evaluate_predictions(normalizer, "xgboost", y_test, y_pred, class_names, parameters, gflops, inference_ms)


def train_cnn(
    normalizer,
    train_images,
    y_train,
    val_images,
    y_val,
    eval_images,
    y_eval,
    class_names,
    args,
    normalization_params,
):
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except (ImportError, OSError) as exc:
        print(f"\n[SKIP] cnn + {normalizer}: cannot import PyTorch ({exc})")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(max(1, args.torch_num_threads))
    torch.manual_seed(args.seed)
    model = SimpleCNN(len(class_names), image_size=args.image_size, dropout=args.cnn_dropout).to(device)
    classes = np.arange(len(class_names))
    class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=args.cnn_label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.cnn_learning_rate,
        weight_decay=args.cnn_weight_decay,
    )
    scheduler = None
    if args.cnn_lr_factor < 1.0:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.cnn_lr_factor,
            patience=args.cnn_lr_patience,
        )

    x_train = torch.from_numpy(train_images.transpose(0, 3, 1, 2)).float()
    x_val = torch.from_numpy(val_images.transpose(0, 3, 1, 2)).float()
    x_eval = torch.from_numpy(eval_images.transpose(0, 3, 1, 2)).float()
    train_dataset = TensorDataset(x_train, torch.from_numpy(y_train).long())
    val_dataset = TensorDataset(x_val, torch.from_numpy(y_val).long())
    eval_dataset = TensorDataset(x_eval, torch.from_numpy(y_eval).long())
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(train_dataset, batch_size=args.cnn_batch_size, shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.cnn_batch_size, shuffle=False, pin_memory=pin_memory)
    eval_loader = DataLoader(eval_dataset, batch_size=args.cnn_batch_size, shuffle=False, pin_memory=pin_memory)

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    epoch_bar = progress(
        range(args.cnn_epochs),
        desc=f"{normalizer} + cnn epochs",
        unit="epoch",
    )
    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        train_batches = progress(
            train_loader,
            desc=f"{normalizer} epoch {epoch + 1} train",
            unit="batch",
            leave=False,
        )
        for images, labels in train_batches:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_batches = progress(
                val_loader,
                desc=f"{normalizer} epoch {epoch + 1} val",
                unit="batch",
                leave=False,
            )
            for images, labels in val_batches:
                images = images.to(device, non_blocking=pin_memory)
                labels = labels.to(device, non_blocking=pin_memory)
                val_loss += criterion(model(images), labels).item() * images.size(0)
        train_loss = total_loss / len(train_dataset)
        val_loss = val_loss / len(val_dataset)
        if scheduler is not None:
            scheduler.step(val_loss)
        learning_rate = optimizer.param_groups[0]["lr"]
        print(
            f"cnn + {normalizer} epoch {epoch + 1}/{args.cnn_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} lr={learning_rate:.6g}"
        )
        if tqdm is not None:
            epoch_bar.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                lr=f"{learning_rate:.2g}",
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.cnn_patience:
                print(f"cnn + {normalizer} early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    checkpoint_path = resolve_path(args.output_dir) / f"best_{normalizer}_cnn.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": normalizer,
            "image_size": args.image_size,
            "cnn_dropout": args.cnn_dropout,
            "class_names": class_names,
            "normalization_params": normalization_params,
        },
        checkpoint_path,
    )
    print(f"Saved CNN checkpoint: {checkpoint_path}")

    y_pred = []
    model.eval()
    start = time.perf_counter()
    with torch.no_grad():
        eval_batches = progress(eval_loader, desc=f"{normalizer} + cnn eval", unit="batch", leave=False)
        for images, _ in eval_batches:
            logits = model(images.to(device, non_blocking=pin_memory))
            y_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
    inference_ms = (time.perf_counter() - start) * 1000.0 / max(1, len(eval_dataset))

    parameters = sum(param.numel() for param in model.parameters())
    gflops = estimate_simple_cnn_gflops(args.image_size, len(class_names))
    result = evaluate_predictions(
        normalizer,
        "cnn",
        y_eval,
        np.array(y_pred),
        class_names,
        parameters,
        gflops,
        inference_ms,
    )
    artifact = {
        "artifact_type": "teavision_simple_cnn",
        "checkpoint": checkpoint_path.name,
        "architecture": "SimpleCNN",
        "normalizer": normalizer,
        "image_size": args.image_size,
        "class_names": class_names,
        "feature_dimension": 128,
        "validation_metrics": {
            "accuracy": result.accuracy,
            "f1_macro": result.f1_macro,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
        },
    }
    artifact_path = checkpoint_path.with_name(f"{normalizer}_cnn_artifact.json")
    artifact_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"Saved CNN artifact manifest: {artifact_path}")
    return result


def estimate_simple_cnn_gflops(image_size, num_classes):
    conv1 = image_size * image_size * 16 * (3 * 3 * 3) * 2
    conv2_size = image_size // 2
    conv2 = conv2_size * conv2_size * 32 * (3 * 3 * 16) * 2
    conv3_size = image_size // 4
    conv3 = conv3_size * conv3_size * 64 * (3 * 3 * 32) * 2
    reduced = image_size // 8
    fc1 = (64 * reduced * reduced) * 128 * 2
    fc2 = 128 * num_classes * 2
    return (conv1 + conv2 + conv3 + fc1 + fc2) / 1e9


def print_result(result):
    print("\n" + "=" * 92)
    print(f"GROUP: normalization={result.normalizer} | model={result.model_name}")
    print("-" * 92)
    print(f"Accuracy           : {result.accuracy:.4f}")
    print(f"Precision macro    : {result.precision_macro:.4f}")
    print(f"Recall macro       : {result.recall_macro:.4f}")
    print(f"F1-Score macro     : {result.f1_macro:.4f}")
    print(f"Precision weighted : {result.precision_weighted:.4f}")
    print(f"Recall weighted    : {result.recall_weighted:.4f}")
    print(f"F1-Score weighted  : {result.f1_weighted:.4f}")
    print(f"Parameters         : {result.parameters:,}")
    gflops = "N/A" if result.gflops is None else f"{result.gflops:.6f}"
    print(f"GFLOPs / image     : {gflops}")
    print(f"Inference Time     : {result.inference_time_ms:.3f} ms/image")
    print("\nPer-class metrics:")
    print(result.per_class.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nConfusion Matrix:")
    print(np.array(result.confusion))


def print_summary(results):
    if not results:
        print("\nNo experiment result was produced.")
        return
    rows = [
        {
            "normalizer": result.normalizer,
            "model": result.model_name,
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "parameters": result.parameters,
            "gflops": result.gflops,
            "inference_ms": result.inference_time_ms,
        }
        for result in results
    ]
    summary = pd.DataFrame(rows).sort_values(["f1_macro", "accuracy"], ascending=False)
    summary["gflops"] = summary["gflops"].apply(lambda value: "N/A" if pd.isna(value) else f"{value:.6f}")
    print("\n" + "=" * 92)
    print("SUMMARY: all groups sorted by F1-Score macro")
    print("=" * 92)
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def save_results(results, output_dir):
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rows = []
    details = []
    for result in results:
        row = {
            "normalizer": result.normalizer,
            "model": result.model_name,
            "accuracy": result.accuracy,
            "precision_macro": result.precision_macro,
            "recall_macro": result.recall_macro,
            "f1_macro": result.f1_macro,
            "precision_weighted": result.precision_weighted,
            "recall_weighted": result.recall_weighted,
            "f1_weighted": result.f1_weighted,
            "parameters": result.parameters,
            "gflops": result.gflops,
            "inference_time_ms": result.inference_time_ms,
        }
        rows.append(row)
        details.append(
            {
                **row,
                "confusion_matrix": result.confusion,
                "per_class": result.per_class.to_dict("records"),
            }
        )

    pd.DataFrame(rows).to_csv(output / "summary.csv", index=False)
    (output / "details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(f"\nSaved results:")
    print(f"- {output / 'summary.csv'}")
    print(f"- {output / 'details.json'}")


def main():
    args = parse_args()
    if not 0.0 <= args.cnn_dropout < 1.0:
        raise ValueError("--cnn-dropout must be in [0, 1).")
    if not 0.0 <= args.cnn_label_smoothing < 1.0:
        raise ValueError("--cnn-label-smoothing must be in [0, 1).")
    if args.cnn_learning_rate <= 0.0:
        raise ValueError("--cnn-learning-rate must be positive.")
    if args.cnn_weight_decay < 0.0:
        raise ValueError("--cnn-weight-decay cannot be negative.")
    if not 0.0 < args.cnn_lr_factor <= 1.0:
        raise ValueError("--cnn-lr-factor must be in (0, 1].")
    if args.all_pairs:
        args.normalizers = ALL_NORMALIZERS.copy()
        args.models = ALL_MODELS.copy()
    np.random.seed(args.seed)
    class_names = load_class_names(args.class_mapping)
    dataset_mean, dataset_std = load_preprocessing_params(args.preprocessing_params)

    print("Loading cleaned splits...")
    train_limit, val_limit, test_limit = split_sample_limits(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        args.max_total_samples,
    )
    if args.max_train_samples is not None:
        train_limit = args.max_train_samples
    if args.max_eval_samples is not None:
        val_limit = args.max_eval_samples
        test_limit = args.max_eval_samples

    train_df = read_split(args.train_csv, max_samples=train_limit, seed=args.seed)
    val_df = read_split(args.val_csv, max_samples=val_limit, seed=args.seed)
    test_df = read_split(args.test_csv, max_samples=test_limit, seed=args.seed)
    eval_df = val_df if args.eval_split == "val" else test_df

    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)} | Test samples: {len(test_df)}")
    print(f"Evaluation split: {args.eval_split}")
    if args.eval_split == "test":
        print("WARNING: test split should only be used once after model/normalizer selection.")
    print(f"Classes: {', '.join(class_names)}")
    print(f"Image size for experiments: {args.image_size}x{args.image_size}")
    print(f"Normalizers: {', '.join(args.normalizers)}")
    print(f"Models: {', '.join(args.models)}")
    experiment_count = len(args.normalizers) * len(args.models)
    print(f"Experiment groups: {experiment_count}")
    all_pair_count = len(ALL_NORMALIZERS) * len(ALL_MODELS)
    if experiment_count < all_pair_count:
        print(
            f"WARNING: This command is running fewer than {all_pair_count} pairs. "
            "Remove --normalizers/--models or add --all-pairs to run the full comparison."
        )
    print(f"CNN batch size: {args.cnn_batch_size} | PyTorch CPU threads: {args.torch_num_threads}")
    print(
        "CNN optimizer: "
        f"lr={args.cnn_learning_rate:g} | weight_decay={args.cnn_weight_decay:g} | "
        f"dropout={args.cnn_dropout:g} | label_smoothing={args.cnn_label_smoothing:g} | "
        f"lr_factor={args.cnn_lr_factor:g}"
    )
    if dataset_mean is not None and dataset_std is not None:
        print(f"Dataset mean/std: {dataset_mean.tolist()} / {dataset_std.tolist()}")

    train_images, y_train = load_images(train_df, args.image_size, split_name="train")
    val_images, y_val = load_images(val_df, args.image_size, split_name="val")
    if args.eval_split == "val":
        eval_images, y_eval = val_images, y_val
    else:
        eval_images, y_eval = load_images(eval_df, args.image_size, split_name=args.eval_split)

    results = []
    normalizer_bar = progress(args.normalizers, desc="Normalization groups", unit="group")
    total_experiments = len(args.normalizers) * len(args.models)
    experiment_index = 0
    experiment_bar = progress(
        total=total_experiments,
        desc="Experiment pairs",
        unit="pair",
        position=1,
    ) if tqdm is not None else None

    def start_experiment(normalizer, model_name):
        nonlocal experiment_index
        experiment_index += 1
        label = f"{normalizer} + {model_name}"
        print("\n" + "-" * 92)
        print(f"EXPERIMENT PAIR {experiment_index}/{total_experiments}: {label}")
        print("-" * 92)
        if experiment_bar is not None:
            experiment_bar.set_postfix(pair=label)

    def finish_experiment():
        if experiment_bar is not None:
            experiment_bar.update(1)

    for normalizer in normalizer_bar:
        if tqdm is not None:
            normalizer_bar.set_postfix(group=normalizer)
        print("\n" + "#" * 92)
        print(f"Normalization group: {normalizer}")
        print("#" * 92)

        x_train = None
        x_eval = None
        feature_models = {"knn", "svm", "random_forest", "xgboost"}
        if feature_models.intersection(args.models):
            x_train, x_eval = normalize_features(normalizer, train_images, eval_images, dataset_mean, dataset_std)
            if "knn" in args.models:
                start_experiment(normalizer, "knn")
                result = train_knn(normalizer, x_train, y_train, x_eval, y_eval, class_names, args.knn_neighbors)
                results.append(result)
                print_result(result)
                finish_experiment()
            if "random_forest" in args.models:
                start_experiment(normalizer, "random_forest")
                result = train_random_forest(normalizer, x_train, y_train, x_eval, y_eval, class_names, args)
                results.append(result)
                print_result(result)
                finish_experiment()
            if "xgboost" in args.models:
                start_experiment(normalizer, "xgboost")
                result = train_xgboost(normalizer, x_train, y_train, x_eval, y_eval, class_names, args)
                if result is not None:
                    results.append(result)
                    print_result(result)
                finish_experiment()
            if "svm" in args.models:
                start_experiment(normalizer, "svm")
                result = train_svm(normalizer, x_train, y_train, x_eval, y_eval, class_names, args.svm_c)
                results.append(result)
                print_result(result)
                finish_experiment()
        if "cnn" in args.models:
            start_experiment(normalizer, "cnn")
            normalization_params = build_cnn_normalization_params(
                normalizer,
                train_images,
                dataset_mean,
                dataset_std,
            )
            cnn_train, cnn_val, cnn_eval = normalize_images_for_cnn(
                normalizer,
                train_images,
                val_images,
                eval_images,
                dataset_mean,
                dataset_std,
            )
            result = train_cnn(
                normalizer,
                cnn_train,
                y_train,
                cnn_val,
                y_val,
                cnn_eval,
                y_eval,
                class_names,
                args,
                normalization_params,
            )
            if result is not None:
                results.append(result)
                print_result(result)
            finish_experiment()
            del cnn_train, cnn_val, cnn_eval
        del x_train, x_eval
        gc.collect()

    if experiment_bar is not None:
        experiment_bar.close()
    print_summary(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
