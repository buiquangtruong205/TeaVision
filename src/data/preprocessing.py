import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def _image_size_tuple(image_size):
    if isinstance(image_size, int):
        return image_size, image_size
    return tuple(image_size)


def build_train_transforms(image_size, mean, std, augmentation_config=None):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        from torchvision import transforms

        size = _image_size_tuple(image_size)
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    config = augmentation_config or {}
    height, width = _image_size_tuple(image_size)
    transforms_list = [A.Resize(height=height, width=width)]
    if config.get("horizontal_flip", True):
        transforms_list.append(A.HorizontalFlip(p=0.5))
    if config.get("random_rotate", 10):
        transforms_list.append(A.Rotate(limit=config.get("random_rotate", 10), p=0.5))
    if config.get("color_jitter", True):
        transforms_list.append(A.ColorJitter(p=0.3))
    transforms_list.extend([A.Normalize(mean=mean, std=std), ToTensorV2()])
    return A.Compose(transforms_list)


def build_eval_transforms(image_size, mean, std):
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError:
        from torchvision import transforms

        size = _image_size_tuple(image_size)
        return transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    height, width = _image_size_tuple(image_size)
    return A.Compose([A.Resize(height=height, width=width), A.Normalize(mean=mean, std=std), ToTensorV2()])


def compute_mean_std(data_loader):
    """Compute channel mean/std from a loader yielding image tensors in [0, 1]."""

    total_sum = None
    total_squared_sum = None
    total_pixels = 0

    for images, _ in data_loader:
        batch = images.float()
        if batch.max() > 1:
            batch = batch / 255.0
        pixels = batch.size(0) * batch.size(2) * batch.size(3)
        channel_sum = batch.sum(dim=[0, 2, 3])
        channel_squared_sum = (batch**2).sum(dim=[0, 2, 3])
        total_sum = channel_sum if total_sum is None else total_sum + channel_sum
        total_squared_sum = channel_squared_sum if total_squared_sum is None else total_squared_sum + channel_squared_sum
        total_pixels += pixels

    mean = total_sum / total_pixels
    variance = (total_squared_sum / total_pixels) - mean**2
    std = variance.clamp(min=1e-12).sqrt()
    return mean.tolist(), std.tolist()


def compute_mean_std_from_csv(csv_file, image_size=None, root_dir=None):
    df = pd.read_csv(csv_file)
    if "image_path" not in df.columns:
        raise ValueError(f"Missing image_path column in {csv_file}")

    root = Path(root_dir) if root_dir else Path.cwd()
    total_sum = np.zeros(3, dtype=np.float64)
    total_squared_sum = np.zeros(3, dtype=np.float64)
    total_pixels = 0

    for image_path in df["image_path"]:
        path = Path(image_path)
        if not path.is_absolute():
            path = root / path
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image_size is not None:
                size = _image_size_tuple(image_size)
                image = image.resize((size[1], size[0]))
            array = np.asarray(image, dtype=np.float32) / 255.0

        total_sum += array.sum(axis=(0, 1))
        total_squared_sum += np.square(array).sum(axis=(0, 1))
        total_pixels += array.shape[0] * array.shape[1]

    mean = total_sum / total_pixels
    variance = (total_squared_sum / total_pixels) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-12))
    return mean.tolist(), std.tolist()


def save_preprocessing_params(path, image_size, mean, std):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    params = {"image_size": image_size, "mean": list(mean), "std": list(std)}
    output.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return params


def load_preprocessing_params(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_train_transform(image_size, mean, std, config=None):
    return build_train_transforms(image_size, mean, std, augmentation_config=config)


def get_eval_transform(image_size, mean, std):
    return build_eval_transforms(image_size, mean, std)


def _legacy_build_train_transforms(image_size, mean, std):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _legacy_build_eval_transforms(image_size, mean, std):
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
