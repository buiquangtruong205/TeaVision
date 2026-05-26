from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class TeaDiseaseDataset(Dataset):
    """PyTorch dataset for tea disease image classification."""

    def __init__(self, samples=None, csv_file=None, transform=None, root_dir=None):
        if samples is None and csv_file is None:
            samples = []
        if samples is not None and csv_file is not None:
            raise ValueError("Provide either samples or csv_file, not both")

        self.root_dir = Path(root_dir) if root_dir else None
        self.samples = self._load_samples(samples, csv_file)
        self.transform = transform

    def _resolve_path(self, image_path):
        path = Path(image_path)
        if path.is_absolute() or self.root_dir is None:
            return path
        return self.root_dir / path

    def _load_samples(self, samples, csv_file):
        if csv_file is None:
            return list(samples)

        df = pd.read_csv(csv_file)
        required_columns = {"image_path", "label"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in {csv_file}: {sorted(missing)}")

        if "is_valid" in df.columns:
            df = df[df["is_valid"].astype(bool)]

        return list(df[["image_path", "label"]].itertuples(index=False, name=None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(self._resolve_path(image_path)).convert("RGB")
        if self.transform:
            image = self._apply_transform(image)
        return image, int(label)

    def _apply_transform(self, image):
        try:
            transformed = self.transform(image=np.array(image))
        except TypeError:
            return self.transform(image)

        if isinstance(transformed, dict) and "image" in transformed:
            return transformed["image"]
        return transformed
