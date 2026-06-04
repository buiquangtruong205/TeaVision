"""Inference service for the exported TeaVision CNN artifact."""

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError

from src.models.leaf_localization import LeafLocalizer, LeafRegion
from src.models.simple_cnn import load_simple_cnn_checkpoint


@dataclass(frozen=True)
class Prediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: dict[str, float]
    leaf_bbox: tuple[int, int, int, int]
    leaf_detected: bool
    localization_confidence: float


class TeaVisionPredictor:
    """Load an exported artifact once and predict tea-leaf disease classes."""

    def __init__(
        self,
        artifact_path: str | Path = "models/cnn_zscore/model_artifact.json",
        device: str | torch.device | None = None,
        localize_leaf: bool = True,
        classify_leaf_crop: bool = False,
    ):
        self.artifact_path = Path(artifact_path).resolve()
        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")

        self.artifact = json.loads(self.artifact_path.read_text(encoding="utf-8"))
        checkpoint_name = self.artifact.get("checkpoint")
        if not checkpoint_name:
            raise ValueError("Model artifact does not define a checkpoint.")

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint_path = self.artifact_path.parent / checkpoint_name
        self.model, self.checkpoint = load_simple_cnn_checkpoint(checkpoint_path, self.device)
        self.image_size = int(self.checkpoint["image_size"])
        self.class_names = list(self.checkpoint["class_names"])
        self.model_version = self.artifact_path.parent.name
        self.localize_leaf = localize_leaf
        self.classify_leaf_crop = classify_leaf_crop
        self.localizer = LeafLocalizer()

        params = self.checkpoint.get("normalization_params")
        if self.checkpoint.get("normalizer") != "zscore" or not params:
            raise ValueError("The deployment predictor requires a CNN + zscore checkpoint.")
        self.mean = self._to_numpy(params["mean"])
        self.scale = self._to_numpy(params["scale"])
        expected_shape = (self.image_size, self.image_size, 3)
        if self.mean.shape != expected_shape or self.scale.shape != expected_shape:
            raise ValueError(f"Expected zscore parameters with shape {expected_shape}.")

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32) / 255.0
        normalized = (pixels - self.mean) / self.scale
        tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0)
        return tensor.to(self.device)

    def predict_image(self, image: Image.Image) -> Prediction:
        image = ImageOps.exif_transpose(image).convert("RGB")
        region = self.localizer.locate(image) if self.localize_leaf else LeafRegion(
            (0, 0, image.width, image.height), False, 0.0
        )
        model_image = image.crop(region.bbox) if region.detected and self.classify_leaf_crop else image
        inputs = self.preprocess(model_image)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(inputs), dim=1)[0].cpu().numpy()

        class_id = int(np.argmax(probabilities))
        return Prediction(
            class_id=class_id,
            class_name=self.class_names[class_id],
            confidence=float(probabilities[class_id]),
            probabilities={
                class_name: float(probabilities[index])
                for index, class_name in enumerate(self.class_names)
            },
            leaf_bbox=region.bbox,
            leaf_detected=region.detected,
            localization_confidence=region.confidence,
        )

    def predict_bytes(self, image_bytes: bytes) -> Prediction:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                return self.predict_image(image)
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc

    def predict_annotated_bytes(self, image_bytes: bytes) -> tuple[Prediction, bytes]:
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                prediction = self.predict_image(image)
                region = LeafRegion(
                    prediction.leaf_bbox,
                    prediction.leaf_detected,
                    prediction.localization_confidence,
                )
                annotated = self.localizer.draw(
                    image, region, prediction.class_name, prediction.confidence
                )
                output = io.BytesIO()
                annotated.save(output, format="JPEG", quality=92)
                return prediction, output.getvalue()
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError("Uploaded file is not a valid image.") from exc
