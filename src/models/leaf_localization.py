"""Classical image processing helpers for locating a tea leaf."""

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image, ImageDraw


@dataclass(frozen=True)
class LeafRegion:
    bbox: tuple[int, int, int, int]
    detected: bool
    confidence: float


class LeafLocalizer:
    """Locate the largest foreground object against a mostly uniform background."""

    def __init__(self, min_area_ratio: float = 0.01, padding_ratio: float = 0.05):
        self.min_area_ratio = min_area_ratio
        self.padding_ratio = padding_ratio

    def locate(self, image: Image.Image) -> LeafRegion:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        height, width = rgb.shape[:2]
        full_bbox = (0, 0, width, height)
        if height < 8 or width < 8:
            return LeafRegion(full_bbox, False, 0.0)

        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        border = np.concatenate(
            (lab[0, :, :], lab[-1, :, :], lab[:, 0, :], lab[:, -1, :]),
            axis=0,
        )
        background = np.median(border, axis=0)
        distance = np.linalg.norm(lab - background, axis=2)
        distance_u8 = np.clip(distance, 0, 255).astype(np.uint8)
        otsu_threshold, mask = cv2.threshold(
            distance_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        if otsu_threshold < 10:
            _, mask = cv2.threshold(distance_u8, 10, 255, cv2.THRESH_BINARY)

        kernel_size = max(3, int(round(min(height, width) * 0.025)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return LeafRegion(full_bbox, False, 0.0)

        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        image_area = float(width * height)
        if area / image_area < self.min_area_ratio:
            return LeafRegion(full_bbox, False, 0.0)

        x, y, box_width, box_height = cv2.boundingRect(contour)
        box_area = float(box_width * box_height)
        fill_ratio = area / box_area if box_area else 0.0
        confidence = float(np.clip(fill_ratio, 0.0, 1.0))

        padding = int(round(max(box_width, box_height) * self.padding_ratio))
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(width, x + box_width + padding)
        bottom = min(height, y + box_height + padding)
        return LeafRegion((left, top, right, bottom), True, confidence)

    @staticmethod
    def draw(
        image: Image.Image,
        region: LeafRegion,
        label: str,
        confidence: float,
    ) -> Image.Image:
        annotated = image.convert("RGB").copy()
        draw = ImageDraw.Draw(annotated)
        left, top, right, bottom = region.bbox
        line_width = max(2, min(annotated.size) // 150)
        color = (30, 220, 70) if region.detected else (255, 170, 0)
        draw.rectangle((left, top, right - 1, bottom - 1), outline=color, width=line_width)
        text = f"{label}: {confidence:.1%}"
        text_bbox = draw.textbbox((left, top), text)
        text_height = text_bbox[3] - text_bbox[1]
        text_width = text_bbox[2] - text_bbox[0]
        text_top = max(0, top - text_height - 6)
        draw.rectangle(
            (left, text_top, min(annotated.width, left + text_width + 8), top),
            fill=color,
        )
        draw.text((left + 4, text_top + 2), text, fill=(0, 0, 0))
        return annotated
