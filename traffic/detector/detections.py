from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torchvision

from traffic.loggers import DataFrameAble


@dataclass
class Detections(DataFrameAble):
    boxes: npt.NDArray[np.float32]
    confidences: npt.NDArray[np.float32]
    class_ids: npt.NDArray[np.int64]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame.from_dict(
            {
                **{f"box_{coord}": self.boxes[:, i] for i, coord in enumerate(["left", "top", "right", "bottom"])},
                **{"confidence": self.confidences, "class_id": self.class_ids},
            }
        )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "Detections":
        return cls(
            boxes=df[[f"box_{coord}" for coord in ["left", "top", "right", "bottom"]]].to_numpy().astype(np.float32),
            confidences=df["confidence"].to_numpy().astype(np.float32),
            class_ids=df["class_id"].to_numpy().astype(np.int64),
        )

    def select(self, idxs: list[int]) -> "Detections":
        return Detections(self.boxes[idxs], self.confidences[idxs], self.class_ids[idxs])


class DetectionsFilter(ABC):
    @abstractmethod
    def _get_filter_idxs(self, detections: Detections) -> list[int]:
        """Get indices for filter."""

    def __call__(self, detections: Detections) -> Detections:
        """Apply the filter."""
        return detections.select(self._get_filter_idxs(detections))


class ConfidenceFilter(DetectionsFilter):
    """Filter to remove low-confidence detections."""

    def __init__(self, min_confidence: float) -> None:
        self.min_confidence = min_confidence

    def _get_filter_idxs(self, detections: Detections) -> list[int]:
        return np.nonzero(detections.confidences > self.min_confidence)[0].tolist()


class ClassFilter(DetectionsFilter):
    """Filter to remove detections of irrelevant classes."""

    def __init__(self, allowed_class_ids: list[int]) -> None:
        self.allowed_class_ids = allowed_class_ids

    def _get_filter_idxs(self, detections: Detections) -> list[int]:
        return np.nonzero(np.isin(detections.class_ids, self.allowed_class_ids))[0].tolist()


class NonMaxSuppressionFilter(DetectionsFilter):
    """Filter to remove detections using non-max suppression."""

    def __init__(self, iou_threshold: float) -> None:
        self.iou_threshold = iou_threshold

    def _get_filter_idxs(self, detections: Detections) -> list[int]:
        """Thin wrapper around torchvision.ops.nms with numpy->torch->numpy conversion."""
        return torchvision.ops.nms(
            torch.tensor(detections.boxes, dtype=torch.float32),
            torch.tensor(detections.confidences, dtype=torch.float32),
            self.iou_threshold,
        ).numpy()
