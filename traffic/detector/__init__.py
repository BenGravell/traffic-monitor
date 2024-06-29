from abc import ABC, abstractmethod
import json
from enum import IntEnum
from dataclasses import dataclass

from traffic.type_defs import Frame, HeightWidth
from traffic.detector.detections import Detections


class DetectorFailure(Exception):
    """Exception raised if a detector failed to compute detections."""


class DetectorType(IntEnum):
    PLAYBACK = 0
    ONNX = 1
    ULTRALYTICS = 2


class DeviceType(IntEnum):
    CPU = 1
    CUDA = 2


class ModelConfigType(IntEnum):
    PLAYBACK = 0
    ONNX = 1
    ONNX_TINY = 2
    ULTRALYTICS = 3
    ULTRALYTICS_TINY = 4


@dataclass
class ModelConfig:
    detector_type: DetectorType
    model_input_image_size: HeightWidth = (0, 0)
    weights_path: str = ""

    @classmethod
    def from_filesystem(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            data = json.load(f)
        # Convert string to enum
        data["detector_type"] = DetectorType[data["detector_type"]]
        return cls(**data)


@dataclass
class RuntimeConfig:
    device_type: DeviceType
    min_confidence: float
    min_intersection_over_union: float
    valid_class_names: list[str]
    model_config_type: ModelConfigType
    run_id: str | None = None
    log_detections: bool = False
    log_tracks: bool = False

    @classmethod
    def from_filesystem(cls, path: str) -> "RuntimeConfig":
        with open(path) as f:
            data = json.load(f)
        # Convert string to enum
        data["device_type"] = DeviceType[data["device_type"]]
        data["model_config_type"] = ModelConfigType[data["model_config_type"]]

        return cls(**data)

    @property
    def model_config_path(self) -> str:
        return f"config/model/static/{self.model_config_type.name.lower()}.json"


class Detector(ABC):
    @abstractmethod
    def __init__(self, runtime_config: RuntimeConfig, model_config: ModelConfig) -> None:
        pass

    @abstractmethod
    def __call__(self, frame: Frame, frame_number: int) -> Detections:
        pass
