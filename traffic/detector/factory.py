from typing import Type

from traffic.detector import Detector, RuntimeConfig, ModelConfig, DetectorType
from traffic.detector.playback import PlaybackDetector
from traffic.detector.onnx import OnnxDetector
from traffic.detector.ultralytics import UltralyticsDetector


DETECTOR_TYPE_MAP: dict[DetectorType, Type[Detector]] = {
    DetectorType.PLAYBACK: PlaybackDetector,
    DetectorType.ONNX: OnnxDetector,
    DetectorType.ULTRALYTICS: UltralyticsDetector,
}


class DetectorFactory:
    @classmethod
    def create(cls, runtime_config: RuntimeConfig) -> Detector:
        model_config = ModelConfig.from_filesystem(runtime_config.model_config_path)
        detector_cls = DETECTOR_TYPE_MAP[model_config.detector_type]
        return detector_cls(runtime_config, model_config)
