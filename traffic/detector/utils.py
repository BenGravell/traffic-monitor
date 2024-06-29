import onnxruntime
import torch

from traffic.paths import ROOT
from traffic.detector import DeviceType

COCO_NAMES_PATH = ROOT / "models/yolov3/onnx/coco.names"
with COCO_NAMES_PATH.open() as file:
    COCO_NAMES = file.read().splitlines()

COCO_NAMES_MAP = dict(enumerate(COCO_NAMES))
COCO_NAMES_INV_MAP = {v: k for k, v in COCO_NAMES_MAP.items()}


def create_onnxruntime_session(weights_path: str, device_type: DeviceType) -> onnxruntime.InferenceSession:
    # Load ONNX file into an ONNX inference session.
    cuda = torch.cuda.is_available() and device_type == DeviceType.CUDA
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    return onnxruntime.InferenceSession(weights_path, providers=providers)


def coco_names_to_ids(class_names: list[str]) -> list[int]:
    return [COCO_NAMES_INV_MAP[name] for name in class_names]
