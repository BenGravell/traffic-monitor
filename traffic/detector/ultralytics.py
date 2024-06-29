from typing import cast

import numpy as np
import numpy.typing as npt
from torch import nn

from traffic.detector import Detector, ModelConfig, RuntimeConfig, DeviceType
from traffic.detector.detections import Detections, ClassFilter, ConfidenceFilter, NonMaxSuppressionFilter
from traffic.detector import utils
import traffic.utils.box_utils as box_utils
from traffic.type_defs import Frame, HeightWidth

# Re-use the same preprocessor class as for ONNX open-source models.
from traffic.detector.onnx import Preprocess


class Primary(nn.Module):
    """PyTorch module for YOLO inference."""

    def __init__(self, weights_path: str, device_type: DeviceType) -> None:
        super().__init__()
        self.session = utils.create_onnxruntime_session(weights_path, device_type)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(self, image: npt.NDArray[np.uint8]) -> list[npt.NDArray[np.float32]]:
        """Performs YOLO inference on an input image tensor.

        Arguments:
        image: numpy array with shape = (channel, height, width)

        Returns:
        Prediction outputs: length-1 list of numpy array with shape (1, 25200, 85) where
        1 = batch size
        25200 = number of boxes
        85 = 4 box coordinates in (x1, y1, x2, y2) format + 1 box confidence + 80 class confidences corresponding to COCO classes.
        """

        return self.session.run(self.output_names, {self.input_names[0]: image})


class Postprocess(nn.Module):
    def __init__(
        self,
        min_confidence: float,
        min_intersection_over_union: float,
        valid_class_ids: list[int],
        input_image_size: HeightWidth,
    ) -> None:
        super().__init__()
        self.min_confidence = min_confidence
        self.min_intersection_over_union = min_intersection_over_union
        self.valid_class_ids = valid_class_ids
        self.input_image_size = input_image_size

    def forward(self, model_output: list[npt.NDArray[np.float32]], frame_shape: HeightWidth) -> Detections:
        # Remove batch dim
        prediction = cast(npt.NDArray[np.float32], model_output[0][0])
        # Extract boxes, scores, and class IDs from the model output.
        boxes = prediction[:, 0:4]
        object_confidences = prediction[:, 4]
        class_ids = np.argmax(prediction[:, 5:], axis=1)
        class_confidences = prediction[:, 5:][np.arange(prediction.shape[0]), class_ids]
        confidences = object_confidences * class_confidences

        # Create detections object.
        detections = Detections(boxes, confidences, class_ids)
        detections = ConfidenceFilter(self.min_confidence)(detections)
        detections = ClassFilter(self.valid_class_ids)(detections)
        # NOTE: Only do geometric transformations on boxes after simple filters to reduce useless work.
        detections.boxes = box_utils.xywh2ltrb(detections.boxes)
        detections.boxes = box_utils.scale_boxes(
            detections.boxes, from_shape=self.input_image_size, to_shape=frame_shape
        ).round()
        detections = NonMaxSuppressionFilter(self.min_intersection_over_union)(detections)

        return detections


class UltralyticsDetector(Detector):
    def __init__(self, runtime_config: RuntimeConfig, model_config: ModelConfig) -> None:
        self.runtime_config = runtime_config
        self.model_config = model_config
        self.preprocess = Preprocess(
            output_image_size=model_config.model_input_image_size,
        )
        self.primary = Primary(
            weights_path=model_config.weights_path,
            device_type=runtime_config.device_type,
        )
        self.postprocess = Postprocess(
            min_confidence=runtime_config.min_confidence,
            min_intersection_over_union=runtime_config.min_intersection_over_union,
            valid_class_ids=utils.coco_names_to_ids(runtime_config.valid_class_names),
            input_image_size=model_config.model_input_image_size,
        )

    def __call__(self, frame: Frame, frame_number: int) -> Detections:
        frame_size: HeightWidth = (frame.shape[0], frame.shape[1])
        model_input_image = self.preprocess(frame)
        model_output = self.primary(model_input_image)
        detections = self.postprocess(model_output, frame_size)
        return detections
