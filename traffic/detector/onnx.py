from typing import cast

import numpy as np
import numpy.typing as npt
from torch import nn
import cv2

from traffic.detector import Detector, ModelConfig, RuntimeConfig, DeviceType
from traffic.detector import utils
from traffic.detector.detections import Detections, ClassFilter, ConfidenceFilter, NonMaxSuppressionFilter
import traffic.utils.image_utils as image_utils
from traffic.type_defs import Frame, HeightWidth
import traffic.utils.box_utils as box_utils


class Preprocess(nn.Module):
    def __init__(self, output_image_size: HeightWidth) -> None:
        super().__init__()
        self.output_image_size = output_image_size

    def forward(self, frame: Frame) -> npt.NDArray[np.float32]:
        # Convert the frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize and add padding to bring image to size expected by model.
        frame_rgb_letterboxed = image_utils.letterbox(frame_rgb, new_size=self.output_image_size)
        # Take range down from np.uint8 (0, 255) to np.float32 (0, 1)
        frame_rgb_letterbox_unitized = frame_rgb_letterboxed.astype(np.float32) / 255
        # Swap axes from (height, width, channels) to (channels, height, width)
        frame_rgb_letterbox_unitized_chw = np.transpose(frame_rgb_letterbox_unitized, [2, 0, 1])
        # Make array contiguous in memory.
        frame_rgb_letterbox_unitized_chw = np.ascontiguousarray(frame_rgb_letterbox_unitized_chw)
        # Expand with singleton batch dim
        frame_rgb_letterbox_unitized_chw_batch = np.expand_dims(frame_rgb_letterbox_unitized_chw, 0)
        return frame_rgb_letterbox_unitized_chw_batch


class Primary(nn.Module):
    """PyTorch module for YOLO inference."""

    def __init__(self, weights_path: str, device_type: DeviceType) -> None:
        super().__init__()
        self.session = utils.create_onnxruntime_session(weights_path, device_type)
        self.input_names = [x.name for x in self.session.get_inputs()]
        self.output_names = [x.name for x in self.session.get_outputs()]

    def forward(
        self,
        image: npt.NDArray[np.float32],
        original_image_size: HeightWidth,
    ) -> list[npt.NDArray[np.float32] | npt.NDArray[np.int32]]:
        """Perform YOLO inference on an input image array.

        Arguments:
        image: numpy array with shape = (channel, height, width)

        Returns:
        Prediction outputs. See https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3#output-of-model for details.

        Output box coordinates are in (left, top, right, bottom) format, and are scaled to be compatible with the original image size.
        """
        original_image_size_arr = np.array(original_image_size, dtype=np.float32).reshape(1, 2)
        return self.session.run(
            self.output_names, {self.input_names[0]: image, self.input_names[1]: original_image_size_arr}
        )


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

    def forward(
        self,
        model_output: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]],
        output_image_size: HeightWidth,
    ) -> Detections:
        # Extract boxes, scores, and class IDs from the model output.
        num_dims_in_model_output = len(model_output[2].shape)
        # Extract predicted positive indexes and class IDs using the proper dimensional indexing.
        # This is needed to handle both onnx and onnx_tiny models;
        # inexplicably, the onnx_tiny model has an extra dimension in the output for some reason.
        if num_dims_in_model_output == 2:
            predicted_positive_idxs = model_output[2][:, 2]
            class_ids = model_output[2][:, 1]
        elif num_dims_in_model_output == 3:
            predicted_positive_idxs = model_output[2][0, :, 2]
            class_ids = model_output[2][0, :, 1]
        else:
            msg = (
                "model_output[2] did not have number of dimensions 2 or 3 as required, but was"
                f" {num_dims_in_model_output}."
            )
            raise ValueError(msg)
        num_predicted_positive = len(predicted_positive_idxs)

        confidences = cast(npt.NDArray[np.float32], model_output[1][0, class_ids, predicted_positive_idxs])

        # Swap top/left and bottom/right idxs of the boxes portion of the prediction.
        # This is based on insepction of the model source code https://github.com/qqwweee/keras-yolo3/blob/master/yolo.py
        # which shows that the raw model outputs are in (top, left, bottom, right) format.
        boxes = np.zeros((num_predicted_positive, 4), dtype=np.float32)
        boxes[:, 0] = model_output[0][0, predicted_positive_idxs, 1]  # left
        boxes[:, 1] = model_output[0][0, predicted_positive_idxs, 0]  # top
        boxes[:, 2] = model_output[0][0, predicted_positive_idxs, 3]  # right
        boxes[:, 3] = model_output[0][0, predicted_positive_idxs, 2]  # bottom

        detections = Detections(boxes, confidences, class_ids.astype(np.int64))
        box_utils.clip_boxes(boxes, output_image_size)
        detections = ConfidenceFilter(self.min_confidence)(detections)
        detections = ClassFilter(self.valid_class_ids)(detections)
        detections = NonMaxSuppressionFilter(self.min_intersection_over_union)(detections)

        return detections


class OnnxDetector(Detector):
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
        frame_size: HeightWidth = (frame[0], frame[1])
        model_input_image = self.preprocess(frame)
        model_output = self.primary(model_input_image, frame_size)
        assert len(model_output) == 3
        model_output_tuple: tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], npt.NDArray[np.int32]] = tuple(
            model_output
        )
        detections = self.postprocess(model_output_tuple, frame_size)
        return detections
