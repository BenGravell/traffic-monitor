from typing import Any
from types import TracebackType


import cv2

from traffic.type_defs import HeightWidth


class VideoCapture:
    def __init__(self, source: str):
        self.source = source
        self.video_capture = None

    def __enter__(self) -> Any:
        self.video_capture = cv2.VideoCapture(self.source)
        assert self.video_capture is not None
        if not self.video_capture.isOpened():
            raise RuntimeError(f"Failed to open video capture source: {self.source}")
        return self.video_capture

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.video_capture is not None:
            self.video_capture.release()
        cv2.destroyAllWindows()


class VideoWriter:
    def __init__(self, destination: str, fps: float, size: HeightWidth):
        self.destination = destination
        self.fps = fps
        self.size = size
        self.video_output = None

    def __enter__(self) -> Any:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_output = cv2.VideoWriter(self.destination, fourcc, self.fps, (self.size[1], self.size[0]))
        assert self.video_output is not None
        if not self.video_output.isOpened():
            raise RuntimeError(f"Failed to open video writer destination: {self.destination}")
        return self.video_output

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if self.video_output is not None:
            self.video_output.release()
