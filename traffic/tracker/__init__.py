from dataclasses import dataclass
import json
from enum import IntEnum
from abc import ABC, abstractmethod
import collections
import typing

import numpy as np
import numpy.typing as npt

from traffic.type_defs import LtrbBoxes


class TrackerType(IntEnum):
    CENTROID = 1


@dataclass
class Track:
    centroid: npt.NDArray[np.float64]
    box: npt.NDArray[np.float64]


@dataclass
class RuntimeConfig:
    tracker_type: TrackerType
    max_frames_disappeared_until_deregistered: int
    max_distance_for_matching: int

    @classmethod
    def from_filesystem(cls, path: str) -> "RuntimeConfig":
        with open(path) as f:
            data = json.load(f)
            # Convert string to enum
            data["tracker_type"] = TrackerType[data["tracker_type"]]
            return cls(**data)


class Tracker(ABC):
    """Base class for detection trackers."""

    def __init__(self, config: RuntimeConfig) -> None:
        self.next_track_id = 0
        self.track_dict: typing.OrderedDict[int, Track] = collections.OrderedDict()
        self.frames_disappeared_dict: typing.OrderedDict[int, int] = collections.OrderedDict()
        self.max_frames_disappeared_until_deregistered = config.max_frames_disappeared_until_deregistered
        self.max_distance_for_matching = config.max_distance_for_matching

    def update_track(self, track: Track, track_id: int) -> None:
        self.track_dict[track_id] = track
        self.frames_disappeared_dict[track_id] = 0

    def register(self, track: Track) -> int:
        track_id = self.next_track_id
        self.update_track(track, track_id)
        self.next_track_id += 1
        return track_id

    def deregister(self, track_id: int) -> None:
        del self.track_dict[track_id]
        del self.frames_disappeared_dict[track_id]

    def on_disappear(self, track_id: int) -> None:
        self.frames_disappeared_dict[track_id] += 1
        if self.frames_disappeared_dict[track_id] > self.max_frames_disappeared_until_deregistered:
            self.deregister(track_id)

    @abstractmethod
    def update(self, arg_boxes: LtrbBoxes) -> dict[int, int]:
        """Update the tracker with detection boxes.

        Arguments:
        arg_boxes: Detection boxes.

        Returns:
        dict from arg box index to track ID.
        """
