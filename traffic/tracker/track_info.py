import dataclasses
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Iterator, Any

import pandas as pd

from traffic.detector import utils as detector_utils
from traffic.region import Region
from traffic.loggers import DataFrameAble


class TrackStatus(IntEnum):
    """Enum representing the status of a track.

    Positive values correspond to active status.
    Negative values correspond to non-active status.
    """

    INACTIVE = -1
    LOST = -2
    NULL = 0
    ACTIVE = 1


@dataclass
class TrackInfo:
    id: int | None = None
    status: TrackStatus = TrackStatus.NULL
    region: Region | str | None = None
    class_id: int | None = None
    confidence: float | None = None
    box: tuple[int, int, int, int] | None = None

    def to_flat_dict(self) -> dict[str, bool | int | float | str]:
        d = dataclasses.asdict(self)
        if isinstance(d["region"], Region):
            d["region"] = d["region"].name
        for i, coord in enumerate(["left", "top", "right", "bottom"]):
            d[f"box_{coord}"] = d["box"][i]
        d.pop("box")
        return d

    @staticmethod
    def format(val: Any, format_func: Callable[..., str]) -> str:
        if val is None:
            return "----"
        return format_func(val)

    @staticmethod
    def format_track_id(track_id: int) -> str:
        return f"{track_id+1:04d}"

    @staticmethod
    def format_status(status: TrackStatus) -> str:
        return status.name

    @staticmethod
    def format_region(region: Region) -> str:
        return region.name

    @staticmethod
    def format_class_id(class_id: int) -> str:
        return detector_utils.COCO_NAMES_MAP[int(class_id)].upper()

    @staticmethod
    def format_confidence(confidence: float) -> str:
        return f"{round(confidence*100)}%"

    def to_display_str(self) -> str:
        vals = [
            self.id,
            self.status,
            self.region,
            self.class_id,
            self.confidence,
        ]
        field_names = [
            "Track ID",
            "Track Stat",
            "Region",
            "Class",
            "Confidence",
        ]
        format_funcs: list[Callable[..., Any]] = [
            self.format_track_id,
            self.format_status,
            self.format_region,
            self.format_class_id,
            self.format_confidence,
        ]
        return "\n".join(
            [
                f"{field_name:10s}: {self.format(val, format_func)}"
                for val, field_name, format_func in zip(vals, field_names, format_funcs)
            ]
        )


class TrackInfoGroup(DataFrameAble):
    def __init__(self, track_infos: list[TrackInfo]) -> None:
        self.track_infos = track_infos

    def __iter__(self) -> Iterator[TrackInfo]:
        return iter(self.track_infos)

    def __getitem__(self, index: int) -> TrackInfo:
        return self.track_infos[index]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([track_info.to_flat_dict() for track_info in self])

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "TrackInfoGroup":
        def track_info_from_row(row: pd.Series) -> TrackInfo:
            return TrackInfo(
                row.id,
                row.status,
                row.region,
                row.class_id,
                row.confidence,
                tuple(row[f"box_{coord}"] for coord in ["left", "top", "right", "bottom"]),
            )

        return cls([track_info_from_row(row) for _, row in df.iterrows()])
