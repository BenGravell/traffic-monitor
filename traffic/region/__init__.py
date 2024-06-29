from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, cast

import numpy as np
import numpy.typing as npt
import cv2

from traffic.utils.colors import BgrColor
from traffic.type_defs import Frame, XyCentroids


@dataclass
class RegionStatistics:
    count_current: int = 0
    count_total: int = 0


class Region:
    def __init__(
        self,
        mask_path: Path | str,
        name: str,
        color: BgrColor,
    ) -> None:
        self.mask = cast(npt.NDArray[np.uint8], cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        self.color = color
        self.name = name
        self.overlay = self.create_overlay()
        self.stats = RegionStatistics()
        self.centroid_idxs_contained: list[int] = []
        self.track_ids_encountered: set[int] = set()

    def create_overlay(self) -> npt.NDArray[np.uint8]:
        overlay = np.zeros_like(cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR), dtype=np.uint8)
        overlay[self.mask == 255] = self.color
        return overlay

    def draw_overlay(self, frame: Frame, alpha: float = 1.0) -> Frame:
        return cv2.addWeighted(frame, 1, self.overlay, alpha, 0)

    def check_points_in_mask(self, points: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
        # Convert mask from np.uint8 to bool
        binary_mask = (self.mask == 255).astype(bool)
        # Round to int and clip to mask bounds.
        points_rounded = np.clip(
            points.round().astype(int),
            a_min=[0, 0],
            a_max=[self.mask.shape[1] - 1, self.mask.shape[0] - 1],
        )
        return binary_mask[points_rounded[:, 1], points_rounded[:, 0]]

    def update_with_centroids(self, centroids: XyCentroids) -> None:
        self.centroid_idxs_contained = list(np.nonzero(self.check_points_in_mask(centroids))[0])

        self.stats.count_current = len(self.centroid_idxs_contained)

    def update_with_tracks(self, centroid_idx_to_track_id_map: dict[int, int]) -> None:
        for centroid_idx in self.centroid_idxs_contained:
            track_id = centroid_idx_to_track_id_map[centroid_idx]
            self.track_ids_encountered.add(track_id)

        self.stats.count_total = len(self.track_ids_encountered)


class RegionGroup:
    def __init__(self, regions: list[Region]) -> None:
        self.region_map = {region.name: region for region in regions}

    def __iter__(self) -> Iterator[Region]:
        return iter(self.region_map.values())

    @property
    def centroid_idxs_contained(self) -> list[int]:
        return sorted(set(idx for region in self for idx in region.centroid_idxs_contained))

    def update_with_centroids(self, centroids: XyCentroids) -> None:
        for region in self:
            region.update_with_centroids(centroids)

    def update_with_tracks(self, centroid_idx_to_track_id_map: dict[int, int]) -> None:
        for region in self:
            region.update_with_tracks(centroid_idx_to_track_id_map)

    def draw_overlay(self, frame: Frame, alpha: float = 1.0) -> Frame:
        for region in self:
            frame = region.draw_overlay(frame, alpha)
        return frame
