from scipy.spatial import distance as dist
import numpy as np

from traffic.utils import box_utils
from traffic.tracker import Tracker, Track
from traffic.type_defs import LtrbBoxes, XyCentroids


class CentroidTracker(Tracker):
    """Detection tracker based on centroid matching."""

    def _create_track(self, idx: int, centroids: XyCentroids, boxes: LtrbBoxes) -> Track:
        return Track(centroid=centroids[idx], box=boxes[idx])

    def update(self, arg_boxes: LtrbBoxes) -> dict[int, int]:
        has_tracks = len(self.track_dict) > 0
        has_arg_boxes = len(arg_boxes) > 0

        # Create an explicit map from track ID to arg box indices
        track_id_to_arg_box_map: dict[int, int] = {}

        # Generate necessary intermediates

        if has_arg_boxes:
            arg_box_centroids = box_utils.centroids_from_boxes(arg_boxes)

        if has_tracks:
            track_ids = list(self.track_dict.keys())
            track_centroids = np.array([track.centroid for track in self.track_dict.values()])

        # Handle 4 cases explicitly and separately for getting unused_rows, unused_cols
        unused_rows: set[int] = set()
        unused_cols: set[int] = set()
        if not has_arg_boxes and not has_tracks:
            # Nothing to do in this case.
            pass

        if not has_arg_boxes and has_tracks:
            # All existing tracks have disappeared.
            unused_rows = {i for i in range(len(track_ids))}
            # There are no *new* boxes because there are no boxes.
            # unused_cols = set()

        if has_arg_boxes and not has_tracks:
            # There are no *disappeared* tracks because there are no tracks.
            # unused_rows = set()
            # All box centroids are new.
            unused_cols = {i for i in range(len(arg_box_centroids))}

        if has_arg_boxes and has_tracks:
            # Compute distance matrix from track centroids to box centroids
            distance_matrix = dist.cdist(track_centroids, arg_box_centroids)
            distance_matrix[distance_matrix > self.max_distance_for_matching] = np.inf

            # Iterate in order of ascending matching distance, assigning track centroids to box centroids,
            # until track-box pairs with distance under the max_distance_for_matching threshold are exhausted.
            used_rows: list[int] = []
            used_cols: list[int] = []
            while np.any(np.isfinite(distance_matrix)):
                urow, ucol = np.unravel_index(np.argmin(distance_matrix), distance_matrix.shape)
                row, col = int(urow), int(ucol)
                used_rows.append(row)
                used_cols.append(col)
                distance_matrix[row, :] = np.inf
                distance_matrix[:, col] = np.inf

            # All track centroids that are assigned get their tracks updated.
            for row, col in zip(used_rows, used_cols):
                track_id = track_ids[row]
                track = self._create_track(col, arg_box_centroids, arg_boxes)
                self.update_track(track, track_id)
                track_id_to_arg_box_map[track_id] = col

            # Tracks that were not assigned have disappeared.
            unused_rows = set(np.arange(len(track_centroids))).difference(used_rows)
            # Boxes that were not assigned are new.
            unused_cols = set(np.arange(len(arg_box_centroids))).difference(used_cols)

        for row in unused_rows:
            track_id = track_ids[row]
            self.on_disappear(track_id)

        for col in unused_cols:
            track = self._create_track(col, arg_box_centroids, arg_boxes)
            track_id = self.register(track)
            track_id_to_arg_box_map[track_id] = col

        return track_id_to_arg_box_map
