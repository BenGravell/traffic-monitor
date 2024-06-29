import uuid
from pathlib import Path

import cv2
import numpy as np
import codetiming

from traffic.detector.factory import DetectorFactory
from traffic.detector import DetectorFailure
from traffic.detector import RuntimeConfig as DetectorRuntimeConfig
from traffic.detector.detections import Detections
from traffic.utils import box_utils
from traffic.tracker import RuntimeConfig as TrackerRuntimeConfig
from traffic.tracker.factory import TrackerFactory
from traffic.utils import image_utils
from traffic.region import Region, RegionGroup
from traffic.utils.video_manager import VideoCapture, VideoWriter
from traffic.utils.colors import BgrColor, Vivid, Monogrey
from traffic.type_defs import Frame, HeightWidth
from traffic.utils.summary import Summary
from traffic.loggers import ParquetDataLogger, NullDataLogger
from traffic.tracker.track_info import TrackInfo, TrackInfoGroup, TrackStatus


class VideoAnalyzer:
    """Class for analyzing videos for traffic object detections and counts."""

    def __init__(
        self, model_runtime_config_path: str, tracker_runtime_config_path: str, output_dir_path: str | None
    ) -> None:
        # Create a unique run ID.
        self.run_id = str(uuid.uuid4())

        # Handle output_dir_path.
        if output_dir_path is None:
            output_dir_path = f"data/runs/{self.run_id}"
        self.output_dir_path = output_dir_path
        Path(self.output_dir_path).mkdir(exist_ok=True, parents=True)

        # Create detector.
        model_runtime_config = DetectorRuntimeConfig.from_filesystem(model_runtime_config_path)
        self.detector = DetectorFactory.create(model_runtime_config)

        # Create tracker.
        tracker_config = TrackerRuntimeConfig.from_filesystem(tracker_runtime_config_path)
        self.tracker = TrackerFactory.create(tracker_config)

        # Create loggers.
        self.detections_logger = ParquetDataLogger() if model_runtime_config.log_detections else NullDataLogger()
        self.tracks_logger = ParquetDataLogger() if model_runtime_config.log_tracks else NullDataLogger()

        # Create detection regions.
        self.region_group = RegionGroup(
            [
                Region(mask_path="data/regions/highway_left.png", name="INCOMING", color=Vivid.BLUE.bgr),
                Region(mask_path="data/regions/highway_right.png", name="OUTGOING", color=Vivid.RED.bgr),
            ]
        )

        # Create summary.
        self.summary = Summary(region_group=self.region_group)
        self.summary_area_height = 200

    def get_track_info_group(
        self,
        detections: Detections,
        arg_box_to_track_id_map: dict[int, int],
        track_id_to_arg_box_map: dict[int, int],
    ) -> TrackInfoGroup:
        track_infos = []

        # Iterate from lowest to highest box bottoms so that nearer objects get drawn on top
        # (since nearer objects are at the bottom of the screen due to camera persepective).
        sort_idxs_box_bottoms = np.argsort(detections.boxes[:, 3])
        for box_idx in sort_idxs_box_bottoms:
            box = detections.boxes[box_idx]
            confidence = detections.confidences[box_idx]
            class_id = detections.class_ids[box_idx]

            region = None
            for r in self.region_group:
                if box_idx in r.centroid_idxs_contained:
                    region = r
                    break

            track_id = arg_box_to_track_id_map.get(box_idx)
            track_status = TrackStatus.INACTIVE if track_id is None else TrackStatus.ACTIVE

            track_infos.append(
                TrackInfo(
                    id=track_id,
                    status=track_status,
                    region=region,
                    class_id=class_id,
                    confidence=confidence,
                    box=tuple(box.astype(int)),
                )
            )

        # Add image annotations for the LOST tracks.
        for track_id in self.tracker.track_dict.keys():
            if track_id not in track_id_to_arg_box_map:
                track = self.tracker.track_dict[track_id]
                track_infos.append(
                    TrackInfo(
                        id=track_id,
                        status=TrackStatus.LOST,
                        box=tuple(track.box.astype(int)),
                    )
                )

        return TrackInfoGroup(track_infos)

    def add_detection_image_annotations(self, frame: Frame, track_info_group: TrackInfoGroup) -> None:
        """Add image annotations for all tracks to an image frame.

        This is an in-place operation on the frame.
        """

        def get_color(track_info: TrackInfo) -> BgrColor:
            if isinstance(track_info.region, Region):
                return track_info.region.color
            if track_info.status == TrackStatus.LOST:
                return Monogrey.DARK.bgr
            return Monogrey.MID.bgr

        def get_alpha(track_info: TrackInfo) -> float:
            return 0.9 if track_info.status == TrackStatus.ACTIVE else 0.7

        for track_info in track_info_group:
            if track_info.box is not None:
                image_utils.box_label(
                    frame,
                    box=track_info.box,
                    label=track_info.to_display_str(),
                    color=get_color(track_info),
                    font_style="Bold",
                    text_color=Monogrey.WHITE.bgr,
                    line_width=2,
                    alpha=get_alpha(track_info),
                )

    def combine_image_and_summary_frames(self, image_frame: Frame, summary_frame: Frame) -> Frame:
        """Combine the main image and summary frames."""
        # Get sizes.
        frame_height, frame_width = image_frame.shape[0:2]
        summary_height, summary_width = summary_frame.shape[0:2]

        # Initialize the whole frame_with_summary array.
        frame_with_summary = np.full((frame_height + self.summary_area_height, frame_width, 3), 255, dtype=np.uint8)

        # Define the position to paste the text image (above the frame).
        position = (int(frame_width / 2 - summary_width / 2), self.summary_area_height - summary_height)

        # Overwrite the patch of the frame above the box with the text image.
        frame_with_summary[position[1] : position[1] + summary_height, position[0] : position[0] + summary_width] = (
            summary_frame
        )

        # Overwrite the frame portion of the final image.
        frame_with_summary[self.summary_area_height :, :] = image_frame

        return frame_with_summary

    def process_frame(self, frame: Frame, frame_number: int) -> Frame:
        """Process a single frame."""
        # Perform detection.
        with codetiming.Timer(text=f"Frame {frame_number:04d} : detection finished in {{:.3f}} seconds"):
            detections = self.detector(frame, frame_number)

        # Add mask overlays.
        frame = self.region_group.draw_overlay(frame, alpha=0.5)

        # Log detections.
        self.detections_logger.store(detections, frame_number)

        # Update detection regions.
        centroids = box_utils.centroids_from_boxes(detections.boxes)
        self.region_group.update_with_centroids(centroids)

        # Update tracker with boxes whose centroids are contained in one of the detection regions.
        centroid_idxs_contained = self.region_group.centroid_idxs_contained
        boxes_in_regions = detections.boxes[centroid_idxs_contained]
        track_id_to_arg_box_map = self.tracker.update(boxes_in_regions)
        arg_box_to_track_id_map = {centroid_idxs_contained[v]: k for k, v in track_id_to_arg_box_map.items()}

        # Update detection region track collectors.
        self.region_group.update_with_tracks(arg_box_to_track_id_map)

        # Get track info.
        track_info_group = self.get_track_info_group(
            detections,
            arg_box_to_track_id_map,
            track_id_to_arg_box_map,
        )

        # Log track info.
        self.tracks_logger.store(track_info_group, frame_number)

        # Add detection image annotations.
        self.add_detection_image_annotations(frame, track_info_group)

        # Update the summary.
        self.summary.update(frame_number)

        # Draw the summary stats in a separate section above the photo frame at the top.
        return self.combine_image_and_summary_frames(image_frame=frame, summary_frame=self.summary.to_image())

    def analyze(self, input_video_path: str) -> None:
        """Analyze an input video.

        Writes:
          - Output video with detections.
          - Detection and track history files.
        """

        with VideoCapture(input_video_path) as video_capture:
            # Get frame width, height, and FPS of input video.
            frame_width = round(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = round(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = video_capture.get(cv2.CAP_PROP_FPS)

            # Write output video using same FPS as input video.
            output_video_size: HeightWidth = (frame_height + self.summary_area_height, frame_width)
            output_video_path = f"{self.output_dir_path}/output.mp4"
            with VideoWriter(output_video_path, fps, output_video_size) as video_output:
                # Main loop iterating over each frame in the input video.
                frame_number = 0
                while True:
                    # Read the next frame from the video.
                    ret, frame = video_capture.read()

                    # Terminate if there was no frame to read.
                    if not ret:
                        print("Input video is exhausted, terminating.")
                        break

                    # Process the current frame.
                    try:
                        output_frame = self.process_frame(frame, frame_number)
                    except DetectorFailure:
                        print("Detector failure encountered, terminating.")
                        break

                    # Write the output frame to the video file.
                    video_output.write(output_frame)

                    # Show the frame.
                    cv2.imshow("Frame", output_frame)

                    # Check for keypress to terminate the loop early or toggle play/pause.
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        print('Caught keypress "q", terminating.')
                        break
                    elif key == ord("k"):
                        cv2.waitKey(0)

                    frame_number += 1

        # Dump log data
        print("Dumping log data...", end="")
        self.detections_logger.dump(f"{self.output_dir_path}/detections.parquet")
        self.tracks_logger.dump(f"{self.output_dir_path}/tracks.parquet")
        print("done.")

        # For convenience, print the output directory at the very end.
        print(f"Find results in {self.output_dir_path}")
