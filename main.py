"""Perform computer vision-based highway traffic detection and counting.

Displays and writes a video at free-running rate, i.e. as soon as each frame is processed with detections it is displayed and written.
Quit with keypress 'q'.
Toggle pause/play with keypress 'k'.
"""

import argparse

from traffic.video_analyzer import VideoAnalyzer


def get_parsed_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input_video_path",
        "-i",
        type=str,
        required=False,
        default="data/example.mp4",
        help="Path in the filesystem to the video to perform traffic detection upon.",
    )
    parser.add_argument(
        "--output_dir_path",
        "-o",
        type=str,
        required=False,
        help=(
            "Path in the filesystem to the directory to write outputs (video and detections data) to. If not specified"
            " (recommended), will write to data/runs/<run_id> where <run_id> is a dynamically generated UUID for the"
            " run."
        ),
    )
    parser.add_argument(
        "--model_runtime_config_path",
        "-m",
        type=str,
        required=False,
        default="config/model/runtime/default.json",
        help="Path in the filesystem to the model runtime configuration file.",
    )
    parser.add_argument(
        "--tracker_runtime_config_path",
        "-t",
        type=str,
        required=False,
        default="config/tracker/runtime/default.json",
        help="Path in the filesystem to the tracker runtime configuration file.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    video_analyzer = VideoAnalyzer(
        args.model_runtime_config_path,
        args.tracker_runtime_config_path,
        args.output_dir_path,
    )
    video_analyzer.analyze(args.input_video_path)


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)
