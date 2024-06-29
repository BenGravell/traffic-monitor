"""Extract the background image from a video."""

import argparse

import numpy as np
import cv2


def extract_background(input_path: str, output_path: str, sample_fps: float) -> None:
    # Sample frames from the video
    print(f"Reading frames from video at {input_path}...", end="")
    cap = cv2.VideoCapture(input_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    num_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    num_frames_between_samples = int(fps / sample_fps)
    num_frames_sampled = num_frames_in_video // num_frames_between_samples
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = np.empty((num_frames_sampled, frame_height, frame_width, 3), dtype=np.uint8)

    for sample_idx in range(num_frames_sampled):
        frame_number = sample_idx * num_frames_between_samples
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        frames[sample_idx] = frame
    cap.release()
    print("done!")

    print("Computing median...", end="")
    background = np.median(frames, axis=0).astype(np.uint8)
    print("done!")

    print(f"Saving background image to {output_path}...", end="")
    cv2.imwrite(output_path, background)
    print("done!")


def main(args: argparse.Namespace) -> None:
    extract_background(
        input_path=args.input_video_path,
        output_path=args.output_image_path,
        sample_fps=args.sample_fps,
    )


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
        help="Path in the filesystem to the video to extract the background from.",
    )
    parser.add_argument(
        "--output_image_path",
        "-o",
        type=str,
        required=False,
        default="data/background/background.png",
        help="Path in the filesystem to write the background image to.",
    )
    parser.add_argument(
        "--sample_fps",
        "-f",
        type=float,
        required=False,
        default=2.0,
        help=(
            "Frames-per-second rate for sampling images from the video. Higher values will use more frames and may"
            " yield higher quality, but will take more time to process."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)
