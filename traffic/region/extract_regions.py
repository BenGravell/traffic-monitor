"""Extract non-background regions from a video."""

import argparse
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.ndimage import center_of_mass
import pandas as pd
import cv2
import PIL.Image
import PIL.ImageFilter


def compute_diff_count_mask(
    background_image_path: str,
    video_path: str,
    min_abs_diff: int,
    min_abs_diff_count: int,
) -> npt.NDArray[np.bool_]:
    # Read the background image
    background = cv2.imread(background_image_path)

    print(f"Computing differences in video at {video_path} from background at {background_image_path}...", end="")
    cap = cv2.VideoCapture(video_path)
    frame_width = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (frame_height, frame_width, 3) != background.shape:
        msg = "Shape mismatch between background and video."
        raise ValueError(msg)

    abs_diff_count = np.zeros(background.shape, dtype=np.int64)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        abs_diff = cv2.absdiff(frame, background)
        abs_diff_count += abs_diff >= min_abs_diff

    cap.release()
    print("done!")

    # Binary mask by checking if diff count exceeded threshold in any of the RGB channels
    mask = np.any(abs_diff_count >= min_abs_diff_count, axis=2)

    return mask


def extract_left_right_mask_images(mask: npt.NDArray[np.bool_]) -> dict[str, PIL.Image.Image]:
    # Find all connected components in the binary mask
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8) * 255, connectivity=8)

    # Consider all components except the first one, which corresponds to the background.
    stat_area_non_background = stats[1:, cv2.CC_STAT_AREA]
    label_idxs_non_background = np.arange(1, num_labels)
    # Pick the top 2 largest regions.
    top_2_largest_label_idxs = label_idxs_non_background[np.argsort(-stat_area_non_background)[0:2]]

    # Assign the left and right portions to separate masks.
    # Infer left and right based on the center of mass of the components.
    center_of_mass_dict = {i: center_of_mass(labels == i) for i in top_2_largest_label_idxs}
    center_of_mass_df = pd.DataFrame.from_dict(center_of_mass_dict, orient="index")
    label_idx_left = int(center_of_mass_df[1].idxmin())
    label_idx_right = int(center_of_mass_df[1].idxmax())

    # Create output masks with the same size as the input.
    def create_output_mask_img(label_idx: int) -> PIL.Image.Image:
        output_mask = np.zeros_like(mask, dtype=np.uint8)
        output_mask[labels == label_idx] = 255
        return PIL.Image.fromarray(output_mask)

    return {"left": create_output_mask_img(label_idx_left), "right": create_output_mask_img(label_idx_right)}


def blur_and_threshold(mask: npt.NDArray[np.bool_], radius: int, threshold: int = 128) -> npt.NDArray[np.bool_]:
    return (
        np.array(PIL.Image.fromarray(mask).convert("L").filter(PIL.ImageFilter.GaussianBlur(radius=radius)))
        >= threshold
    )


def hide_top(mask: npt.NDArray[np.bool_], top_fraction: float) -> npt.NDArray[np.bool_]:
    frame_height = mask.shape[0]
    mask[0 : int(top_fraction * frame_height)] = False
    return mask


def main(args: argparse.Namespace) -> None:
    # Get base mask
    mask = compute_diff_count_mask(
        background_image_path=args.input_background_image_path,
        video_path=args.input_video_path,
        min_abs_diff=args.min_abs_diff,
        min_abs_diff_count=args.min_abs_diff_count,
    )

    # Blur and threshold to remove tiny spurious regions.
    mask = blur_and_threshold(mask, radius=args.blur_radius)

    # Chop off the top of the image where the highway gets far away and vehicles are too small to detect reliably.
    mask = hide_top(mask, top_fraction=args.top_fraction)

    # Get the left and right highway masks.
    highway_mask_map = extract_left_right_mask_images(mask)

    # Write to files.
    region_dir_path = Path(args.output_region_mask_image_dir_path)
    region_dir_path.mkdir(parents=True, exist_ok=True)
    for key, image in highway_mask_map.items():
        image.save(region_dir_path / f"highway_{key}.png")


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
        help="Path in the filesystem to the video to extract regions from.",
    )
    parser.add_argument(
        "--input_background_image_path",
        "-b",
        type=str,
        required=False,
        default="data/background/background.png",
        help="Path in the filesystem to the background image.",
    )
    parser.add_argument(
        "--output_region_mask_image_dir_path",
        "-o",
        type=str,
        required=False,
        default="data/regions",
        help="Path in the filesystem for the directory in which to write region mask images.",
    )
    parser.add_argument(
        "--min_abs_diff",
        type=int,
        required=False,
        default=64,
        help=(
            "Minimum absolute difference in pixel value to consider as 'diverged'. Only 'diverged' pixels go towards"
            " the count for determining non-background pixels."
        ),
    )
    parser.add_argument(
        "--min_abs_diff_count",
        type=int,
        required=False,
        default=1,
        help="Minimum count of frames with 'diverged' values for each pixel to count as non-background.",
    )
    parser.add_argument(
        "--blur_radius",
        type=int,
        required=False,
        default=5,
        help=(
            "Radius for Gaussian blur to use in smoothing out the raw mask. Passed as radius kwarg to"
            " PIL.ImageFilter.GaussianBlur()."
        ),
    )
    parser.add_argument(
        "--top_fraction",
        type=float,
        required=False,
        default=0.6,
        help=(
            "Proportion of the height of the image from the top edge to exclude from region masks. The greater the"
            " number, the more of the image will be excluded. Used to exclude noisy detections from participating in"
            " tracking. Must be between 0 and 1."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parsed_args()
    main(args)
