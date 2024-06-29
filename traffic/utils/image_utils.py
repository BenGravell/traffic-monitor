"""Image utilities."""

from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from PIL.Image import Resampling
import cv2

from traffic.paths import ROOT
from traffic.type_defs import HeightWidth, Frame
from traffic.utils.colors import RgbColor, BgrColor, Monogrey


FontStyle: TypeAlias = Literal["Regular"] | Literal["Italic"] | Literal["Bold"] | Literal["BoldItalic"]


def letterbox(
    image_arr: npt.NDArray[np.uint8],
    new_size: HeightWidth,
    resample: Resampling = Resampling.BICUBIC,
    pad_color: RgbColor = Monogrey.MID.rgb,
) -> npt.NDArray[np.uint8]:
    """Resize image with unchanged aspect ratio using padding.

    Arguments:
    ---------
    image_arr: Image data array of shape (height, width, RGB).
    new_size: Target size to resize to (height, width).
    resample: Resampling method to use.
    pad_color: Color of the padded area.

    Takes inspiration from letterbox utilities in:
    1. https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/yolov3
    2. https://github.com/ultralytics/yolov3

    """
    # Convert RGB array to PIL Image
    # NOTE: PIL Image size is (width, height, num_channels)
    image = Image.fromarray(image_arr)
    scale = np.min(new_size / np.flip(image.size))
    unpad_size = (np.flip(image.size) * scale).astype(int)
    half_pad = ((new_size - unpad_size) / 2).astype(int)

    # Resize
    image = image.resize(np.flip(unpad_size), resample=resample)

    # Add padding
    new_image = Image.new("RGB", tuple(np.flip(new_size)), pad_color)
    new_image.paste(image, tuple(np.flip(half_pad)))

    return np.array(new_image)


def get_font_path(style: FontStyle) -> Path:
    return ROOT / f"fonts/IBM_Plex_Mono/IBMPlexMono-{style}.ttf"


def draw_text(
    text: str,
    font_size: int = 16,
    font_style: FontStyle = "Regular",
    text_color: RgbColor = Monogrey.WHITE.rgb,
    bkgd_color: RgbColor = Monogrey.BLACK.rgb,
    padding: int = 0,
) -> Image.Image:
    # Create a font
    font = ImageFont.truetype(get_font_path(font_style), font_size)

    # Get the dimensions of the bounding box that tightly fits the text
    temp_image = Image.new("RGB", (0, 0))
    temp_draw = ImageDraw.Draw(temp_image)
    _, _, text_width, text_height = temp_draw.multiline_textbbox((0, 0), text, font=font)

    # Add padding and create background image
    width, height = text_width + 2 * padding, text_height + 2 * padding
    image = Image.new("RGB", (width, height), bkgd_color)
    draw = ImageDraw.Draw(image)

    # Draw the text on the background
    text_x, text_y = padding, padding
    draw.multiline_text((text_x, text_y), text, fill=text_color, font=font)

    return image


def box_label(
    frame: Frame,
    box: tuple[int, int, int, int],
    label: str,
    color: BgrColor = Monogrey.BLACK.rgb,
    text_color: BgrColor = Monogrey.WHITE.rgb,
    font_size: int = 12,
    font_style: FontStyle = "Regular",
    text_padding: int = 4,
    line_width: int = 4,
    alpha: float = 1.0,
) -> None:
    """Add one labeled box to an image frame.

    This is an in-place operation on the frame.
    """
    # Draw a rectangle for the bounding box.
    p1, p2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(frame, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)

    # Draw centroid dot
    centroid = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    cv2.circle(frame, centroid, 9, (0, 0, 0), -1)
    cv2.circle(frame, centroid, 7, (255, 255, 255), -1)
    cv2.circle(frame, centroid, 5, color, -1)

    # Draw a text label
    text_im_pil = draw_text(
        label,
        text_color=text_color[::-1],
        bkgd_color=color[::-1],
        font_style=font_style,
        font_size=font_size,
        padding=text_padding,
    )
    text_im_width, text_im_height = text_im_pil.size

    # Extract frame shape
    frame_height, frame_width = frame.shape[0:2]

    # Convert the Pillow image to an OpenCV-compatible format
    text_im_cv = cv2.cvtColor(np.array(text_im_pil), cv2.COLOR_RGB2BGR)

    # Define the position to paste the text image
    skip_text = False
    # Align text box with left edge of bounding box.
    position_x = box[0]
    # Place text box above bounding box.
    position_y = box[1] - text_im_height

    # Right of the text box would fall outside the frame - align the text box with the right edge of the bounding box.
    if position_x + text_im_width >= frame_width:
        position_x = box[2] - text_im_width
    # Top of the text box would fall outside the frame - move the text box below the bounding box.
    if position_y < 0:
        position_y = box[3]

    # If we still cannot fit the box in the frame, give up and skip.
    if (position_x < 0) or (position_y > frame_height):
        skip_text = True

    if not skip_text:
        # Draw the text image
        position = (position_x, position_y)
        frame[position[1] : position[1] + text_im_height, position[0] : position[0] + text_im_width] = (
            ((1 - alpha) * frame[position[1] : position[1] + text_im_height, position[0] : position[0] + text_im_width])
            + (alpha * text_im_cv)
        ).astype(np.uint8)
