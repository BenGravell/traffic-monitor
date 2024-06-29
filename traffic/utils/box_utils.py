import numpy as np

from traffic.type_defs import LtrbBoxes, XywhBoxes, HeightWidth, XyCentroids


def xywh2ltrb(xywh: XywhBoxes) -> LtrbBoxes:
    """Converts box format from [(center_x, center_y, width, height) to (left, top, right, bottom)."""
    ltrb = np.zeros_like(xywh)
    center_x = xywh[..., 0]
    center_y = xywh[..., 1]
    half_wide = xywh[..., 2] / 2
    half_high = xywh[..., 3] / 2
    ltrb[..., 0] = center_x - half_wide  # left
    ltrb[..., 1] = center_y - half_high  # top
    ltrb[..., 2] = center_x + half_wide  # right
    ltrb[..., 3] = center_y + half_high  # bottom
    return ltrb


def clip_boxes(boxes: LtrbBoxes, size: HeightWidth) -> None:
    """Clips bounding boxes to within the specified image size.

    In-place operation.

    Arguments:
    ---------
    boxes: (N, 4) array of box coordinates in (left, top, right, bottom) format.
    size: (height, width) tuple of frame size to clip boxes to.

    """
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, size[1])  # left, right
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, size[0])  # top, bottom


def scale_boxes(
    boxes: LtrbBoxes,
    from_shape: HeightWidth,
    to_shape: HeightWidth,
) -> LtrbBoxes:
    """Scale boxes from one frame shape to another.

    Arguments:
    ---------
    boxes: Bounding boxes in (left, top, right, bottom) format.
    from_shape: Frame shape (height, width) for the arg boxes.
    to_shape: Frame shape (height, width) for the returned boxes.

    Returns:
    -------
    Boxes in (left, top, right, bottom) format, scaled to conform to the to_shape.

    """
    gain = min(from_shape[0] / to_shape[0], from_shape[1] / to_shape[1])
    pad = (from_shape[1] - to_shape[1] * gain) / 2, (from_shape[0] - to_shape[0] * gain) / 2

    boxes[..., [0, 2]] -= pad[0]
    boxes[..., [1, 3]] -= pad[1]
    boxes[..., 0:4] /= gain
    clip_boxes(boxes, to_shape)
    return boxes


def centroids_from_boxes(boxes: LtrbBoxes) -> XyCentroids:
    return np.vstack([np.mean(boxes[:, [0, 2]], axis=1), np.mean(boxes[:, [1, 3]], axis=1)]).T
