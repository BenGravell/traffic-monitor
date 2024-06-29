import numpy as np
import cv2

import traffic.utils.image_utils as image_utils
from traffic.region import RegionGroup
from traffic.type_defs import Frame


class Summary:
    def __init__(self, region_group: RegionGroup, frame_number: int = 0):
        self.region_group = region_group
        self.frame_number = frame_number

    def update(self, frame_number: int) -> None:
        self.frame_number = frame_number

    def to_display_str(self) -> str:
        headers = [f"Frame # {self.frame_number:04d}", "Incoming", "Outgoing"]
        values = [
            [
                "Current",
                f'{self.region_group.region_map["INCOMING"].stats.count_current:4d}',
                f'{self.region_group.region_map["OUTGOING"].stats.count_current:4d}',
            ],
            [
                "Total",
                f'{self.region_group.region_map["INCOMING"].stats.count_total:4d}',
                f'{self.region_group.region_map["OUTGOING"].stats.count_total:4d}',
            ],
        ]

        column_widths = [max(len(str(val)) for val in col) for col in zip(headers, *values)]
        horizontal_lines = ["-" * col_width for col_width in column_widths]

        def format_row(row: list[str]) -> str:
            return "  |  ".join(f"{val:<{width}}" for val, width in zip(row, column_widths))

        display_str_lines = [format_row(row) for row in [headers, horizontal_lines, *values]]
        display_str_lines[1] = display_str_lines[1].replace(" ", "-")

        return "\n".join(display_str_lines)

    def to_image(self) -> Frame:
        summary_im_pil = image_utils.draw_text(
            self.to_display_str(),
            text_color=(0, 0, 0),
            bkgd_color=(255, 255, 255),
            font_style="Regular",
            font_size=30,
            padding=20,
        )
        return cv2.cvtColor(np.array(summary_im_pil), cv2.COLOR_RGB2BGR)
