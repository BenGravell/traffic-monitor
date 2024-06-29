from typing import TypeAlias, cast
from PIL import ImageColor

RgbColor: TypeAlias = tuple[int, int, int]
BgrColor: TypeAlias = tuple[int, int, int]
HexColor: TypeAlias = str


class Color:
    def __init__(self, r: int, g: int, b: int) -> None:
        self.r = r
        self.g = g
        self.b = b

    @classmethod
    def from_rgb(cls, rgb: RgbColor) -> "Color":
        return cls(rgb[0], rgb[1], rgb[2])

    @classmethod
    def from_bgr(cls, bgr: BgrColor) -> "Color":
        return cls(bgr[2], bgr[1], bgr[0])

    @classmethod
    def from_hex(cls, hex: str) -> "Color":
        return cls(*ImageColor.getrgb(hex))

    @property
    def rgb(self) -> RgbColor:
        return cast(RgbColor, (self.r, self.g, self.b))

    @property
    def bgr(self) -> BgrColor:
        return cast(BgrColor, (self.b, self.g, self.r))

    @property
    def hex(self) -> HexColor:
        return cast(HexColor, "#%02x%02x%02x" % self.rgb)


class Vivid:
    RED = Color(214, 40, 40)
    BLUE = Color(40, 40, 214)


class Atom:
    RED = Color.from_hex("#c13f21")
    ORANGE = Color.from_hex("#d36e2d")
    YELLOW = Color.from_hex("#dda032")
    GREEN = Color.from_hex("#78af9f")
    BLUE = Color.from_hex("#659cc8")
    PURPLE = Color.from_hex("#584b4f")
    LIGHT = Color.from_hex("#eed9b7")
    DARK = Color.from_hex("#343233")


class Monokai:
    RED = Color.from_hex("#ff6188")
    ORANGE = Color.from_hex("#fc9867")
    YELLOW = Color.from_hex("#ffd866")
    GREEN = Color.from_hex("#a9dc76")
    BLUE = Color.from_hex("#78dce8")
    PURPLE = Color.from_hex("#ab9df2")


class Monogrey:
    WHITE = Color(255, 255, 255)
    LIGHT = Color(192, 192, 192)
    MID = Color(128, 128, 128)
    DARK = Color(64, 64, 64)
    BLACK = Color(0, 0, 0)
