from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt

Frame: TypeAlias = npt.NDArray[np.uint8]

XywhBoxes: TypeAlias = npt.NDArray[Any]
LtrbBoxes: TypeAlias = npt.NDArray[Any]

XyCentroids: TypeAlias = npt.NDArray[Any]

HeightWidth: TypeAlias = tuple[int, int]
