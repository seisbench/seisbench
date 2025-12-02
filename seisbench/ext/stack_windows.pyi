from typing import Literal

import numpy as np

def stack_windows(
    pred_windows: np.ndarray,
    offsets: np.ndarray,
    method: Literal["avg", "max"] = "avg",
    n_threads: int = 1,
) -> np.ndarray: ...
