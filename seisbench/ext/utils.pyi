from typing import Literal, Sequence

import numpy as np

def stack_windows(
    windows: np.ndarray | Sequence[np.ndarray],
    offsets: np.ndarray,
    method: Literal["avg", "max"] = "avg",
) -> np.ndarray: ...
def get_edge_indices(
    offsets: np.ndarray,
    edge_value: float = 0.0,
) -> tuple[int, int]: ...
