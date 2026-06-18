import copy
import numpy as np

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import h5py


class FixedDASWindow:
    """
    A simple windower that returns fixed windows.
    In addition, the windower rewrites all annotations to point to the correct samples after window
    selection. Window position and shape can be set either at initialization or separately in each call.
    The latter is primarily intended for more complicated windowers inheriting from this class.

    This class mirrors in most aspects :py:class:`~seisbench.generate.windows.FixedWindow`.

    .. warning ::

        After windowing, the window will always be in memory. It is therefore advisable not to load unnecessarily large
        windows.

    :param p0: Tuple with the indices of the first corner.
    :param shape: Tuple with the output shape.
    :param strategy: Strategy if the requested window is not fully contained in the data.
                     See :py:class:`~seisbench.generate.windows.FixedWindow` for details on the available strategies.
    :param key: The keys for reading from and writing to the state dict.
                If key is a single string, the corresponding entry in state dict is modified.
                Otherwise, a 2-tuple is expected, with the first string indicating the key to
                read from and the second one the key to write to.
    """

    def __init__(
        self,
        p0: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
        strategy: str = "fail",
        key: str | tuple[str, str] = "X",
    ):
        self.p0 = p0
        self.shape = shape
        self.strategy = strategy
        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        if value not in ["fail", "pad", "move", "variable"]:
            raise ValueError(
                f"Unknown strategy '{value}'. Options are 'fail', 'pad', 'move', 'variable'"
            )

        self._strategy = value

    def __call__(
        self,
        state_dict: dict[str, Any],
        p0: tuple[int, ...] | None = None,
        shape: tuple[int, ...] | None = None,
    ):
        record, metadata = state_dict[self.key[0]]
        p0, shape = self._validate_parameters(record, p0, shape)

        if self.key[0] != self.key[1]:
            # Ensure metadata is not modified inplace unless input and output key are anyhow identical
            metadata = copy.deepcopy(metadata)

        if self.strategy == "pad":
            padding = [
                (min(s, max(0, -p)), min(s, max(0, p + s - L)))
                for p, s, L in zip(p0, shape, record.shape)
            ]  # Left and right padding
        else:
            padding = None

        window = record[
            tuple(slice(max(p, 0), max(p + s, 0)) for p, s in zip(p0, shape))
        ]

        annotations = metadata["__annotations__"]
        for k in annotations:
            annotations[k] = annotations[k][
                max(0, p0[1]) : max(0, p0[1] + shape[1])
            ] - max(p0[0], 0)

        if padding is not None:
            window = np.pad(window, padding, "constant", constant_values=0)
            for k in annotations:
                # Type conversion is required to allow putting NaN values in case the inputs are ints
                annotations[k] = np.pad(
                    annotations[k].astype(np.float32),
                    padding[1:2],
                    "constant",
                    constant_values=np.nan,
                ) - min(0, p0[0])

        metadata["__annotations__"] = annotations

        window = np.asarray(
            window
        )  # For consistency, ensure that array is always in memory.

        state_dict[self.key[1]] = (window, metadata)

    def _validate_parameters(
        self,
        record: np.ndarray | h5py.Dataset,
        p0: tuple[int, ...] | None,
        shape: tuple[int, ...] | None,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        if p0 is None:
            p0 = self.p0
        if shape is None:
            shape = self.shape

        if p0 is None:
            raise ValueError("Start position must be set in either init or call.")
        if shape is None:
            raise ValueError("Window length must be set in either init or call.")

        if not (len(p0) == len(shape) == record.ndim):
            raise ValueError(
                f"Inconsistent dimension number between p0, target shape, and input record. "
                f"({len(p0)}, {len(shape)}, {record.ndim})"
            )

        if self.strategy in ["fail", "move"] and any(
            s > L for s, L in zip(shape, record.shape)
        ):
            raise ValueError(
                "Requested window is larger than input record. This is not permitted for the strategies "
                "fail and move."
            )

        if any(p < 0 for p in p0) and self.strategy == "fail":
            raise ValueError("Negative indexing is not supported for strategy fail.")

        elif self.strategy == "move":
            p0 = tuple(max(p, 0) for p in p0)
            # This can never result in negative values, because otherwise the check above would have failed
            p0 = tuple(min(p, L - s) for p, s, L in zip(p0, shape, record.shape))

        elif self.strategy == "variable":
            new_p0, new_shape = [], []
            for p, s in zip(p0, shape):
                if p < 0:
                    s = max(0, s + p)  # Shorten target window len
                    p = 0
                new_p0.append(p)
                new_shape.append(s)
            p0 = tuple(new_p0)
            shape = tuple(new_shape)

        if self.strategy == "fail" and any(
            p + s > L for p, s, L in zip(p0, shape, record.shape)
        ):
            raise ValueError(
                "Requested window is not fully contained in record. This is not permitted "
                "for the strategy fail."
            )

        return p0, shape
