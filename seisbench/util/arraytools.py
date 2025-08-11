import numpy as np
import torch


def torch_detrend(x: torch.Tensor) -> torch.Tensor:
    """
    Detrends a tensor along the last axis using a linear fit.

    :param x: Array to detrend
    :returns: Detrended array
    """
    m = x.mean(dim=-1, keepdim=True)
    samples = torch.linspace(-1, 1, x.shape[-1], device=x.device, dtype=x.dtype)
    slope = ((x - m) * samples).sum(dim=-1, keepdims=True) / (samples * samples).sum(
        dim=-1, keepdims=True
    )

    return x - m - slope * samples


def pad_packed_sequence(seq: list[np.ndarray]) -> np.ndarray:
    """
    Packs a list of arrays into one array by adding a new first dimension and padding where necessary.

    :param seq: List of numpy arrays
    :return: Combined arrays
    """
    max_size = np.array([max([x.shape[i] for x in seq]) for i in range(seq[0].ndim)])

    new_seq = []
    for i, elem in enumerate(seq):
        d = max_size - np.array(elem.shape)
        if (d != 0).any():
            pad = [(0, d_dim) for d_dim in d]
            new_seq.append(np.pad(elem, pad, "constant", constant_values=0))
        else:
            new_seq.append(elem)

    return np.stack(new_seq, axis=0)
