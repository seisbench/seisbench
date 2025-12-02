import numpy as np

from seisbench.ext import stack_windows


def test_arrange_average():
    n_windows = 100
    n_samples = 1000
    n_channels = 3

    rnd = np.random.RandomState(42)
    pred_windows = rnd.randn(n_windows, n_samples, n_channels).astype(np.float32)
    offsets = rnd.randint(0, n_samples * 300, size=n_windows)

    _ = stack_windows(
        pred_windows=pred_windows,
        offsets=offsets,
        method="avg",
        n_threads=1,
    )


def test_arrange_max():
    n_windows = 100
    n_samples = 1000
    n_channels = 3

    rnd = np.random.RandomState(42)
    pred_windows = rnd.randn(n_windows, n_samples, n_channels).astype(np.float32)
    offsets = rnd.randint(0, n_samples * 300, size=n_windows)

    _ = stack_windows(
        pred_windows=pred_windows,
        offsets=offsets,
        method="max",
        n_threads=1,
    )
