from abc import abstractmethod, ABC
import torch
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


class WaveformGenerator(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self.list_ids = [i for i in range(len(dataset))]

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    @abstractmethod
    def __getitem__(self):
        pass


class PytorchWindowGenerator(WaveformGenerator, Dataset):
    """
    PyTorch compatible sequential window generator.
    """

    def __init__(
        self,
        dataset,
        windowlen,
        timestep,
        processing={"shuffle": False, "normalize": None},
    ):
        super().__init__(dataset)
        self.processing = processing
        self.windowlen = windowlen
        self.timestep = timestep

        self._idx = 0
        self.window_starts = []
        self.bool_mask = np.zeros(len(dataset), dtype=np.bool)

    def create_windows(self, X, y):
        """
        Create N sequential windows from trace data.
        """
        self.n_windows = (X.shape[1] - self.windowlen) // self.timestep
        X_windows = []
        y_windows = []
        # Reset window start idxs
        self.window_starts = (
            np.array([i for i in range(self.n_windows)]) * self.timestep
        )
        for i in range(self.n_windows):
            # Data is transformed to [N, C, W]
            X_windows.append(
                X.T[i * self.timestep : i * self.timestep + self.windowlen].T
            )
            y_windows.append(
                y.T[i * self.timestep : i * self.timestep + self.windowlen].T
            )

        return np.array(X_windows), np.array(y_windows)

    def sequential_windower(self, X, y):
        """
        Sequentially windows a training-label pair.
        :param X: Training example, format (W, C)
        :param y: Label example, format (W, C)
        :return: (X, y) Windowed training timeseries, format (N, C, W)
        :return type: tuple(torch.Tensor, torch.Tensor)
        """

        X_windows, y_windows = self.create_windows(X, y)

        if self.processing["shuffle"]:
            shuffle_idxs = [i for i in range(self.n_windows)]
            np.random.shuffle(shuffle_idxs)
            self.window_starts = np.array(shuffle_idxs) * self.timestep
            X_windows = np.array(X_windows[shuffle_idxs], dtype=float)
            y_windows = np.array(y_windows[shuffle_idxs], dtype=float)

        elif "normalize" in self.processing.keys():

            if self.processing["normalize"] == "window":
                X_windows = np.array(
                    [window / np.amax(window) for window in X_windows], dtype=float
                )
                X_windows.clip(1e-15, 1 - 1e-15)

            elif self.processing["normalize"] == "std":
                X_windows = np.array(
                    [window / np.std(window) for window in X_windows], dtype=float
                )

        return X_windows, y_windows

    def __getitem__(self, index):
        idx = self.list_ids[index]
        self.bool_mask[idx] = True

        X = self.dataset.get_waveforms(mask=self.bool_mask)
        y = self.dataset.get_waveforms(mask=self.bool_mask)

        # NOTE: Here would be where an operation can remove zero padding on traces
        # This also requires a change to PyTorch code, just use standard format (NCW) for now

        X, y = self.sequential_windower(X.squeeze(), y.squeeze())
        self.bool_mask[idx] = False
        return X, y
