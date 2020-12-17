from abc import abstractmethod, ABC
import torch
from copy import deepcopy
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from collections import OrderedDict


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


class SupervisedLabel(ABC):
    def __init__(self, y):
        self.y = y

    @abstractmethod
    def label(self):
        pass


class TrainingExample:
    """
    Training example base class.
    """

    def __init__(self, X, y=None):
        self.X = X
        self.y = y
        self._processing = OrderedDict()

    def apply_processing(self):
        """
        Apply processing methods. Updates training (X) and label (y) data in-state.
        """
        for k, v in self._processing.items():
            if v["include_y"]:
                self.X, self.y = self._processing[k]["func"](
                    self.X, self.y, *v["args"], **v["kwargs"]
                )
            else:
                self.X = self._processing[k]["func"](self.X, *v["args"], **v["kwargs"])

    def augment(self, f, *args, **kwargs):
        """
        Decorator method to add processing steps.
        Any processing methods must include training data `X` as the first input variable.
        If label-data `y` also requires augmentation, this must be included as the second input-variable.
        """
        if not "X" is f.__code__.co_varnames[0]:
            raise TypeError("X is required as first argument entry.")
        if "y" in f.__code__.co_varnames:
            requires_y = True
            if not "y" in f.__code__.co_varnames[1]:
                raise TypeError("y is required as second argument entry.")
        else:
            requires_y = False
        self._processing.update(
            {
                f.__name__: {
                    "func": f,
                    "args": args,
                    "kwargs": kwargs,
                    "include_y": requires_y,
                }
            }
        )

        def wrap_process(*args, **kwargs):
            return f(*args, **kwargs)

        return wrap_process


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

        train = TrainingExample(X=X_windows, y=y_windows)

        if self.processing["shuffle"]:

            @train.augment
            def shuffle(X, y):
                shuffle_idxs = [i for i in range(self.n_windows)]
                np.random.shuffle(shuffle_idxs)
                self.window_starts = np.array(shuffle_idxs) * self.timestep
                X = np.array(X[shuffle_idxs], dtype=float)
                y = np.array(y[shuffle_idxs], dtype=float)
                return X, y

        if "normalize" in self.processing.keys():
            if self.processing["normalize"] == "window":

                @train.augment
                def window_normalize(X):
                    X = np.array(
                        [window / np.amax(window) for window in X], dtype=float
                    )
                    return X

            elif self.processing["normalize"] == "std":

                @train.augment
                def std_normalize(X):
                    X = np.array([window / np.std(window) for window in X], dtype=float)
                    return X

        train.apply_processing()
        return train.X, train.y

    def __getitem__(self, index):
        idx = self.list_ids[index]

        X = self.dataset.get_waveforms(idx=idx)
        y = self.dataset.get_waveforms(idx=idx)

        # NOTE: Here would be where an operation can remove zero padding on traces
        # This also requires a change to PyTorch code, just use standard format (NCW) for now

        X, y = self.sequential_windower(X, y)
        return X, y
