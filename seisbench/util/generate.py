from torch.utils.data import Dataset
import numpy as np


class GenericGenerator(Dataset):
    def __init__(self, dataset):
        self._augmentations = []
        self.dataset = dataset
        super().__init__()

    def augmentation(self, f):
        """
        Decorator for augmentations.
        """
        self._augmentations.append(f)

        return f

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        state_dict = {
            "waveforms": self.dataset.get_waveforms(idx),
            "metadata": self.dataset.metadata.iloc[idx].to_dict(),
        }

        # Recursive application of augmentation processing methods
        for func in self._augmentations:
            func(state_dict)

        return state_dict


class SlidingWindow:
    def __init__(self, windowlen=600, timestep=200):
        self.windowlen = windowlen
        self.timestep = timestep

    def __call__(self, state_dict):
        windowlen = self.windowlen
        timestep = self.timestep

        n_windows = (state_dict["waveforms"].shape[1] - windowlen) // timestep
        X_windows = []
        y_windows = []

        for i in range(n_windows):
            # Data is transformed to [N, C, W]
            X_windows.append(
                state_dict["waveforms"].T[i * timestep : i * timestep + windowlen].T
            )
            y_windows.append(
                state_dict["waveforms"].T[i * timestep : i * timestep + windowlen].T
            )
        # Add sequential windows to state_dict
        state_dict["X"] = np.array(X_windows)
        state_dict["y"] = np.array(y_windows)


class Normalize:
    def __init__(self, norm_type="window"):
        self.norm_type = norm_type

    def __call__(self, state_dict):
        if self.norm_type == "window":
            state_dict["X"] = np.array(
                [window / np.amax(window) for window in state_dict["X"]], dtype=float
            )
        elif self.norm_type == "std":
            state_dict["X"] = np.array(
                [window / np.std(window) for window in state_dict["X"]], dtype=float
            )


class Demean:
    def __init__(self, axis=1, key="X"):
        self.axis = axis
        self.key = key

    def __call__(self, state_dict):
        state_dict[self.key] = state_dict[self.key].astype(float) - np.mean(
            state_dict[self.key], axis=self.axis, keepdims=True
        )
