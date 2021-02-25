import torch
import numpy as np
from torch.utils.data import Dataset
from collections import OrderedDict
import inspect


class GenericGenerator(Dataset):
    def __init__(self, dataset):
        self._augmentations = OrderedDict()
        self.dataset = dataset
        super().__init__()

    def augmentation(self, f):
        """
        Decorator for augmentations.
        """
        # Track augmentation methods and submitted params
        self._augmentations[f.__name__] = {"func": f, "params": {}}

        # TODO: See whether simple check is required that state_dict is passed as first arg
        for param, value in inspect.signature(f).parameters.items():
            if param is not "state_dict":
                self._augmentations[f.__name__]["params"][param] = value.default

        def wrap_process(self, *args, **kwargs):
            return f(*args, **kwargs)

        return wrap_process

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
        for func_name, func_params in self._augmentations.items():

            func_params["func"](state_dict, **func_params["params"])

        return state_dict
