from torch.utils.data import Dataset


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
        state_dict = self.dataset.get_sample(idx)

        # Recursive application of augmentation processing methods
        for func in self._augmentations:
            func(state_dict)

        return state_dict
