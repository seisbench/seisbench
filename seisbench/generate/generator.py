from torch.utils.data import Dataset


class GenericGenerator(Dataset):
    def __init__(self, dataset):
        """
        A generic data generator which can be used to build preprocessing and data augmentation pipelines.
        The data generator subclasses the pytorch Dataset class and can therefore be used directly with DataLoaders
        in pytorch. The processing pipeline of the generator is defined through a series of processing steps or
        augmentations. For each data sample, the generator calls the augmentations in order.
        Information between the augmentation steps is passed through a state dict.
        The state dict is a python dictionary mapping keys to a tuple (data, metadata).
        In getitem, the generator automatically populates the initial dictionary with the waveforms
        and the corresponding metadata for the row from the underlying data set using the key "X".
        After applying all augmentation, the generator removes all metadata information.
        This means that the output dict only maps keys to the data part.
        Any metadata that should be output needs to explicitly be written to data.

        Augmentation can be either callable classes of functions.
        Functions are usually best suited for simple operations, while callable classes offer more configuration
        options. SeisBench already offers a set of standard augmentations for augmentation and preprocessing,
        e.g., for window selection, data normalization or different label encodings,
        which should cover many common use cases.
        For details on implementing custom augmentations we suggest looking at the examples provided.

        SeisBench augmentations by default always work on the key "X".
        Label generating augmentations by default put labels into the key "y".
        However, for more complex workflows, the augmentations can be adjusted using the key argument.
        This allows in particular none-sequential augmentation sequences.

        :param dataset: The underlying SeisBench data set.
        :type dataset: seisbench.data.WaveformDataset or seisbench.data.MultiWaveformDataset
        """
        self._augmentations = []
        self.dataset = dataset
        super().__init__()

    def augmentation(self, f):
        """
        Decorator for augmentations.
        """
        self._augmentations.append(f)

        return f

    def add_augmentations(self, augmentations):
        """
        Adds a list of augmentations to the generator. Can not be used as decorator.

        :param augmentations: List of augmentations
        :type augmentations: list[callable]
        """
        if not isinstance(augmentations, list):
            raise TypeError(
                "The argument of add_augmentations must be a list of augmentations."
            )

        self._augmentations.extend(augmentations)

    def __str__(self):
        summary = f"GenericGenerator with {len(self._augmentations)} augmentations:\n"
        for i, aug in enumerate(self._augmentations):
            summary += f" {i + 1}.\t{str(aug)}\n"
        return summary

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        state_dict = {"X": self.dataset.get_sample(idx)}

        # Recursive application of augmentation processing methods
        for func in self._augmentations:
            func(state_dict)

        # Remove all metadata from the output
        state_dict = {k: v[0] for k, v in state_dict.items()}

        return state_dict


class SteeredGenerator(GenericGenerator):
    """
    This data generator follows the same principles as the :py:func:`~GenericGenerator`.
    However, in contrast to the :py:func:`~GenericGenerator` the generator is controlled by a dataframe with control
    information. Each row in the control dataframe corresponds to one example output by the generator.
    The dataframe holds two types of information. First, information identifying the traces, provided using the
    `trace_name` (required), `trace_chunk` (optional), and `trace_dataset` (optional). See the description of
    :py:func:`~seisbench.data.base.WaveformDataset.get_idx_from_trace_name` for details.
    Second, additional information for the augmentations.
    This additional information is stored in `state_dict["_control_"]` as a dict.
    This generator is particularly useful for evaluation, e.g., when extracting predefined windows from a trace.

    Note that the "_control_" group will usually not be modified by augmentations.
    This means, that for example after a window selection, sample positions might be off.
    To automatically handle these, you must explicitly put the relevant control information
    into the state_dict metadata of the relevant key.

    .. warning::
        This generator should in most cases not be used for changing label distributions by resampling the dataset.
        For this application, we recommend using a
        `pytorch Sampler <https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler>`_.

    :param dataset: The underlying SeisBench data set
    :type dataset: seisbench.data.WaveformDataset or seisbench.data.MultiWaveformDataset
    :param metadata: The additional information as pandas dataframe.
                     Each row corresponds to one sample from the generator.
    :type metadata: pandas.DataFrame
    """

    def __init__(self, dataset, metadata):
        self.metadata = metadata
        super().__init__(dataset)

    def __str__(self):
        summary = f"SteeredGenerator with {len(self._augmentations)} augmentations:\n"
        for i, aug in enumerate(self._augmentations):
            summary += f" {i + 1}.\t{str(aug)}\n"
        return summary

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        control = self.metadata.iloc[idx].to_dict()
        kwargs = {
            "trace_name": control["trace_name"],
            "chunk": control.get("trace_chunk", None),
            "dataset": control.get("trace_dataset", None),
        }
        data_idx = self.dataset.get_idx_from_trace_name(**kwargs)
        state_dict = {"X": self.dataset.get_sample(data_idx), "_control_": control}

        # Recursive application of augmentation processing methods
        for func in self._augmentations:
            func(state_dict)

        # Remove control information
        del state_dict["_control_"]
        # Remove all metadata from the output
        state_dict = {k: v[0] for k, v in state_dict.items()}

        return state_dict
