import seisbench

from abc import abstractmethod, ABC
from pathlib import Path
import torch.nn as nn


class SeisBenchModel(nn.Module):
    def __init__(self, name, citation=None):
        super().__init__()
        self._name = name
        self._citation = citation

    @property
    def name(self):
        return self._name

    @property
    def citation(self):
        return self._citation

    def _model_path(self):
        return Path(seisbench.cache_root, "models", self.name.lower())

    def load_pretrained(self, name):
        weight_path = self._model_path() / f"{name}.pt"
        if not weight_path.is_file():
            seisbench.logger.info(f"Weight file {name} not in cache. Downloading...")
            # TODO: Download from SeisBench endpoint
            raise NotImplementedError("Downloading models is not yet implemented")

        self.load(weight_path)


class WaveformModel(ABC):
    """
    Abstract interface for models processing waveforms.
    Class does not inherit from SeisBenchModel to allow implementation of non-DL models with this interface.
    """

    @abstractmethod
    def annotate(self, stream, *args, **kwargs):
        """
        Annotates a stream
        :param stream: Obspy stream to annotate
        :param args:
        :param kwargs:
        :return: Obspy stream of annotations
        """
        pass

    @abstractmethod
    def classify(self, stream, *args, **kwargs):
        """
        Classifies the stream. As there are no
        :param stream: Obspy stream to classify
        :param args:
        :param kwargs:
        :return: A classification for the full stream, e.g., signal/noise or source magnitude
        """
        pass
