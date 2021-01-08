import seisbench
import seisbench.util as util

from abc import abstractmethod, ABC
from pathlib import Path
import os
import torch
import torch.nn as nn


class SeisBenchModel(nn.Module):
    def __init__(self, name, citation=None):
        super().__init__()
        self._name = name
        self._citation = citation
        self._weights_docstring = None

    @property
    def name(self):
        return self._name

    @property
    def citation(self):
        return self._citation

    @property
    def weights_docstring(self):
        return self._weights_docstring

    def _model_path(self):
        return Path(seisbench.cache_root, "models", self.name.lower())

    def _remote_path(self):
        return os.path.join(seisbench.remote_root, "models", self.name.lower())

    def load_pretrained(self, name):
        weight_path = self._model_path() / f"{name}.pt"
        doc_path = self._model_path() / f"{name}.txt"
        if not weight_path.is_file():
            seisbench.logger.info(f"Weight file {name} not in cache. Downloading...")
            weight_path.parent.mkdir(exist_ok=True, parents=True)

            remote_weight_path = os.path.join(self._remote_path(), f"{name}.pt")
            util.download_http(remote_weight_path, weight_path)

            remote_doc_path = os.path.join(self._remote_path(), f"{name}.txt")
            try:
                util.download_http(remote_doc_path, doc_path, progress_bar=False)
            except ValueError:
                pass

        self.load_state_dict(torch.load(weight_path))

        if doc_path.is_file():
            with open(doc_path, "r") as f:
                self._weights_docstring = f.read()
        else:
            self._weights_docstring = ""


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
