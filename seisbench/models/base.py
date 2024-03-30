import asyncio
import json
import math
import os
import re
import tempfile
import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from queue import PriorityQueue
from typing import Any, Optional
from urllib.parse import urljoin

import bottleneck as bn
import nest_asyncio
import numpy as np
import obspy
import torch
import torch.nn as nn
import torch.nn.functional as F
from obspy.signal.trigger import trigger_onset
from packaging import version

import seisbench
import seisbench.util as util
from seisbench.util import in_notebook

if in_notebook():
    # Jupyter notebooks have their own asyncio loop and will crash `annotate/classify`
    # if not patched with nest_asyncio
    nest_asyncio.apply()


def _cache_migration_v0_v3():
    """
    Migrates model cache from v0 to v3 if necessary
    """
    if seisbench.cache_model_root.is_dir():
        return  # Migration already done

    if not (seisbench.cache_root / "models").is_dir():
        return  # No legacy cache

    seisbench.logger.info("Migrating model cache to version 3")

    # Move cache
    seisbench.cache_model_root.mkdir(parents=True)
    for path in (seisbench.cache_root / "models").iterdir():
        if path.name == "v3":
            continue

        path.rename(seisbench.cache_model_root / path.name)

    if (seisbench.cache_model_root / "phasenet").is_dir():
        # Rename phasenet to phasenetlight
        (seisbench.cache_model_root / "phasenet").rename(
            seisbench.cache_model_root / "phasenetlight"
        )


class SeisBenchModel(nn.Module):
    """
    Base SeisBench model interface for processing waveforms.

    :param citation: Citation reference, defaults to None.
    :type citation: str, optional
    """

    # The model can list combination of weights and versions that should cause a warning.
    # Each entry is a 3-tuple:
    # - weights name regex
    # - weights_version
    # - warning message
    _weight_warnings = []

    def __init__(self, citation=None):
        super().__init__()
        self._citation = citation
        self._weights_docstring = None
        self._weights_version = None
        self._weights_metadata = None

    def __str__(self):
        return f"SeisBench model\t\t{self.name}\n\n{super().__str__()}"

    @property
    def name(self):
        return self._name_internal()

    @classmethod
    def _name_internal(cls):
        return cls.__name__

    @property
    def citation(self):
        return self._citation

    @property
    def weights_docstring(self):
        return self._weights_docstring

    @property
    def weights_version(self):
        return self._weights_version

    @classmethod
    def _model_path(cls):
        return Path(seisbench.cache_model_root, cls._name_internal().lower())

    @classmethod
    def _remote_path(cls):
        return urljoin(seisbench.remote_model_root, cls._name_internal().lower())

    @classmethod
    def _pretrained_path(cls, name, version_str=""):
        if version_str != "":
            version_str = ".v" + version_str
        weight_path = cls._model_path() / f"{name}.pt{version_str}"
        metadata_path = cls._model_path() / f"{name}.json{version_str}"

        return weight_path, metadata_path

    @classmethod
    def from_pretrained(
        cls, name, version_str="latest", update=False, force=False, wait_for_file=False
    ):
        """
        Load pretrained model with weights.

        A pretrained model weights consists of two files. A weights file [name].pt and a [name].json config file.
        The config file can (and should) contain the following entries, even though all arguments are optional:

        - "docstring": A string documenting the pipeline. Usually also contains information on the author.
        - "model_args": Argument dictionary passed to the init function of the pipeline.
        - "seisbench_requirement": The minimal version of SeisBench required to use the weights file.
        - "default_args": Default args for the :py:func:`annotate`/:py:func:`classify` functions.
          These arguments will supersede any potential constructor settings.
        - "version": The version string of the model. For **all but the latest version**, version names should
          furthermore be denoted in the file names, i.e., the files should end with the suffix ".v[VERSION]".
          If no version is specified in the json, the assumed version string is "1".

        .. warning::
            Even though the version is set to "latest" by default, this will only use the latest version locally
            available. Only if no weight is available locally, the remote repository will be queried. This behaviour
            is implemented for privacy reasons, as it avoids contacting the remote repository for every call of the
            function. To explicitly update to the latest version from the remote repository, set `update=True`.

        :param name: Model name prefix.
        :type name: str
        :param version_str: Version of the weights to load. Either a version string or "latest". The "latest" model is
                            the model with the highest version number.
        :type version_str: str
        :param force: Force execution of download callback, defaults to False
        :type force: bool, optional
        :param update: If true, downloads potential new weights file and config from the remote repository.
                       The old files are retained with their version suffix.
        :type update: bool
        :param wait_for_file: Whether to wait on partially downloaded files, defaults to False
        :type wait_for_file: bool, optional
        :return: Model instance
        :rtype: SeisBenchModel
        """
        cls._cleanup_local_repository()
        _cache_migration_v0_v3()

        if version_str == "latest":
            versions = cls.list_versions(name, remote=update)
            # Always query remote versions if cache is empty
            if len(versions) == 0:
                versions = cls.list_versions(name, remote=True)

            if len(versions) == 0:
                raise ValueError(f"No version for weight '{name}' available.")
            version_str = max(versions, key=version.parse)

        cls._version_warnings(name, version_str)

        weight_path, metadata_path = cls._pretrained_path(name, version_str)

        cls._ensure_weight_files(
            name, version_str, weight_path, metadata_path, force, wait_for_file
        )

        return cls.load(weight_path.with_name(name), version_str=version_str)

    @classmethod
    def _version_warnings(cls, name: str, version_str: str):
        """
        Check if the current weight should issue a warning
        """
        for name_regex, weight_version, warning_str in cls._weight_warnings:
            if not re.fullmatch(name_regex, name):
                continue

            if not weight_version == version_str:
                continue

            seisbench.logger.warning(f"Weight version warning: {warning_str}")

    @classmethod
    def _cleanup_local_repository(cls):
        """
        Cleans up local weights by moving all files without weight suffix to the correct weight suffix.

        Function required to keep compatibility to caches created with seisbench==0.1.x
        """
        model_path = cls._model_path()
        if not model_path.is_dir():
            # No need to cleanup if model path does not yet exist
            return

        files = [
            x.name[:-5] for x in model_path.iterdir() if x.name.endswith(".json")
        ]  # Files without version tag

        for file in files:
            metadata_path = model_path / (file + ".json")
            weight_path = model_path / (file + ".pt")

            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)

            file_version = weights_metadata.get("version", "1")

            weight_path_new = weight_path.parent / (
                weight_path.name + ".v" + file_version
            )
            metadata_path_new = metadata_path.parent / (
                metadata_path.name + ".v" + file_version
            )

            weight_path.rename(weight_path_new)
            metadata_path.rename(metadata_path_new)

    @classmethod
    def _ensure_weight_files(
        cls, name, version_str, weight_path, metadata_path, force, wait_for_file
    ):
        """
        Checks whether weight files are available and downloads them otherwise
        """

        def download_callback(files):
            weight_path, metadata_path = files
            seisbench.logger.info(
                f"Weight file {weight_path.name} not in cache. Downloading..."
            )
            weight_path.parent.mkdir(exist_ok=True, parents=True)

            remote_metadata_name, remote_weight_name = cls._get_remote_names(
                name, version_str
            )

            remote_weight_path = f"{cls._remote_path()}/{remote_weight_name}"
            remote_metadata_path = f"{cls._remote_path()}/{remote_metadata_name}"

            util.download_http(remote_weight_path, weight_path)
            util.download_http(remote_metadata_path, metadata_path, progress_bar=False)

        seisbench.util.callback_if_uncached(
            [weight_path, metadata_path],
            download_callback,
            force=force,
            wait_for_file=wait_for_file,
        )

    @classmethod
    def _get_remote_names(cls, name, version_str):
        """
        Determines the file names of weight and metadata file on the remote repository. This function is required as
        the remote version might not have a suffix.
        """
        remote_weight_name = f"{name}.pt.v{version_str}"
        remote_metadata_name = f"{name}.json.v{version_str}"
        remote_listing = seisbench.util.ls_webdav(cls._remote_path())
        if remote_metadata_name not in remote_listing:
            # Version not in repository under version name, check file without version suffix
            if f"{name}.json" in remote_listing:
                remote_version = cls._get_version_of_remote_without_suffix(name)
                if remote_version == version_str:
                    remote_weight_name = f"{name}.pt"
                    remote_metadata_name = f"{name}.json"
                else:
                    raise ValueError(
                        f"Version '{version_str}' of weight '{name}' is not available."
                    )
            else:
                raise ValueError(
                    f"Version '{version_str}' of weight '{name}' is not available."
                )
        return remote_metadata_name, remote_weight_name

    @classmethod
    def list_pretrained(cls, details=False, remote=True):
        """
        Returns list of available pretrained weights and optionally their docstrings.

        :param details: If true, instead of a returning only a list, also return their docstrings.
                        By default, returns the docstring of the "latest" version for each weight.
                        Note that this requires to download the json files for each model in the background
                        and is therefore slower. Defaults to false.
        :type details: bool
        :param remote: If true, reports both locally available weights and versions in the remote repository.
                       Otherwise only reports local versions.
        :type remote: bool
        :return: List of available weights or dict of weights and their docstrings
        :rtype: list or dict
        """
        cls._cleanup_local_repository()
        _cache_migration_v0_v3()

        # Idea: If details, copy all "latest" configs to a temp directory

        model_path = cls._model_path()
        model_path.mkdir(
            parents=True, exist_ok=True
        )  # Create directory if not existent
        weights = [
            cls._parse_weight_filename(x)[0]
            for x in model_path.iterdir()
            if cls._parse_weight_filename(x)[0] is not None
        ]

        if remote:
            remote_path = cls._remote_path()
            weights += [
                cls._parse_weight_filename(x)[0]
                for x in seisbench.util.ls_webdav(remote_path)
                if cls._parse_weight_filename(x)[0] is not None
            ]

        # Unique
        weights = sorted(list(set(weights)))

        if details:
            return {
                name: cls._get_latest_docstring(name, remote=remote) for name in weights
            }
        else:
            return weights

    @classmethod
    def _get_latest_docstring(cls, name, remote):
        """
        Get the latest docstring for a given weight name.

        Assumes that there is at least one version of the weight available locally (remote=False) or
        locally/remotely (remote=True).
        """
        versions = cls.list_versions(name, remote=remote)
        version_str = max(versions, key=version.parse)

        _, metadata_path = cls._pretrained_path(name, version_str)

        if metadata_path.is_file():
            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)
        else:
            remote_metadata_name, _ = cls._get_remote_names(name, version_str)
            remote_metadata_path = f"{cls._remote_path()}/{remote_metadata_name}"
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata_path = Path(tmpdir) / f"{name}.json"
                util.download_http(
                    remote_metadata_path, metadata_path, progress_bar=False
                )
                with open(metadata_path, "r") as f:
                    weights_metadata = json.load(f)

        return weights_metadata.get("docstring", None)

    @staticmethod
    def _parse_weight_filename(filename):
        """
        Parses filename into weight name, file type and version string.
        Returns None, None, None if file can not be parsed.
        """
        if isinstance(filename, Path):
            filename = filename.name

        name = None
        version_str = None

        for ftype in ["json", "pt"]:
            p = filename.find(f".{ftype}")
            if p != -1:
                name = filename[:p]
                remainder = filename[p + len(ftype) + 1 :]
                if len(remainder) > 0:
                    if remainder[:2] != ".v":
                        return None, None, None
                    version_str = remainder[2:]
                break

        if name is None and version_str is None:
            ftype = None

        return name, ftype, version_str

    @classmethod
    def list_versions(cls, name, remote=True):
        """
        Returns list of available versions for a given weight name.

        :param name: Name of the queried weight
        :type name: str
        :param remote: If true, reports both locally available versions and versions in the remote repository.
                       Otherwise only reports local versions.
        :type remote: bool
        :return: List of available versions
        :rtype: list[str]
        """
        cls._cleanup_local_repository()
        _cache_migration_v0_v3()

        if cls._model_path().is_dir():
            files = [x.name for x in cls._model_path().iterdir()]
            versions = cls._get_versions_from_files(name, files)
        else:
            versions = []

        if remote:
            remote_path = cls._remote_path()
            files = seisbench.util.ls_webdav(remote_path)
            remote_versions = cls._get_versions_from_files(name, files)

            if "" in remote_versions:
                remote_versions = [x for x in remote_versions if x != ""]
                # Need to download config file to check version
                file_version = cls._get_version_of_remote_without_suffix(name)
                remote_versions.append(file_version)

            versions = list(set(versions + remote_versions))

        return sorted(versions)

    @classmethod
    def _get_version_of_remote_without_suffix(cls, name):
        """
        Gets the version of the config in the remote repository without a version suffix in the file name.
        Assumes this config exists.
        """
        remote_metadata_path = f"{cls._remote_path()}/{name}.json"
        with tempfile.TemporaryDirectory() as tmpdir:
            metadata_path = Path(tmpdir) / "metadata.json"
            util.download_http(remote_metadata_path, metadata_path, progress_bar=False)

            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)
        file_version = weights_metadata.get("version", "1")
        return file_version

    @staticmethod
    def _get_versions_from_files(name, files):
        """
        Calculates the available versions from a list of files.

        :param name: Name of the queried weight
        :type name: str
        :param files: List of files
        :type files: list[str]
        :return: List of available versions
        :rtype: list[str]
        """
        configs = [x for x in files if x.startswith(f"{name}.json")]
        prefix_len = len(f"{name}.json.v")
        return sorted([config[prefix_len:] for config in configs])

    @classmethod
    def load(cls, path, version_str=None):
        """
        Load a SeisBench model from local path.

        For more information on the SeisBench model format see py:func:`save`.

        :param path: Define the path to the SeisBench model.
        :type path: pathlib.Path ot str
        :param version_str: Version string of the model. If none, no version string is appended.
        :type version_str: str, None
        :return: Model instance
        :rtype: SeisBenchModel
        """
        path_json, path_pt = cls._get_weights_file_paths(path, version_str)

        # Load model metadata
        with open(path_json, "r") as f:
            weights_metadata = json.load(f)
        # Load model weights
        model_weights = torch.load(f"{path_pt}")

        model_args = weights_metadata.get("model_args", {})
        model = cls(**model_args)
        model._weights_metadata = weights_metadata
        model._parse_metadata()

        model.load_state_dict(model_weights)

        return model

    def save(self, path, weights_docstring="", version_str=None):
        """
        Save a SeisBench model locally.

        SeisBench models are stored inside the directory 'path'. SeisBench models are saved in 2 parts,
        the model configuration is stored in JSON format [path][.json], and the underlying model weights
        in PyTorch format [path][.pt]. Where 'path' is the output path to store. The suffixes are appended
        to the path parameter automatically.

        In addition, the models can have a version string which is appended to the json and the pt path.
        For example, setting `version_str="1"` will append `.v1` to the file names.

        The model config should contain the following information, which is automatically created from
        the model instance state:
            - "weights_docstring": A string documenting the pipeline. Usually also contains information on the author.
            - "model_args": Argument dictionary passed to the init function of the pipeline.
            - "seisbench_requirement": The minimal version of SeisBench required to use the weights file.
            - "default_args": Default args for the :py:func:`annotate`/:py:func:`classify` functions.

        Non-serializable arguments (e.g. functions) cannot be saved to JSON, so are not converted.

        :param path: Define the path to the output model.
        :type path: pathlib.Path or str
        :param weights_docstring: Documentation for the model weights (training details, author etc.)
        :type weights_docstring: str, default to ''
        :param version_str: Version string of the model. If none, no version string is appended.
        :type version_str: str, None
        """
        path_json, path_pt = self._get_weights_file_paths(path, version_str)

        def _contains_callable_recursive(dict_obj):
            """
            Recursive lookup through dictionary to check wheter any values are callable.
            """
            for k, v in dict_obj.items():
                if callable(v):
                    return True
                if isinstance(v, dict):
                    obj = _contains_callable_recursive(v)
                    if obj is not None:
                        return obj

        model_args = self.get_model_args()

        if not model_args:
            seisbench.logger.warning(
                "No 'model_args' found. "
                "Saving any model parameters should be done manually within abstractmethod: `get_model_args`. "
                "Have you implemented `get_model_args`?. "
                "If this is the desired behaviour, and you have no parameters for your model, please ignore."
            )

        parsed_model_args = {}
        for k, v in model_args.items():
            if k not in (
                "__class__",
                "self",
                "default_args",
                "_weights_metadata",
                "_weights_docstring",
            ):
                # Check for non-serlizable types
                _flagged_callable = False
                if isinstance(v, set):
                    # Sets converted
                    parsed_model_args.update({k: list(v)})
                    continue
                if callable(v):
                    # Callables not stored in JSON
                    _flagged_callable = True
                if isinstance(v, dict):
                    # Check inside nested dicts for callables
                    if _contains_callable_recursive(v):
                        _flagged_callable = True

                if not _flagged_callable:
                    parsed_model_args.update({k: v})
                else:
                    seisbench.logger.warning(
                        f"{k} parameter is a non-serilizable object, cannot be saved to JSON config file."
                    )

        model_metadata = {
            "docstring": weights_docstring,
            "model_args": parsed_model_args,
            "seisbench_requirement": seisbench.__version__,
            "default_args": self.__dict__.get("default_args", ""),
        }

        # Save weights
        torch.save(self.state_dict(), path_pt)
        # Save model metadata
        with open(path_json, "w") as json_fp:
            json.dump(model_metadata, json_fp, indent=4)

        seisbench.logger.debug(f"Saved {self.name} model at {path}")

    @staticmethod
    def _get_weights_file_paths(path, version_str):
        """
        Return file names by parsing the path and version_str. For details, see save or load methods.
        """
        if isinstance(path, str):
            path = Path(path)

        if version_str is None:
            version_suffix = ""
        else:
            version_suffix = f".v{version_str}"

        base_name = path.name
        path_json = path.with_name(base_name + ".json" + version_suffix)
        path_pt = path.with_name(base_name + ".pt" + version_suffix)

        return path_json, path_pt

    def _parse_metadata(self):
        # Load docstring
        self._weights_docstring = self._weights_metadata.get("docstring", "")
        self._weights_version = self._weights_metadata.get("version", "1")

        # Check version requirement
        self._check_version_requirement()

        # Parse default args - Config default_args supersede constructor args
        default_args = self._weights_metadata.get("default_args", {})
        self.default_args.update(default_args)

    def _check_version_requirement(self):
        seisbench_requirement = self._weights_metadata.get(
            "seisbench_requirement", None
        )
        if seisbench_requirement is not None:
            if version.parse(seisbench_requirement) > version.parse(
                seisbench.__version__
            ):
                raise ValueError(
                    f"Weights require seisbench version at least {seisbench_requirement}, "
                    f"but the installed version is {seisbench.__version__}."
                )

    @abstractmethod
    def get_model_args(self):
        """
        Obtain all model parameters for saving.

        :return: Dictionary of all parameters for a model to store during saving.
        :rtype: Dict
        """
        return {"citation": self._citation}


class WaveformModel(SeisBenchModel, ABC):
    """
    Abstract interface for models processing waveforms.
    Based on the properties specified by inheriting models, WaveformModel automatically provides the respective
    :py:func:`annotate`/:py:func:`classify` functions.
    Both functions take obspy streams as input.
    The :py:func:`annotate` function has a rather strictly defined output, i.e.,
    it always outputs obspy streams with the annotations.
    These can for example be functions of pick probability over time.
    In contrast, the :py:func:`classify` function can tailor it's output to the model type.
    For example, a picking model might output picks, while a magnitude estimation model might only output
    a scalar magnitude.
    Internally, :py:func:`classify` will usually rely on :py:func:`annotate` and simply add steps to it's output.

    For details see the documentation of these functions.

    .. document_args:: seisbench.models WaveformModel

    :param component_order: Specify component order (e.g. ['ZNE']), defaults to None.
    :type component_order: list, optional
    :param sampling_rate: Sampling rate of the model, defaults to None.
                          If sampling rate is not None, the annotate and classify functions will automatically resample
                          incoming traces and validate correct sampling rate if the model overwrites
                          :py:func:`annotate_stream_pre`.
    :type sampling_rate: float
    :param output_type: The type of output from the model. Current options are:

                        - "point" for a point prediction, i.e., the probability of containing a pick in the window
                          or of a pick at a certain location. This will provide an :py:func:`annotate` function.
                          If an :py:func:`classify_aggregate` function is provided by the inheriting model,
                          this will also provide a :py:func:`classify` function.
                        - "array" for prediction curves, i.e., probabilities over time for the arrival of certain wave
                          types. This will provide an :py:func:`annotate` function.
                          If an :py:func:`classify_aggregate` function is provided by the inheriting model,
                          this will also provide a :py:func:`classify` function.
                        - "regression" for a regression value, i.e., the sample of the arrival within a window.
                          This will only provide a :py:func:`classify` function.

    :type output_type: str
    :param default_args: Default arguments to use in annotate and classify functions
    :type default_args: dict[str, any]
    :param in_samples: Number of input samples in time
    :type in_samples: int
    :param pred_sample: For a "point" prediction: sample number of the sample in a window for which the prediction is
                        valid. For an "array" prediction: a tuple of first and last sample defining the prediction
                        range. Note that the number of output samples and input samples within the given range are
                        not required to agree.
    :type pred_sample: int, tuple
    :param labels: Labels for the different predictions in the output, e.g., Noise, P, S. If a function is passed,
                   it will be called for every label generation and be provided with the stats of the trace that was
                   annotated.
    :type labels: list or string or callable
    :param filter_args: Arguments to be passed to :py:func:`obspy.filter` in :py:func:`annotate_stream_pre`
    :type filter_args: tuple
    :param filter_kwargs: Keyword arguments to be passed to :py:func:`obspy.filter` in :py:func:`annotate_stream_pre`
    :type filter_kwargs: dict
    :param grouping: Level of grouping for annotating streams. Supports "instrument", "channel" and "full".
                     Alternatively, a custom GroupingHelper can be passed.
    :type grouping: Union[str, GroupingHelper]
    :param allow_padding: If True, annotate will pad different windows if they have different sizes.
                          This is useful, for example, for multi-station methods.
    :type allow_padding: bool
    :param kwargs: Kwargs are passed to the superclass
    """

    # Optional arguments for annotate/classify: key -> (documentation, default_value)
    _annotate_args = {
        "batch_size": ("Batch size for the model", 256),
        "overlap": (
            "Overlap between prediction windows in samples (only for window prediction models)",
            0,
        ),
        "stacking": (
            "Stacking method for overlapping windows (only for window prediction models). "
            "Options are 'max' and 'avg'. ",
            "avg",
        ),
        "stride": ("Stride in samples (only for point prediction models)", 1),
        "strict": (
            "If true, only annotate if recordings for all components are available, "
            "otherwise impute missing data with zeros.",
            False,
        ),
        "flexible_horizontal_components": (
            "If true, accepts traces with Z12 components as ZNE and vice versa. "
            "This is usually acceptable for rotationally invariant models, "
            "e.g., most picking models.",
            True,
        ),
    }

    _stack_options = {
        "avg",
        "max",
    }  # Known stacking options - mutable and accessible for docs.

    def __init__(
        self,
        component_order=None,
        sampling_rate=None,
        output_type=None,
        default_args=None,
        in_samples=None,
        pred_sample=0,
        labels=None,
        filter_args=None,
        filter_kwargs=None,
        grouping="instrument",
        allow_padding=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if component_order is None:
            self._component_order = seisbench.config["component_order"]
        else:
            self._component_order = component_order

        self.sampling_rate = sampling_rate
        self.output_type = output_type
        if default_args is None:
            self.default_args = {}
        else:
            self.default_args = default_args
        self.in_samples = in_samples
        self.pred_sample = pred_sample
        self.filter_args = filter_args
        self.filter_kwargs = filter_kwargs

        if isinstance(grouping, str):
            if grouping == "channel":
                if component_order is not None:
                    seisbench.logger.warning(
                        "Grouping is 'channel' but component_order is given. "
                        "component_order will be ignored, as every channel is treated separately."
                    )
                self._component_order = None
            grouping = GroupingHelper(grouping)

        self._grouping: GroupingHelper = grouping
        self.allow_padding = allow_padding

        # Validate pred sample
        if output_type == "point" and not isinstance(pred_sample, (int, float)):
            raise TypeError(
                "For output type 'point', pred_sample needs to be a scalar."
            )
        if output_type == "array":
            if not isinstance(pred_sample, (list, tuple)) or not len(pred_sample) == 2:
                raise TypeError(
                    "For output type 'array', pred_sample needs to be a tuple of length 2."
                )
            if pred_sample[0] < 0 or pred_sample[1] < 0:
                raise ValueError(
                    "For output type 'array', both entries of pred_sample need to be non-negative."
                )

        self.labels = labels

        self._annotate_function_mapping = {
            "point": (
                self._async_cut_fragments_point,
                self._async_reassemble_blocks_point,
            ),
            "array": (
                self._async_cut_fragments_array,
                self._async_reassemble_blocks_array,
            ),
        }

        if self.output_type == "point" and self._grouping.grouping == "full":
            raise NotImplementedError(
                "Point outputs with full grouping are currently not implemented."
            )

        self._annotate_functions = self._annotate_function_mapping.get(
            output_type, None
        )

    def __str__(self):
        return f"Component order:\t{self.component_order}\n{super().__str__()}"

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def component_order(self):
        return self._component_order

    def _argdict_get_with_default(self, argdict, key):
        return argdict.get(key, self._annotate_args.get(key)[1])

    def annotate(
        self,
        stream,
        copy=True,
        **kwargs,
    ):
        """
        Annotates an obspy stream using the model based on the configuration of the WaveformModel superclass.
        For example, for a picking model, annotate will give a characteristic function/probability function for picks
        over time.
        The annotate function contains multiple subfunctions, which can be overwritten individually by inheriting
        models to accommodate their requirements. These functions are:

        - :py:func:`annotate_stream_pre`
        - :py:func:`annotate_stream_validate`
        - :py:func:`annotate_batch_pre`
        - :py:func:`annotate_batch_post`

        Please see the respective documentation for details on their functionality, inputs and outputs.

        .. hint::
            If your machine is equipped with a GPU, this function will usually run faster when making use of the GPU.
            Just call `model.cuda()`. In addition, you might want to increase the batch size by passing the `batch_size`
            argument to the function. Possible values might be 2048 or 4096 (or larger if your GPU permits).

        .. warning::
            Even though the `asyncio` implementation itself is not parallel, this does not guarantee that only a single
            CPU core will be used, as the underlying libraries (pytorch, numpy, scipy, ...) might be parallelised.
            If you need to limit the parallelism of these libraries, check their documentation, e.g.,
            `here <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#tuning-the-number-of-threads>`_
            or
            `here <https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy>`_.
            Bear in mind that a lower number of threads might occasionally improve runtime performance, as it limits
            overheads, e.g.,
            `here <https://github.com/pytorch/pytorch/issues/3146>`_.


        :param stream: Obspy stream to annotate
        :type stream: obspy.core.Stream
        :param copy: If true, copies the input stream. Otherwise, the input stream is modified in place.
        :type copy: bool
        :param kwargs:
        :return: Obspy stream of annotations
        """
        if "parallelism" in kwargs:
            seisbench.logger.warning(
                "The `parallelism` argument has been deprecated in favour of batch processing."
            )

        call = self.annotate_async(stream, copy=copy, **kwargs)
        return asyncio.run(call)

    def _verify_argdict(self, argdict):
        for key in argdict.keys():
            if not any(
                re.fullmatch(pattern.replace("*", ".*"), key)
                for pattern in self._annotate_args.keys()
            ):
                seisbench.logger.warning(f"Unknown argument '{key}' will be ignored.")

    async def annotate_async(self, stream, copy=True, **kwargs):
        """
        `annotate` implementation based on asyncio
        Parameters as for :py:func:`annotate`.
        """
        self._verify_argdict(kwargs)

        cut_fragments, reassemble_blocks = self._get_annotate_functions()

        # Kwargs overwrite default args
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        if copy:
            stream = stream.copy()
        stream.merge(-1)

        output = obspy.Stream()
        if len(stream) == 0:
            return output

        # Preprocess stream, e.g., filter/resample
        self.annotate_stream_pre(stream, argdict)

        # Validate stream
        stream = self.annotate_stream_validate(stream, argdict)

        if len(stream) == 0:
            return output

        # Sampling rate of the data. Equal to self.sampling_rate is this is not None
        sampling_rate = stream[0].stats.sampling_rate
        argdict["sampling_rate"] = sampling_rate

        # Group stream
        strict = self._argdict_get_with_default(argdict, "strict")
        flexible_horizontal_components = self._argdict_get_with_default(
            argdict, "flexible_horizontal_components"
        )
        comp_dict, _ = self._build_comp_dict(stream, flexible_horizontal_components)
        groups = self._grouping.group_stream(
            stream,
            strict=strict,
            min_length_s=(self.in_samples - 1) / sampling_rate,
            comp_dict=comp_dict,
        )

        # Queues for multiprocessing
        batch_size = self._argdict_get_with_default(argdict, "batch_size")
        queue_groups = asyncio.Queue()  # Waveform groups
        queue_raw_blocks = (
            asyncio.Queue()
        )  # Waveforms as blocks of arrays and their metadata
        queue_raw_fragments = asyncio.Queue(
            4 * batch_size
        )  # Raw waveform fragments with the correct input size
        queue_postprocessed_pred = (
            asyncio.Queue()
        )  # Queue for raw (but unbatched) predictions
        queue_pred_blocks = asyncio.Queue()  # Queue for blocks of predictions
        queue_results = asyncio.Queue()  # Results streams

        process_streams_to_arrays = asyncio.create_task(
            self._async_streams_to_arrays(
                queue_groups,
                queue_raw_blocks,
                argdict,
            )
        )
        process_cut_fragments = asyncio.create_task(
            cut_fragments(queue_raw_blocks, queue_raw_fragments, argdict)
        )
        process_predict = asyncio.create_task(
            self._async_predict(queue_raw_fragments, queue_postprocessed_pred, argdict)
        )
        process_reassemble_blocks = asyncio.create_task(
            reassemble_blocks(queue_postprocessed_pred, queue_pred_blocks, argdict)
        )
        process_predictions_to_streams = asyncio.create_task(
            self._async_predictions_to_streams(queue_pred_blocks, queue_results)
        )

        for group in groups:
            await queue_groups.put(group)
        await queue_groups.put(None)

        await process_streams_to_arrays
        await queue_raw_blocks.put(None)
        await process_cut_fragments
        await queue_raw_fragments.put(None)
        await process_predict
        await queue_postprocessed_pred.put(None)
        await process_reassemble_blocks
        await queue_pred_blocks.put(None)
        await process_predictions_to_streams

        while True:
            try:
                output += queue_results.get_nowait()
            except asyncio.QueueEmpty:
                break

        return output

    def _get_annotate_functions(self):
        if self._annotate_functions is None:
            raise NotImplementedError(
                "This model has no annotate function implemented."
            )
        cut_fragments, reassemble_blocks = self._annotate_functions
        return cut_fragments, reassemble_blocks

    async def _async_streams_to_arrays(
        self,
        queue_in,
        queue_out,
        argdict,
    ):
        """
        Wrapper around :py:func:`stream_to_array`, adding the functionality to read from and write to queues.
        :param queue_in: Input queue
        :param queue_out: Output queue
        :return: None
        """
        group = await queue_in.get()
        while group is not None:
            t0, block, stations = self.stream_to_array(
                group,
                argdict,
            )
            await queue_out.put((t0, block, stations))
            group = await queue_in.get()

    async def _async_predict(self, queue_in, queue_out, argdict):
        """
        Prediction function, gathering predictions until a batch is full and handing them to :py:func:`_predict_buffer`.
        :param queue_in: Input queue
        :param queue_out: Output queue
        :param argdict: Dictionary of arguments
        :return: None
        """
        buffer = []
        batch_size = self._argdict_get_with_default(argdict, "batch_size")

        elem = await queue_in.get()
        while True:
            if elem is not None:
                buffer.append(elem)

            if len(buffer) == batch_size or (elem is None and len(buffer) > 0):
                pred = await asyncio.to_thread(
                    self._predict_buffer,
                    [window for window, _ in buffer],
                    argdict=argdict,
                )
                for pred_window, (_, metadata) in zip(pred, buffer):
                    await queue_out.put((pred_window, metadata))
                buffer = []

            if elem is None:
                break

            elem = await queue_in.get()

    async def _async_predictions_to_streams(self, queue_in, queue_out):
        """
        Wrapper with queue IO functionality around :py:func:`_predictions_to_stream`
        :param queue_in: Input queue
        :param queue_out: Output queue
        :return: None
        """
        elem = await queue_in.get()
        while elem is not None:
            (pred_rate, pred_time, preds), stations = elem
            await queue_out.put(
                self._predictions_to_stream(pred_rate, pred_time, preds, stations)
            )
            elem = await queue_in.get()

    async def _async_cut_fragments_point(self, queue_in, queue_out, argdict):
        """
        Wrapper with queue IO functionality around :py:func:`_cut_fragments_point`
        """
        elem = await queue_in.get()
        while elem is not None:
            t0, block, stations = elem

            for output_elem in self._cut_fragments_point(t0, block, stations, argdict):
                await queue_out.put(output_elem)

            elem = await queue_in.get()

    def _cut_fragments_point(self, t0, block, stations, argdict):
        """
        Cuts numpy arrays into fragments for point prediction models.

        :param t0:
        :param block:
        :param stations:
        :param argdict:
        :return:
        """
        stride = self._argdict_get_with_default(argdict, "stride")
        starts = np.arange(0, block.shape[-1] - self.in_samples + 1, stride)
        if len(starts) == 0:
            seisbench.logger.warning(
                "Parts of the input stream consist of fragments shorter than the number "
                "of input samples. Output might be empty."
            )
            return

        bucket_id = np.random.randint(1000000)

        # Generate windows and preprocess
        for s in starts:
            window = block[..., s : s + self.in_samples]
            # The combination of stations and t0 is a unique identifier
            # s can be used to reassemble the block, len(starts) allows to identify if the block is complete yet
            metadata = (t0, s, len(starts), stations, bucket_id)
            yield window, metadata

    async def _async_reassemble_blocks_point(self, queue_in, queue_out, argdict):
        """
        Wrapper with queue IO functionality around :py:func:`_reassemble_blocks_point`
        """
        buffer = defaultdict(list)  # Buffers predictions until a block is complete

        elem = await queue_in.get()
        while elem is not None:
            output = self._reassemble_blocks_point(elem, buffer, argdict)
            if output is not None:
                await queue_out.put(output)

            elem = await queue_in.get()

    def _reassemble_blocks_point(self, elem, buffer, argdict):
        """
        Reassembles point predictions into numpy arrays. Returns None except if a buffer was processed.
        """
        stride = self._argdict_get_with_default(argdict, "stride")

        window, metadata = elem
        t0, s, len_starts, stations, bucket_id = metadata
        key = f"{t0}_{'__'.join(stations)}"

        output = None

        buffer[key].append(elem)
        if len(buffer[key]) == len_starts:
            preds = [(s, window) for window, (_, s, _, _, _) in buffer[key]]
            preds = sorted(
                preds, key=lambda x: x[0]
            )  # Sort by start (overwrite keys to make sure window is never used as key)
            preds = [window for s, window in preds]
            preds = np.stack(preds, axis=0)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            pred_time = t0 + self.pred_sample / argdict["sampling_rate"]
            pred_rate = argdict["sampling_rate"] / stride

            output = ((pred_rate, pred_time, preds), stations)

            del buffer[key]

        return output

    async def _async_cut_fragments_array(self, queue_in, queue_out, argdict):
        """
        Wrapper with queue IO functionality around :py:func:`_cut_fragments_array`
        """
        while True:
            elem = await queue_in.get()
            if elem is None:
                break

            for output_elem in self._cut_fragments_array(elem, argdict):
                await queue_out.put(output_elem)

    def _cut_fragments_array(self, elem, argdict):
        """
        Cuts numpy arrays into fragments for array prediction models.
        """
        overlap = self._argdict_get_with_default(argdict, "overlap")

        t0, block, stations = elem

        bucket_id = np.random.randint(int(1e9))

        starts = np.arange(
            0, block.shape[-1] - self.in_samples + 1, self.in_samples - overlap
        )
        if len(starts) == 0:
            seisbench.logger.warning(
                "Parts of the input stream consist of fragments shorter than the number "
                "of input samples. Output might be empty."
            )
            return

        # Add one more trace to the end
        if starts[-1] + self.in_samples < block.shape[-1]:
            starts = np.concatenate([starts, [block.shape[-1] - self.in_samples]])

        # Generate windows and preprocess
        for s in starts:
            window = block[..., s : s + self.in_samples]
            # The combination of stations and t0 is a unique identifier
            # s can be used to reassemble the block, len(starts) allows to identify if the block is complete yet
            metadata = (t0, s, len(starts), stations, bucket_id)
            yield window, metadata

    async def _async_reassemble_blocks_array(self, queue_in, queue_out, argdict):
        """
        Wrapper with queue IO functionality around :py:func:`_reassemble_blocks_array`
        """
        buffer = defaultdict(list)  # Buffers predictions until a block is complete

        while True:
            elem = await queue_in.get()
            if elem is None:
                break

            output_elem = self._reassemble_blocks_array(elem, buffer, argdict)
            if output_elem is not None:
                await queue_out.put(output_elem)

    def _reassemble_blocks_array(self, elem, buffer, argdict):
        """
        Reassembles array predictions into numpy arrays.
        """
        overlap = self._argdict_get_with_default(argdict, "overlap")
        stack_method = self._argdict_get_with_default(
            argdict, "stacking"
        ).lower()  # This is a breaking change for v 0.3 - see PR#99
        assert (
            stack_method in self._stack_options
        ), f"Stacking method {stack_method} unknown. Known options are: {self._stack_options}"
        window, metadata = elem
        t0, s, len_starts, stations, bucket_id = metadata
        key = f"{t0}_{'__'.join(stations)}"
        buffer[key].append(elem)

        output = None

        if len(buffer[key]) == len_starts:
            preds = [(s, window) for window, (_, s, _, _, _) in buffer[key]]
            preds = sorted(
                preds, key=lambda x: x[0]
            )  # Sort by start (overwrite keys to make sure window is never used as key)
            starts = [s for s, window in preds]
            preds = [window for s, window in preds]
            preds = [self._add_grouping_dimensions(pred) for pred in preds]

            # Number of prediction samples per input sample
            prediction_sample_factor = preds[0].shape[1] / (
                self.pred_sample[1] - self.pred_sample[0]
            )

            # Maximum number of predictions covering a point
            coverage = int(np.ceil(self.in_samples / (self.in_samples - overlap) + 1))

            pred_length = int(
                np.ceil((np.max(starts) + self.in_samples) * prediction_sample_factor)
            )
            pred_merge = (
                np.zeros_like(
                    preds[0],
                    shape=(
                        preds[0].shape[0],
                        pred_length,
                        preds[0].shape[-1],
                        coverage,
                    ),
                )
                * np.nan
            )
            for i, (pred, start) in enumerate(zip(preds, starts)):
                pred_start = int(start * prediction_sample_factor)
                pred_merge[
                    :, pred_start : pred_start + pred.shape[1], :, i % coverage
                ] = pred

            with warnings.catch_warnings():
                if stack_method == "avg":
                    warnings.filterwarnings(
                        action="ignore", message="Mean of empty slice"
                    )
                    preds = bn.nanmean(pred_merge, axis=-1)
                elif stack_method == "max":
                    warnings.filterwarnings(action="ignore", message="All-NaN")
                    preds = bn.nanmax(pred_merge, axis=-1)
                # Case of stack_method not in avg or max is caught by assert above

            if self._grouping.grouping == "channel":
                preds = preds[0, :, 0]
            elif self._grouping.grouping == "instrument":
                preds = preds[0]

            pred_time = t0 + self.pred_sample[0] / argdict["sampling_rate"]
            pred_rate = argdict["sampling_rate"] * prediction_sample_factor

            output = ((pred_rate, pred_time, preds), stations)

            del buffer[key]

        return output

    def _predict_buffer(self, buffer, argdict):
        """
        Batches model inputs, runs preprocess, prediction and postprocess, and unbatches output

        :param buffer: List of inputs to the model
        :return: Unpacked predictions
        """
        if self.allow_padding:
            fragments = seisbench.util.pad_packed_sequence(buffer)
        else:
            fragments = np.stack(buffer, axis=0)
        fragments = torch.tensor(fragments, device=self.device, dtype=torch.float32)

        train_mode = self.training
        try:
            self.eval()
            with torch.no_grad():
                preprocessed = self.annotate_batch_pre(fragments, argdict=argdict)
                if isinstance(preprocessed, tuple):  # Contains piggyback information
                    assert len(preprocessed) == 2
                    preprocessed, piggyback = preprocessed
                else:
                    piggyback = None

                preds = self(preprocessed)

                preds = self.annotate_batch_post(
                    preds, piggyback=piggyback, argdict=argdict
                )
        finally:
            if train_mode:
                self.train()

        # Explicit synchronisation can help profiling the stack
        # if torch.cuda.is_available():
        #    torch.cuda.synchronize()

        preds = self._recursive_torch_to_numpy(preds)
        # Unbatch window predictions
        reshaped_preds = [pred for pred in self._recursive_slice_pred(preds)]
        return reshaped_preds

    def _predictions_to_stream(self, pred_rate, pred_time, pred, stations):
        """
        Converts a set of predictions to obspy streams

        :param pred_rates: Sampling rates of the prediction arrays
        :param pred_times: Start time of each prediction array
        :param preds: The prediction arrays, each with shape (samples, channels)
        :param stations: The list of stations as strings in format NET.STA.LOC or NET.STA.LOC.CHA
        :return: Obspy stream of predictions
        """
        output = obspy.Stream()

        pred = self._add_grouping_dimensions(pred)

        # Define and store default labels
        if self.labels is None:
            self.labels = list(range(pred.shape[-1]))

        for station_idx, trace_id in enumerate(stations):
            for channel_idx in range(pred.shape[-1]):
                if callable(self.labels):
                    label = self.labels(stations)
                else:
                    label = self.labels[channel_idx]

                trimmed_pred, f, _ = self._trim_nan(pred[station_idx, :, channel_idx])
                trimmed_start = pred_time + f / pred_rate
                network, station, location = trace_id.split(".")[:3]
                output.append(
                    obspy.Trace(
                        trimmed_pred,
                        {
                            "starttime": trimmed_start,
                            "sampling_rate": pred_rate,
                            "network": network,
                            "station": station,
                            "location": location,
                            "channel": f"{self.__class__.__name__}_{label}",
                        },
                    )
                )
        return output

    def _add_grouping_dimensions(self, pred):
        """
        Add fake dimensions to make pred shape (stations, samples, channels)
        """
        if self._grouping.grouping == "instrument":
            pred = np.expand_dims(pred, 0)
        if self._grouping.grouping == "channel":
            pred = np.expand_dims(np.expand_dims(pred, -1), 0)
        return pred

    def annotate_stream_pre(self, stream, argdict):
        """
        Runs preprocessing on stream level for the annotate function, e.g., filtering or resampling.
        By default, this function will resample all traces if a sampling rate for the model is provided.
        Furthermore, if a filter is specified in the class, the filter will be executed.
        As annotate create a copy of the input stream, this function can safely modify the stream inplace.
        Inheriting classes should overwrite this function if necessary.
        To keep the default functionality, a call to the overwritten method can be included.

        :param stream: Input stream
        :type stream: obspy.Stream
        :param argdict: Dictionary of arguments
        :return: Preprocessed stream
        """
        self._filter_stream(stream)

        if self.sampling_rate is not None:
            self.resample(stream, self.sampling_rate)
        return stream

    def _filter_stream(self, stream):
        """
        Filters stream according to filter_args and filter_kwargs.
        By default, these are directly passed to `obspy.stream.filter(*filter_arg, **filter_kwargs)`.
        In addition, separate filtering for different channels can be defined.
        This is done by making `filter_args` a dict from channel regex to the actual filter arguments.
        In this case, `filter_kwargs` is expected to be a dict with the same keys.
        For example, `filter_args = {"??Z": ("highpass",)}` and `filter_kwargs = {"??Z": {"freq": 1}}`
        would high-pass filter only the vertical components at 1 Hz.
        """
        # TODO: This should check for gaps and ensure that these are zeroed at the end of processing
        if self.filter_args is not None or self.filter_kwargs is not None:
            if isinstance(self.filter_args, dict):
                for key, filter_args in self.filter_args.items():
                    substream = stream.select(channel=key)
                    if key not in self.filter_kwargs:
                        raise ValueError(
                            f"Invalid filter definition. Key '{key}' in args but not in kwargs."
                        )
                    self._filter_stream_single(
                        filter_args, self.filter_kwargs[key], substream
                    )

            else:
                self._filter_stream_single(self.filter_args, self.filter_kwargs, stream)

    @staticmethod
    def _filter_stream_single(filter_args, filter_kwargs, stream):
        if filter_args is None:
            filter_args = ()
        if filter_kwargs is None:
            filter_kwargs = {}

        stream.filter(*filter_args, **filter_kwargs)

    def annotate_stream_validate(self, stream, argdict):
        """
        Validates stream for the annotate function.
        This function should raise an exception if the stream is invalid.
        By default, this function will check if the sampling rate fits the provided one, unless it is None,
        and check for mismatching traces, i.e., traces covering the same time range on the same instrument with
        different values.
        Inheriting classes should overwrite this function if necessary.
        To keep the default functionality, a call to the overwritten method can be included.

        :param stream: Input stream
        :type stream: obspy.Stream
        :param argdict: Dictionary of arguments
        :return: None
        """
        if self.sampling_rate is not None:
            if any(trace.stats.sampling_rate != self.sampling_rate for trace in stream):
                raise ValueError(
                    f"Detected traces with mismatching sampling rate. "
                    f"Expected {self.sampling_rate} Hz for all traces."
                )

        return self.sanitize_mismatching_overlapping_records(stream)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Runs preprocessing on batch level for the annotate function, e.g., normalization.
        By default, returns the input batch unmodified.
        Optionally, this can return a tuple of the preprocessed batch and piggyback information that is passed to
        :py:func:`annotate_batch_post`.
        This can for example be used to transfer normalization information.
        Inheriting classes should overwrite this function if necessary.

        :param batch: Input batch
        :param argdict: Dictionary of arguments
        :return: Preprocessed batch and optionally piggyback information that is passed to :py:func:`annotate_batch_post`
        """
        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Runs postprocessing on the predictions of a window for the annotate function, e.g., reformatting them.
        By default, returns the original prediction.
        Inheriting classes should overwrite this function if necessary.

        :param batch: Predictions for the batch. The data type depends on the model.
        :param argdict: Dictionary of arguments
        :param piggyback: Piggyback information, by default None.
        :return: Postprocessed predictions
        """
        return batch

    @staticmethod
    def _trim_nan(x):
        """
        Removes all starting and trailing nan values from a 1D array and returns the new array and the number of NaNs
        removed per side.
        """
        mask = ~np.isnan(x)
        valid = np.nonzero(mask == True)[0]
        mask[valid[0] : valid[-1]] = True
        _end = len(x)
        x = x[mask]

        return x, valid[0], _end - (1 + valid[-1])

    def _recursive_torch_to_numpy(self, x):
        """
        Recursively converts torch.Tensor objects to numpy arrays while preserving any overarching tuple
        or list structure.
        :param x:
        :return:
        """
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        elif isinstance(x, list):
            return [self._recursive_torch_to_numpy(y) for y in x]
        elif isinstance(x, tuple):
            return tuple([self._recursive_torch_to_numpy(y) for y in x])
        elif isinstance(x, np.ndarray):
            return x
        else:
            raise ValueError(f"Can't unpack object of type {type(x)}.")

    def _recursive_slice_pred(self, x):
        """
        Converts batched predictions into a list of single predictions, assuming batch axis is first in all cases.
        Preserves overarching tuple and list structures
        :param x:
        :return:
        """
        if isinstance(x, np.ndarray):
            return list(y for y in x)
        elif isinstance(x, list):
            return [
                list(entry)
                for entry in zip(*[self._recursive_slice_pred(y) for y in x])
            ]
        elif isinstance(x, tuple):
            return [entry for entry in zip(*[self._recursive_slice_pred(y) for y in x])]
        else:
            raise ValueError(f"Can't unpack object of type {type(x)}.")

    async def classify_async(
        self, stream: obspy.Stream, **kwargs
    ) -> util.ClassifyOutput:
        """
        Async interface to the :py:func:`classify` function. See details there.
        """
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        stream = self.classify_stream_pre(stream, argdict)
        annotations = await self.annotate_async(stream, **argdict)
        return self.classify_aggregate(annotations, argdict)

    def _classify_parallel(self, stream: obspy.Stream, **kwargs) -> util.ClassifyOutput:
        argdict = self.default_args.copy()
        argdict.update(kwargs)

        stream = self.classify_stream_pre(stream, argdict)
        annotations = self.annotate(stream, **argdict)
        return self.classify_aggregate(annotations, argdict)

    def classify(
        self, stream: obspy.Stream, parallelism: Optional[int] = None, **kwargs
    ) -> util.ClassifyOutput:
        """
        Classifies the stream. The classification can contain any information,
        but should be consistent with existing models.

        :param stream: Obspy stream to classify
        :type stream: obspy.core.Stream
        :param kwargs:
        :return: A classification for the full stream, e.g., a list of picks or the source magnitude.
        """
        if parallelism is None:
            return asyncio.run(self.classify_async(stream, **kwargs))
        else:
            return self._classify_parallel(stream, parallelism=parallelism, **kwargs)

    def classify_stream_pre(self, stream, argdict):
        """
        Runs preprocessing on stream level for the classify function, e.g., subselecting traces.
        By default, this function will simply return the input stream.
        In contrast to :py:func:`annotate_stream_pre`, this function operates on the original input stream.
        The stream should therefore not be modified in place.
        Note that :py:func:`annotate_stream_pre` will be executed on the output of this stream
        within the :py:func:`classify` function.

        :param stream: Input stream
        :type stream: obspy.Stream
        :param argdict: Dictionary of arguments
        :return: Preprocessed stream
        """
        return stream

    def classify_aggregate(self, annotations, argdict) -> util.ClassifyOutput:
        """
        An aggregation function that converts the annotation streams returned by :py:func:`annotate` into
        a classification. A classification consists of a ClassifyOutput, essentialy a namespace that can hold
        an arbitrary set of keys. However, when implementing a model which already exists in similar form,
        we recommend using the same output format. For example, all pick outputs should have
        the same format.

        :param annotations: Annotations returned from :py:func:`annotate`
        :param argdict: Dictionary of arguments
        :return: Classification object
        """
        return util.ClassifyOutput(self.name)

    @staticmethod
    def resample(stream, sampling_rate):
        """
        Perform inplace resampling of stream to a given sampling rate.

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :param sampling_rate: Sampling rate (sps) to resample to
        :type sampling_rate: float
        """
        del_list = []
        for i, trace in enumerate(stream):
            if trace.stats.sampling_rate == sampling_rate:
                continue
            if trace.stats.sampling_rate % sampling_rate == 0:
                trace.filter("lowpass", freq=sampling_rate * 0.5, zerophase=True)
                trace.decimate(
                    int(trace.stats.sampling_rate / sampling_rate), no_filter=True
                )
            else:
                trace.resample(sampling_rate, no_filter=True)

        for i in del_list:
            del stream[i]

    @staticmethod
    def sanitize_mismatching_overlapping_records(stream):
        """
        Detects if for any id the stream contains overlapping traces that do not match.
        If yes, all mismatching parts are removed and a warning is issued.

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :return: The stream object without mismatching traces
        :rtype: obspy.core.Stream
        """
        stream.merge(-1)  # Ensures overlapping matching traces are merged
        original_num_traces = len(stream)

        ids = defaultdict(list)
        for trace_idx, trace in enumerate(stream):
            ids[trace.id].append((trace_idx, trace))

        del_idx = []
        for traces in ids.values():
            # Go through all traces in order, keep stack of active elements.
            # If more than one element is active, all events are discarded until the stack is empty again.
            start_times = sorted(
                [
                    (trace.stats.starttime, trace.stats.endtime, trace_idx)
                    for trace_idx, trace in traces
                ],
                key=lambda x: x[0],
            )

            p = 0
            conflict = False
            active_traces = PriorityQueue()
            while p < len(start_times):
                # Note that as active_traces.queue is a heap, the first element is always the smallest one
                act_start_time, act_end_time, act_trace_idx = start_times[p]
                while (
                    not active_traces.empty()
                    and active_traces.queue[0][0] <= act_start_time
                ):
                    _, trace_idx = active_traces.get()
                    if conflict:
                        del_idx.append(trace_idx)

                if active_traces.qsize() == 0:
                    conflict = False

                active_traces.put((act_end_time, act_trace_idx))
                p += 1

                if active_traces.qsize() > 1:
                    conflict = True

            if conflict:
                while not active_traces.empty():
                    _, trace_idx = active_traces.get()
                    del_idx.append(trace_idx)

        for idx in sorted(del_idx, reverse=True):
            # Reverse order to ensure that the different deletions do not interfere with each other
            del stream[idx]

        if not original_num_traces == len(stream):
            seisbench.logger.warning(
                "Detected multiple records for the same time and component that did not agree. "
                "All mismatching traces will be ignored."
            )

        return stream

    def stream_to_array(
        self,
        stream,
        argdict,
    ):
        """
        Converts streams into a start time and a numpy array.
        Assumes:

        - All traces within a group can be put into an array, i.e, the strict parameter is already enforced.
          Every remaining gap is intended to be filled with zeros.
          The selection/cutting of intervals has already been done by :py:func:`GroupingHelper.group_stream`.
        - No overlapping traces of the same component exist
        - All traces have the same sampling rate

        :param stream: Input stream
        :type stream: obspy.core.Stream
        :param argdict: Dictionary of arguments
        :return: output_times: Start times for each array
        :return: output_data: Arrays with waveforms
        """
        flexible_horizontal_components = self._argdict_get_with_default(
            argdict, "flexible_horizontal_components"
        )

        comp_dict, component_order = self._build_comp_dict(
            stream, flexible_horizontal_components
        )

        if self._grouping.grouping == "channel":

            def get_station_key(trace: obspy.Trace) -> str:
                return trace.id

        else:

            def get_station_key(trace: obspy.Trace) -> str:
                return self._grouping.trace_id_without_component(trace)

        stations = np.unique([get_station_key(trace) for trace in stream])
        station_dict = {station: i for i, station in enumerate(stations)}

        sampling_rate = stream[0].stats.sampling_rate
        t0 = min(trace.stats.starttime for trace in stream)
        t1 = max(trace.stats.endtime for trace in stream)

        data = np.zeros(
            (len(stations), len(component_order), int((t1 - t0) * sampling_rate + 2))
        )  # +2 avoids fractional errors

        for trace in stream:
            p = int((trace.stats.starttime - t0) * sampling_rate)
            if trace.id[-1] not in comp_dict:
                continue
            comp_idx = comp_dict[trace.id[-1]]
            sta_idx = station_dict[get_station_key(trace)]
            data[sta_idx, comp_idx, p : p + len(trace.data)] = trace.data

        data = data[:, :, :-1]  # Remove fractional error +1
        if self._grouping.grouping == "channel":
            data = data[0, 0]  # Remove station and channel dimension
        elif self._grouping.grouping == "instrument":
            data = data[0]  # Remove station dimension

        return t0, data, stations

    def _build_comp_dict(
        self, stream: obspy.Stream, flexible_horizontal_components: bool
    ):
        """
        Build mapping of component codes to indices taking into account flexible_horizontal_components
        """
        if self._component_order is None:
            # Use the provided component
            component_order = stream[0].id[-1]
        else:
            component_order = self._component_order

        comp_dict = {c: i for i, c in enumerate(component_order)}
        matches = [
            ("1", "N"),
            ("2", "E"),
        ]  # Component regarded as identical if flexible_horizontal_components is True.
        if flexible_horizontal_components:
            for a, b in matches:
                if a in comp_dict:
                    comp_dict[b] = comp_dict[a]
                elif b in comp_dict:
                    comp_dict[a] = comp_dict[b]

            # Maps traces to the components existing. Allows to warn for mixed use of ZNE and Z12.
            existing_trace_components = defaultdict(list)

            for trace in stream:
                if trace.id[-1] in comp_dict and len(trace.data) > 0:
                    existing_trace_components[
                        self._grouping.trace_id_without_component(trace)
                    ].append(trace.id[-1])

            for trace, components in existing_trace_components.items():
                for a, b in matches:
                    if a in components and b in components:
                        seisbench.logger.warning(
                            f"Station {trace} has both {a} and {b} components. "
                            f"This might lead to undefined behavior. "
                            f"Please preselect the relevant components "
                            f"or set flexible_horizontal_components=False."
                        )

        return comp_dict, component_order

    @staticmethod
    def picks_from_annotations(annotations, threshold, phase) -> util.PickList:
        """
        Converts the annotations streams for a single phase to discrete picks using a classical trigger on/off.
        The lower threshold is set to half the higher threshold.
        Picks are represented by :py:class:`~seisbench.util.annotations.Pick` objects.
        The pick start_time and end_time are set to the trigger on and off times.

        :param annotations: Stream of annotations
        :param threshold: Higher threshold for trigger
        :param phase: Phase to label, only relevant for output phase labelling
        :return: List of picks
        """
        picks = []
        for trace in annotations:
            trace_id = (
                f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            )
            triggers = trigger_onset(trace.data, threshold, threshold / 2)
            times = trace.times()
            for s0, s1 in triggers:
                t0 = trace.stats.starttime + times[s0]
                t1 = trace.stats.starttime + times[s1]

                peak_value = np.max(trace.data[s0 : s1 + 1])
                s_peak = s0 + np.argmax(trace.data[s0 : s1 + 1])
                t_peak = trace.stats.starttime + times[s_peak]

                pick = util.Pick(
                    trace_id=trace_id,
                    start_time=t0,
                    end_time=t1,
                    peak_time=t_peak,
                    peak_value=peak_value,
                    phase=phase,
                )
                picks.append(pick)

        return util.PickList(sorted(picks))

    @staticmethod
    def detections_from_annotations(annotations, threshold) -> util.DetectionList:
        """
        Converts the annotations streams for a single phase to discrete detections using a classical trigger on/off.
        The lower threshold is set to half the higher threshold.
        Detections are represented by :py:class:`~seisbench.util.annotations.Detection` objects.
        The detection start_time and end_time are set to the trigger on and off times.

        :param annotations: Stream of annotations
        :param threshold: Higher threshold for trigger
        :return: List of detections
        """
        detections = []
        for trace in annotations:
            trace_id = (
                f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"
            )
            triggers = trigger_onset(trace.data, threshold, threshold / 2)
            times = trace.times()
            for s0, s1 in triggers:
                t0 = trace.stats.starttime + times[s0]
                t1 = trace.stats.starttime + times[s1]
                peak_value = np.max(trace.data[s0 : s1 + 1])

                detection = util.Detection(
                    trace_id=trace_id, start_time=t0, end_time=t1, peak_value=peak_value
                )
                detections.append(detection)

        return util.DetectionList(sorted(detections))

    def get_model_args(self):
        model_args = super().get_model_args()
        model_args = {
            **model_args,
            **{
                "sampling_rate": self.sampling_rate,
                "output_type": self.output_type,
                "component_order": self.component_order,
                "default_args": self.default_args,
                "in_samples": self.in_samples,
                "pred_sample": self.pred_sample,
                "labels": self.labels,
                "filter_args": self.filter_args,
                "filter_kwargs": self.filter_kwargs,
                "grouping": self._grouping.grouping,
            },
        }
        return model_args


class WaveformPipeline(ABC):
    """
    A waveform pipeline is a collection of models that together expose an :py:func:`annotate` and
    a :py:func:`classify` function. Examples of waveform pipelines would be multi-step picking models,
    conducting first a detection with one model and then a pick identification with a second model.
    This could also easily be extended by adding further models, e.g., estimating magnitude for each detection.

    In contrast to :py:class:`WaveformModel`, a waveform pipeline is not a pytorch module and has no forward function.
    This also means, that all components of a pipeline will usually be trained separately. As a rule of thumb,
    if the pipeline can be trained end to end, it should most likely rather be a :py:class:`WaveformModel`.
    For a waveform pipeline, the :py:func:`annotate` and :py:func:`classify` functions are not automatically generated,
    but need to be implemented manually.

    Waveform pipelines offer functionality for downloading pipeline configurations from the SeisBench repository.
    Similarly to :py:class:`SeisBenchModel`, waveform pipelines expose a :py:func:`from_pretrained` function,
    that will download the configuration for a pipeline and its components.

    To implement a waveform pipeline, this class needs to be subclassed. This class will throw an exception when
    trying to instantiate.

    .. warning::
        In contrast to :py:class:`SeisBenchModel` this class does not yet feature versioning for weights. By default,
        all underlying models will use the latest, locally available version. This functionality will eventually be
        added. Please raise an issue on Github if you require this functionality.

    :param components: Dictionary of components contained in the model. This should contain all models used in the
                       pipeline.
    :type components: dict [str, SeisBenchModel]
    :param citation: Citation reference, defaults to None.
    :type citation: str, optional
    """

    def __init__(self, components, citation=None):
        self.components = components
        self._citation = citation
        self._docstring = None

    @classmethod
    @abstractmethod
    def component_classes(cls):
        """
        Returns a mapping of component names to their classes. This function needs to be defined in each pipeline,
        as it is required to load configurations.

        :return: Dictionary mapping component names to their classes.
        :rtype: Dict[str, SeisBenchModel classes]
        """
        return {}

    @property
    def docstring(self):
        return self._docstring

    @property
    def citation(self):
        return self._citation

    @property
    def name(self):
        return self._name_internal()

    @classmethod
    def _name_internal(cls):
        return cls.__name__

    @classmethod
    def _remote_path(cls):
        return urljoin(
            seisbench.remote_root, "pipelines/" + cls._name_internal().lower()
        )

    @classmethod
    def _local_path(cls):
        return Path(seisbench.cache_root, "pipelines", cls._name_internal().lower())

    @classmethod
    def _config_path(cls, name):
        return cls._local_path() / f"{name}.json"

    def annotate(self, stream, **kwargs):
        raise NotImplementedError("This class does not expose an annotate function.")

    def classify(self, stream, **kwargs):
        raise NotImplementedError("This class does not expose a classify function.")

    @classmethod
    def from_pretrained(cls, name, force=False, wait_for_file=False):
        """
        Load pipeline from configuration. Automatically loads all dependent pretrained models weights.

        A pipeline configuration is a json file. On the top level, it has three entries:

        - "components": A dictionary listing all contained models and the pretrained weight to use for this model.
                        The instances of these classes will be created using the
                        :py:func:`~SeisBenchModel.from_pretrained` method.
                        The components need to match the components from the dictionary returned by
                        :py:func:`component_classes`.
        - "docstring": A string documenting the pipeline. Usually also contains information on the author.
        - "model_args": Argument dictionary passed to the init function of the pipeline. (optional)

        :param name: Configuration name
        :type name: str
        :param force: Force execution of download callback, defaults to False
        :type force: bool, optional
        :param wait_for_file: Whether to wait on partially downloaded files, defaults to False
        :type wait_for_file: bool, optional
        :return: Pipeline instance
        :rtype: WaveformPipeline
        """
        config_path = cls._config_path(name)
        cls._local_path().mkdir(parents=True, exist_ok=True)

        def download_callback(config_path):
            remote_config_path = os.path.join(cls._remote_path(), f"{name}.json")
            util.download_http(remote_config_path, config_path, progress_bar=False)

        seisbench.util.callback_if_uncached(
            config_path,
            download_callback,
            force=force,
            wait_for_file=wait_for_file,
        )

        with open(config_path, "r") as f:
            config = json.load(f)

        component_classes = cls.component_classes()
        component_weights = config.get("components", {})
        if sorted(component_weights.keys()) != sorted(component_classes.keys()):
            raise ValueError(
                "Invalid configuration. Components don't match between configuration and class."
            )

        components = {
            key: component_classes[key].from_pretrained(
                component_weights[key], force=force, wait_for_file=wait_for_file
            )
            for key in component_weights
        }

        model_args = config.get("model_args", {})
        model = cls(components, **model_args)

        model._docstring = config.get("docstring", None)

        return model

    @classmethod
    def list_pretrained(cls, details=False):
        """
        Returns list of available configurations and optionally their docstrings.

        :param details: If true, instead of a returning only a list, also return their docstrings.
                        Note that this requires to download the json files for each model in the background
                        and is therefore slower. Defaults to false.
        :type details: bool
        :return: List of available weights or dict of weights and their docstrings
        :rtype: list or dict
        """
        remote_path = cls._remote_path()

        try:
            configurations = [
                x[:-5]
                for x in seisbench.util.ls_webdav(remote_path)
                if x.endswith(".json")
            ]
        except ValueError:
            # No weights available
            configurations = []

        if details:
            detail_configurations = {}

            # Create path if necessary
            cls._local_path().mkdir(parents=True, exist_ok=True)

            for configuration in configurations:

                def download_callback(config_path):
                    remote_config_path = os.path.join(
                        cls._remote_path(), f"{configuration}.json"
                    )
                    seisbench.util.download_http(
                        remote_config_path, config_path, progress_bar=False
                    )

                config_path = cls._config_path(configuration)

                seisbench.util.callback_if_uncached(config_path, download_callback)

                with open(config_path, "r") as f:
                    config = json.load(f)
                detail_configurations[configuration] = config.get("docstring", None)

            configurations = detail_configurations

        return configurations


class Conv1dSame(nn.Module):
    """
    Add PyTorch compatible support for Tensorflow/Keras padding option: padding='same'.

    Discussions regarding feature implementation:
    https://discuss.pytorch.org/t/converting-tensorflow-model-to-pytorch-issue-with-padding/84224
    https://github.com/pytorch/pytorch/issues/3867#issuecomment-598264120

    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()
        self.cut_last_element = (
            kernel_size % 2 == 0 and stride == 1 and dilation % 2 == 1
        )
        self.padding = math.ceil((1 - stride + dilation * (kernel_size - 1)) / 2)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding + 1,
            stride=stride,
            dilation=dilation,
        )

    def forward(self, x):
        if self.cut_last_element:
            return self.conv(x)[:, :, :-1]
        else:
            return self.conv(x)


def hard_sigmoid(x):
    return torch.clip(0.2 * x + 0.5, 0, 1)


class ActivationLSTMCell(nn.Module):
    """
    LSTM Cell using variable gating activation, by default hard sigmoid

    If gate_activation=torch.sigmoid this is the standard LSTM cell

    Uses recurrent dropout strategy from https://arxiv.org/abs/1603.05118 to match Keras implementation.
    """

    def __init__(
        self, input_size, hidden_size, gate_activation=hard_sigmoid, recurrent_dropout=0
    ):
        super(ActivationLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_activation = gate_activation
        self.recurrent_dropout = recurrent_dropout

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for param in [self.weight_hh, self.weight_ih]:
                for idx in range(4):
                    mul = param.shape[0] // 4
                    torch.nn.init.xavier_uniform_(param[idx * mul : (idx + 1) * mul])

    def forward(self, input, state):
        if state is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
            cx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
        else:
            hx, cx = state
        gates = (
            torch.mm(input, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(hx, self.weight_hh.t())
            + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.gate_activation(ingate)
        forgetgate = self.gate_activation(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = self.gate_activation(outgate)

        if self.recurrent_dropout > 0:
            cellgate = F.dropout(cellgate, p=self.recurrent_dropout)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class CustomLSTM(nn.Module):
    """
    LSTM to be used with custom cells
    """

    def __init__(self, cell, *cell_args, bidirectional=True, **cell_kwargs):
        super(CustomLSTM, self).__init__()
        self.cell_f = cell(*cell_args, **cell_kwargs)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cell_b = cell(*cell_args, **cell_kwargs)

    def forward(self, input, state=None):
        # Forward
        state_f = state
        outputs_f = []
        for i in range(len(input)):
            out, state_f = self.cell_f(input[i], state_f)
            outputs_f += [out]

        outputs_f = torch.stack(outputs_f)

        if not self.bidirectional:
            return outputs_f, None

        # Backward
        state_b = state
        outputs_b = []
        l = input.shape[0] - 1
        for i in range(len(input)):
            out, state_b = self.cell_b(input[l - i], state_b)
            outputs_b += [out]

        outputs_b = torch.flip(torch.stack(outputs_b), dims=[0])

        output = torch.cat([outputs_f, outputs_b], dim=-1)

        # Keep second argument for consistency with PyTorch LSTM
        return output, None


class GroupingHelper:
    """
    A helper class for grouping streams for the annotate function.
    In most cases, no direct interaction with this class is required.
    However, when implementing new models, subclassing this helper allows for more flexibility.
    """

    def __init__(self, grouping: str) -> None:
        self._grouping = grouping

        self._grouping_functions = {
            "instrument": self._group_instrument,
            "channel": self._group_channel,
            "full": self._group_full,
        }

        if grouping not in self._grouping_functions:
            raise ValueError(f"Unknown grouping parameter '{self.grouping}'.")

    @property
    def grouping(self):
        return self._grouping

    def group_stream(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
    ) -> list[list[obspy.Trace]]:
        """
        Perform grouping of input stream.
        In addition, enforces the strict mode, i.e, if strict=True only keeps segments where all components are available,
        and discards segments that are too short.
        For grouping=channel no checks are performed.

        :param stream: Input stream
        :param strict: If streams should be treated strict as for waveform model.
                       Only applied if grouping is "full".
        :param min_length_s: Minimum length of a segment in seconds.
                             Only applied if grouping is "full".
        :param comp_dict: Mapping of component characters to int.
                          Only used if grouping is "full".
        :return: Grouped list of list traces.
        """
        return self._grouping_functions[self.grouping](
            stream, strict, min_length_s, comp_dict
        )

    @staticmethod
    def trace_id_without_component(trace: obspy.Trace):
        return f"{trace.stats.network}.{trace.stats.station}.{trace.stats.location}"

    def _group_instrument(
        self, stream: obspy.Stream, *args, **kwargs
    ) -> list[list[obspy.Trace]]:
        pre_groups = defaultdict(list)
        for trace in stream:
            pre_groups[self.trace_id_without_component(trace)].append(trace)

        groups = []
        for group in pre_groups.values():
            groups.extend(self._group_full(obspy.Stream(group), *args, **kwargs))

        return groups

    def _group_channel(
        self, stream: obspy.Stream, *args, **kwargs
    ) -> list[list[obspy.Trace]]:
        pre_groups = defaultdict(list)
        for trace in stream:
            pre_groups[trace.id].append(trace)

        groups = []
        for group in pre_groups.values():
            groups.extend(
                self._group_full(obspy.Stream(group), *args, channel=True, **kwargs)
            )

        return groups

    def _group_full(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
        channel: bool = False,
    ) -> list[list[obspy.Trace]]:
        intervals = self._get_intervals(
            stream, strict, min_length_s, comp_dict, channel=channel
        )

        return self._assemble_groups(stream, intervals)

    @staticmethod
    def _bin_search_idx(coords: list[obspy.UTCDateTime], t: obspy.UTCDateTime) -> int:
        mini = 0
        maxi = len(coords)

        while (maxi - mini) != 1:
            m = (maxi + mini) // 2
            if coords[m] > t:
                maxi = m
            else:
                mini = m
        return mini

    @staticmethod
    def _align_fractional_samples(stream: obspy.Stream) -> None:
        """
        Shifts the starttime of every member to a valid fractional second according to the sampling rate.
        Assumes there is a hypothetical sample at UTCDateTime(0).

        For example, at 20 Hz sampling rate:
        0.05 is okay
        0.06 is not
        """
        for trace in stream:
            ts = trace.stats.starttime.timestamp
            ts *= trace.stats.sampling_rate
            ts = np.round(ts) / trace.stats.sampling_rate
            trace.stats.starttime = obspy.UTCDateTime(ts)

    def _get_intervals(
        self,
        stream: obspy.Stream,
        strict: bool,
        min_length_s: float,
        comp_dict: dict[str, int],
        channel: bool = False,
    ) -> list[tuple[list[str], obspy.UTCDateTime, obspy.UTCDateTime]]:
        if channel:
            strict = False

        self._align_fractional_samples(stream)

        # Do coordinate compression
        coords = np.unique(
            [trace.stats.starttime for trace in stream]
            + [trace.stats.endtime for trace in stream]
        )
        coords = sorted(list(coords))
        if len(coords) <= 1:
            return []

        if channel:
            n_comp = 1
            stations = sorted(list(set(trace.id for trace in stream)))
        else:
            n_comp = max(comp_dict.values()) + 1
            stations = sorted(
                list(set(self.trace_id_without_component(trace) for trace in stream))
            )

        sta_dict = {sta: i for i, sta in enumerate(stations)}

        covered = np.zeros((len(stations), n_comp, len(coords) - 1), dtype=bool)

        for trace in stream:
            p0 = self._bin_search_idx(coords, trace.stats.starttime)
            p1 = self._bin_search_idx(coords, trace.stats.endtime)

            if channel:
                comp_idx = 0
                sta_idx = sta_dict[trace.id]
            else:
                if trace.id[-1] not in comp_dict:
                    continue

                comp_idx = comp_dict[trace.id[-1]]
                sta_idx = sta_dict[self.trace_id_without_component(trace)]

            covered[sta_idx, comp_idx, p0:p1] = True

        if strict:
            covered = covered.all(axis=1)
        else:
            covered = covered.any(axis=1)

        covered, coords = self._merge_intervals(covered, coords, min_length_s)

        intervals = []

        act = covered[:, 0]
        start_i = 0
        for i in range(1, covered.shape[1]):
            if np.all(act == covered[:, i]):
                # Same station coverage in both blocks, merge the intervals
                continue
            else:
                if act.any():
                    interval_stations = [
                        sta for sta, m in zip(stations, act) if m
                    ]  # Active stations in interval
                    t0 = coords[start_i]
                    t1 = coords[i]

                    intervals.append((interval_stations, t0, t1))

                start_i = i
                act = covered[:, i]

        if act.any():
            interval_stations = [
                sta for sta, m in zip(stations, act) if m
            ]  # Active stations in interval
            t0 = coords[start_i]
            t1 = coords[covered.shape[1]]
            intervals.append((interval_stations, t0, t1))

        return intervals

    def _merge_intervals(
        self, covered: np.ndarray, coords: list[obspy.UTCDateTime], min_length_s: float
    ) -> tuple[np.ndarray, list[obspy.UTCDateTime]]:
        # Goal: Maximize "stations * time" while ensuring no segment is too short
        # Use a greedy approach for maximizing which will not always lead to the globally optimal results
        # but usually to reasonably good results.
        has_warned = np.zeros(1, dtype=bool)

        def encompassing_interval(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime):
            p0 = self._bin_search_idx(coords, t0)
            # Move index to actual left border of this segment
            while p0 > 0:
                if (covered[:, p0 - 1] == covered[:, p0]).all():
                    p0 -= 1
                else:
                    break

            p1 = self._bin_search_idx(coords, t1)
            if coords[p1] != t1:
                # This ensures that coords[p1] is greater or equal than t1
                p1 += 1
            # Move index to actual right border of this segment
            while p1 < covered.shape[-1] - 1:
                if (covered[:, p1 + 1] == covered[:, p1]).all():
                    p1 += 1
                else:
                    break

            return p0, p1

        def merge_costs(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime) -> float:
            if t0 < coords[0] or t1 > coords[-1]:
                # The interval is not actually fully covered
                return np.inf

            p0, p1 = encompassing_interval(t0, t1)

            cost = 0

            merged_cover = np.all(
                covered[:, p0:p1], axis=1
            )  # Stations present in the whole interval
            if (
                not merged_cover.any()
            ):  # This is never better than just deleting the center interval
                return np.inf

            for p in range(p0, p1):
                if t0 - coords[p] > min_length_s:  # Left border profits from splitting
                    delta_t = t0 - (coords[p + 1] - min_length_s)
                elif (
                    coords[p + 1] - t1 > min_length_s
                ):  # Right border profits from splitting
                    delta_t = t1 - (coords[p] + min_length_s)
                else:
                    delta_t = coords[p + 1] - coords[p]
                delta_sta = np.sum(covered[:, p]) - np.sum(merged_cover)
                cost += delta_t * delta_sta

            return cost

        def merge_interval(t0: obspy.UTCDateTime, t1: obspy.UTCDateTime) -> None:
            p0, p1 = encompassing_interval(t0, t1)
            merged_cover = np.all(covered[:, p0:p1], axis=1)
            for p in range(p0, p1):
                if t0 - coords[p] > min_length_s:  # Left border profits from splitting
                    coords[p + 1] = t0  # End interval earlier
                elif (
                    coords[p + 1] - t1 > min_length_s
                ):  # Right border profits from splitting
                    coords[p] = t1  # Start interval later
                else:
                    covered[:, p] = merged_cover

        def fix_interval_if_necessary(act: np.ndarray, p0: int, p1: int):
            t0 = coords[p0]
            t1 = coords[p1]

            if act.any() and t1 - t0 < min_length_s:
                # Fixing required
                # Iterate over all reasonable merging times and find the cheapest
                # Reasonable merge intervals:
                # - every interval covering the target, starting either at a coord to the left
                #   or ending at a coord to the right
                # - if corner interval has at least min_length_s left, make new coord
                # - else, just merge intervals

                if not has_warned[0]:
                    has_warned[0] = True
                    seisbench.logger.warning(
                        "Parts of the input stream consist of fragments shorter than the number "
                        "of input samples or misaligned traces. Output might be empty."
                    )

                candidate_starts = []
                for p in range(p0, -1, -1):
                    if t1 - coords[p] > min_length_s:
                        break
                    candidate_starts.append(coords[p])
                for p in range(p0 + 1, len(coords)):
                    if coords[p] - t0 > min_length_s:
                        break
                    candidate_starts.append(coords[p] - min_length_s)

                candidate_starts = np.unique(candidate_starts)

                candidate_merge_costs = [
                    merge_costs(t, t + min_length_s) for t in candidate_starts
                ]

                if np.isinf(np.min(candidate_merge_costs)):
                    # Only delete if there is no other option.
                    # This improves coverage over time
                    act &= False
                else:
                    best_merge = np.argmin(candidate_merge_costs)
                    merge_interval(
                        candidate_starts[best_merge],
                        candidate_starts[best_merge] + min_length_s,
                    )

        act = covered[:, 0]
        start_i = 0

        for i in range(1, covered.shape[1]):
            if np.all(act == covered[:, i]):
                # Same station coverage in both blocks, merge the intervals
                continue
            else:
                fix_interval_if_necessary(act, start_i, i)

                start_i = i
                act = covered[:, i]

        fix_interval_if_necessary(act, start_i, covered.shape[1])

        return covered, coords

    @staticmethod
    def _assemble_groups(
        stream: obspy.Stream,
        intervals: list[tuple[list[str], obspy.UTCDateTime, obspy.UTCDateTime]],
    ) -> list[list[obspy.Trace]]:
        groups = []
        for stations, t0, t1 in intervals:
            sub = stream.slice(t0, t1)
            group = []
            for station in stations:
                for trace in sub.select(id=station + "*"):
                    group.append(trace)
            groups.append(group)

        return groups
