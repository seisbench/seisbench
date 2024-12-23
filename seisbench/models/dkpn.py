import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import seisbench.util as sbu
from seisbench.models.base import WaveformModel, _cache_migration_v0_v3

# ---------  For PreProc class
from obspy.core import Trace
import copy
from obspy.signal.filter import bandpass
from scipy.signal import lfilter


class DKPN(WaveformModel):
    """
    .. document_args:: seisbench.models PhaseNet
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
    ]

    def __init__(
        self,
        in_channels=5,
        classes=3,
        phases="PSN",
        sampling_rate=100,
        norm="peak",
        grouping="channel",
        **kwargs,
    ):
        citation = (
            "Lomax, Anthony, Matteo Bagagli, Sonja Gaviano, Spina Cianetti, "
            "Dario Jozinović, Alberto Michelini, Christopher Zerafa, and "
            "Carlo Giunchi. 'Effects on a Deep-Learning, Seismic Arrival-Time "
            "Picker of Domain-Knowledge Based Preprocessing of Input "
            "Seismograms.' Seismica 3, no. 1 (2024)."
        )

        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            default_args={
                "overlap": 1500,
                "P_threshold": 0.2,
                "S_threshold": 0.2,
                #
                "fp_stabilization": 4,  # sec
                "t_long": 4,
                "freqmin": 0.5,
                "corner": 1,
                "perc_taper": 0.1,
                "mode": "rms",
                "clip": -999,
                "log": True,
                "normalize": False,
                "polarization_win_len": 1,
                "use_amax_only": False},
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(8, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

        # DKPN attributes for taking care of windows CF and probs
        self.batch_ = []
        self.batch_cfs = []
        self.batch_probs = []
        self.stream_cfs = None

    def __reset_predict(self):
        self.batch_ = []
        self.batch_cfs = []
        self.batch_probs = []
        self.stream_cfs = None

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    # ================================================================
    # ==================================  PROCESS STREAM
    # ================================================================

    def annotate_stream_pre(self, stream, argdict):
        """ ARGDICT represent the default dict,merged (eventually)
            with the key:value pairs of the "*.json" specifying the model
        """
        self.__reset_predict()

        # -----------------   Pre-pend the original code of SEISBENCH (v0.2)
        if self.filter_args is not None or self.filter_kwargs is not None:
            if self.filter_args is None:
                filter_args = ()
            else:
                filter_args = self.filter_args

            if self.filter_kwargs is None:
                filter_kwargs = {}
            else:
                filter_kwargs = self.filter_kwargs

            stream.filter(*filter_args, **filter_kwargs)

        _prpr = PreProc(**argdict)
        _prpr.work(stream)
        stream = _prpr.get_stream()
        self.stream_cfs = stream.copy()
        return stream

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:

        assert batch.shape[1] == 5
        self.batch_.append(batch.detach().clone())

        # ===================================================  STD
        # --- NORMALIZE  FP-CFs matrix with the matrix STD
        batch[:, 0:3, :] = batch[:, 0:3, :] / (
            torch.std(batch[:, 0:3, :], axis=1, keepdims=True) + 1e-10
            )
        # --- NORMALIZE  MODULUS matrix with the its STD
        batch[:, 4, :] = batch[:, 4, :] / (
            torch.std(batch[:, 4, :], axis=-1, keepdims=True) + 1e-10
            )

        # --- The incidence remains untouched
        self.batch_cfs.append(batch.detach().clone())  # Store WINDOW CFS

        return batch

    def annotate_batch_post(
            self, batch: torch.Tensor, piggyback=None, argdict=None):
        # Transpose predictions to correct shape
        batch = torch.transpose(batch, -1, -2)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        self.batch_probs.append(batch.detach().clone())
        return batch

    def extract_cf(self, stream):
        _prpr = PreProc(**self.default_args)
        _prpr.work(stream.copy())
        return _prpr.get_stream()

    def get_cf_windows(self):
        """ This method extract the single's windows used in CNN process """
        return self.batch_cfs

    def get_probs_windows(self):
        """ This method extract the single's windows used in CNN process """
        return self.batch_probs

    # ================================================================
    # ==================================  END  PROCESS STREAM
    # ================================================================

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "sampling_rate",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate

        return model_args

    def get_defaults(self):
        return self.default_args

    def set_dkpn_parameter(self, pardict):
        """ Must provide the full dictionary with all keys """
        if not isinstance(pardict, dict):
            raise TypeError("I need a dict-type object. Received: %s" % type(pardict))
        #
        self.default_args = pardict

    @classmethod
    def from_pretrained_expand(
        cls, name, version_str="latest", update=False, force=False, wait_for_file=False
    ):
        """
        Load pretrained model with weights and copy the input channel weights that match the Z component to a new,
        4th dimension that is used to process the hydrophone component of the input trace.
        For further instructions, see :py:func:`~seisbench.models.base.SeisBenchModel.from_pretrained`. This method
        differs from :py:func:`~seisbench.models.base.SeisBenchModel.from_pretrained` in that it does not call helper
        functions to load the model weights. Instead it covers the same logic and, in addition, takes intermediate
        steps to insert a new `in_channels` dimension to the loaded model and copy weights.
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

        weight_path, metadata_path = cls._pretrained_path(name, version_str)

        cls._ensure_weight_files(
            name, version_str, weight_path, metadata_path, force, wait_for_file
        )

        if metadata_path.is_file():
            with open(metadata_path, "r") as f:
                weights_metadata = json.load(f)
        else:
            weights_metadata = {}
        model_args = weights_metadata.get("model_args", {})
        model_args["in_channels"] = 4
        model = cls(**model_args)

        model._weights_metadata = weights_metadata
        model._parse_metadata()

        state_dict = torch.load(weight_path)
        old_weight = state_dict["inc.weight"]
        state_dict["inc.weight"] = torch.zeros(
            old_weight.shape[0], old_weight.shape[1] + 1, old_weight.shape[2]
        ).type_as(old_weight)
        state_dict["inc.weight"][:, :3, ...] = old_weight
        state_dict["inc.weight"][:, 3, ...] = old_weight[:, 0, ...]
        model.load_state_dict(state_dict)
        return model


# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================  PREPROC
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================


class PreProc(object):
    def __init__(self, **kwargs):
        self.stream = None  # --> Instantiated everytime in the work() class-method
        self.__dict__.update(kwargs)
        self.eps = 1e-10

    def __call__(self, state_dict):
        """
        Provides an augmentation for seisbench.generate.generator.GenericGenerator.add_augmentations().
        """

        labels, metadata_labels = state_dict["y"]
        waveforms, metadata_waveforms = state_dict["X"]  # metadata is the relevant row-field of CSV
        waveforms = self.__matrix_cfs__(waveforms)

        # ================================  Emulates WINDOW-LEVEL NORM
        # --- Remove FP stabilization
        fstab = int(metadata_waveforms["trace_sampling_rate_hz"]*self.fp_stabilization)
        waveforms = waveforms[:, fstab:(3001+fstab)]
        labels = labels[:, fstab:(3001+fstab)]

        # --- NORMALIZE  FP-CF matrix with the matrix STD
        waveforms[0:3, :] = waveforms[0:3, :] / (
            np.std(waveforms[0:3, :], axis=1, keepdims=True) + 1e-10
            )

        # --- NORMALIZE  MODULUS matrix with the its STD
        waveforms[4, :] = waveforms[4, :] / (
            np.std(waveforms[4, :], axis=-1, keepdims=True) + 1e-10
            )

        state_dict["X"] = (waveforms, metadata_waveforms)  # OVERRIDE MATRIX --> OUTPUT
        state_dict["y"] = (labels, metadata_labels)  # OVERRIDE MATRIX --> OUTPUT

    def __matrix_cfs__(self, MM):
        """ Given a matrix in input, it returns the new CFs only """

        # --- DEMEAN
        MM = MM - np.mean(MM, axis=1, keepdims=True)

        # --- NORMALIZE  input stream matrix (3C) with the matrix standard deviation
        MM = MM / (
            np.std(MM, axis=1, keepdims=True) + self.eps
            )

        # ===========================  GO CALCULATE
        fp_cf_waveforms = []
        fp_band_data = []
        fp_max_arrays_band = []
        fp_max_arrays_val = []

        for _idx, arr in enumerate(MM):

            # ====================
            # 1) Run FBsummary
            summary = FBSummary(arr,
                                sampling_rate=100.0,
                                npts=arr.shape[0],
                                t_long=self.t_long,
                                freqmin=self.freqmin,
                                corner=self.corner,
                                perc_taper=self.perc_taper,
                                mode=self.mode)

            fp_cf_waveforms.append(summary.summary)

            fp_band_data.append(summary.BF)
            fp_max_array_band = np.argmax(summary.FC, axis=0)
            fp_max_arrays_band.append(fp_max_array_band)
            fp_max_array_val = summary.summary
            fp_max_arrays_val.append(fp_max_array_val)

        fp_cf_waveforms = np.asarray(fp_cf_waveforms)

        if self.clip > 0.0:
            fp_cf_waveforms = np.clip(fp_cf_waveforms, a_min=None, a_max=self.clip)
        if self.log:
            fp_cf_waveforms = np.log10(fp_cf_waveforms + 1.0)
        if self.normalize:
            fp_cf_waveforms = fp_cf_waveforms / (
                np.amax(np.abs(fp_cf_waveforms), axis=(0, 1), keepdims=True) + self.eps)

        # ====================
        # 2) Run polarization
        fp_band_data = np.asarray(fp_band_data)
        fp_max_arrays_band = np.asarray(fp_max_arrays_band)     # [nchan, npts]
        fp_max_arrays_band_amax = np.amax(fp_max_arrays_band, axis=0)       # [npts]
        band_amax = -1
        if self.use_amax_only:
            fp_max_arrays_val = np.asarray(fp_max_arrays_val)       # [nchan, npts]
            fp_max_arrays_val_argmax_ndx = np.argmax(fp_max_arrays_val)
            index_amax = np.unravel_index(fp_max_arrays_val_argmax_ndx, fp_max_arrays_val.shape)
            band_amax = fp_max_arrays_band[index_amax]

        result_dict = self.PolarizationFP(fp_band_data, fp_max_arrays_band_amax, band_amax)

        # --- Add Incidence
        fp_cf_waveforms = np.append(fp_cf_waveforms,
                                    [result_dict["incidence"]],
                                    axis=0)

        # --- Add Modulus
        fp_cf_waveforms = np.append(fp_cf_waveforms,
                                    [result_dict["modulus"]],
                                    axis=0)

        # --- Bring everything to float32
        fp_cf_waveforms = fp_cf_waveforms.astype("float32")

        return fp_cf_waveforms

    def get_stream(self):
        return self.stream

    def work(self, stream):
        """
        Main order of pre-processing stages
        """

        self.stream = stream
        self.stream.merge(method=0,
                          fill_value="interpolate",
                          interpolation_samples=0)
        # Ensure channel order ZNE
        self.stream.sort(keys=["channel"], reverse=True)
        # Perform a linear detrend on the data  (PhasePapy)
        self.stream.detrend('linear')

        # --- Create a matrix to easen the operation
        MM = np.zeros([3, len(self.stream[0].data)])
        for idx in range(3):
            MM[idx, :] = self.stream[idx].data

        fp_cf_waveforms = self.__matrix_cfs__(MM)

        # Post CF calculations done at a window level.
        # ====================  Populate the out-stream
        for _idx, _cfdata in enumerate(fp_cf_waveforms):
            if _idx < 3:
                self.stream[_idx].data = _cfdata
                self.stream[_idx].stats.channel = (
                    "CF"+self.stream[_idx].stats.channel[-1]
                    )

            elif _idx == 3:
                # Take the metadata from the first trace
                # they should be equal with the other
                # (i.e. staname, sampling rate ...)
                _cfstats = copy.deepcopy(self.stream[0].stats)
                _cfstats.channel = "CFI"  # just modify the NAME --> incidence

                self.stream += Trace(data=_cfdata,
                                     header=_cfstats)
            elif _idx == 4:
                # Take the metadata from the first trace
                # they should be equal with the other
                # (i.e. staname, sampling rate ...)
                _cfstats = copy.deepcopy(self.stream[0].stats)
                _cfstats.channel = "CFM"  # just modify the NAME --> incidence

                self.stream += Trace(data=_cfdata,
                                     header=_cfstats)

        # # ==================== FP_ STABILIZATION
        # print("DEBUG: removing the first %.2f seconds of trace" % self.fp_stabilization)
        # for _tr in self.stream:
        #     _tr.trim(_tr.stats.starttime + self.fp_stabilization,
        #              _tr.stats.endtime)

    def PolarizationFP(self, fp_band_data, fp_max_arrays_band_amax,
                       band_amax=-1):

        nband = band_amax
        res = []

        if self.use_amax_only:
            res = self.incidence_modulus(fp_band_data[0][nband], fp_band_data[1][nband], fp_band_data[2][nband])
        else:
            trace_len = len(fp_band_data[0][0])
            data = fp_band_data[:, fp_max_arrays_band_amax, np.arange(trace_len, dtype=int)]
            res = self.incidence_modulus(data[0], data[1], data[2])
        #
        result_dict = {
                    "incidence": res[0],
                    "modulus": res[1],
                    }
        return result_dict

    def incidence_modulus(self, vertical, north, east):
        """
        Computes the single point, instantaneous particle-motion azimuth, incidence and magnitude of a 3-comp time-series

        :param vertical: ZNE trace data
        :type float
        :returns:  incidence (-1.0 (down) -> 0 (horiz) -> 1.0 (up)) and modulus
        """

        hxy = np.hypot(north, east)
        modulus = np.hypot(hxy, vertical)
        # if no horizontal signal, set incidence to 0
        if np.max(hxy) > np.max(vertical) / 1000.0:
            incidence = np.arctan2(vertical, hxy)   # -pi (down) -> 0 (horiz) -> pi (up)
            incidence = incidence / (np.pi / 2.0)   # -1.0 (down) -> 0 (horiz) -> 1.0 (up)  # BUGFIX
        else:
            incidence = np.zeros_like(vertical)
        if self.log:
            modulus = np.log10(modulus + 1.0)
        if self.normalize:
            modulus = modulus / (np.max(modulus + 1e-6))    # normalized

        return [incidence, modulus]


# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================  FBSummary
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================
# ====================================================================


class FBSummary():
    """
    The class defines all CFn bands data and statistics.
    """

    def __init__(self,
                 data,
                 npts=3001,
                 sampling_rate=100.0,
                 t_long=5,
                 freqmin=1,
                 corner=1,
                 perc_taper=0.1,
                 mode='rms',
                 t_ma=20,
                 nsigma=6,
                 t_up=0.78,
                 nr_len=2,
                 nr_coeff=2,
                 pol_len=10,
                 pol_coeff=10,
                 uncert_coeff=3):

        # #====OLD
        # self.tr = tr
        # self.stats = self.tr.stats
        self.data = data
        self.npts = npts
        self.sampling_rate = sampling_rate
        self.delta = 1/self.sampling_rate
        self.summary = None

        # --------------------------------
        self.t_long = t_long
        self.freqmin = freqmin
        self.cnr = corner
        self.perc_taper = perc_taper
        self.statistics_mode = mode
        self.t_ma = t_ma
        self.nsigma = nsigma
        self.t_up = t_up
        self.nr_len = nr_len
        self.nr_coeff = nr_coeff
        self.pol_len = pol_len
        self.pol_coeff = pol_coeff
        self.uncert_len = self.t_ma
        self.uncert_coeff = uncert_coeff
        # --------------------------------

        self.FC, self.BF = self._statistics_decay()
        self.summary = np.amax(self.FC, axis=0)

    def _rms(self, x, axis=None):
        """ Function to calculate the root mean square value of an array.
        """
        return np.sqrt(np.mean(x**2, axis=axis))

    def _N_bands(self):
        """ Determine number of band n_bands in term of sampling rate.
        """
        Nyquist = self.sampling_rate / 2.0
        n_bands = int(np.log2(Nyquist / 1.5 / self.freqmin)) + 1
        return n_bands

    def filter(self):
        """ Filter data for each band.
        """
        n_bands = self._N_bands()
        # create zeros 2D array for BF
        BF = np.zeros(shape=(n_bands, self.npts))

        for j in range(n_bands):
            octave_high = (self.freqmin + self.freqmin * 2.0) / 2.0 * (2**j)
            octave_low = octave_high / 2.0
            BF[j] = bandpass(self.data, octave_low, octave_high,
                             self.sampling_rate,
                             corners=self.cnr,
                             zerophase=False)
        return BF

    def get_summary(self):
        return self.summary

    def _statistics_decay(self):
        """ Calculate statistics for each band.
        """
        n_bands = self._N_bands()

        npts_t_long = int(self.t_long / self.delta)
        decay_const = 1.0 - (1.0 / npts_t_long)
        # one_minus_decay_const = 1.0 - decay_const
        decay_factor = self.delta / self.t_long
        decay_const = 1.0 - decay_factor

        BF = self.filter()

        E = np.power(BF, 2)

        aveE = np.zeros(shape=(n_bands, self.npts))
        if self.statistics_mode == 'rms':  # ALomax #
            rmsE = np.zeros(shape=(n_bands, self.npts))
        elif self.statistics_mode == 'std':  # ALomax #
            sigmaE = np.zeros(shape=(n_bands, self.npts))

        if self.statistics_mode == 'rms':
            E_2 = np.power(E, 2)
            aveE = lfilter([decay_factor], [1.0, -decay_const], E_2, axis=1)
            sqrt_aveE = np.sqrt(aveE)
            rmsE = lfilter([decay_factor], [1.0, -decay_const], sqrt_aveE, axis=1)
        elif self.statistics_mode == 'std':
            raise NotImplementedError(
                self.__class__.__name__ + "._statistics_decay(statistics_mode=='std')")

        if self.statistics_mode == 'rms':
            FC = np.abs(E)/(rmsE + 1.0e-6)
        elif self.statistics_mode == 'std':
            FC = np.abs(E-aveE)/(sigmaE)

        S = self.t_long
        L = int(round(S/self.delta, 0))
        for k in range(L):
            FC[:, k] = 0

        return FC, BF
