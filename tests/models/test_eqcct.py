import numpy as np
import obspy
import pytest
import torch
from obspy import UTCDateTime

import seisbench.models as sbm
import seisbench.util as sbu


def _synthetic_zne_stream(
    *, npts: int = 12_000, sampling_rate: float = 100.0
) -> obspy.Stream:
    """Long enough for EQCCTP/EQCCTS (6000-sample windows at 100 Hz)."""
    t0 = UTCDateTime(0)
    base = dict(
        network="SB",
        station="TEST",
        location="00",
        sampling_rate=sampling_rate,
        starttime=t0,
    )
    rng = np.random.default_rng(0)
    z = rng.standard_normal(npts).astype(np.float64)
    n = rng.standard_normal(npts).astype(np.float64)
    e = rng.standard_normal(npts).astype(np.float64)
    return obspy.Stream(
        [
            obspy.Trace(data=z, header={**base, "channel": "HHZ"}),
            obspy.Trace(data=n, header={**base, "channel": "HHN"}),
            obspy.Trace(data=e, header={**base, "channel": "HHE"}),
        ]
    )


@pytest.mark.parametrize("norm", ["std", "peak"])
def test_eqcctp_annotate_classify(norm):
    """EQCCTP annotate/classify run end-to-end (no pretrained weights)."""
    model = sbm.EQCCTP(norm=norm)
    stream = _synthetic_zne_stream()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(stream)
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


@pytest.mark.parametrize("norm", ["std", "peak"])
def test_eqccts_annotate_classify(norm):
    """EQCCTS annotate/classify run end-to-end (no pretrained weights)."""
    model = sbm.EQCCTS(norm=norm)
    stream = _synthetic_zne_stream()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(stream)
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_eqcctp_forward_shape():
    model = sbm.EQCCTP()
    x = torch.randn(2, 3, model.in_samples)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, len(model.labels), model.in_samples)


def test_eqccts_forward_shape():
    model = sbm.EQCCTS()
    x = torch.randn(2, 3, model.in_samples)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (2, len(model.labels), model.in_samples)


def test_eqcctp_get_model_args():
    model = sbm.EQCCTP()
    args = model.get_model_args()
    assert "sampling_rate" in args
    assert "norm" in args
    assert args["norm"] == "peak"


def test_export_names_in_models_package():
    assert hasattr(sbm, "EQCCTP")
    assert hasattr(sbm, "EQCCTS")
    assert sbm.EQCCTP.__name__ == "EQCCTP"
    assert sbm.EQCCTS.__name__ == "EQCCTS"


def test_eqcctp_weights_cache_dirname_is_eqcct():
    assert sbm.EQCCTP._model_path().name == "eqcct"


def test_eqccts_weights_cache_dirname_is_eqccts():
    assert sbm.EQCCTS._model_path().name == "eqccts"
