import pytest
import numpy as np
import xdas

import seisbench.models as sbm


@pytest.mark.parametrize("model", [sbm.PhaseNet, sbm.EQTransformer])
@pytest.mark.parametrize("blinding", [(0, 0), (100, 100)])
def test_das_wrapper_annotate(model, blinding):
    model_3c = model(sampling_rate=100)
    model = sbm.DASWaveformModelWrapper(model_3c)

    n_samples = 7000
    n_channels = 50

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:01:00", "us"),
            ],
        }
    )

    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    callback = sbm.InMemoryCollectionCallback()

    model.annotate(da, callback, blinding=blinding)

    for phase in "PS":
        ann = callback.get_results_dict()[phase]
        assert ann.shape == (n_samples - sum(blinding), n_channels)


def test_filter_conversion():
    model = sbm.PhaseNet(filter_args=None)
    assert sbm.DASWaveformModelWrapper._get_filter_args(model, (0.005, 0.02)) is None

    model = sbm.PhaseNet(
        filter_args=["bandpass"],
        filter_kwargs={
            "freqmin": 1.0,
            "freqmax": 2.0,
            "corners": 2,
            "zerophase": False,
        },
    )
    filter_type, filter_kwargs = sbm.DASWaveformModelWrapper._get_filter_args(
        model, (0.005, 0.02)
    )
    assert filter_type == "iirfilter"
    assert filter_kwargs == {
        "N": 2,
        "Wn": [1.0, 2.0],
        "btype": "bandpass",
        "ftype": "butter",
    }

    # Double corners for zerophase filters
    model = sbm.PhaseNet(
        filter_args=["bandpass"],
        filter_kwargs={
            "freqmin": 1.0,
            "freqmax": 2.0,
            "corners": 2,
            "zerophase": True,
        },
    )
    filter_type, filter_kwargs = sbm.DASWaveformModelWrapper._get_filter_args(
        model, (0.005, 0.02)
    )
    assert filter_type == "iirfilter"
    assert filter_kwargs == {
        "N": 4,
        "Wn": [1.0, 2.0],
        "btype": "bandpass",
        "ftype": "butter",
    }

    # Adjust corner for consistency with Nyquist
    model = sbm.PhaseNet(
        filter_args=["bandpass"],
        filter_kwargs={
            "freqmin": 1.0,
            "freqmax": 45.0,
            "corners": 2,
            "zerophase": True,
        },
    )
    filter_type, filter_kwargs = sbm.DASWaveformModelWrapper._get_filter_args(
        model, (0.005, 0.02)
    )
    assert filter_type == "iirfilter"
    assert filter_kwargs == {
        "N": 4,
        "Wn": [1.0, 0.999999 * 25],
        "btype": "bandpass",
        "ftype": "butter",
    }
