import seisbench.generate
import seisbench.generate.labeling
from seisbench.generate import (
    Normalize,
    Filter,
    FixedWindow,
    SlidingWindow,
    FilterKeys,
    WindowAroundSample,
    RandomWindow,
    SteeredWindow,
    ChangeDtype,
    ProbabilisticLabeller,
    ProbabilisticPointLabeller,
    StandardLabeller,
    DetectionLabeller,
    StepLabeller,
    Copy,
)
from seisbench.data import DummyDataset

import numpy as np
import pandas as pd
import copy
import scipy.signal
import logging
import pytest
from unittest.mock import patch, MagicMock


def test_normalize():
    np.random.seed(42)
    base_state_dict = {"X": (10 * np.random.rand(3, 1000), None)}

    # No error on int
    norm = Normalize()
    state_dict = {"X": (np.random.randint(0, 10, 1000), None)}
    norm(state_dict)

    # Demean single axis
    norm = Normalize(demean_axis=-1)
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"][0], axis=-1) < 1e-10).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["X"][0], axis=-1), 1).all()

    # Demean multiple axis
    norm = Normalize(demean_axis=(0, 1))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert not (
        np.mean(state_dict["X"][0], axis=-1) < 1e-10
    ).all()  # Axis are not individually
    assert np.mean(state_dict["X"][0]) < 1e-10  # Axis are normalized jointly

    # Detrend
    norm = Normalize(detrend_axis=-1)
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    # Detrending was applied
    assert np.allclose(
        state_dict["X"][0], scipy.signal.detrend(base_state_dict["X"][0], axis=-1)
    )
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["X"][0], axis=-1), 1).all()

    # Peak normalization
    norm = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak")
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"][0], axis=-1) < 1e-10).all()
    assert np.isclose(np.max(np.abs(state_dict["X"][0]), axis=-1), 1).all()

    # std normalization
    norm = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std")
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"][0], axis=-1) < 1e-10).all()
    assert np.isclose(np.std(state_dict["X"][0], axis=-1), 1).all()

    # Different key
    norm = Normalize(demean_axis=-1, key="Y")
    state_dict = {"Y": (10 * np.random.rand(3, 1000), None)}
    norm(state_dict)
    assert (np.mean(state_dict["Y"][0], axis=-1) < 1e-10).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["Y"][0], axis=-1), 1).all()

    # No inplace modification - demean
    norm = Normalize(demean_axis=-1, key=("X", "Y"))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0]).all()

    # No inplace modification - detrend
    norm = Normalize(detrend_axis=-1, key=("X", "Y"))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0]).all()

    # No inplace modification - peak normalization
    norm = Normalize(amp_norm_axis=-1, amp_norm_type="peak", key=("X", "Y"))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0]).all()

    # No inplace modification - std normalization
    norm = Normalize(amp_norm_axis=-1, amp_norm_type="std", key=("X", "Y"))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0]).all()

    # No NaN when normalizing zeros
    norm = Normalize(amp_norm_axis=-1, key=("X", "Y"))
    state_dict = {"X": (np.zeros((3, 1000)), None)}
    norm(state_dict)
    assert not (np.isnan(state_dict["X"][0])).any()

    # Unknown normalization type
    with pytest.raises(ValueError):
        Normalize(amp_norm_type="Unknown normalization type")


def test_filter():
    np.random.seed(42)
    base_state_dict = {
        "X": (10 * np.random.rand(3, 1000), {"trace_sampling_rate_hz": 20})
    }

    # lowpass - forward_backward=False
    filt = Filter(2, 1, "lowpass", forward_backward=False)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(2, 1, "lowpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfilt(sos, base_state_dict["X"][0])
    assert (state_dict["X"][0] == X_comp).all()

    # lowpass - forward_backward=True
    filt = Filter(2, 1, "lowpass", forward_backward=True)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(2, 1, "lowpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfiltfilt(sos, base_state_dict["X"][0])
    assert (state_dict["X"][0] == X_comp).all()

    # bandpass - multiple frequencies
    filt = Filter(1, (0.5, 2), "bandpass", forward_backward=True)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(1, (0.5, 2), "bandpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfiltfilt(sos, base_state_dict["X"][0])
    assert (state_dict["X"][0] == X_comp).all()


def test_filter_sampling_rate_list():
    np.random.seed(42)

    filt = Filter(2, 1, "lowpass")
    state_dict = {
        "X": (10 * np.random.rand(3, 1000), {"trace_sampling_rate_hz": [20, 20]})
    }
    filt(state_dict)

    state_dict = {
        "X": (10 * np.random.rand(3, 1000), {"trace_sampling_rate_hz": [20, 25]})
    }
    with pytest.raises(NotImplementedError):
        filt(state_dict)


def test_fixed_window():
    np.random.seed(42)
    base_state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {"trace_sampling_rate_hz": 20, "trace_p_arrival_sample": 500},
        ),
    }

    # Hard coded selection
    window = FixedWindow(0, 600)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert "X" in state_dict
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, :600]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 500

    # p0 dynamic selection
    window = FixedWindow(windowlen=600)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict, p0=100)
    assert "X" in state_dict
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 100:700]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 400

    # p0 and windowlen dynamic selection
    window = FixedWindow()
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict, 200, 500)
    assert "X" in state_dict
    assert state_dict["X"][0].shape == (3, 500)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 200:700]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 300

    # Insufficient selection - p0
    window = FixedWindow(windowlen=500)
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "Start position" in str(e)

    # Insufficient selection - windowlen
    window = FixedWindow(p0=0)
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "Window length" in str(e)

    # p0 negative, strategy fail
    window = FixedWindow(p0=-1, windowlen=600, strategy="fail")
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "Negative indexing is not supported" in str(e)

    # Invalid strategy
    with pytest.raises(ValueError) as e:
        FixedWindow(strategy="invalid")
    assert "Unknown strategy" in str(e)

    # Strategy "fail"
    window = FixedWindow(p0=900, windowlen=600, strategy="fail")
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "longer than available" in str(e)

    # Strategy "pad"
    window = FixedWindow(p0=900, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0][:, :100] == base_state_dict["X"][0][:, 900:]).all()
    assert (state_dict["X"][0][:, 100:] == 0).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == -400

    # Strategy "pad" p0 > trace_length
    window = FixedWindow(p0=1100, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == 0).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == -500

    # Strategy "pad" p0 < 0, p0 + windowlen > 0
    window = FixedWindow(p0=-100, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0][:, :100] == 0).all()
    assert (state_dict["X"][0][:, 100:] == base_state_dict["X"][0][:, :500]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 600

    # Strategy "pad" p0 < 0, p0 + windowlen < 0
    window = FixedWindow(p0=-700, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == 0).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 1200

    # Strategy "pad" p0 < 0, p0 + windowlen > trace_length
    window = FixedWindow(p0=-100, windowlen=1200, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 1200)
    assert (state_dict["X"][0][:, :100] == 0).all()
    assert (state_dict["X"][0][:, 100:-100] == base_state_dict["X"][0]).all()
    assert (state_dict["X"][0][:, -100:] == 0).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 600

    # Strategy "move" - right aligned
    window = FixedWindow(p0=700, windowlen=600, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, -600:]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 100

    # Strategy "move" - left aligned
    window = FixedWindow(p0=-100, windowlen=600, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, :600]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 500

    # Strategy "move" - total size too short
    window = FixedWindow(p0=0, windowlen=1200, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "Total trace length" in str(e)

    # Strategy "move" - total size too short - p0 negative
    window = FixedWindow(p0=-300, windowlen=1200, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    with pytest.raises(ValueError) as e:
        window(state_dict)
    assert "Total trace length" in str(e)

    # Strategy "variable"
    window = FixedWindow(p0=700, windowlen=600, strategy="variable")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 300)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, -300:]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == -200

    # Strategy "variable" - p0 negative - non-empty output
    window = FixedWindow(p0=-100, windowlen=600, strategy="variable")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 500)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, :500]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 500

    # Strategy "variable" - p0 negative - empty output
    window = FixedWindow(p0=-700, windowlen=600, strategy="variable")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 0)
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 500

    # Strategy "variable" - p0 negative - p0 + windowlen > trace_length
    window = FixedWindow(p0=-100, windowlen=1200, strategy="variable")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 1000)
    assert (state_dict["X"][0] == base_state_dict["X"][0]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 500


def test_filter_keys():
    # No keys specified
    with pytest.raises(ValueError) as e:
        FilterKeys()
    assert "Exactly one of include or exclude must be specified" in str(e)

    # Include and exclude keys specified
    with pytest.raises(ValueError) as e:
        FilterKeys(include=["a"], exclude=["b"])
    assert "Exactly one of include or exclude must be specified" in str(e)

    # Include keys
    filter = FilterKeys(include=["a", "b", "c"])
    state_dict = {"a": 1, "b": 2, "c": 3, "d": 4}
    filter(state_dict)
    assert set(state_dict.keys()) == {"a", "b", "c"}

    # Exclude keys
    filter = FilterKeys(exclude=["a", "c"])
    state_dict = {"a": 1, "b": 2, "c": 3, "d": 4}
    filter(state_dict)
    assert set(state_dict.keys()) == {"b", "d"}


def test_sliding_window():
    np.random.seed(42)
    base_state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {"trace_sampling_rate_hz": 20, "trace_p_arrival_sample": 500},
        )
    }

    # Zero windows
    window = SlidingWindow(timestep=100, windowlen=1100)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (0, 3, 1100)

    # Fitting at the end
    window = SlidingWindow(timestep=100, windowlen=200)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (9, 3, 200)
    for i in range(9):
        assert (
            state_dict["X"][0][i] == base_state_dict["X"][0][:, i * 100 : 200 + i * 100]
        ).all()
    assert (
        state_dict["X"][1]["trace_p_arrival_sample"] == np.arange(500, -400, -100)
    ).all()

    # Not fitting at the end
    window = SlidingWindow(timestep=101, windowlen=200)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (8, 3, 200)
    for i in range(8):
        assert (
            state_dict["X"][0][i] == base_state_dict["X"][0][:, i * 101 : 200 + i * 101]
        ).all()
    assert (
        state_dict["X"][1]["trace_p_arrival_sample"] == np.arange(500, -300, -101)
    ).all()


def test_window_around_sample():
    np.random.seed(42)
    base_state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {"trace_p_arrival_sample": 300, "trace_s_arrival_sample": 700},
        )
    }

    # Unknown selection strategy
    with pytest.raises(ValueError):
        WindowAroundSample("trace_p_arrival_sample", selection="unknown")

    # Single key, rewritten to list
    window = WindowAroundSample(
        "trace_p_arrival_sample", samples_before=100, windowlen=200
    )
    assert window.metadata_keys == ["trace_p_arrival_sample"]
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 200:400]).all()

    # Three key, two valid, one not in metadata, strategy first
    window = WindowAroundSample(
        ["trace_p_arrival_sample", "trace_s_arrival_sample", "trace_g_arrival_sample"],
        samples_before=100,
        windowlen=200,
        selection="first",
    )
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 200:400]).all()

    # Two keys, both valid, strategy random
    window = WindowAroundSample(
        ["trace_p_arrival_sample", "trace_s_arrival_sample"],
        samples_before=100,
        windowlen=200,
        selection="random",
    )
    with patch("numpy.random.choice") as choice:
        state_dict = copy.deepcopy(base_state_dict)
        choice.return_value = 300
        window(state_dict)
        assert (state_dict["X"][0] == base_state_dict["X"][0][:, 200:400]).all()
        choice.assert_called_once_with([300, 700])

        choice.reset_mock()
        state_dict = copy.deepcopy(base_state_dict)
        choice.return_value = 700
        window(state_dict)
        assert (state_dict["X"][0] == base_state_dict["X"][0][:, 600:800]).all()
        choice.assert_called_once_with([300, 700])

    # Two keys, one valid
    window = WindowAroundSample(
        ["trace_p_arrival_sample", "trace_s_arrival_sample"],
        samples_before=100,
        windowlen=200,
    )
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["X"][1]["trace_p_arrival_sample"] = np.nan
    window(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 600:800]).all()

    # Two keys, none valid
    window = WindowAroundSample(
        ["trace_p_arrival_sample", "trace_s_arrival_sample"],
        samples_before=100,
        windowlen=200,
        selection="first",
    )
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["X"][1]["trace_p_arrival_sample"] = np.nan
    state_dict["X"][1]["trace_s_arrival_sample"] = np.nan
    window(state_dict)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, :200]).all()


def test_random_window():
    np.random.seed(42)
    base_state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {"trace_p_arrival_sample": 300, "trace_s_arrival_sample": 700},
        )
    }

    # Fails on low >= high
    with pytest.raises(ValueError):
        RandomWindow(500, 100)
    with pytest.raises(ValueError):
        RandomWindow(100, 100)

    # Works for high - low == windowlen
    RandomWindow(100, 500, windowlen=400)
    # Fails on high - low < windowlen
    with pytest.raises(ValueError):
        RandomWindow(100, 500, windowlen=401)
    # Fails on high < windowlen
    with pytest.raises(ValueError):
        RandomWindow(None, 500, windowlen=501)

    # Works if trace aligns with right side
    window = RandomWindow(600, None, windowlen=400)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)

    # Fails if trace too short
    with pytest.raises(ValueError):
        window = RandomWindow(600, None, windowlen=401)
        state_dict = copy.deepcopy(base_state_dict)
        window(state_dict)

    # Works as expected
    with patch("numpy.random.randint") as randint:
        window = RandomWindow(100, None, windowlen=400)
        randint.return_value = 205
        state_dict = copy.deepcopy(base_state_dict)
        window(state_dict)
        randint.assert_called_with(100, 601)
        assert (state_dict["X"][0] == base_state_dict["X"][0][:, 205:605]).all()
        assert state_dict["X"][1]["trace_p_arrival_sample"] == 95
        assert state_dict["X"][1]["trace_s_arrival_sample"] == 495

    # Strategy pad, low > high - windowlen
    with patch("numpy.random.randint") as randint:
        window = RandomWindow(100, 300, windowlen=400, strategy="pad")
        randint.return_value = -40
        state_dict = copy.deepcopy(base_state_dict)
        window(state_dict)
        randint.assert_called_with(-100, 101)
        assert (state_dict["X"][0][:, :140] == 0).all()
        assert (
            state_dict["X"][0][:, 140:340] == base_state_dict["X"][0][:, 100:300]
        ).all()
        assert (state_dict["X"][0][:, 340:] == 0).all()
        assert state_dict["X"][1]["trace_p_arrival_sample"] == 340
        assert state_dict["X"][1]["trace_s_arrival_sample"] == 740

    # Strategy variable, low > high - windowlen
    window = RandomWindow(100, 300, windowlen=400, strategy="variable")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 200)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, 100:300]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 200
    assert state_dict["X"][1]["trace_s_arrival_sample"] == 600


def test_change_dtype():
    change = ChangeDtype(dtype=np.float64)
    state_dict = {"X": (np.ones(10, dtype=np.float32), {"A": 1})}
    change(state_dict)
    assert state_dict["X"][0].dtype == np.float64

    change = ChangeDtype(dtype=np.float64, key=("X", "X2"))
    state_dict = {"X": (np.ones(10, dtype=np.float32), {"A": 1})}
    change(state_dict)
    assert state_dict["X"][0].dtype == np.float32
    assert state_dict["X2"][0].dtype == np.float64
    assert (
        state_dict["X"][1] is not state_dict["X2"][1]
    )  # Metadata was copied and not only a pointer was copies


def test_probabilistic_pick_labeller():
    for shape in ["gaussian", "triangle", "box"]:
        np.random.seed(42)
        state_dict = {
            "X": (
                10 * np.random.rand(3, 1000),
                {
                    "trace_p_arrival_sample": 500,
                    "trace_s_arrival_sample": 700,
                    "trace_g_arrival_sample": np.nan,
                },
            )
        }

        # Assumes standard config['dimension_order'] = 'NCW'
        # Test label construction for single window, handling NaN values
        labeller = ProbabilisticLabeller(dim=0, shape=shape)
        labeller(state_dict)

        assert state_dict["y"][0].shape == (4, 1000)
        if shape == "box":
            assert np.array_equiv(state_dict["y"][0][1][490:510], np.ones(20))
            assert np.array_equiv(state_dict["y"][0][2][690:710], np.ones(20))
        else:
            assert np.argmax(state_dict["y"][0], axis=1)[1] == 499
            assert np.argmax(state_dict["y"][0], axis=1)[2] == 699
        assert (
            state_dict["y"][0][0] == 0
        ).all()  # Check that NaN picks are interpreted as not present

        # Fails when multi_class specified and channel dim sum > 1
        with pytest.raises(ValueError):
            labeller = ProbabilisticLabeller(dim=1, shape=shape)
            labeller(state_dict)

        # Test label construction for multiple windows
        state_dict = {
            "X": (
                10 * np.random.rand(5, 3, 1000),
                {
                    "trace_p_arrival_sample": np.array([500] * 5),
                    "trace_s_arrival_sample": np.array([700] * 5),
                    "trace_g_arrival_sample": np.array([500, 500, 200, np.nan, 500]),
                },
            )
        }
        labeller = ProbabilisticLabeller(dim=1, shape=shape)
        labeller(state_dict)

        assert state_dict["y"][0].shape == (5, 4, 1000)
        if shape == "box":
            assert np.array_equiv(state_dict["y"][0][3, 1, 490:510], np.ones(20))
            assert np.array_equiv(state_dict["y"][0][3, 2, 690:710], np.ones(20))
            assert np.array_equiv(
                state_dict["y"][0][2, 0, 190:210], np.ones(20)
            )  # Entry with pick
        else:
            assert np.argmax(state_dict["y"][0][3, :, :], axis=-1)[1] == 499
            assert np.argmax(state_dict["y"][0][3, :, :], axis=-1)[2] == 699
            assert (
                np.argmax(state_dict["y"][0][2, :, :], axis=-1)[0] == 199
            )  # Entry with pick
        assert (state_dict["y"][0][3, 0, :] == 0).all()

        # Fails if single sample provided for multiple windows
        state_dict = {
            "X": (
                10 * np.random.rand(5, 3, 1000),
                {
                    "trace_p_arrival_sample": 500,
                    "trace_s_arrival_sample": 700,
                },
            )
        }
        with pytest.raises(ValueError):
            labeller = ProbabilisticLabeller(dim=1, shape=shape)
            labeller(state_dict)

        state_dict["X"] = np.random.rand(10, 5, 3, 1000)

        # Fails if non-compatible input data dimensions are provided
        with pytest.raises(ValueError):
            labeller = ProbabilisticLabeller(dim=1, shape=shape)
            labeller(state_dict)


def test_step_labeller():
    np.random.seed(42)
    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 500,
                "trace_s_arrival_sample": -100,
                "trace_g_arrival_sample": np.nan,
            },
        )
    }

    # Assumes standard config['dimension_order'] = 'NCW'
    # Test label construction for single window, handling NaN values
    labeller = StepLabeller()
    labeller(state_dict)

    assert state_dict["y"][0].shape == (3, 1000)
    assert (state_dict["y"][0][1, :500] == 0).all()
    assert (state_dict["y"][0][1, 500:] == 1).all()
    assert (state_dict["y"][0][2] == 1).all()
    assert (state_dict["y"][0][0] == 0).all()

    # Test label construction for multiple windows
    state_dict = {
        "X": (
            10 * np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": np.array([500] * 5),
                "trace_s_arrival_sample": np.array([700] * 5),
                "trace_g_arrival_sample": np.array([500, 500, 200, np.nan, 500]),
            },
        )
    }
    labeller = StepLabeller(dim=1)
    labeller(state_dict)

    assert state_dict["y"][0].shape == (5, 3, 1000)
    assert (state_dict["y"][0][:, 1, :500] == 0).all()
    assert (state_dict["y"][0][:, 1, 500:] == 1).all()
    assert (state_dict["y"][0][:, 2, :700] == 0).all()
    assert (state_dict["y"][0][:, 2, 700:] == 1).all()

    assert (state_dict["y"][0][0, 0, :500] == 0).all()
    assert (state_dict["y"][0][0, 0, 500:] == 1).all()
    assert (state_dict["y"][0][3, 0, :] == 0).all()

    # Fails if single sample provided for multiple windows
    state_dict = {
        "X": (
            10 * np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": 500,
                "trace_s_arrival_sample": 700,
            },
        )
    }
    with pytest.raises(ValueError):
        labeller = StepLabeller()
        labeller(state_dict)

    state_dict["X"] = np.random.rand(10, 5, 3, 1000)

    # Fails if non-compatible input data dimensions are provided
    with pytest.raises(ValueError):
        labeller = StepLabeller()
        labeller(state_dict)


def test_probabilistic_pick_labeller_overlap():
    # Test that even in case of an overlap, the total probability mass is exactly 1.
    np.random.seed(42)
    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 500,
                "trace_s_arrival_sample": 500,
                "trace_g_arrival_sample": np.nan,
            },
        )
    }

    labeller = ProbabilisticLabeller(dim=0, sigma=50)
    labeller(state_dict)

    assert np.allclose(np.sum(state_dict["y"][0], axis=0), 1)  # Sum is always 1
    assert np.isclose(
        np.min(state_dict["y"][0]), 0
    )  # Minimum is close to 0, in particular noise is never negative


def test_probabilistic_pick_labeller_pickgroups():
    np.random.seed(42)
    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {
                "trace_Pn_arrival_sample": 300,
                "trace_Pg_arrival_sample": 900,
                "trace_P1_arrival_sample": np.nan,  # Missing value is simply ignored
                "trace_S_arrival_sample": 700,
            },
        )
    }

    label_columns = {
        "trace_Pn_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",  # Sample not present in the metadata is ignored
        "trace_S_arrival_sample": "S",
    }

    labeller = ProbabilisticLabeller(dim=0, label_columns=label_columns)
    labeller(state_dict)

    assert state_dict["y"][0].shape == (3, 1000)
    assert np.isclose(state_dict["y"][0][0, 299], 1)  # P picks
    assert np.isclose(state_dict["y"][0][0, 899], 1)  # P picks
    assert np.argmax(state_dict["y"][0], axis=1)[1] == 699  # S pick


def test_standard_pick_labeller():
    np.random.seed(42)

    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {
                "trace_Pn_arrival_sample": 300,
                "trace_Pg_arrival_sample": 900,
                "trace_P1_arrival_sample": np.nan,  # Missing value is simply ignored
                "trace_S_arrival_sample": 500,
            },
        )
    }

    label_columns = {
        "trace_Pn_arrival_sample": "P",
        "trace_Pg_arrival_sample": "P",
        "trace_P1_arrival_sample": "P",
        "trace_PmP_arrival_sample": "P",  # Sample not present in the metadata is ignored
        "trace_S_arrival_sample": "S",
    }

    def check_labels_against_ids(labeller, ids, labels):
        assert len(labels) == len(ids.reshape(-1))
        for label, i in zip(labels, ids.reshape(-1)):
            assert label == labeller.labels[i]

    # Check 'label-first' strategy on overlap
    labeller = StandardLabeller(on_overlap="label-first", label_columns=label_columns)
    labeller(state_dict)
    assert state_dict["y"][0].shape == (1,)
    check_labels_against_ids(labeller, state_dict["y"][0], ["P"])
    assert labeller.labels == ["P", "S", "Noise"]

    # Check 'fixed-relevance' strategy on overlap
    labeller = StandardLabeller(
        on_overlap="fixed-relevance", label_columns=label_columns
    )
    labeller(state_dict)
    check_labels_against_ids(labeller, state_dict["y"][0], ["S"])


def test_standard_labeller_low_high():
    np.random.seed(42)

    # Test label construction for multiple windows
    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {"trace_p_arrival_sample": [320, -100, 1200, 500, 600]},
        )
    }

    labeller = StandardLabeller()  # Low is None, high is None

    labeller(state_dict)
    assert (state_dict["y"][0][:, 0] == [0, 1, 1, 0, 0]).all()

    labeller.low = 400
    labeller(state_dict)
    assert (state_dict["y"][0][:, 0] == [1, 1, 1, 0, 0]).all()

    labeller.low = -600
    labeller(state_dict)
    assert (state_dict["y"][0][:, 0] == [1, 1, 1, 0, 0]).all()

    labeller.high = 600
    labeller(state_dict)
    assert (state_dict["y"][0][:, 0] == [1, 1, 1, 0, 1]).all()

    labeller.high = -400
    labeller(state_dict)
    assert (state_dict["y"][0][:, 0] == [1, 1, 1, 0, 1]).all()


def test_standard_pick_labeller_pickgroups():
    np.random.seed(42)

    # Test label construction for multiple windows
    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": [320, -100, 490, 220, 440],
                "trace_s_arrival_sample": [540, 880, 810, 380, 740],
                "trace_g_arrival_sample": [np.nan] * 5,
            },
        )
    }

    def check_labels_against_ids(labeller, ids, labels):
        assert len(labels) == len(ids.reshape(-1))
        for label, i in zip(labels, ids.reshape(-1)):
            assert label == labeller.labels[i]

    # Check 'label-first' strategy on overlap
    labeller = StandardLabeller(on_overlap="label-first")
    labeller(state_dict)
    assert state_dict["y"][0].shape == (5, 1)
    check_labels_against_ids(labeller, state_dict["y"][0], ["p", "s", "p", "p", "p"])
    assert labeller.labels == ["g", "p", "s", "Noise"]

    # Check 'fixed-relevance' strategy on overlap
    labeller = StandardLabeller(on_overlap="fixed-relevance")
    labeller(state_dict)
    assert state_dict["y"][0].shape == (5, 1)
    check_labels_against_ids(labeller, state_dict["y"][0], ["s", "s", "p", "s", "p"])
    assert labeller.labels == ["g", "p", "s", "Noise"]

    # Check 'random' strategy on overlap
    np.random.seed(42)
    labeller = StandardLabeller(on_overlap="random")
    labeller(state_dict)
    assert state_dict["y"][0].shape == (5, 1)
    check_labels_against_ids(labeller, state_dict["y"][0], ["p", "s", "s", "p", "p"])
    assert labeller.labels == ["g", "p", "s", "Noise"]

    # Fails if single sample provided for multiple windows
    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": 500,
                "trace_s_arrival_sample": 700,
            },
        )
    }
    with pytest.raises(IndexError):
        labeller = StandardLabeller(dim=1)
        labeller(state_dict)

    # Fails if non-compatible input data dimensions are provided
    state_dict["X"] = np.random.rand(10, 5, 3, 1000)
    with pytest.raises(ValueError):
        labeller = StandardLabeller(dim=1)
        labeller(state_dict)


def test_colums_to_dict_and_labels():
    label_columns = ["trace_p_arrival_sample", "trace_s_arrival_sample"]
    (
        label_columns,
        labels,
        label_ids,
    ) = seisbench.generate.labeling.PickLabeller._colums_to_dict_and_labels(
        label_columns
    )

    assert label_columns == {
        "trace_p_arrival_sample": "p",
        "trace_s_arrival_sample": "s",
    }
    assert labels == ["p", "s", "Noise"]
    assert label_ids == {"p": 0, "s": 1, "Noise": 2}

    label_columns = {
        "trace_p_arrival_sample": "p",
        "trace_Pg_arrival_sample": "p",
        "trace_s_arrival_sample": "s",
    }
    (
        label_columns,
        labels,
        label_ids,
    ) = seisbench.generate.labeling.PickLabeller._colums_to_dict_and_labels(
        label_columns
    )

    assert label_columns == {
        "trace_p_arrival_sample": "p",
        "trace_Pg_arrival_sample": "p",
        "trace_s_arrival_sample": "s",
    }
    assert labels == ["p", "s", "Noise"]
    assert label_ids == {"p": 0, "s": 1, "Noise": 2}


def test_swap_dimension_order():

    arr = np.zeros((6, 3, 100))
    assert ProbabilisticLabeller._swap_dimension_order(
        arr, current_dim="NCW", expected_dim="CNW"
    ).shape == (3, 6, 100)
    assert ProbabilisticLabeller._swap_dimension_order(
        arr, current_dim="NCW", expected_dim="WCN"
    ).shape == (100, 3, 6)
    assert ProbabilisticLabeller._swap_dimension_order(
        arr, current_dim="NCW", expected_dim="CWN"
    ).shape == (3, 100, 6)

    arr = np.zeros((6, 3, 100, 5))
    assert ProbabilisticLabeller._swap_dimension_order(
        arr, current_dim="abcd", expected_dim="cabd"
    ).shape == (100, 6, 3, 5)

    arr = np.zeros((100, 3))
    assert ProbabilisticLabeller._swap_dimension_order(
        arr, current_dim="CW", expected_dim="WC"
    ).shape == (3, 100)


def test_autoidentify_pick_labels():
    state_dict = {
        "trace_Pn_arrival_sample": None,
        "trace_Pg_arrival_sample": None,
        "trace_Sg_arrival_sample": None,
        "abc": None,
    }

    assert seisbench.generate.labeling.PickLabeller._auto_identify_picklabels(
        state_dict
    ) == [
        "trace_Pg_arrival_sample",
        "trace_Pn_arrival_sample",
        "trace_Sg_arrival_sample",
    ]


def test_add_augmentations():
    # Tests mixing of augmentation decorator and add_augmentations
    generator = seisbench.generate.GenericGenerator(None)

    @generator.augmentation
    def f():
        return

    generator.add_augmentations([1, 2])

    @generator.augmentation
    def g():
        return

    assert generator._augmentations == [f, 1, 2, g]


def test_oneof():
    def aug1(state_dict):
        state_dict["a"] = 1

    def aug2(state_dict):
        state_dict["b"] = 2

    # Automatic uniform distribution
    oneof = seisbench.generate.OneOf([aug1, aug2])
    assert np.allclose(oneof.probabilities, 1 / 2)

    # ValueError for incompatible lists
    with pytest.raises(ValueError):
        seisbench.generate.OneOf([aug1, aug2], [4, 1, 1])

    # Automatic norm of probabilities to 1
    oneof = seisbench.generate.OneOf([aug1, aug2], [4, 1])
    assert np.allclose(oneof.probabilities, [0.8, 0.2])

    # Correct probabilities are passed to numpy.random.choice
    with patch("numpy.random.choice") as choice:
        choice.return_value = aug1
        state_dict = {}
        oneof(state_dict)
        assert state_dict == {"a": 1}
        assert choice.call_args[0] == ([aug1, aug2],)
        assert np.allclose(choice.call_args[1]["p"], [0.8, 0.2])


def test_detection_labeller_parameters():
    labeller = seisbench.generate.labeling.DetectionLabeller("P", "S")
    assert labeller.p_phases == ["P"]
    assert labeller.s_phases == ["S"]

    labeller = seisbench.generate.labeling.DetectionLabeller(["P1", "P2"], ["S"])
    assert labeller.p_phases == ["P1", "P2"]
    assert labeller.s_phases == ["S"]


def test_detection_labeller_3d():
    np.random.seed(42)

    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": [100, 150, np.nan, 100, np.nan],
                "trace_p2_arrival_sample": [np.nan, 100, np.nan, 120, np.nan],
                "trace_s_arrival_sample": [200, 300, 810, np.nan, np.nan],
            },
        )
    }

    target0 = np.zeros(1000)
    target0[100:340] = 1
    target1 = np.zeros(1000)
    target1[100:580] = 1

    p_phases = [f"trace_{x}_arrival_sample" for x in ["p", "p2", "p3"]]
    s_phases = "trace_s_arrival_sample"
    labeller = seisbench.generate.labeling.DetectionLabeller(
        p_phases, s_phases, factor=1.4
    )
    labeller(state_dict)
    y = state_dict["y"][0][:, 0, :]
    assert y.shape == (5, 1000)
    assert (y[[2, 3, 4]] == 0).all()
    assert np.allclose(y[0], target0)
    assert np.allclose(y[1], target1)


def test_detection_labeller_3d_fixed():
    np.random.seed(42)

    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": np.array([100, 150, np.nan, 900, np.nan]),
            },
        )
    }

    target0 = np.zeros(1000)
    target0[100:300] = 1
    target1 = np.zeros(1000)
    target1[150:350] = 1
    target3 = np.zeros(1000)
    target3[900:] = 1

    p_phases = [f"trace_{x}_arrival_sample" for x in ["p", "p2", "p3"]]
    labeller = seisbench.generate.labeling.DetectionLabeller(p_phases, fixed_window=200)
    labeller(state_dict)
    y = state_dict["y"][0][:, 0, :]
    assert y.shape == (5, 1000)
    assert (y[[2, 4]] == 0).all()
    print(np.sum(y[0]))
    print(np.sum(target0))
    assert np.allclose(y[0], target0)
    assert np.allclose(y[1], target1)
    assert np.allclose(y[3], target3)


def test_detection_labeller_2d():
    np.random.seed(42)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 100,
                "trace_p2_arrival_sample": np.nan,
                "trace_s_arrival_sample": 200,
            },
        )
    }

    target = np.zeros(1000)
    target[100:340] = 1

    p_phases = [f"trace_{x}_arrival_sample" for x in ["p", "p2", "p3"]]
    s_phases = "trace_s_arrival_sample"
    labeller = seisbench.generate.labeling.DetectionLabeller(
        p_phases, s_phases, factor=1.4
    )
    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, target)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": np.nan,
                "trace_p2_arrival_sample": np.nan,
                "trace_s_arrival_sample": 200,
            },
        )
    }

    target = np.zeros(1000)
    target[100:340] = 1
    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, 0)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 100,
                "trace_p2_arrival_sample": np.nan,
                "trace_s_arrival_sample": np.nan,
            },
        )
    }

    target = np.zeros(1000)
    target[100:340] = 1
    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, 0)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 150,
                "trace_p2_arrival_sample": 100,
                "trace_s_arrival_sample": 200,
            },
        )
    }

    target = np.zeros(1000)
    target[100:340] = 1

    p_phases = [f"trace_{x}_arrival_sample" for x in ["p", "p2", "p3"]]
    s_phases = "trace_s_arrival_sample"
    labeller = seisbench.generate.labeling.DetectionLabeller(
        p_phases, s_phases, factor=1.4
    )
    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, target)


def test_detection_labeller_2d_fixed():
    np.random.seed(42)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 100,
            },
        )
    }

    target = np.zeros(1000)
    target[100:300] = 1

    p_phases = [f"trace_{x}_arrival_sample" for x in ["p", "p2", "p3"]]
    labeller = DetectionLabeller(p_phases, fixed_window=200)
    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, target)

    state_dict = {
        "X": (
            np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": np.nan,
            },
        )
    }

    labeller(state_dict)
    y = state_dict["y"][0][0, :]
    assert y.shape == (1000,)
    assert np.allclose(y, 0)


def test_channel_dropout():
    np.random.seed(42)

    dropout = seisbench.generate.ChannelDropout(axis=1)  # Positively defined axis
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    dropout(state_dict)
    assert state_dict["X"][0].shape == (5, 3, 1000)

    dropout = seisbench.generate.ChannelDropout(axis=-2)  # Negatively defined axis
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    dropout(state_dict)
    assert state_dict["X"][0].shape == (5, 3, 1000)

    with patch("numpy.random.randint") as randint:
        randint.return_value = 2
        state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
        dropout(state_dict)
        x = state_dict["X"][0]
        assert x.shape == (5, 3, 1000)
        assert (
            np.sum((x == 0).all(axis=(0, 2))) == 2
        )  # Exactly two axis have been dropped

    # No inplace modification occurs for different input and output keys
    dropout = seisbench.generate.ChannelDropout(axis=1, key=("X", "y"))  # Check key
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    new_state_dict = copy.deepcopy(state_dict)
    dropout(new_state_dict)

    assert (state_dict["X"][0] == new_state_dict["X"][0]).all()
    assert state_dict["X"][1] == new_state_dict["X"][1]
    assert "y" in new_state_dict


def test_add_gap():
    np.random.seed(42)

    gap = seisbench.generate.AddGap(axis=2)  # Positively defined axis
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    gap(state_dict)
    assert state_dict["X"][0].shape == (5, 3, 1000)

    gap = seisbench.generate.AddGap(axis=-1)  # Negatively defined axis
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    gap(state_dict)
    assert state_dict["X"][0].shape == (5, 3, 1000)

    with patch("numpy.random.randint") as randint:
        randint.side_effect = [100, 200]
        state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
        gap(state_dict)
        assert state_dict["X"][0].shape == (5, 3, 1000)
        assert (state_dict["X"][0][:, :, 100:200] == 0).all()  # Correct part is blinded
        assert not (
            state_dict["X"][0][:, :, :100] == 0
        ).all()  # Before are non-zero entries
        assert not (
            state_dict["X"][0][:, :, 200:] == 0
        ).all()  # After are non-zero entries

    # No inplace modification occurs for different input and output keys
    gap = seisbench.generate.ChannelDropout(axis=1, key=("X", "y"))  # Check key
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    new_state_dict = copy.deepcopy(state_dict)
    gap(new_state_dict)

    assert (state_dict["X"][0] == new_state_dict["X"][0]).all()
    assert state_dict["X"][1] == new_state_dict["X"][1]
    assert "y" in new_state_dict


def test_random_array_rotation_keys():
    rot = seisbench.generate.RandomArrayRotation(keys="X")
    assert rot.keys == [("X", "X")]

    rot = seisbench.generate.RandomArrayRotation(keys=("X", "y"))
    assert rot.keys == [("X", "y")]

    rot = seisbench.generate.RandomArrayRotation(keys=["X", ("X1", "X2")])
    assert rot.keys == [("X", "X"), ("X1", "X2")]


def test_random_array_rotation():
    np.random.seed(42)

    roll = seisbench.generate.RandomArrayRotation(
        keys=["X", "y"], axis=-1
    )  # Broadcast axis
    state_dict = {
        "X": (np.random.rand(5, 3, 1000), {}),
        "y": (np.random.rand(3, 1000), {}),
    }
    roll(state_dict)

    # Incompatible axis
    state_dict = {
        "X": (np.random.rand(5, 3, 1000), {}),
        "y": (np.random.rand(3, 2000), {}),
    }
    with pytest.raises(ValueError):
        roll(state_dict)

    # Check identical rolling
    roll = seisbench.generate.RandomArrayRotation(
        keys=[("X", "X2"), ("y", "y2")], axis=[-1, 1]
    )
    state_dict = {
        "X": (np.random.rand(5, 3, 1000), {}),
        "y": (np.random.rand(3, 1000, 1), {}),
    }
    with patch("numpy.random.randint") as randint:
        randint.return_value = 105
        roll(state_dict)
        randint.assert_called_once()

    assert (np.roll(state_dict["X"][0], 105, axis=-1) == state_dict["X2"][0]).all()
    assert (np.roll(state_dict["y"][0], 105, axis=1) == state_dict["y2"][0]).all()


def test_gaussian_noise():
    np.random.seed(42)

    noise = seisbench.generate.GaussianNoise()
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    noise(state_dict)
    assert state_dict["X"][0].shape == (5, 3, 1000)

    noise = seisbench.generate.GaussianNoise(key=("X", "y"))
    state_dict = {"X": (np.random.rand(5, 3, 1000), {})}
    with patch("numpy.random.uniform") as uniform:
        uniform.return_value = 0.15
        noise(state_dict)
        uniform.assert_called_once_with(*noise.scale)

    x = state_dict["X"][0]
    y = state_dict["y"][0]
    assert (
        0.14 < np.std(y - x) < 0.16
    )  # Bounds are rather liberal to ensure a stable test


def test_probabilistic_point_labeller():
    np.random.seed(42)
    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {
                "trace_p_arrival_sample": 500,
                "trace_s_arrival_sample": 700,
            },
        )
    }

    # Assumes standard config['dimension_order'] = 'NCW'
    # Test label construction for single window, handling NaN values
    labeller = ProbabilisticPointLabeller(sigma=100)
    labeller(state_dict)

    p = 1  # P probability
    s = np.exp(-2)  # S probability
    y = state_dict["y"][0]
    assert y.shape == (3,)
    assert np.isclose(np.sum(y), 1)
    assert np.isclose(y[0], p / (p + s), atol=1e-2, rtol=1e-2)
    assert np.isclose(y[1], s / (p + s), atol=1e-2, rtol=1e-2)
    assert np.isclose(y[2], 0)

    # Label position between p and s
    labeller = ProbabilisticPointLabeller(sigma=100, position=0.6)
    labeller(state_dict)

    y = state_dict["y"][0]
    assert y.shape == (3,)
    assert np.isclose(np.sum(y), 1)
    assert np.isclose(y[0], 0.5, atol=1e-2, rtol=1e-2)
    assert np.isclose(y[1], 0.5, atol=1e-2, rtol=1e-2)
    assert np.isclose(y[2], 0)

    # Test label construction for multiple windows
    state_dict = {
        "X": (
            np.random.rand(5, 3, 1000),
            {
                "trace_p_arrival_sample": [320, -100, 490, 220, 440],
                "trace_s_arrival_sample": [540, 880, 810, 380, 740],
            },
        )
    }
    labeller = ProbabilisticPointLabeller(sigma=100, dim=1)
    labeller(state_dict)

    y = state_dict["y"][0]
    assert y.shape == (5, 3)
    assert np.allclose(np.sum(y, axis=-1), 1)


def test_detection_labeller_warning_fixed_and_s_phase(caplog):
    with caplog.at_level(logging.WARNING):
        DetectionLabeller(p_phases=["P"], s_phases=["S"])

    assert caplog.text == ""

    with caplog.at_level(logging.WARNING):
        DetectionLabeller(p_phases=["P"], fixed_window=100)

    assert caplog.text == ""

    with caplog.at_level(logging.WARNING):
        DetectionLabeller(p_phases=["P"], s_phases=["S"], fixed_window=100)

    assert (
        "Provided both S phases and fixed window length to DetectionLabeller."
        in caplog.text
    )


def test_steered_generator():
    dummy = DummyDataset()

    def raise_on_check(state_dict):
        if "X" in state_dict and "_control_" in state_dict:
            if state_dict["_control_"]["x"] == 100:
                raise ValueError()

    augmentation = MagicMock(side_effect=raise_on_check)

    metadata = [
        {"trace_name": dummy["trace_name"].values[i], "x": 2 * i}
        for i in range(50, 100)
    ]
    metadata = pd.DataFrame(metadata)

    generator = seisbench.generate.SteeredGenerator(dummy, metadata)
    generator.augmentation(augmentation)

    assert len(generator) == 50

    with pytest.raises(ValueError):
        generator[0]


@pytest.mark.parametrize("case", [0, 1, 2, 3])
def test_steered_generator_keys(case):
    if case == 0:
        metadata = pd.DataFrame([{"trace_name": "a"}])
        target = {"trace_name": "a", "chunk": None, "dataset": None}
    elif case == 1:
        metadata = pd.DataFrame([{"trace_name": "a", "trace_chunk": "1"}])
        target = {"trace_name": "a", "chunk": "1", "dataset": None}
    elif case == 2:
        metadata = pd.DataFrame([{"trace_name": "a", "trace_dataset": "d"}])
        target = {"trace_name": "a", "chunk": None, "dataset": "d"}
    else:
        metadata = pd.DataFrame(
            [{"trace_name": "a", "trace_chunk": "1", "trace_dataset": "d"}]
        )
        target = {"trace_name": "a", "chunk": "1", "dataset": "d"}

    data = DummyDataset()

    generator = seisbench.generate.SteeredGenerator(data, metadata)

    with patch(
        "seisbench.data.base.WaveformDataset.get_idx_from_trace_name"
    ) as get_idx:
        get_idx.return_value = 0
        generator[0]
        get_idx.assert_called_once_with(**target)


def test_steered_window():
    np.random.seed(42)
    base_state_dict = {"X": (np.random.rand(3, 1000), {})}

    # Variable window size
    window = SteeredWindow(windowlen=None, strategy="pad")

    # Sample shorter than trace
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 100, "end_sample": 900}
    window(state_dict)

    assert np.all(state_dict["X"][0] == base_state_dict["X"][0][:, 100:900])
    assert np.all(state_dict["window_borders"][0] == [0, 800])

    # Sample longer than trace
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": -100, "end_sample": 1100}
    window(state_dict)

    assert state_dict["X"][0].shape == (3, 1200)
    assert np.all(state_dict["X"][0][:, :1000] == base_state_dict["X"][0])
    assert np.all(state_dict["window_borders"][0] == [-100, 1100])

    # Fixed window size
    window = SteeredWindow(windowlen=500)

    # Range longer than window length
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 100, "end_sample": 900}
    # Requested window is too long
    with pytest.raises(ValueError):
        window(state_dict)

    # Range with sufficient samples around
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 200, "end_sample": 500}

    window(state_dict)

    assert state_dict["X"][0].shape == (3, 500)
    assert np.all(state_dict["X"][0] == base_state_dict["X"][0][:, 100:600])
    assert np.all(state_dict["window_borders"][0] == [100, 400])

    # Range with insufficient samples at the end
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 700, "end_sample": 900}

    window(state_dict)

    assert state_dict["X"][0].shape == (3, 500)
    assert np.all(state_dict["X"][0] == base_state_dict["X"][0][:, 500:])
    assert np.all(state_dict["window_borders"][0] == [200, 400])

    # Range with insufficient samples at the start
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 50, "end_sample": 200}

    window(state_dict)

    assert state_dict["X"][0].shape == (3, 500)
    assert np.all(state_dict["X"][0] == base_state_dict["X"][0][:, :500])
    assert np.all(state_dict["window_borders"][0] == [50, 200])

    # Range longer than trace samples
    window = SteeredWindow(windowlen=1200, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    state_dict["_control_"] = {"start_sample": 50, "end_sample": 200}

    window(state_dict)

    assert state_dict["X"][0].shape == (3, 1200)
    assert np.all(state_dict["X"][0][:, :1000] == base_state_dict["X"][0])
    assert np.all(state_dict["window_borders"][0] == [50, 200])


def test_standard_labeller_no_labels_in_metadata():
    # Checks that the labeller works correct if no label columns are present in the metadata
    np.random.seed(42)

    state_dict = {
        "X": (
            10 * np.random.rand(3, 1000),
            {},
        )
    }

    label_columns = {
        "trace_Pn_arrival_sample": "P",
        "trace_S_arrival_sample": "S",
    }

    # Check 'label-first' strategy on overlap
    labeller = StandardLabeller(label_columns=label_columns)
    labeller(state_dict)

    assert state_dict["y"][0] == 2  # Labeled as noise


def test_copy():

    state_dict = {"X": (5 * np.random.rand(3, 1000), None)}

    # Default params copy
    copy_default = Copy(key=("X", "Xc"))
    copy_default(state_dict)
    assert "Xc" in state_dict.keys()

    # Write to new key copy
    copy_new_key = Copy(key=("X", "abc"))
    copy_new_key(state_dict)
    assert "abc" in state_dict.keys()

    assert set(["Xc", "X", "abc"]) == set(state_dict.keys())

    # Wrong key selection
    copy_wrong_key = Copy(key=("z", "Xc"))
    with pytest.raises(KeyError):
        copy_wrong_key(state_dict)

    # Check deepcopy ok
    state_dict["X"][0][0, 500] = 2
    assert state_dict["Xc"][0][0, 500] != 2
