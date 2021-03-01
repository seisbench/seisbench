from seisbench.generate import (
    Normalize,
    Filter,
    FixedWindow,
    SlidingWindow,
    FilterKeys,
    WindowAroundSample,
    RandomWindow,
)

import numpy as np
import copy
import scipy.signal
import pytest
from unittest.mock import patch


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
    assert (
        state_dict["X"][0] == scipy.signal.detrend(base_state_dict["X"][0], axis=-1)
    ).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["X"][0], axis=-1), 1).all()

    # Peak normalization
    norm = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak")
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"][0], axis=-1) < 1e-10).all()
    assert np.isclose(np.max(state_dict["X"][0], axis=-1), 1).all()

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

    # p0 negative
    window = FixedWindow(p0=-1, windowlen=600)
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

    # Strategy "pad" out of bounds p0
    window = FixedWindow(p0=1100, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == 0).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == -500

    # Strategy "move"
    window = FixedWindow(p0=700, windowlen=600, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"][0].shape == (3, 600)
    assert (state_dict["X"][0] == base_state_dict["X"][0][:, -600:]).all()
    assert state_dict["X"][1]["trace_p_arrival_sample"] == 100

    # Strategy "move" - total size too short
    window = FixedWindow(p0=0, windowlen=1200, strategy="move")
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

    # Two key, both valid, strategy first
    window = WindowAroundSample(
        ["trace_p_arrival_sample", "trace_s_arrival_sample"],
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
    with pytest.raises(ValueError):
        window(state_dict)


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
