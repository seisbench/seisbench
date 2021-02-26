from seisbench.generate import Normalize, Filter, FixedWindow

import numpy as np
import copy
import scipy.signal
import pytest


def test_normalize():
    np.random.seed(42)
    base_state_dict = {"X": 10 * np.random.rand(3, 1000)}

    # No error on int
    norm = Normalize()
    state_dict = {"X": np.random.randint(0, 10, 1000)}
    norm(state_dict)
    assert state_dict["X"].dtype.char not in np.typecodes["AllInteger"]

    # Demean single axis
    norm = Normalize(demean_axis=-1)
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"], axis=-1) < 1e-10).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["X"], axis=-1), 1).all()

    # Demean multiple axis
    norm = Normalize(demean_axis=(0, 1))
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert not (
        np.mean(state_dict["X"], axis=-1) < 1e-10
    ).all()  # Axis are not individually
    assert np.mean(state_dict["X"]) < 1e-10  # Axis are normalized jointly

    # Detrend
    norm = Normalize(detrend_axis=-1)
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    # Detrending was applied
    assert (
        state_dict["X"] == scipy.signal.detrend(base_state_dict["X"], axis=-1)
    ).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["X"], axis=-1), 1).all()

    # Peak normalization
    norm = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="peak")
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"], axis=-1) < 1e-10).all()
    assert np.isclose(np.max(state_dict["X"], axis=-1), 1).all()

    # std normalization
    norm = Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type="std")
    state_dict = copy.deepcopy(base_state_dict)
    norm(state_dict)
    assert (np.mean(state_dict["X"], axis=-1) < 1e-10).all()
    assert np.isclose(np.std(state_dict["X"], axis=-1), 1).all()

    # Different key
    norm = Normalize(demean_axis=-1, key="Y")
    state_dict = {"Y": 10 * np.random.rand(3, 1000)}
    norm(state_dict)
    assert (np.mean(state_dict["Y"], axis=-1) < 1e-10).all()
    # No std normalization has been applied. Data generation ensures std >> 1 is fulfilled.
    assert not np.isclose(np.std(state_dict["Y"], axis=-1), 1).all()

    # Unknown normalization type
    with pytest.raises(ValueError):
        Normalize(amp_norm_type="Unknown normalization type")


def test_filter():
    np.random.seed(42)
    base_state_dict = {
        "X": 10 * np.random.rand(3, 1000),
        "metadata": {"trace_sampling_rate_hz": 20},
    }

    # lowpass - forward_backward=False
    filt = Filter(2, 1, "lowpass", forward_backward=False)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(2, 1, "lowpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfilt(sos, base_state_dict["X"])
    assert (state_dict["X"] == X_comp).all()

    # lowpass - forward_backward=True
    filt = Filter(2, 1, "lowpass", forward_backward=True)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(2, 1, "lowpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfiltfilt(sos, base_state_dict["X"])
    assert (state_dict["X"] == X_comp).all()

    # bandpass - multiple frequencies
    filt = Filter(1, (0.5, 2), "bandpass", forward_backward=True)
    state_dict = copy.deepcopy(base_state_dict)
    filt(state_dict)
    sos = scipy.signal.butter(1, (0.5, 2), "bandpass", output="sos", fs=20)
    X_comp = scipy.signal.sosfiltfilt(sos, base_state_dict["X"])
    assert (state_dict["X"] == X_comp).all()


def test_fixed_window():
    np.random.seed(42)
    base_state_dict = {
        "waveforms": 10 * np.random.rand(3, 1000),
        "metadata": {"trace_sampling_rate_hz": 20, "trace_p_arrival_sample": 500},
    }

    # Hard coded selection
    window = FixedWindow(0, 600)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert "X" in state_dict
    assert state_dict["X"].shape == (3, 600)
    assert (state_dict["X"] == state_dict["waveforms"][:, :600]).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == 500

    # p0 dynamic selection
    window = FixedWindow(windowlen=600)
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict, p0=100)
    assert "X" in state_dict
    assert state_dict["X"].shape == (3, 600)
    assert (state_dict["X"] == state_dict["waveforms"][:, 100:700]).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == 400

    # p0 and windowlen dynamic selection
    window = FixedWindow()
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict, 200, 500)
    assert "X" in state_dict
    assert state_dict["X"].shape == (3, 500)
    assert (state_dict["X"] == state_dict["waveforms"][:, 200:700]).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == 300

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
    assert state_dict["X"].shape == (3, 600)
    assert (state_dict["X"][:, :100] == state_dict["waveforms"][:, 900:]).all()
    assert (state_dict["X"][:, 100:] == 0).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == -400

    # Strategy "pad" out of bounds p0
    window = FixedWindow(p0=1100, windowlen=600, strategy="pad")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"].shape == (3, 600)
    assert (state_dict["X"] == 0).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == -500

    # Strategy "move"
    window = FixedWindow(p0=700, windowlen=600, strategy="move")
    state_dict = copy.deepcopy(base_state_dict)
    window(state_dict)
    assert state_dict["X"].shape == (3, 600)
    assert (state_dict["X"] == state_dict["waveforms"][:, -600:]).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == 100

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
    assert state_dict["X"].shape == (3, 300)
    assert (state_dict["X"] == state_dict["waveforms"][:, -300:]).all()
    assert state_dict["metadata"]["trace_p_arrival_sample"] == -200
