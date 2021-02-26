from seisbench.generate import Normalize

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
