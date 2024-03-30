"""
This file checks that the following functions are equivalent:
- annotate_window_pre and annotate_batch_pre
- annotate_window_post and annotate_batch_post

As this is essentially a one-of static check, it can be deleted soon.
When deleting this, all annotate_window_pre and annotate_window_post functions should be deleted too.
Jannes Münchmeyer - 2024/03
"""

import numpy as np
import pytest
import torch

import seisbench.models as sbm


@pytest.mark.parametrize("norm_detrend", [True, False])
@pytest.mark.parametrize("norm_amp_per_comp", [True, False])
@pytest.mark.parametrize("norm", ["std", "peak"])
def test_eqtransformer_pre(norm_detrend, norm_amp_per_comp, norm):
    model = sbm.EQTransformer(
        norm_detrend=norm_detrend, norm_amp_per_comp=norm_amp_per_comp, norm=norm
    )
    windows = np.random.random((5, 3, 6000))

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_eqtransformer_post():
    model = sbm.EQTransformer()
    pred = np.random.rand(5, 3, 6000)

    pred_t = torch.tensor(pred)
    batched = model.annotate_batch_post(
        [pred_t[:, 0], pred_t[:, 1], pred_t[:, 2]], {}, {}
    ).numpy()
    single = np.stack(
        [
            model.annotate_window_post([window[0], window[1], window[2]], {}, {})
            for window in pred
        ]
    )

    assert np.allclose(batched, single, equal_nan=True)


@pytest.mark.parametrize("norm_detrend", [True, False])
@pytest.mark.parametrize("norm_amp_per_comp", [True, False])
@pytest.mark.parametrize("norm", ["std", "peak"])
@pytest.mark.parametrize("model", [sbm.PhaseNet, sbm.PhaseNetLight])
def test_phasenet_pre(norm_detrend, norm_amp_per_comp, norm, model):
    model = model(
        norm_detrend=norm_detrend, norm_amp_per_comp=norm_amp_per_comp, norm=norm
    )
    windows = np.random.random((5, 3, 3001))

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("model", [sbm.PhaseNet, sbm.PhaseNetLight])
def test_phasenet_post(model):
    model = model()
    pred = np.random.rand(5, 3, 6000)

    pred_t = torch.tensor(pred)
    batched = model.annotate_batch_post(pred_t, {}, {}).numpy()
    single = np.stack([model.annotate_window_post(window, {}, {}) for window in pred])

    assert np.allclose(batched, single, equal_nan=True)


def test_gpd_pre():
    model = sbm.GPD()
    windows = np.random.random((5, 3, 400))

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_gpd_post():
    model = sbm.GPD()
    pred = np.random.rand(5, 3)

    pred_t = torch.tensor(pred)
    batched = model.annotate_batch_post(pred_t, {}, {}).numpy()
    single = np.stack([model.annotate_window_post(window, {}, {}) for window in pred])

    assert np.allclose(batched, single, equal_nan=True)


def test_cred_pre():
    model = sbm.CRED()
    windows = np.random.random((5, 3, 3000))

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_cred_post():
    model = sbm.CRED()
    windows = np.random.random((5, 3, 3000))

    batched = model.annotate_batch_post(torch.tensor(windows), {}, {}).numpy()
    single = np.stack(
        [model.annotate_window_post(window, {}, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_basicphaseae_pre():
    model = sbm.BasicPhaseAE()
    windows = np.random.random((5, 3, 600))

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_basicphaseae_post():
    model = sbm.BasicPhaseAE()
    pred = np.random.rand(5, 3, 600)

    pred_t = torch.tensor(pred)
    batched = model.annotate_batch_post(pred_t, {}, {}).numpy()
    single = np.stack([model.annotate_window_post(window, {}, {}) for window in pred])

    assert np.allclose(batched, single, equal_nan=True)


def test_deepdenoiser_pre():
    model = sbm.DeepDenoiser()
    windows = np.random.random((5, 3000))

    batched, batched_aux = model.annotate_batch_pre(torch.tensor(windows), {})
    batched = batched.numpy()
    batched_aux = batched_aux.numpy()
    single_full = [model.annotate_window_pre(window, {}) for window in windows]
    single = np.stack([x[0] for x in single_full])
    single_aux = np.stack([x[1] for x in single_full])

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)
    assert np.allclose(batched_aux, single_aux, rtol=1e-3, atol=1e-3)


def test_deepdenoiser_post():
    model = sbm.DeepDenoiser()
    x = np.random.random((5, 2, 31, 201))
    aux = np.random.random((5, 2, 31, 201))

    batched = model.annotate_batch_post(torch.tensor(x), torch.tensor(aux), {})
    single = [model.annotate_window_post(wx, waux, {}) for wx, waux in zip(x, aux)]

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("max_stations", [4, 10])  # Same as shape or bigger than shape
@pytest.mark.parametrize("norm", ["std", "peak"])
def test_phaseteam_pre(max_stations, norm):
    model = sbm.PhaseTEAM(norm=norm, max_stations=max_stations)
    windows = np.random.rand(5, 4, 3, 3001)

    batched = model.annotate_batch_pre(torch.tensor(windows), {}).numpy()
    single = np.stack(
        [model.annotate_window_pre(window, {}) for window in windows], axis=0
    )

    assert np.allclose(batched, single, rtol=1e-3, atol=1e-3)


def test_phaseteam_post():
    model = sbm.PhaseTEAM()
    pred = np.random.rand(5, 3, 3001)

    pred_t = torch.tensor(pred)
    batched = model.annotate_batch_post(pred_t, {}, {}).numpy()
    single = np.stack([model.annotate_window_post(window, {}, {}) for window in pred])

    assert np.allclose(batched, single, equal_nan=True)
