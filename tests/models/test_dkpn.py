from unittest.mock import patch

import numpy as np
import obspy
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import seisbench.generate as sbg
import seisbench.models as sbm
import seisbench.util as sbu


def _waveforms(components="ZNE", n_samples=3401, sampling_rate=100.0):
    t = np.arange(n_samples) / sampling_rate
    values = {
        "Z": np.sin(2 * np.pi * 3 * t),
        "N": np.cos(2 * np.pi * 4 * t),
        "E": np.sin(2 * np.pi * 5 * t),
        "1": np.cos(2 * np.pi * 4 * t),
        "2": np.sin(2 * np.pi * 5 * t),
    }
    return np.stack([values[component] for component in components]).astype(np.float64)


def _stream(components="ZNE", n_samples=3201, sampling_rate=100.0):
    start = obspy.UTCDateTime(2020, 1, 1)
    waveforms = _waveforms(components, n_samples, sampling_rate)

    stream = obspy.Stream()
    for idx, component in enumerate(components):
        stream += obspy.Trace(
            waveforms[idx],
            {
                "network": "XX",
                "station": "DKPN",
                "location": "",
                "channel": f"BH{component}",
                "sampling_rate": sampling_rate,
                "starttime": start,
            },
        )

    return stream


class _WaveformDataset:
    def __init__(self, n_examples=2):
        self.n_examples = n_examples

    def __len__(self):
        return self.n_examples

    def get_sample(self, idx):
        metadata = {
            "trace_component_order": "ZNE",
            "trace_sampling_rate_hz": 100.0,
            "trace_p_arrival_sample": 900 + idx,
            "trace_s_arrival_sample": 1400 + idx,
        }
        return _waveforms(n_samples=3401), metadata


def test_dkpn_constructor_and_forward():
    model = sbm.DKPN()

    assert isinstance(model, sbm.PhaseNet)
    assert model.labels == "PSN"
    assert model.sampling_rate == 100
    assert model.in_samples == 3001
    assert model.component_order == "ZNEIM"

    x = torch.randn(2, 5, 3001)
    y = model(x)
    assert y.shape == (2, 3, 3001)
    np.testing.assert_allclose(
        y.sum(dim=1).detach().numpy(), np.ones((2, 3001)), rtol=1e-5
    )


def test_dkpn_from_pretrained_uses_seisbench_cache(tmp_path):
    model_orig = sbm.DKPN()
    model_orig.save(tmp_path / "test", version_str="1")

    checkpoint = torch.load(tmp_path / "test.pt.v1", map_location="cpu")
    assert list(checkpoint.keys()) == list(model_orig.state_dict().keys())

    with patch("seisbench.models.DKPN._model_path") as model_path:
        model_path.return_value = tmp_path
        assert sbm.DKPN.list_pretrained(remote=False) == ["test"]
        assert sbm.DKPN.list_versions("test", remote=False) == ["1"]
        model = sbm.DKPN.from_pretrained(
            "test", version_str="1", update=False, wait_for_file=True
        )

    assert isinstance(model, sbm.DKPN)
    assert isinstance(model, sbm.PhaseNet)
    assert model.get_model_args() == model_orig.get_model_args()
    assert model.labels == "PSN"
    assert model.component_order == "ZNEIM"
    assert model.in_channels == 5
    assert model.classes == 3
    assert model.sampling_rate == 100

    for key in [
        "P_threshold",
        "S_threshold",
        "blinding",
        "t_long",
        "freqmin",
        "corner",
        "mode",
    ]:
        assert key in model.default_args

    for name, parameter in model_orig.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], parameter)

    model.eval()
    x = torch.randn(1, 5, 3001)
    y = model(x)
    assert y.shape == (1, 3, 3001)


def test_dkpn_preprocessor_creates_feature_stream():
    model = sbm.DKPN()

    features = model.annotate_stream_pre(_stream(), model.default_args.copy())

    assert len(features) == 5
    assert [trace.stats.channel for trace in features] == [
        "CFZ",
        "CFN",
        "CFE",
        "CFI",
        "CFM",
    ]
    assert [trace.id[-1] for trace in features] == list("ZNEIM")
    assert all(trace.data.dtype == np.float32 for trace in features)
    assert np.isfinite(np.stack([trace.data for trace in features])).all()


def test_dkpn_batch_pre_keeps_incidence_unchanged():
    model = sbm.DKPN()
    batch = torch.randn(2, 5, 3001)
    incidence = batch[:, 3, :].clone()

    normalized = model.annotate_batch_pre(batch, {})

    torch.testing.assert_close(normalized[:, 3, :], incidence)


def test_dkpn_batch_pre_rejects_raw_three_component_batches():
    model = sbm.DKPN()
    batch = torch.randn(2, 3, 3001)

    with pytest.raises(ValueError, match="DKPNPreprocessor"):
        model.annotate_batch_pre(batch, {})


def test_dkpn_generator_preprocessor_creates_feature_tensor_and_shifts_metadata():
    state_dict = {
        "X": (
            _waveforms(n_samples=3401),
            {
                "trace_component_order": "ZNE",
                "trace_sampling_rate_hz": 100.0,
                "trace_p_arrival_sample": 900,
                "trace_s_arrival_sample": 1400,
            },
        )
    }

    sbg.DKPNPreprocessor(output_samples=3001)(state_dict)

    features, metadata = state_dict["X"]
    assert features.shape == (5, 3001)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert metadata["trace_component_order"] == "ZNEIM"
    assert metadata["trace_p_arrival_sample"] == 500
    assert metadata["trace_s_arrival_sample"] == 1000


def test_dkpn_generator_preprocessor_accepts_flexible_horizontal_components():
    state_dict = {
        "X": (
            _waveforms("Z12"),
            {
                "trace_component_order": "Z12",
                "trace_sampling_rate_hz": 100.0,
            },
        )
    }

    sbg.DKPNPreprocessor()(state_dict)

    features, metadata = state_dict["X"]
    assert features.shape == (5, 3401)
    assert metadata["trace_component_order"] == "ZNEIM"


def test_dkpn_generator_preprocessor_rejects_incomplete_components():
    state_dict = {
        "X": (
            _waveforms("ZN"),
            {
                "trace_component_order": "ZN",
                "trace_sampling_rate_hz": 100.0,
            },
        )
    }

    with pytest.raises(ValueError, match="complete ZNE"):
        sbg.DKPNPreprocessor()(state_dict)


def test_dkpn_stream_and_generator_preprocessors_match():
    model = sbm.DKPN()
    stream = _stream(n_samples=3401)
    feature_stream = model.annotate_stream_pre(stream.copy(), model.default_args.copy())

    state_dict = {
        "X": (
            _waveforms(n_samples=3401),
            {
                "trace_component_order": "ZNE",
                "trace_sampling_rate_hz": 100.0,
            },
        )
    }
    sbg.DKPNPreprocessor()(state_dict)
    features, _ = state_dict["X"]

    stream_features = np.stack([trace.data for trace in feature_stream])
    np.testing.assert_allclose(features, stream_features, rtol=1e-5, atol=1e-6)


def test_dkpn_generator_training_batch_and_backward_step():
    model = sbm.DKPN()
    generator = sbg.GenericGenerator(_WaveformDataset())
    generator.add_augmentations(
        [
            sbg.DKPNPreprocessor(output_samples=model.in_samples),
            sbg.ProbabilisticLabeller(
                label_columns={
                    "trace_p_arrival_sample": "P",
                    "trace_s_arrival_sample": "S",
                },
                model_labels=model.labels,
                sigma=10,
                dim=0,
            ),
            sbg.ChangeDtype(np.float32, key="X"),
            sbg.ChangeDtype(np.float32, key="y"),
        ]
    )
    loader = DataLoader(generator, batch_size=2)
    batch = next(iter(loader))

    assert batch["X"].shape == (2, 5, 3001)
    assert batch["y"].shape == (2, 3, 3001)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    x = model.annotate_batch_pre(batch["X"].clone(), model.default_args.copy())
    probabilities = model(x).clamp_min(1e-7)
    loss = F.nll_loss(probabilities.log(), batch["y"].argmax(dim=1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)


def test_dkpn_annotate_and_classify():
    model = sbm.DKPN()
    stream = _stream()

    annotations = model.annotate(stream)
    assert len(annotations) == 3
    assert {trace.stats.channel for trace in annotations} == {
        "DKPN_P",
        "DKPN_S",
        "DKPN_N",
    }

    output = model.classify(stream)
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name


def test_dkpn_save_load(tmp_path):
    model_orig = sbm.DKPN()
    stream = _stream()

    model_orig.save(tmp_path / "dkpn")
    model_load = sbm.DKPN.load(tmp_path / "dkpn")

    assert model_load.get_model_args() == model_orig.get_model_args()

    pred_orig = model_orig.annotate(stream)
    pred_load = model_load.annotate(stream)

    assert len(pred_orig) == len(pred_load)
    for trace_orig, trace_load in zip(pred_orig, pred_load):
        assert trace_orig.id == trace_load.id
        np.testing.assert_allclose(trace_orig.data, trace_load.data)


def test_dkpn_accepts_flexible_horizontal_components():
    model = sbm.DKPN()

    features = model.annotate_stream_pre(
        _stream("Z12"), {**model.default_args, "flexible_horizontal_components": True}
    )

    assert [trace.stats.channel for trace in features] == [
        "CFZ",
        "CFN",
        "CFE",
        "CFI",
        "CFM",
    ]


def test_dkpn_skips_incomplete_component_groups(caplog):
    model = sbm.DKPN()

    with caplog.at_level("WARNING"):
        features = model.annotate_stream_pre(_stream("ZN"), model.default_args.copy())

    assert len(features) == 0
    assert "complete 3-component" in caplog.text


@pytest.mark.parametrize("pre, post", [(100, 0), (0, 100), (100, 100)])
def test_dkpn_annotate_batch_post_blinding(pre, post):
    model = sbm.DKPN()
    pred = torch.ones((2, 3, 1000))

    blinded = model.annotate_batch_post(
        pred, None, argdict={"blinding": (pre, post)}
    ).numpy()

    if pre:
        assert np.isnan(blinded[:, :pre]).all()
    if post:
        assert np.isnan(blinded[:, -post:]).all()
    assert np.isfinite(blinded[:, pre : 1000 - post]).all()
