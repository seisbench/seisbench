import numpy as np
import obspy
import pytest
import torch

import seisbench.models as sbm
import seisbench.util as sbu


@pytest.mark.parametrize("variant", ["s", "m", "l"])
def test_seist_forward_shape(variant):
    model = sbm.SeisT(variant=variant, in_samples=1200)
    x = torch.randn(2, 3, model.in_samples)
    model.eval()
    with torch.no_grad():
        y = model(x)
        y_logits = model(x, logits=True)

    assert y.shape == (2, len(model.labels), model.in_samples)
    assert model.labels == ["Detection", "P", "S"]
    assert torch.all((0 <= y) & (y <= 1))
    assert torch.allclose(torch.sigmoid(y_logits), y)


def test_seist_variable_length_input():
    # The architecture is fully convolutional/attentional and works for other input lengths than in_samples
    model = sbm.SeisT(variant="s", in_samples=1200)
    model.eval()
    with torch.no_grad():
        y = model(torch.randn(1, 3, 1000))
    assert y.shape == (1, 3, 1000)


def test_seist_presets_and_overrides():
    model = sbm.SeisT(variant="s", in_samples=600, layer_channels=[16, 16, 32, 32])
    assert model.arch["layer_channels"] == [16, 16, 32, 32]
    assert model.arch["layer_blocks"] == [2, 2, 3, 2]  # From preset

    with pytest.raises(ValueError):
        sbm.SeisT(variant="xl")
    with pytest.raises(ValueError):
        sbm.SeisT(norm="none")
    with pytest.raises(ValueError):
        sbm.SeisT(classes=3, phases="PS")


def test_seist_phases_none():
    model = sbm.SeisT(variant="s", in_samples=600, classes=3, phases=None)
    assert model.labels == ["Detection", "0", "1", "2"]
    assert model.phases == ["0", "1", "2"]


@pytest.mark.parametrize("norm", ["std", "peak"])
def test_seist_annotate_batch_pre(norm):
    model = sbm.SeisT(variant="s", in_samples=600, norm=norm)
    batch = torch.rand(4, 3, 600) * 100 + 50  # Non-zero mean, arbitrary scale
    out = model.annotate_batch_pre(batch, {})

    assert out.shape == batch.shape
    assert torch.allclose(out.mean(dim=-1), torch.zeros(4, 3), atol=1e-4)
    if norm == "std":
        assert torch.allclose(out.std(dim=-1), torch.ones(4, 3), atol=1e-4)
    else:
        assert torch.allclose(out.abs().amax(dim=-1), torch.ones(4, 3), atol=1e-4)


def test_seist_annotate_batch_post():
    model = sbm.SeisT(variant="s", in_samples=600)
    pred = torch.rand(5, 3, 600)

    out = model.annotate_batch_post(pred.clone(), None, {"blinding": (0, 0)})
    assert out.shape == (5, 600, 3)
    assert torch.allclose(out, pred.transpose(-1, -2))

    out = model.annotate_batch_post(pred.clone(), None, {"blinding": (100, 50)})
    assert torch.isnan(out[:, :100]).all()
    assert torch.isnan(out[:, -50:]).all()
    assert not torch.isnan(out[:, 100:-50]).any()


@pytest.mark.parametrize("norm", ["std", "peak"])
def test_seist_annotate_classify(norm):
    # Tests that the annotate/classify functions run without crashes and produce outputs
    model = sbm.SeisT(variant="s", in_samples=1000, norm=norm)
    stream = obspy.read()

    annotations = model.annotate(stream, blinding=(50, 50), overlap=500)
    assert len(annotations) == 3
    assert {tr.stats.channel for tr in annotations} == {
        "SeisT_Detection",
        "SeisT_P",
        "SeisT_S",
    }

    output = model.classify(stream, blinding=(50, 50), overlap=500)
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert isinstance(output.detections, sbu.DetectionList)
    assert output.creator == model.name


def test_seist_classify_aggregate():
    model = sbm.SeisT(variant="s", in_samples=600)

    def make_trace(channel, data):
        trace = obspy.Trace(np.asarray(data, dtype=float))
        trace.stats.network = "SB"
        trace.stats.station = "ABC"
        trace.stats.channel = channel
        trace.stats.sampling_rate = 100
        return trace

    det = np.zeros(600)
    det[100:400] = 1.0
    p = np.zeros(600)
    p[150:160] = np.linspace(0, 1, 10)
    s = np.zeros(600)
    s[300:310] = np.linspace(0, 0.2, 10)  # Below the pick threshold

    annotations = obspy.Stream(
        [
            make_trace("SeisT_Detection", det),
            make_trace("SeisT_P", p),
            make_trace("SeisT_S", s),
        ]
    )
    output = model.classify_aggregate(annotations, {})

    assert len(output.detections) == 1
    assert len(output.picks) == 1
    assert output.picks[0].phase == "P"
    assert output.picks[0].peak_value == pytest.approx(1.0)

    output = model.classify_aggregate(annotations, {"S_threshold": 0.1})
    assert len(output.picks) == 2


def test_seist_get_model_args_save_load(tmp_path):
    model = sbm.SeisT(
        variant="s",
        in_samples=800,
        sampling_rate=50,
        norm="peak",
        head_dims=[8, 8, 8, 8],
    )
    model.eval()
    args = model.get_model_args()
    for key in ["citation", "output_type", "default_args", "pred_sample", "labels"]:
        assert key not in args
    assert args["variant"] == "s"
    assert args["in_samples"] == 800
    assert args["sampling_rate"] == 50
    assert args["norm"] == "peak"
    assert args["head_dims"] == [8, 8, 8, 8]
    assert args["phases"] == "PS"

    model.save(tmp_path / "seist_test", weights_docstring="test weights")
    loaded = sbm.SeisT.load(tmp_path / "seist_test")
    loaded.eval()

    assert loaded.arch == model.arch
    assert loaded.in_samples == model.in_samples
    assert loaded.sampling_rate == model.sampling_rate
    assert loaded.labels == model.labels

    x = torch.randn(2, 3, 800)
    with torch.no_grad():
        assert torch.allclose(model(x), loaded(x))


def test_seist_backbone_state_dict_matches_original_layout():
    # Weights from the original repository can be loaded by prefixing the keys with "backbone."
    model = sbm.SeisT(variant="m", in_samples=600)
    keys = list(model.state_dict().keys())
    assert all(k.startswith("backbone.") for k in keys)
    assert any(k.startswith("backbone.stem.") for k in keys)
    assert any(k.startswith("backbone.encoder_layers.") for k in keys)
    assert any(k.startswith("backbone.out_head.") for k in keys)


def test_export_names_in_models_package():
    assert hasattr(sbm, "SeisT")
    assert sbm.SeisT.__name__ == "SeisT"
