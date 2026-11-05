import numpy as np
import obspy
import pytest
import torch

import seisbench.models as sbm
import seisbench.util as sbu


@pytest.mark.parametrize(
    "logits",
    [True, False],
)
@pytest.mark.parametrize(
    "model",
    [
        sbm.EquiPol(sampling_rate=100, in_channels=1, component_order="Z"),
        sbm.RPNet(sampling_rate=100),
    ],
)
def test_equivariance_property(logits, model):
    x = torch.rand(5, 1, 400)

    model.eval()
    assert model(x, logits=logits).shape == (5, 2)
    assert torch.allclose(model(x, logits=logits)[:, 0], model(-x, logits=logits)[:, 1])
    assert torch.allclose(model(x, logits=logits)[:, 1], model(-x, logits=logits)[:, 0])


@pytest.mark.parametrize(
    "picklist",
    [True, False],
)
@pytest.mark.parametrize(
    "model",
    [sbm.EquiPol(sampling_rate=100), sbm.RPNet(sampling_rate=100)],
)
def test_annotate_classify(picklist: bool, model):
    stream = obspy.read()
    picks = [
        ("BW.RJOB.", stream[0].stats.starttime + 5),  # Valid
        ("BW.RJOB.", stream[0].stats.starttime + 1),  # On the boundary
        ("BW.RJOB.", stream[0].stats.starttime + 100),  # Out of window
        ("BW.RJOB.AD", stream[0].stats.starttime + 5),  # Not existing
    ]

    if picklist:
        picks = sbu.PickList(
            [
                sbu.Pick(trace_id=trace_id, peak_time=time, phase="P")
                for trace_id, time in picks
            ]
        )

    polarities = model.annotate(stream, picks=picks)
    assert isinstance(polarities[0], np.ndarray)
    for i in range(1, 4):
        assert polarities[i] is None

    polarities = model.classify(stream, picks=picks, threshold=0.5).picks
    assert polarities[0].polarity is not None
    assert polarities[0].polarity_value is not None
    for i in range(1, 4):
        assert polarities[i].polarity is None
        assert polarities[i].polarity_value is None
