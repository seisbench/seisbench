import obspy
import pytest

import seisbench.models as sbm
import seisbench.util as sbu


@pytest.mark.parametrize("norm", ["std", "peak"])
def test_annotate_phasenet(norm):
    # Tests that the annotate/classify functions run without crashes and annotate produces an output
    model = sbm.Skynet(
        sampling_rate=2000,
        norm=norm,
    )  # Higher sampling rate ensures trace is long enough
    stream = obspy.read()

    annotations = model.annotate(stream)
    assert len(annotations) > 0
    output = model.classify(
        stream
    )  # Ensures classify succeeds even though labels are unknown
    assert isinstance(output, sbu.ClassifyOutput)
    assert isinstance(output.picks, sbu.PickList)
    assert output.creator == model.name
