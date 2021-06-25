import seisbench.data as sbd

import pytest
from transcode_dataset import get_component_order_mapping


def test_get_component_order_mapping():
    dataset = sbd.DummyDataset(missing_components="pad")

    # Wrong missing components
    with pytest.raises(AssertionError):
        get_component_order_mapping(dataset)

    dataset.missing_components = "ignore"

    dataset._metadata.loc[0, "trace_component_order"] = "Z"
    dataset._metadata.loc[1, "trace_component_order"] = "ZN"
    dataset._metadata.loc[2, "trace_component_order"] = "EN"
    dataset._metadata.loc[3, "trace_component_order"] = "NEZ"
    dataset._metadata.loc[4, "trace_component_order"] = "NZ"

    mapping = get_component_order_mapping(dataset)
    assert len(mapping) == 6
    assert mapping["Z"] == "Z"
    assert mapping["ZN"] == "ZN"
    assert mapping["ZNE"] == "ZNE"
    assert mapping["EN"] == "NE"
    assert mapping["NEZ"] == "ZNE"
    assert mapping["NZ"] == "ZN"

    # Data loss, as "H" is not in output
    dataset._metadata.loc[1, "trace_component_order"] = "ZH"
    with pytest.raises(ValueError):
        get_component_order_mapping(dataset)
