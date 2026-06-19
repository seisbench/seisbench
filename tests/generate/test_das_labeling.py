import numpy as np
import pytest

import seisbench.generate as sbg


@pytest.fixture
def record():
    return np.zeros((100, 4), dtype=np.float32)


@pytest.fixture
def metadata():
    return {
        "__annotations__": {
            "P": np.array([50, 50, 50, 50], dtype=float),
        }
    }


# ---------------------------------------------------------------------------
# _normalize_label_specification
# ---------------------------------------------------------------------------


def test_normalize_list_annotation_mapping():
    mapping, labels = sbg.ProbabilisticDASLabeller._normalize_label_specification(
        ["P", "S"]
    )

    assert mapping == {"P": "P", "S": "S"}
    assert labels == ["P", "S", "noise"]


def test_normalize_dict_annotation_mapping():
    mapping, labels = sbg.ProbabilisticDASLabeller._normalize_label_specification(
        {"Pg": "P", "Sg": "S"}
    )

    assert mapping == {"Pg": "P", "Sg": "S"}
    assert labels == ["P", "S", "noise"]


def test_normalize_without_noise():
    mapping, labels = sbg.ProbabilisticDASLabeller._normalize_label_specification(
        ["P", "S"],
        noise_map=False,
    )

    assert labels == ["P", "S"]


def test_output_labels_get_noise_appended():
    mapping, labels = sbg.ProbabilisticDASLabeller._normalize_label_specification(
        ["P"],
        noise_map=True,
        output_labels=["P"],
    )

    assert labels == ["P", "noise"]


# ---------------------------------------------------------------------------
# __str__
# ---------------------------------------------------------------------------


def test_str():
    labeller = sbg.ProbabilisticDASLabeller(["P"], sigma=5, label_shape="triangle")

    assert str(labeller) == ("ProbabilisticDASLabeller (label_shape=triangle, sigma=5)")


# ---------------------------------------------------------------------------
# _build_single_label_map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["gaussian", "triangle", "box"])
def test_build_single_label_map_shape(shape):
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        label_shape=shape,
        sigma=10,
    )

    ann_values = np.array([50, 50, 50, 50], dtype=float)

    label = labeller._build_single_label_map(
        ann_values,
        (100, 4),
    )

    assert label.shape == (100, 4)
    assert np.all(label >= 0)
    assert np.all(label <= 1)


def test_build_single_label_map_peak_at_pick():
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        label_shape="gaussian",
        sigma=10,
    )

    ann_values = np.array([50, 50, 50, 50], dtype=float)

    label = labeller._build_single_label_map(
        ann_values,
        (100, 4),
    )

    np.testing.assert_allclose(label[50], 1.0)


def test_build_single_label_map_nan_columns_are_zero():
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        label_shape="gaussian",
        sigma=10,
    )

    ann_values = np.array([50, np.nan, 50, np.nan])

    label = labeller._build_single_label_map(
        ann_values,
        (100, 4),
    )

    assert np.all(label[:, 1] == 0)
    assert np.all(label[:, 3] == 0)


def test_unknown_label_shape():
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        label_shape="invalid",
    )

    with pytest.raises(ValueError, match="Unknown label shape"):
        labeller._build_single_label_map(
            np.array([50, 50]),
            (100, 2),
        )


# ---------------------------------------------------------------------------
# label
# ---------------------------------------------------------------------------


def test_label_output_shape(record, metadata):
    labeller = sbg.ProbabilisticDASLabeller(["P"])

    labels = labeller.label(record, metadata)

    assert labels.shape == (2, 100, 4)  # P + noise


def test_label_without_noise_map(record, metadata):
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        noise_map=False,
    )

    labels = labeller.label(record, metadata)

    assert labels.shape == (1, 100, 4)
    assert np.max(labels) == pytest.approx(1.0)


def test_noise_map_sums_to_one(record, metadata):
    labeller = sbg.ProbabilisticDASLabeller(["P"])

    labels = labeller.label(record, metadata)

    np.testing.assert_allclose(
        np.sum(labels, axis=0),
        np.ones(record.shape),
    )


def test_unmapped_annotations_are_ignored(record):
    metadata = {
        "__annotations__": {
            "UNKNOWN": np.array([50, 50, 50, 50]),
        }
    }

    labeller = sbg.ProbabilisticDASLabeller(["P"])

    labels = labeller.label(record, metadata)

    assert np.all(labels[0] == 0)
    assert np.all(labels[1] == 1)


def test_multiple_annotations_same_output_label(record):
    metadata = {
        "__annotations__": {
            "Pg": np.array([40, 40, 40, 40]),
            "Pn": np.array([40, 40, 40, 40]),
        }
    }

    labeller = sbg.ProbabilisticDASLabeller(
        {
            "Pg": "P",
            "Pn": "P",
        }
    )

    labels = labeller.label(record, metadata)

    # Values should be clipped to 1 after accumulation
    assert np.max(labels[0]) <= 1.0


# ---------------------------------------------------------------------------
# __call__
# ---------------------------------------------------------------------------


def test_call_adds_output_to_state_dict(record, metadata):
    labeller = sbg.ProbabilisticDASLabeller(["P"])

    state_dict = {
        "X": (record, metadata),
    }

    labeller(state_dict)

    assert "y" in state_dict

    y, y_metadata = state_dict["y"]

    assert y.shape == (2, 100, 4)
    assert y_metadata is not metadata


def test_call_respects_custom_keys(record, metadata):
    labeller = sbg.ProbabilisticDASLabeller(
        ["P"],
        key=("input", "target"),
    )

    state_dict = {
        "input": (record, metadata),
    }

    labeller(state_dict)

    assert "target" in state_dict
