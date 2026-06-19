import copy
import numpy as np
import pytest

import seisbench.generate as sbg


@pytest.mark.parametrize("strategy", ["fail", "pad", "move", "variable"])
@pytest.mark.parametrize(
    "ps0", [(-10, 5), (-10, 20), (5, 20), (10, 600), (610, 20), (-10, 620)]
)
@pytest.mark.parametrize(
    "ps1", [(-10, 5), (-10, 20), (5, 20), (10, 600), (610, 20), (-10, 620)]
)
def test_das_windowing(strategy, ps0, ps1):
    """
    Test cases are designed to cover the following scenarios across both axis
    p0, shape = (0, -10), (20, 5)  # to the left
    p0, shape = (0, -10), (20, 20)  # left overlap
    p0, shape = (0, 0), (20, 20)  # contained
    p0, shape = (0, 10), (20, 600)  # right overlap
    p0, shape = (0, 610), (20, 20)  # to the right
    p0, shape = (0, -10), (20, 620)  # full cover
    """
    np.random.seed(42)
    record = np.random.rand(500, 600)
    annotation = np.random.randint(0, 500, 600)
    base_state_dict = {
        "X": (
            record,
            {"__annotations__": {"P": annotation}},
        ),
    }

    p0 = (ps0[0], ps1[0])
    shape = (ps0[1], ps1[1])

    window = sbg.FixedDASWindow(
        p0=p0,
        shape=shape,
        strategy=strategy,
    )
    state_dict = copy.deepcopy(base_state_dict)

    should_fail = False
    if strategy == "fail" and (p0[0] < 0 or p0[1] < 0):
        should_fail = True

    if strategy == "move" and (shape[0] > 500 or shape[1] > 600):
        should_fail = True

    if strategy == "fail" and (p0[0] + shape[0] > 500 or p0[1] + shape[1] > 600):
        should_fail = True

    if should_fail:
        with pytest.raises(ValueError):
            window(state_dict)

    else:
        window(state_dict)
        res_record = state_dict["X"][0]
        res_annotiation = state_dict["X"][1]["__annotations__"]["P"]

        assert res_annotiation.shape[0] == res_record.shape[1]

        # Check shapes
        if strategy != "variable":
            assert res_record.shape == shape
        else:
            x_overlap = set(range(p0[0], p0[0] + shape[0])) & set(
                range(record.shape[0])
            )
            y_overlap = set(range(p0[1], p0[1] + shape[1])) & set(
                range(record.shape[1])
            )

            shape_x = max(x_overlap) - min(x_overlap) + 1 if len(x_overlap) > 0 else 0
            shape_y = max(y_overlap) - min(y_overlap) + 1 if len(y_overlap) > 0 else 0

            assert res_record.shape == (shape_x, shape_y)

        # Check data
        if strategy == "fail":
            assert np.all(
                res_record == record[p0[0] : p0[0] + shape[0], p0[1] : p0[1] + shape[1]]
            )
            assert np.all(
                res_annotiation == annotation[p0[1] : p0[1] + shape[1]] - p0[0]
            )

        elif strategy == "move":
            true_p0x = min(max(p0[0], 0), record.shape[0] - shape[0])
            true_p0y = min(max(p0[1], 0), record.shape[1] - shape[1])
            shape_x, shape_y = shape
            assert np.all(
                res_record
                == record[true_p0x : true_p0x + shape_x, true_p0y : true_p0y + shape_y]
            )
            assert np.all(
                res_annotiation == annotation[true_p0y : true_p0y + shape_y] - true_p0x
            )

        elif strategy == "variable":
            x_overlap = set(range(p0[0], p0[0] + shape[0])) & set(
                range(record.shape[0])
            )
            y_overlap = set(range(p0[1], p0[1] + shape[1])) & set(
                range(record.shape[1])
            )
            if len(x_overlap) > 0 and len(y_overlap) > 0:
                true_p0x = min(x_overlap)
                true_p0y = min(y_overlap)
                shape_x = max(x_overlap) - min(x_overlap) + 1
                shape_y = max(y_overlap) - min(y_overlap) + 1

                assert np.all(
                    res_record
                    == record[
                        true_p0x : true_p0x + shape_x, true_p0y : true_p0y + shape_y
                    ]
                )
                assert np.all(
                    res_annotiation
                    == annotation[true_p0y : true_p0y + shape_y] - true_p0x
                )

        elif strategy == "pad":
            if p0[0] < 0:
                assert np.all(res_record[: -p0[0]] == 0)

            if p0[1] < 0:
                assert np.all(res_record[:, : -p0[1]] == 0)
                assert np.all(np.isnan(res_annotiation[: -p0[1]]))

            if p0[0] + shape[0] > record.shape[0]:
                assert np.all(res_record[record.shape[0] - p0[0] - shape[0] :] == 0)

            if p0[1] + shape[1] > record.shape[1]:
                assert np.all(res_record[:, record.shape[1] - p0[1] - shape[1] :] == 0)
                assert np.all(
                    np.isnan(res_annotiation[record.shape[1] - p0[1] - shape[1] :])
                )

            # Where was the data extracted from
            true_p0x = max(0, p0[0])
            true_p0y = max(0, p0[1])
            # Where was the data written to in res
            res_p0x = max(0, -p0[0])
            res_p0y = max(0, -p0[1])

            # What is the actual number of samples written along each axis
            x_overlap = set(range(p0[0], p0[0] + shape[0])) & set(
                range(record.shape[0])
            )
            y_overlap = set(range(p0[1], p0[1] + shape[1])) & set(
                range(record.shape[1])
            )
            shape_x = max(x_overlap) - min(x_overlap) + 1 if len(x_overlap) > 0 else 0
            shape_y = max(y_overlap) - min(y_overlap) + 1 if len(y_overlap) > 0 else 0

            assert np.all(
                res_record[res_p0x : res_p0x + shape_x, res_p0y : res_p0y + shape_y]
                == record[true_p0x : true_p0x + shape_x, true_p0y : true_p0y + shape_y]
            )
            assert np.all(
                res_annotiation[res_p0y : res_p0y + shape_y]
                == annotation[true_p0y : true_p0y + shape_y] - p0[0]
            )


@pytest.mark.parametrize("strategy", ["fail", "pad", "move", "variable"])
@pytest.mark.parametrize("contains_annotation", [True, False])
def test_random_das_window(strategy, contains_annotation):
    np.random.seed(42)
    record = np.random.rand(500, 600)
    annotation1 = np.random.randint(0, 500, 600)
    annotation2 = np.random.randint(0, 500, 600)
    base_state_dict = {
        "X": (
            record,
            {"__annotations__": {"P": annotation1, "S": annotation2}},
        ),
    }

    window = sbg.RandomDASWindow(
        shape=(100, 200),
        strategy=strategy,
        contains_annotation=contains_annotation,
    )

    for _ in range(100):
        state_dict = copy.deepcopy(base_state_dict)
        window(state_dict)
        assert state_dict["X"][0].shape == window.shape


@pytest.mark.parametrize("strategy", ["fail", "pad", "move", "variable"])
def test_random_das_window_empty_annotations(strategy):
    np.random.seed(42)
    record = np.random.rand(500, 600)
    annotation1 = np.random.randint(0, 500, 600) * np.nan  # Not valid
    annotation2 = np.random.randint(0, 500, 600) + 1000  # Out of range
    annotation3 = np.random.randint(0, 500, 600) - 1000  # Out of range
    base_state_dict = {
        "X": (
            record,
            {"__annotations__": {"P": annotation1, "S": annotation2, "T": annotation3}},
        ),
    }

    window = sbg.RandomDASWindow(
        shape=(100, 200),
        strategy=strategy,
        contains_annotation=True,
    )

    for _ in range(100):
        state_dict = copy.deepcopy(base_state_dict)
        # Key point is that it does not crash
        window(state_dict)
        assert state_dict["X"][0].shape == window.shape
