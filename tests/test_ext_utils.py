import asyncio
from typing import Literal, Type

import numpy as np
import obspy
import pytest
from matplotlib import pyplot as plt

from seisbench import models
from seisbench.ext.utils import get_edge_indices
from seisbench.models.base import _trim_nan, _trim_zeros
from seisbench.models.utils import PredictionSegment, PredictionsStacked

_MODELS_DICT: dict[str, Type[models.WaveformModel]] = {
    "PhaseNet": models.PhaseNet,
    "EQTransformer": models.EQTransformer,
}

MODEL_NAMES = list(_MODELS_DICT.keys())
MINUTE = 60.0

Implementation = Literal["numpy", "C"]


def fill_zeros(pred_segment: PredictionsStacked) -> None:
    pred_segment.data[np.isnan(pred_segment.data)] = 0.0


def get_predictions(
    model_name: str = "PhaseNet",
) -> tuple[list[list[PredictionSegment]], models.WaveformModel]:
    model = _MODELS_DICT[model_name]()

    stream = obspy.Stream()
    length_seconds = MINUTE * 60 * 5
    sampling_rate = 100.0
    for channel in ("HHZ", "HHN", "HHE"):
        tr = obspy.Trace(
            data=np.random.randn(int(sampling_rate * length_seconds)).astype(
                np.float32
            ),
            header={
                "network": "XX",
                "station": "TEST",
                "channel": channel,
                "sampling_rate": sampling_rate,
                "starttime": obspy.UTCDateTime(2025, 1, 1),
            },
        )
        stream.append(tr)
    # Sampling rate of the data. Equal to self.sampling_rate is this is not None
    sampling_rate = stream[0].stats.sampling_rate

    # Group stream
    comp_dict, _ = model._build_comp_dict(
        stream,
        flexible_horizontal_components=True,
    )

    argdict = {"sampling_rate": sampling_rate}

    async def produce_predictions():
        groups = model._grouping.group_stream(
            stream,
            strict=False,
            min_length_s=(model.in_samples - 1) / sampling_rate,
            comp_dict=comp_dict,
        )
        try:
            train_mode = model.training
            model.eval()

            groups = model._iter_groups(groups, argdict)
            traces = model._iter_fragments_array(groups, argdict)
            predictions_blocks = model._iter_predictions(traces, argdict)
            predictions = [predictions async for predictions in predictions_blocks]
        finally:
            if train_mode:
                model.train()
        return predictions

    predictions = asyncio.run(produce_predictions())
    return predictions, model


@pytest.mark.parametrize("stacking", ["max", "avg"])
@pytest.mark.parametrize("implementation", ["numpy", "C"])
def test_stacking_benchmark(
    benchmark,
    stacking: Literal["avg", "max"],
    implementation: Implementation,
):
    predictions, model = get_predictions()
    predictions = predictions[0]

    benchmark.group = stacking

    def stack_numpy():
        model._stack_predictions_array(
            predictions,
            {"stacking": stacking, "sampling_rate": 100.0},
        )

    def stack_c():
        model._stack_predictions_array_ext(
            predictions,
            {"stacking": stacking, "sampling_rate": 100.0},
        )

    func = stack_numpy if implementation == "numpy" else stack_c
    benchmark(func)


@pytest.mark.parametrize("stacking", ["avg", "max"])
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_stacking(stacking, model_name):
    predictions, model = get_predictions(model_name)

    for prediction in predictions:
        res = model._stack_predictions_array(
            prediction, {"stacking": stacking, "sampling_rate": 100.0}
        )
        res_ext = model._stack_predictions_array_ext(
            prediction, {"stacking": stacking, "sampling_rate": 100.0}
        )

        np.testing.assert_allclose(res.data, res_ext.data, rtol=1e-5)


@pytest.mark.skip
@pytest.mark.parametrize("method", ["avg", "max"])
def test_arrange_blocks_plot(method):
    predictions, model = get_predictions()

    predictions = predictions[0]
    res = model._stack_predictions_array(
        predictions,
        {"stacking": method, "sampling_rate": 100.0},
    )
    res_ext = model._stack_predictions_array_ext(
        predictions,
        {"stacking": method, "sampling_rate": 100.0},
    )

    plt.figure()
    plt.plot(res.data[:, 0], label="internal")
    plt.plot(res_ext.data[:, 0], label="ext")
    plt.legend()
    plt.title(f"Stacking method: {method}")
    plt.show()

    np.testing.assert_allclose(res.data, res_ext.data, rtol=1e-5)


def test_trim_edges() -> None:
    array = np.zeros(100, dtype=np.float32)
    array[20:80] = 1.0

    begin_edge, end_edge = get_edge_indices(array, edge_value=0.0)
    assert begin_edge == 20
    assert end_edge == 80

    _, removed_begin, removed_end = _trim_zeros(array)
    assert removed_begin == 20
    assert array.size - removed_end == 80

    array[array == 0.0] = np.nan

    begin_edge, end_edge = get_edge_indices(array, edge_value=np.nan)
    assert begin_edge == 20
    assert end_edge == 80

    _, removed_begin, removed_end = _trim_nan(array)
    assert removed_begin == 20
    assert array.size - removed_end == 80


@pytest.mark.benchmark(group="trim_edges")
@pytest.mark.parametrize("implementation", ["numpy", "C"])
def test_trim_edges_benchmark(
    benchmark,
    implementation: Implementation,
):
    length = MINUTE * 60 * 10  # 60 minutes
    array = np.ones(int(length), dtype=np.float32)
    array[:300] = 0.0
    array[-500:] = 0.0

    def trim_zeros_numpy():
        _trim_zeros(array)

    def trim_zeros_c():
        begin_edge, end_edge = get_edge_indices(array, edge_value=0.0)
        return array[begin_edge:end_edge]

    func = trim_zeros_numpy if implementation == "numpy" else trim_zeros_c
    benchmark(func)
