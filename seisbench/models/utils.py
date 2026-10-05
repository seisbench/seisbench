from __future__ import annotations

import asyncio
import contextlib
from typing import Any, AsyncGenerator, Awaitable, Callable, Literal, NamedTuple

import numpy as np
import obspy

Key = tuple[float, str]

_QUEUE_DONE = object()


async def iter_queue_worker(
    worker: Callable[[asyncio.Queue], Awaitable[None]],
    source: AsyncGenerator | None = None,
) -> AsyncGenerator[Any]:
    """
    Runs ``worker`` as a task that puts its results into a queue and yields them in order.

    Exceptions raised in the worker are re-raised in the consumer instead of leaving it
    waiting for the queue forever. If the consumer stops early, e.g., because a later
    stage of the pipeline failed or the annotation was cancelled, the worker task is
    cancelled and awaited, so the upstream pipeline is cleaned up before this
    generator closes.

    :param worker: Coroutine function that receives the output queue
    :param source: Async generator consumed by the worker. It is closed when the worker
                   finishes, so that its own worker is cancelled if this worker fails.
    :return: Async generator over the queue elements
    """
    queue = asyncio.Queue()

    async def run() -> None:
        try:
            if source is None:
                await worker(queue)
            else:
                async with contextlib.aclosing(source):
                    await worker(queue)
        finally:
            queue.put_nowait(_QUEUE_DONE)

    task = asyncio.create_task(run())
    try:
        while (elem := await queue.get()) is not _QUEUE_DONE:
            yield elem
        await task  # Re-raises exceptions from the worker
    finally:
        task.cancel()  # No-op if the worker already finished
        # Waiting retrieves the worker's exception if it was not re-raised above,
        # e.g., when a later stage failed first
        await asyncio.gather(task, return_exceptions=True)


class GroupedTraceData(NamedTuple):
    data: np.ndarray
    stations: list[str]
    component_order: list[str]
    start_time: obspy.UTCDateTime
    sampling_rate: float
    grouping: Literal["instrument", "station", "full"]

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_components(self) -> int:
        return len(self.component_order)

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]


class TraceSegment(NamedTuple):
    data: np.ndarray
    key: Key
    start_time: obspy.UTCDateTime
    window_offset: int
    n_windows: int
    stations: list[str]
    in_samples: int
    pred_sample: tuple[int, int]

    @property
    def n_samples(self) -> int:
        if self.data.ndim == 1:
            return self.data.shape[0]
        return self.data.shape[1]

    @property
    def n_channels(self) -> int:
        if self.data.ndim == 1:
            return 1
        return self.data.shape[0]


class PredictionSegment(TraceSegment):
    @classmethod
    def from_trace_segment(
        cls,
        predictions: np.ndarray,
        segment: TraceSegment,
    ) -> PredictionSegment:
        return cls(
            data=predictions,
            key=segment.key,
            start_time=segment.start_time,
            window_offset=segment.window_offset,
            n_windows=segment.n_windows,
            stations=segment.stations,
            in_samples=segment.in_samples,
            pred_sample=segment.pred_sample,
        )

    @property
    def n_samples(self) -> int:
        if self.data.ndim == 1:
            return self.data.shape[0]
        else:
            return self.data.shape[-2]

    @property
    def n_channels(self) -> int:
        if self.data.ndim == 1:
            return 1
        else:
            return self.data.shape[-1]


class PredictionsStacked(NamedTuple):
    data: np.ndarray
    stations: list[str]
    start_time: obspy.UTCDateTime
    sampling_rate: float

    @property
    def n_stations(self) -> int:
        return len(self.stations)

    @property
    def n_samples(self) -> int:
        return self.data.shape[-1]
