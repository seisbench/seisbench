import math

import numpy as np
import obspy
from obspy import ObsPyException
from obspy.clients.fdsn import Client as FDSNClient
from obspy.clients.fdsn.client import FDSNException, FDSNNoDataException
from obspy.io.mseed import ObsPyMSEEDError

import seisbench


def trace_has_spikes(data, factor=25, quantile=0.975):
    """
    Checks for bit flip errors in the data using a simple quantile rule

    :param data: Data array
    :type data: np.ndarray
    :param factor: Maximum allowed factor between peak and quantile
    :type factor: float
    :param quantile: Quantile to check. Must be between 0 and 1.
    :type quantile: float
    """
    q = np.quantile(np.abs(data), quantile, axis=1, keepdims=True)
    return np.any(data > q * factor)


def stream_to_array(stream, component_order):
    """
    Converts stream of single station waveforms into a numpy array according to a given component order.
    If trace start and end times disagree between component traces, remaining parts are filled with zeros.
    Also returns completeness, i.e., the fraction of samples in the output that actually contain data.
    Assumes all traces to have the same sampling rate.

    :param stream: Stream to convert
    :type stream: obspy.Stream
    :param component_order: Component order
    :type component_order: str
    :return: starttime, data, completeness
    :rtype: UTCDateTime, np.ndarray, float
    """
    starttime = min(trace.stats.starttime for trace in stream)
    endtime = max(trace.stats.endtime for trace in stream)
    sampling_rate = stream[0].stats.sampling_rate

    samples = int((endtime - starttime) * sampling_rate) + 1

    completeness = 0.0
    data = np.zeros((len(component_order), samples), dtype="float64")
    for c_idx, c in enumerate(component_order):
        c_stream = stream.select(channel=f"??{c}")
        if len(c_stream) > 1:
            # If multiple traces are found, issue a warning and write them into the data ordered by their length
            seisbench.logger.warning(
                f"Found multiple traces for {c_stream[0].id} starting at {stream[0].stats.starttime}. "
                f"Completeness will be wrong in case of overlapping traces."
            )
            c_stream = sorted(c_stream, key=lambda x: x.stats.npts)

        c_completeness = 0.0
        for trace in c_stream:
            start_sample = int((trace.stats.starttime - starttime) * sampling_rate)
            tr_length = min(len(trace.data), samples - start_sample)
            data[c_idx, start_sample : start_sample + tr_length] = trace.data[
                :tr_length
            ]
            c_completeness += tr_length

        completeness += min(1.0, c_completeness / samples)

    data -= np.mean(data, axis=1, keepdims=True)

    completeness /= len(component_order)
    return starttime, data, completeness


def rotate_stream_to_zne(stream, inventory):
    """
    Tries to rotate the stream to ZNE inplace. There are several possible failures, which are silently ignored.

    :param stream: Stream to rotate
    :type stream: obspy.Stream
    :param inventory: Inventory object
    :type inventory: obspy.Inventory
    """
    try:
        stream.rotate("->ZNE", inventory=inventory)
    except ValueError:
        pass
    except NotImplementedError:
        pass
    except AttributeError:
        pass
    except ObsPyException:
        pass
    except Exception:
        # Required, because obspy throws a plain Exception for missing channel metadata
        pass


def waveform_id_to_network_station_location(waveform_id):
    """
    Takes a waveform_id as string in the format Network.Station.Location.Channel and
    returns a string with channel dropped. If the waveform_id does not conform to the format,
    the input string is returned.

    :param waveform_id: Waveform ID in format Network.Station.Location.Channel
    :type waveform_id: str
    :return: Waveform ID in format Network.Station.Location
    :rtype: str
    """
    parts = waveform_id.split(".")
    if len(parts) != 4:
        return waveform_id
    else:
        return ".".join(parts[:-1])


def fdsn_get_bulk_safe(client: FDSNClient, bulk: list[tuple]) -> obspy.Stream:
    """
    A wrapper around obspy's get_waveforms_bulk that does error handling
    and tries to download as much data as possible.

    :param client: An obspy FDSN client
    :param bulk: A bulk request as for get_waveforms_bulk
    """
    return _fdsn_get_bulk_safe(client, bulk, failed=False)


def _fdsn_get_bulk_safe(
    client: FDSNClient, bulk: list[tuple], failed: bool = False
) -> obspy.Stream:
    """
    Internal wrapper to hide the failed parameter from the user.
    """
    if len(bulk) == 0 or (failed and len(bulk) == 1):
        return obspy.Stream()

    new_failed = False
    if len(bulk) == 1:
        new_failed = True

    try:
        return client.get_waveforms_bulk(bulk)
    except FDSNNoDataException:
        return obspy.Stream()
    except (FDSNException, ObsPyMSEEDError, Exception):
        # Send off requests for each half to isolate the error
        m = len(bulk) // 2
        return _fdsn_get_bulk_safe(client, bulk[:m], new_failed) + _fdsn_get_bulk_safe(
            client, bulk[m:], new_failed
        )


def _round_py2(x: float) -> float:
    if x > 0:
        return float(math.floor((x) + 0.5))
    else:
        return float(math.ceil((x) - 0.5))


def stream_slice(
    stream: obspy.Stream,
    starttime: obspy.UTCDateTime,
    endtime: obspy.UTCDateTime,
) -> obspy.Stream:
    """Slice an ObsPy Stream object between starttime and endtime.

    This is a lightweight implementation of `obspy.Stream.slice` that ensures with
    default paramters `Stream.slice(..., keep_empty_traces=False, nearest_sample=True)`.

    The performance gain of ~7x is achieved by avoiding deepcopies of traces and
    combining `Trace._ltrim` and `Trace._rtrim` into a single operation.

    Difference for edge cases compared to `Stream.slice` is that times are rounded
    with Python3 rounding (round half to even) instead of ObsPy's custom rounding
    (round half away from zero). This can lead to differences of 1 sample in edge
    cases.

    :param stream: The input ObsPy Stream.
    :type stream: obspy.Stream
    :param starttime: The start time for slicing.
    :type starttime: obspy.UTCDateTime
    :param endtime: The end time for slicing.
    :type endtime: obspy.UTCDateTime

    :return: The sliced ObsPy Stream.
    :rtype: obspy.Stream
    """
    if len(stream) == 0:
        raise ValueError("Stream is empty and cannot be sliced.")

    if starttime >= endtime:
        raise ValueError("starttime must be earlier than endtime.")

    first = stream[0].stats

    # select start/end time fitting to a sample point on the first trace
    delta = _round_py2((starttime - first.starttime) * first.sampling_rate)
    starttime = first.starttime + delta * first.delta

    delta = _round_py2((endtime - first.endtime) * first.sampling_rate)
    endtime = first.endtime + delta * first.delta  # delta is negative!

    out = obspy.Stream()
    for tr in stream:
        stats = tr.stats
        sampling_rate = stats.sampling_rate

        samples_start = int(_round_py2((starttime - stats.starttime) * sampling_rate))
        samples_start = max(samples_start, 0)
        if samples_start >= stats.npts - 1:
            continue
        slice_starttime = stats.starttime + samples_start * stats.delta

        samples_end = (
            int(_round_py2((endtime - slice_starttime) * sampling_rate))
            + samples_start
            + 1
        )
        samples_end = min(samples_end, stats.npts)

        n_samples = samples_end - samples_start
        if n_samples <= 0:
            continue

        sliced_trace = obspy.Trace(
            tr.data[samples_start:samples_end],
            {
                "network": stats.network,
                "station": stats.station,
                "location": stats.location,
                "channel": stats.channel,
                "starttime": slice_starttime,
                "sampling_rate": sampling_rate,
                "t_offset": stats.get("t_offset", 0.0),
            },
        )
        out.append(sliced_trace)
    return out
