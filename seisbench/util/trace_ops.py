import seisbench

import numpy as np
from obspy import ObsPyException


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
            l = min(len(trace.data), samples - start_sample)
            data[c_idx, start_sample : start_sample + l] = trace.data[:l]
            c_completeness += l

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
