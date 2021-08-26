class Pick:
    """
    This class serves as container for storing pick information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the pick was generated from
    :type trace_id: str
    :param start_time: Onset time of the pick
    :type start_time: UTCDateTime
    :param end_time: End time of the pick
    :type end_time: UTCDateTime
    :param peak_time: Peak time of the characteristic function for the pick
    :type peak_time: UTCDateTime
    :param peak_value: Peak value of the characteristic function for the pick
    :type peak_value: float
    :param phase: Phase hint
    :type phase: str
    """

    def __init__(
        self,
        trace_id,
        start_time,
        end_time=None,
        peak_time=None,
        peak_value=None,
        phase=None,
    ):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_value = peak_value
        self.phase = phase

        if end_time is not None and peak_time is not None:
            if not start_time <= peak_time <= end_time:
                raise ValueError("peak_time must be between start_time and end_time.")

    def __lt__(self, other):
        """
        Compares start time, end time and trace id in this order.
        """
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False

    def __str__(self):
        parts = [self.trace_id]
        if self.peak_time is None:
            parts.append(str(self.start_time))
        else:
            parts.append(str(self.peak_time))

        if self.phase is not None:
            parts.append(str(self.phase))

        return "\t".join(parts)


class Detection:
    """
    This class serves as container for storing detection information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the detection was generated from
    :type trace_id: str
    :param start_time: Onset time of the detection
    :type start_time: UTCDateTime
    :param end_time: End time of the detection
    :type end_time: UTCDateTime
    :param peak_value: Peak value of the characteristic function for the detection
    :type peak_value: float
    """

    def __init__(self, trace_id, start_time, end_time, peak_value=None):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_value = peak_value

    def __lt__(self, other):
        """
        Compares start time, end time and trace id in this order.
        """
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False

    def __str__(self):
        parts = [self.trace_id, str(self.start_time), str(self.end_time)]

        return "\t".join(parts)
