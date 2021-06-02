class Pick:
    """
    This class serves as container for storing pick information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the pick was generated from
    :param start_time: Onset time of the pick
    :param end_time: End time of the pick
    :param peak_time: Peak time of the characteristic function for the pick
    :param peak_value: Peak value of the characteristic function for the pick
    :param phase: Phase hint
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
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False


class Detection:
    """
    This class serves as container for storing detection information.
    Defines an ordering based on start time, end time and trace id.

    :param trace_id: Id of the trace the detection was generated from
    :param start_time: Onset time of the detection
    :param end_time: End time of the detection
    :param peak_value: Peak value of the characteristic function for the detection
    """

    def __init__(self, trace_id, start_time, end_time, peak_value=None):
        self.trace_id = trace_id
        self.start_time = start_time
        self.end_time = end_time
        self.peak_value = peak_value

    def __lt__(self, other):
        if self.start_time < other.start_time:
            return True
        if self.end_time < other.end_time:
            return True
        if self.trace_id < other.trace_id:
            return True
        return False
