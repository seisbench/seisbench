import re
from types import SimpleNamespace

MAX_ENTRIES_STR = 10


class ClassifyOutput(SimpleNamespace):
    """
    A general container to hold the outputs of the classify function of SeisBench models.
    This allows each model to provide a different set of outputs while keeping a consistent output type.
    For example, EQTransformer can output picks and detections, while PhaseNet only provides detections.

    :param creator: The model creating the output.
    :param kwargs: All outputs of the model
    """

    def __init__(self, creator: str, **kwargs):
        self.creator = creator
        super().__init__(**kwargs)

    @staticmethod
    def _raise_breaking_change_error():
        raise NotImplementedError(
            "This method is not implemented. "
            "Most likely you are trying to use an old interface to access the list of picks. "
            "To access the list of picks, just use `classify_output.picks` instead."
        )

    def __iter__(self):
        self._raise_breaking_change_error()

    def __getitem__(self, item):
        self._raise_breaking_change_error()


class PickList(list):
    """
    A list of Pick objects with convenience functions for selecting and printing
    """

    def __str__(self) -> str:
        return f"PickList with {len(self)} entries:\n\n" + self._rep_entries()

    def _rep_entries(self):
        if len(self) <= MAX_ENTRIES_STR:
            return "\n".join([str(pick) for pick in self])
        else:
            return "\n".join(
                [str(pick) for pick in self[:3]]
                + ["..."]
                + [str(pick) for pick in self[-3:]]
            )

    def __repr__(self):
        return str(self)

    def select(
        self, trace_id: str = None, min_confidence: float = None, phase: str = None
    ):
        """
        Select specific picks. Only arguments provided will be used to filter.

        :param trace_id: A regular expression to match against the trace id.
                         The string is directly passed to the `re` module in Python, i.e.,
                         characters like dots need to be escapes and wildcards are represented
                         using `.*`.
        :param min_confidence: The minimum confidence values.
                               Picks without confidence value are discarded.
        :param phase: The phase of the pick. Only exact matches will be returned.
                      Picks without phase information are discarded.
        """
        filtered: list[Pick] = [x for x in self]
        if trace_id is not None:
            exp = re.compile(trace_id)
            filtered = [x for x in filtered if exp.fullmatch(x.trace_id)]

        if min_confidence is not None:
            filtered = [
                x
                for x in filtered
                if x.peak_value is not None and x.peak_value >= min_confidence
            ]

        if phase is not None:
            filtered = [x for x in filtered if x.phase == phase]

        return PickList(filtered)


class DetectionList(PickList):
    """
    A list of Detection objects with convenience functions for selecting and printing
    """

    def __str__(self) -> str:
        return f"DetectionList with {len(self)} entries:\n\n" + self._rep_entries()

    def select(self, trace_id: str = None, min_confidence: float = None):
        """
        Select specific detections. Only arguments provided will be used to filter.

        :param trace_id: A regular expression to match against the trace id.
                         The string is directly passed to the `re` module in Python, i.e.,
                         characters like dots need to be escapes and wildcards are represented
                         using `.*`.
        :param min_confidence: The minimum confidence values.
                               Detections without confidence value are discarded.
        """
        return DetectionList(
            super().select(trace_id=trace_id, min_confidence=min_confidence)
        )


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
