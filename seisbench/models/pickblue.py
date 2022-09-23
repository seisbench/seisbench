import scipy.signal
import numpy as np


class PickBlue:

    def __init__(
        self,
        type='eqtransformer',
    ):
        citation = (
            """ """
        )
        if type == 'eqtransformer':
            ...
        self._filt_args = (1, .5, "highpass", False)
        self.axis = -1
        super().__init__(
            citation=citation,
            output_type="array",
            default_args={"overlap": 1800, "blinding": (500, 500)},
            in_samples=in_samples,
            pred_sample=(0, in_samples),
            labels=["Detection"] + list(phases),
            sampling_rate=sampling_rate,
            in_channels=in_channels,
            **kwargs,
        )
