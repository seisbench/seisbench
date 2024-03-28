from .eqtransformer import EQTransformer



class OBSTransformer(EQTransformer):
    """
    Initialize an instance of OBSTransformer model.
    OBSTransformer is built based on the original (non-conservative) EqTransformer model.
    """


    def __init__(
        self,
        lstm_blocks=2,
        drop_rate=0.2,
        original_compatible=True,
        **kwargs,
    ):
        super().__init__(
            lstm_blocks = lstm_blocks,
            drop_rate = drop_rate,
            original_compatible = original_compatible,
            **kwargs
        )
        self._citation = (
            "Niksejel, A. and Zhang, M., 2024. OBSTransformer: a deep-learning seismic "
            "phase picker for OBS data using automated labelling and transfer learning."
            "Geophysical Journal International, p.ggae049, https://doi.org/10.1093/gji/ggae049."
        )