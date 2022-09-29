from .eqtransformer import EQTransformer
from .phasenet import PhaseNet


class PickBlue:

    def __new__(cls, *args, base='eqtransformer', **kwargs):
        if base.lower() == 'eqtransformer':
            return EQTransformer.from_pretrained("obs")
        elif base.lower() == 'phasenet':
            return PhaseNet.from_pretrained("obs")
        else:
            raise ValueError(f"'{base}' is no valid base class of PickBlue. Choose 'EQTransformer' or 'PhaseNet'!")
