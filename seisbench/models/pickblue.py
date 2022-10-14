from .eqtransformer import EQTransformer
from .phasenet import PhaseNet


class PickBlue:

    def __new__(cls, *args, base='eqtransformer', **kwargs):
        if base.lower() == 'eqtransformer':
            return EQTransformer.from_pretrained("obs")
        elif base.lower() == 'phasenet':
            pn_model = PhaseNet.from_pretrained("obs")
            pn_model.labels = 'PSN'  # We trained PhaseNet using different label order 
            return pn_model
        else:
            raise ValueError(f"'{base}' is no valid base class of PickBlue. Choose 'EQTransformer' or 'PhaseNet'!")
