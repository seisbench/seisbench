from .eqtransformer import EQTransformer
from .phasenet import PhaseNet


def PickBlue(base: str = "phasenet", **kwargs):
    """
    Initialize a PickBlue model. All `kwargs` are passed to `from_pretrained`.

    :param base: Base model to use. Currently, supports either `eqtransformer` or `phasenet`.
    """
    if base.lower() == "eqtransformer":
        return EQTransformer.from_pretrained("obs", **kwargs)
    elif base.lower() == "phasenet":
        return PhaseNet.from_pretrained("obs", **kwargs)
    else:
        raise ValueError(
            f"'{base}' is no valid base class of PickBlue. Choose 'EQTransformer' or 'PhaseNet'!"
        )
