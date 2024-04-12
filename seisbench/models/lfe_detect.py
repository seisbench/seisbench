from .phasenet import VariableLengthPhaseNet


class LFEDetect(VariableLengthPhaseNet):
    """
    This detection and phase picking model for low-frequency earthquakes (LFEs) is based on PhaseNet.
    Please note that, for the time being, LFE detection models do not reach the quality of EQ detection models.

    .. document_args:: seisbench.models LFEDetect
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._citation = (
            "Münchmeyer, J., Giffard-Roisin, S., Malfante, M., Frank, W., Poli, P., Marsan, D., Socquet A. (2024). "
            "Deep learning detects uncataloged low-frequency earthquakes across regions. Seismica."
        )
