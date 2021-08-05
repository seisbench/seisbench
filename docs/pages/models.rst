SeisBench Model API
===================

Any SeisBench Model should subclass :py:class:`~seisbench.models.base.SeisBenchModel`.
This top-level interface encompasses all models applied in SeisBench. If the
model exploits waveform information (e.g. used in phase picking or event detection), the
:py:class:`~seisbench.models.base.WaveformModel` base class should instead be subclassed.


.. code-block:: python

    from seisbench.data import WaveformModel

    class PickingModel(WaveformModel):
        """
        A simple example picking model.
        """
        pass


:py:class:`~seisbench.models.base.SeisBenchModel` inherits from
`torch.nn.Module <https://pytorch.org/docs/stable/generated/torch.nn.Module.html>`__, allowing for the construction
of DL models in addition to those which use more 'traditional' methods (e.g. STA/LTA).


.. code-block:: python

    from seisbench.data import WaveformModel

    class DLPickingModel(WaveformModel):
        """
        A simple example DL-based picking model.
        """
        pass

:py:class:`~seisbench.models.base.WaveformModel` is an abstract interface for processing waveforms. Based on the
properties specified by the inheriting models, WaveformModel automatically provides the respective functions to convert
input waveform streams into predictions.