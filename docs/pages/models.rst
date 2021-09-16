.. _models:

SeisBench Model API
===================

SeisBench model structure
-------------------------

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
of deep learning-based models natively through PyTorch, in addition to those which use more 'traditional' methods (e.g. STA/LTA).


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


Models integrated into SeisBench
--------------------------------

You don't have to build models from scratch if you don't want to. SeisBench integrates the following notable models from the literature
for you to use. Again, as they inherit from the common SeisBench model interface, all these deep learning models are constructed through
PyTorch. Where possible, the original trained weights are imported and made available. These can be accessed via the ``from_pretrained``
method. For a more in-depth explanation, see the :ref:`examples`.

+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| Integrated models                                         | Task                                  | Reference                                        |
+===========================================================+=======================================+==================================================+
| :py:class:`~seisbench.models.aepicker.BasicPhaseAE`       | Phase Picking                         |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| :py:class:`~seisbench.models.cred.CRED`                   | Earthquake Detection                  |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| :py:class:`~seisbench.models.dpp.DPP`                     | Phase Picking                         |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| :py:class:`~seisbench.models.eqtransformer.EQTransformer` | Earthquake Detection/Phase Picking    |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| :py:class:`~seisbench.models.gpd.GPD`                     | Phase Picking                         |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+
| :py:class:`~seisbench.models.phasenet.PhaseNet`           | Phase Picking                         |                                                  |
+-----------------------------------------------------------+---------------------------------------+--------------------------------------------------+

Currently integrated models are limited to picking and detection works, but you can build ML models in SeisBench to perform general seismic tasks such as:
magnitude and source parameter estimation, hypocentre determination etc.  