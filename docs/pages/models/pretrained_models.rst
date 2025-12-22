.. _pretrained_models:

Loading pretrained models
=========================

For annotating waveforms in a meaningful way, trained model weights are required.
SeisBench offers a range of pretrained model weights through a common interface.
Model weights are downloaded on the first use and cached locally afterwards.
For some model weights, multiple versions are available.
For details on accessing these, check the documentation at :py:class:`~seisbench.models.base.SeisBenchModel.from_pretrained`.

.. code-block:: python

    import seisbench.models as sbm

    sbm.PhaseNet.list_pretrained()                  # Get available models
    model = sbm.PhaseNet.from_pretrained("geofon")  # Load the model trained on GEOFON

Pretrained models can not only be used for annotating data, but also offer a great starting point for transfer learning.
