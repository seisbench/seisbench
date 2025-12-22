.. _das_models:

SeisBench DAS Models
====================

.. dasexperimentalhint::

Overview
--------

SeisBench offers the abstract class :py:class:`~seisbench.models.das_base.DASModel` that every SeisBench model working
on DAS data should inherit from. As for classical waveform data, these models have two interfaces. First, a each model
is implemented in PyTorch and works as a regular PyTorch model. This interface is particularly useful for training
the model. Second, aimed specifically for people aiming to apply trainer models, there are the ``annotate`` and
``classify`` functions. These functions take xdas DataArrays as input, allowing to process all data that can be read by
xdas. In the background, SeisBench will take care of chunking, reassambling, batching, processing on GPU (optional),
and even handling larger-than-memory data. To accommodate the particularities of DAS data, the interface is slightly
different than for regular waveform data.

The ``annotate`` function takes an xdas DataArra object and a callback as input. The callback processes the outputs of
the model, for example, the pick probability maps, and transforms them into the desired information. We use callbacks
as the output data will often be too large to fit in memory. SeisBench comes with a range of pre-implemented callbacks,
for example, a picking callback (:py:class:`~seisbench.models.das_base.DASPickingCallback`).

.. code-block:: python

    data = xdas.open_dataarray("mydata")
    callback = DASPickingCallback()
    annotations = model.annotate(stream, callback)  # Execute the model
    callback.get_results_dict()                     # The resulting picks

There are also callbacks to get the full output of the model, either in-memory
(:py:class:`~seisbench.models.das_base.InMemoryCollectionCallback`) or written to disk
(:py:class:`~seisbench.models.das_base.WriterCallback`).
If you want to use multiple callbacks at once, you can combine them using :py:class:`~seisbench.models.das_base.MultiCallback`.

The ``classify`` function works similar to ``annotate``. However, instead of providing a callback, you only provide the
data and the model will use the default callback defined by the developer and directly return the associated output.
For example, a picking model will give you a list of picks just as for the regular SeisBench waveform models.

.. code-block:: python

    stream = xdas.open_dataarray("mydata")
    outputs = model.classify(stream)                # Returns a list of picks
    print(outputs)

In contrast to the functions for regular waveforms, DAS models only take one array per call to
``annotate``/``classify``. This behavior was chosen as for DAS data the processing itself is much heavier than for
1D seismic waveforms and therefore the potential gains from batching across multiple input arrays are miniscule.
You can simply iterate over all your inputs.
For advanced users SeisBench exposes the asynchronous implementations ``annotate_async`` and ``classify_async``.
Both functions are implemented to release the GIL during heavy IO or numerical operations.

Models integrated into SeisBench
--------------------------------

SeisBench integrates the following DAS models. Most models offer pretrained weights through the ``from_pretrained``
method.

+-------------------------------------------------------------------+-------------------------------------------+
| Integrated models                                                 | Task                                      |
+===================================================================+===========================================+
| :py:class:`~seisbench.models.das_wrapper.DASWaveformModelWrapper` | Phase Picking                             |
+-------------------------------------------------------------------+-------------------------------------------+
