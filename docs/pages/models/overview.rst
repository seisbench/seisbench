.. _model_overview:

SeisBench Models
================

The SeisBench model module provides a collection of deep learning models for seismic data analysis. These models
cover a wide range of tasks including phase picking, earthquake detection, denoising, and depth phase identification.
SeisBench offers both classical waveform models for standard seismic data and models for Distributed
Acoustic Sensing (DAS) data. SeisBench models support accelerated processing on GPUs and Apple's MPS.

All models in SeisBench expose two unified interfaces, making it easy to load pretrained weights, apply models to new
data, and train models on custom datasets. With the first interface, implementing the ``annotate`` and ``classify``
functions, models can be applied directly to data in convenient formats such as ObsPy streams for waveform data or xdas
arrays for DAS data. The second interface provides direct, low-level access to the Pytoch implementation of the models
underneath and can be used for training the models.

.. admonition:: Hint

    If you are looking to quickly apply pretrained models to your seismic data, check out the :ref:`examples`
    section for practical tutorials on using these models for phase picking, detection, and other tasks.

.. toctree::
   :maxdepth: 1

   waveform_models.rst
   das_models.rst
   pretrained_models.rst
   speed_up.rst
