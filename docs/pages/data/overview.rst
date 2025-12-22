.. _dataset_overview:

SeisBench Datasets and Data Tools
=================================

The SeisBench data module handles the access to data for model training and evaluation. Currently, SeisBench supports
two types of data formats. One for classical seismic data, recorded, e.g., with a seismometer, strong motion instrument,
or geophone, and one for Distributed Acoustic Sensing (DAS) records. The data specifications and available benchmark
datasets for both of the formats are explained on the subpages.

.. admonition:: Hint

    If you are only looking to apply models, but not to train or quantitatively evaluate them, you probably will
    not need the functionality described here. Instead, most likely the model functionality to be applied
    directly to seismic and DAS records in convenient formats (through obspy/xdas) will be what you need. Check out
    the :ref:`examples` and the :ref:`model_overview`.

.. toctree::
   :maxdepth: 1

   data_format.rst
   benchmark_datasets.rst
   dataset_inspection.rst
   data_format_das.rst
   benchmark_das_datasets.rst
