What is SeisBench?
==================


Unifying access to seismological data
-------------------------------------
SeisBench standardises access to publicly available seismological datasets for machine learning,
enabling easy comparison and benchmarking of ML models. All datasets are stored in the following format.

.. figure::  ../_static/dataset_format.svg
   :align:   center

Where each row of metadata.csv contains all appropriate parameter information for a given raw waveform, and the waveforms.hdf5 
file stores the raw trace data, indexed by trace_name. As the metadata is stored in table-oriented format, the data is read in 
and integrated with the high-level data analysis library `pandas <https://pandas.pydata.org/>`__ providing rapid filtering and 
general comparison functionality. 

Getting started
-------------------------------------
Here is a quick run-through of the general structure of the SeisBench API. A :py:class:`~seisbench.data.base.WaveformDataset`
reads the metadata provided in metadata.csv as a ``pandas.DataFrame``. The metadata can then be easily
filtered based on users' preferences, and the corresponding raw waveforms obtained. 

For instructions on how to install SeisBench, 
please see the :ref:`installation<installation_and_usage>` page.

.. code-block:: python

    import seisbench.data

    # When requesting the dataset the first time, this will download the dataset.
    # Afterwards it will load the cached version from ~/.seisbench/dummydataset.
    # The SeisBench path can be set with the environment variable SEISBENCH_CACHE_ROOT

    dummy = seisbench.data.DummyDataset()
    print(len(dummy))
    print(dummy.metadata)

    dummy.filter(dummy["source_magnitude"] > 2)
    waveforms = dummy.get_waveforms()
    print(waveforms.shape)

The example uses one of the pre-compiled :ref:`benchmark datasets<benchmark_datasets>`. These are downloadable, publicly available waveform datasets for machine learning in seismology. 
Any :py:class:`~seisbench.data.base.BenchmarkDataset` are cached on download and placed in the configurable cache path ``SEISBENCH_CACHE_ROOT`` which by default is :code:`$HOME/.seisbench`.

