.. _benchmark_das_datasets:

Benchmark DAS Datasets
======================

.. dasexperimentalhint::

SeisBench facilitates the downloading of a suite of publicly available DAS datasets
for training machine learning models. An overview of the contents of each dataset is below,
along with the corresponding citation.

RandomDASDataset
----------------

The :py:class:`~seisbench.data.das_base.RandomDASDataset` contains, as the name might already suggest, random data.
This data is in no way physically meaningful. However, it has the correct format of a SeisBench DAS data set an contains
the keys and structures you'd expect, including (completely random) P and S pick labels. It is intended exclusively for
testing purposes.
