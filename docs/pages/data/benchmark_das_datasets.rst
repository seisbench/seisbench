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

MLSubDAS
----------------

The :py:class:`~seisbench.data.mlsubdas.MLSubDAS` contains 1236 DAS records from cables predominantly in Alaska, Chile,
and Japan. The records have been annotated semi-automatically: first, each individual trace has been picked using
classical three-component models. Afterwards, picks have been corrected manually. Nonetheless, the dataset has clear
imperfections: pick fronts are often incomplete, sometimes inconsistent, and can be missing completely, in particular,
for P arrivals. In addition, pick residuals in excess of 1 s occur. This should be kept in mind when using this dataset.

.. warning::

    Dataset size: **~545Gb**

.. admonition:: Citation

    Xiao, H., Tilmann, F., van den Ende, M., Rivet, D., Loureiro, A., Tsuji, T., ... & Denolle, M. A. (2026).
    DeepSubDAS: an earthquake phase picker from submarine distributed acoustic sensing data.
    Geophysical Journal International, 245(2), ggag061.

    https://doi.org/10.1093/gji/ggag061
