.. _data_format_das:

The SeisBench Data Format for DAS
=================================

.. dasexperimentalhint::

Overview
--------

SeisBench DAS dataset are in many ways similar to the SeisBench datasets for classical seismic data and many aspects
of the designs are shared. At the same time, several differences exist to account for the particularities of DAS data.
Each dataset is stored in a single folder and consists of pairs of files: a metadata file and a records file.
The metadata file is in `parquet format`, a tabular data format that can be read with all common libraries, e.g.,
`pandas`. In contrast to the csv format used for regular datasets, parquet has faster read performance and is type-safe.
The records file is an hdf5 file, containing the actual DAS record, as well as the annotations like phase picks.
As a general rule, annotations that are common to the whole trace, e.g., the magnitude of the event, should be an entry
in the metadata, while annotations that are channel-specific, e.g., the P wave arrival times, should be stored in the
records file.

When opening a DAS datasets, SeisBench will load the metadata into memory, but not load the underlying data. This makes
handling of large datasets possible. When training models, data will typically be loaded on the fly and evicted from
memory after use, allowing to work with larger-than-memory datasets.

Chunking
--------

As DAS datasets can quickly become large and handling large files is inconvenient, chunking is an essential part of this
data format. Each chunk consists of a metadata and a records file, following the naming scheme
``metadata_$CHUNK.parquet`` and ``records_$CHUNK.hdf5``. Each pair of files is self-contained, i.e., metadata entries
can only refer to the records in the corresponding hdf5 file.
Datasets should contain a ``chunks`` file in the folder of the dataset listing all available chunks separated by
line breaks. However, SeisBench will also try automatically inferring the available chunks from the data.
When loading datasets, the chunks to load can be specified. If no chunks are specified, all chunks are loaded.

Metadata naming scheme
----------------------

The metadata naming scheme closely follows the one for classical datasets:

    ``CATEGORY_PARAMETER_UNIT``.

The categories are **record**, **source**, **instrument** and **path**.

- ``record`` contains all information and annotations for the record, such as the start time, the sampling rate, and the
  channel spacing. It replaces the ``trace`` category for classical datasets.
- ``source`` describes the source of the waves, such as an earthquake.
  All rows with the same ``source_id`` should agree in all source parameters.
- ``instrument`` contains all information about the recording instrument and fibre. It is the analog to the ``station``
  category.
- ``path`` is used for all properties of the propagation path. Note that some of these might become annotations instead,
  i.e., be stored in the records instead of the metadata, as they might be different per channel.

All names should be in snake case, i.e., lowercase and using underscores as separators.
Exceptions regarding capitalization can be made where common, e.g., seismometer components, units, wave phases.

The ``PARAMETER`` and ``UNIT`` work as for classical datasets. Please see the table of common parameters there and the
existing DAS datasets for examples.

.. note::
    Please note that not every dataset needs to integrate all of the metadata columns!
    The only strictly required column is ``record_name`` as it is used for referencing the record in the hdf5 data file.
