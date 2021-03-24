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

Metadata naming schema
-----------------------
When building custom datasets, metadata columns should conform to the following general naming schema: 

    ``CATEGORY_PARAMETER_UNIT``.

The ``CATEGORY`` defines which object the parameter describes.

The categories are **trace**, **source**, **station** and **path**.

- :ref:`trace<trace>` contains all information and annotations for the trace, such as the start time, the picks or ground motion parameters.
- :ref:`source<source>` describes the source of the waves, such as an earthquake. Source parameters will be empty for noise traces.
  All rows with the same `source_id` should agree in all source parameters.
- :ref:`station<station>` contains all information about the station recording the trace, such as station and network code or station location. 
- :ref:`path<path>` is used for all properties of the propagation path, such as travel times or distances.

The ``PARAMETER`` describes the provided information, e.g., `latitude` or `longitude`.
Parameter names should be as self-contained as possible.

The ``UNIT`` defines the unit in which the information is provided.
Example unit identifiers would be `m`, `cm`, `s`, `counts` or `samples`.
For division in units use `p`, for example `mps` for meters per second or `mps2` for meter per second squared.
The unit should only be omitted if the parameter is unit-less, such as for trace ids or station codes.

All names should be in snake case, i.e., lowercase and using underscores as separators.
Exceptions regarding capitalization can be made where common, e.g., seismometer components, units, wave phases.

.. note::
    Please note that not every dataset needs to integrate all of the metadata columns!
    The only strictly required column is `trace_name` as it is used for referencing the trace in the hdf5 data file.


.. _trace:

Trace parameters
-----------------

+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| Parameter name                | Comment                                                                                                                | 
+===============================+========================================================================================================================+
| trace_name                    | A unique identifier for the trace.                                                                                     |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_start_time              | If possible following ISO 8601:2004.                                                                                   |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_sampling_rate_hz        | Sampling rate of the trace. If sampling rate is constant across all traces in the data set,                            |
|                               | it can also be specified in the `data_format` group in the hdf5 data file.                                             |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_dt_s                    | Time difference between two samples. Will be ignored if sampling rate is provided.                                     |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_npts                    | Number of samples in the trace.                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_channel                 | Channel from which the data was obtained without the component identifier, e.g., `HH`, `HN`, `BH`.                     |
|                               | If you're planning to build a dataset with multiple channels for each trace, please get in touch with the developers.  |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_category                | e.g. earthquake, noise, mine blast.                                                                                    |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_p_arrival_sample        | Sample in trace at which P-phase arrives.                                                                              |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_p_status                | e.g. manual/automatic.                                                                                                 |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_p_weight                | Weighting factor assigned to P-phase pick.                                                                             |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_p_uncertainty_s         | Uncertainty of P-phase pick in seconds.                                                                                |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_s_arrival_sample        | Sample in trace at which S-phase arrives.                                                                              |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_s_status                | e.g. manual/automatic.                                                                                                 |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_s_weight                | Weighting factor assigned to S-phase pick.                                                                             |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_s_uncertainty_s         | Uncertainty of S-phase pick in seconds.                                                                                |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_polarity                |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_coda_end_sample         | Total no. of samples in trace.                                                                                         |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_snr_db                  | Signal-to-noise ratio of trace in decibels.                                                                            |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_snr_db                | SNR on the Z component in decibels, similar for other components.                                                      |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_completeness            | Fraction of samples in the trace, which were not filled with placeholder values (between 0 and 1).                     |
|                               | Placeholder values occure for example in case of recording gaps or missing component traces.                           |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_pga_perg                | PGA in precent g on the horizontal components.                                                                         |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_pga_cmps2               | PGA in cm / s ** 2 on the horizontal components.                                                                       |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_pga_cmps2             | PGA in cm / s ** 2 on the Z component. Similar for other components.                                                   |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_pgv_cmps                | PGV in cm / s on the horizontal components.                                                                            |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_pgv_cmps              | PGV in cm / s on the Z component. Similar for other components.                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_E_sa0.3s_perg           | Spectral acceleration at t=0.3s in percent g.                                                                          |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_pga_time                | If possible following ISO 8601:2004. Similar for components.                                                           |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_median_counts         |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_mean_counts           |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_rms_counts            |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_min_counts            |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_max_counts            |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_lower_quartile_counts |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_upper_quartile_counts |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+
| trace_Z_spikes                |                                                                                                                        |
+-------------------------------+------------------------------------------------------------------------------------------------------------------------+


.. _source:

Source parameters
--------------------

+-----------------------------------------+----------------------------------------------------------------------------------------------+
| Parameter name                          | Comment                                                                                      | 
+=========================================+==============================================================================================+
| source_id                               | A unique identifier for the source trace.                                                    |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_origin_time                      | Origin time of source.                                                                       |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_origin_uncertainty_sec           | Uncertainty of source origin time in seconds.                                                |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_latitude_deg                     | Source latitude coordinate in degrees.                                                       |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_latitude_uncertainty_deg         | Uncertainty of source latitude coordinate in degrees.                                        |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_longitude_deg                    | Source longitude coordinate in degrees.                                                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_longitude_uncertainty_deg        |  Uncertainty of source longitude coordinate in degrees.                                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_error_sec                        | Error association with source location. ??                                                   |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_gap_deg                          | Azimuthal gap from the source determination.                                                 |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_horizontal_uncertainty_km        | Epicentral uncertainity of source location in kilometers.                                    |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_depth_km                         | Source depth in kilometers.                                                                  |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_depth_uncertainty_km             | Uncertainty of source depth in kilometers.                                                   |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_magnitude                        | Magnitude value association with source.                                                     |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_magnitude_type                   | Type of magnitude caluculation used when assigning magnitude to source.                      |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_magnitude_author                 | Author of magnitude calculation.                                                             |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_t_azimuth        | Focal mechanism should be described using azimuth, plunge and length of                      |
|                                         | the three principal axis. While this might be less common than describing the fault plane,   |
|                                         | this allows to accurately describe non-double-couple focal mechanism.                        |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_t_plunge         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_t_length         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_p_azimuth        |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_p_plunge         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_p_length         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_n_azimuth        |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_n_plunge         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_n_length         |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_eval_mode        |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+
| source_focal_mechanism_scalar_moment_Nm |                                                                                              |
+-----------------------------------------+----------------------------------------------------------------------------------------------+


.. _station:

Station parameters
--------------------

+--------------------------------+--------------------------------------------+
| Parameter name                 | Comment                                    | 
+================================+============================================+
| station_code                   | Instrument station code.                   |
+--------------------------------+--------------------------------------------+
| station_network_code           | Instrument network code.                   |
+--------------------------------+--------------------------------------------+
| station_location_code          | Instrument location code.                  |
+--------------------------------+--------------------------------------------+
| station_latitude_deg           | Instrument latitude in degrees.            |
+--------------------------------+--------------------------------------------+
| station_longitude_deg          | Instrument latitude in degrees.            |
+--------------------------------+--------------------------------------------+
| station_elevation_m            | Instrument latitude in m.                  |
+--------------------------------+--------------------------------------------+
| station_sensitivity_counts_spm | Instrument sensitivity in counts * s/m.    |
+--------------------------------+--------------------------------------------+



.. _path:

Path parameters
--------------------

+--------------------------------+---------------------------------------------------------------+
| Parameter name                 | Comment                                                       | 
+================================+===============================================================+
| path_p_travel_s                | Travel-time for P-phase in seconds.                           |
+--------------------------------+---------------------------------------------------------------+
| path_p_residual_s              | Residual of P-phase against some prediction in seconds        |
+--------------------------------+---------------------------------------------------------------+
| path_weight_phase_location_p   | Weight assigned to P-phase in location procedure in seconds.  |
+--------------------------------+---------------------------------------------------------------+
| path_s_travel_s                | Travel-time for S-phase in seconds.                           |
+--------------------------------+---------------------------------------------------------------+
| path_s_residual_s              |Residual of P-phase against some prediction in seconds.        |
+--------------------------------+---------------------------------------------------------------+
| path_weight_phase_location_s   | Weight assigned to P-phase in location procedure in seconds.  |
+--------------------------------+---------------------------------------------------------------+
| path_azimuth_deg               | Azimuth of phase path from source to reciever in degrees.     |
+--------------------------------+---------------------------------------------------------------+
| path_back_azimuth_deg          | Backazimuth of phase path from source to reciever in degrees. |
+--------------------------------+---------------------------------------------------------------+
| path_ep_distance_km            | Epicentral distance of source reciever path in kilometers.    |
+--------------------------------+---------------------------------------------------------------+
| path_hyp_distance_km           | Hypocentral distance of source reciever path in kilometers.   |
+--------------------------------+---------------------------------------------------------------+


