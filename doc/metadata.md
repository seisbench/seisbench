# Metadata naming guide

This document gives guidance on how names of the metadata columns should be chosen.
There is a general naming scheme, which all names should adhere, and in addition a collection of names which are already in use.
When implementing a new dataset, it is strongly encouraged to reuse the names given here.
If no name for the attribute is given in this guide, please construct a name according to the general naming scheme.

Please note that not every dataset needs to intergrate all of the metadata columns!
The only strictly required column is `trace_name` as it is used for referencing the trace in the hdf5 data file.

## Naming scheme
All column names should follow the structure `CATEGORY_PARAMETER_UNIT`.
The `CATEGORY` defines which object the parameter describes.
The categories are `trace`, `source`, `station` and `path`.
- `trace` contains all information and annotations for the trace, such as the start time, the picks or ground motion parameters.
- `source` describes the source of the waves, such as an earthquake. Source parameters will be empty for noise traces.
  All rows with the same `source_id` should agree in all source parameters.
- `station` contains all information about the station recording the trace, such as station and network code or station location. 
  Similar to `source`, all rows describing the same stations should agree in all station parameters (except for actual changes in the station setup over time).
- `path` is used for all properties of the propagation path, like travel times or distances.
Theses parameters will usually be derived from a combination of source, station and trace parameters.

The `PARAMETER` describes the provided information, e.g., `latitude` or `longitude`.
Parameter names should be as self-contained as possible.

The `UNIT` defines the unit, in which the information is provided.
Example unit identifiers would be `m`, `cm`, `s`, `counts` or `samples`.
For division in units use `p`, for example `mps` for meters per second or `mps2` for meter per second squared.
The unit should only be omitted if the parameter is unit-less, as for trace ids or station codes.

All names should be in snake case, i.e., lowercase and using underscores as separators.
Exceptions regarding capitalization can be made where common, e.g., seismometer components, units, wave phases.

### Trace parameters

|Parameter name|Comment|
|---|---|
|trace_name|A unique identifier for the trace|
|trace_start_time|If possible following ISO 8601:2004|
|trace_sampling_rate_hz|Sampling rate of the trace. If sampling rate is constant across all traces in the data set, it can also be specified in the `data_format` group in the hdf5 data file.|
|trace_dt_s|Time difference between too samples. Will be ignored if sampling rate is provided.|
|trace_npts|Number of samples in the trace|
|trace_channel|Channel from which the data was obtained without the component identifier, e.g., `HH`, `HN`, `BH`. If you're planning to build a dataset with multiple channels for each trace, please get in touch with the developers.|
|trace_category|e.g. earthquake, noise, mine blast|
|trace_p_arrival_sample||
|trace_p_status|e.g. manual (hand-picked), automatic (from an autopicker), estimated (inferred from source parameters and velocity model)|
|trace_p_weight||
|trace_p_uncertainty_s||
|trace_s_arrival_sample||
|trace_s_status||
|trace_s_weight||
|trace_s_uncertainty_s||
|trace_polarity||
|trace_coda_end_sample||
|trace_snr_db||
|trace_Z_snr_db|SNR on the Z component, similar for other components|
|trace_completeness|Fraction of samples in the trace, which were not filled with placeholder values (between 0 and 1). Placeholder values occure for example in case of recording gaps or missing component traces.|
|trace_pga_perg|PGA in precent g on the horizontal components|
|trace_pga_cmps2|PGA in cm / s ** 2 on the horizontal components|
|trace_Z_pga_cmps2|PGA in cm / s ** 2 on the Z component. Similar for other components.|
|trace_pgv_cmps|PGV in cm / s on the horizontal components|
|trace_Z_pgv_cmps|PGV in cm / s on the Z component. Similar for other components.|
|trace_E_sa0.3s_perg|Spectral acceleration at t=0.3s in percent g|
|trace_pga_time|If possible following ISO 8601:2004. Similar for components|
|trace_Z_median_counts||
|trace_Z_mean_counts||
|trace_Z_rms_counts||
|trace_Z_min_counts||
|trace_Z_max_counts||
|trace_Z_lower_quartile_counts||
|trace_Z_upper_quartile_counts||
|trace_Z_spikes|Number of spikes|

### Station parameters

|Parameter name|Comment|
|---|---|
|station_code||
|station_network_code||
|station_location_code||
|station_latitude_deg||
|station_longitude_deg||
|station_elevation_m||
|station_sensitivity_counts_spm|Instrument sensitivity in counts * s/m.|

### Source parameters

|Parameter name|Comment|
|---|---|
|source_id|A unique identifier for the source trace|
|source_origin_time||
|source_origin_uncertainty_sec||
|source_latitude_deg||
|source_latitude_uncertainty_deg||
|source_longitude_deg||
|source_longitude_uncertainty_deg||
|source_error_sec||
|source_gap_deg|Azimuthal gap from the source determination|
|source_horizontal_uncertainty_km||
|source_depth_km||
|source_depth_uncertainty_km||
|source_magnitude|Without units, as magnitude units is implicit|
|source_magnitude_type||
|source_magnitude_author||
|source_focal_mechanism_t_azimuth|Focal mechanism should be described using azimuth, plunge and length of the three principal axis. While this might be less common than describing the fault plane, this allows to accurately describe non-double-couple focal mechanism.|
|source_focal_mechanism_t_plunge||
|source_focal_mechanism_t_length||
|source_focal_mechanism_p_azimuth||
|source_focal_mechanism_p_plunge||
|source_focal_mechanism_p_length||
|source_focal_mechanism_n_azimuth||
|source_focal_mechanism_n_plunge||
|source_focal_mechanism_n_length||
|source_focal_mechanism_eval_mode|e.g. manual/automatic|
|source_focal_mechanism_scalar_moment_Nm|| 

### Path parameters

|Parameter name|Comment|
|---|---|
|path_p_travel_s||
|path_p_residual_s||
|path_weight_phase_location_p||
|path_s_travel_s||
|path_s_residual_s||
|path_weight_phase_location_s||
|path_azimuth_deg||
|path_back_azimuth_deg||
|path_ep_distance_km||
|path_hyp_distance_km||

## Relation to STEAD and QuakeML identifiers
The table lists names for SeisBench and the analog in the STEAD format.
In addition it provides names for QuakeML where applicable.

|SeisBench|STEAD|QuakeML|
|---|---|---|
|trace_start_time|trace_start_time||
|trace_category|trace_category||
|trace_name|trace_name||
|trace_p_arrival_sample|p_arrival_sample||
|trace_p_status|p_status|Pick.evaluationMode, pick.evaluationStatus|
|trace_p_weight|p_weight|Arrival.weight|
|path_p_travel_sec|p_travel_sec||
|trace_s_arrival_sample|s_arrival_sample||
|trace_s_status|s_status|Pick.evaluationMode, pick.evaluationStatus|
|trace_s_weight|s_weight|Arrival.weight|
|path_s_travel_sec|s_travel_sec||
|path_back_azimuth_deg|back_azimuth_deg||
|trace_snr_db|snr_db||
|trace_coda_end_sample|coda_end_sample||
|station_network_code|network_code|Network.code|
|station_code|receiver_code|Station.code|
|trace_channel|receiver_type||
|station_latitude_deg|receiver_latitude|Station.latitude|
|station_longitude_deg|receiver_longitude|Station.longitude|
|station_elevation_m|receiver_elevation_m|Station.elevation|
|source_id|source_id|Event.publicID|
|source_origin_time|source_origin_time|Origin.time.value|
|source_origin_uncertainty_sec|source_origin_uncertainty_sec|Origin.time.uncertainty|
|source_latitude_deg|source_latitude|Origin.latitude.value|
|source_longitude_deg|source_longitude|Origin.longitude.value|
|source_error_sec|source_error_sec|Origin.quality.standardError|
|source_gap_deg|source_gap_deg|Origin.quality.azimuthalGap|
|source_horizontal_uncertainty_km|source_horizontal_uncertainty_km|Origin.originUncertainty.horizontalUncertainty / 1000|
|source_depth_km|source_depth_km|Origin.depth.value / 1000|
|source_depth_uncertainty_km|source_depth_uncertainty_km|Origin.depth.uncertainty / 1000|
|source_magnitude|source_magnitude|Magnitude.mag.value|
|source_magnitude_type|source_magnitude_type|Magnitude.type|
|source_magnitude_author|source_magnitude_author|Magnitude.creationInfo.author|
