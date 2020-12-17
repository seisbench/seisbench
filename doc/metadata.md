# Metadata naming guide

This document gives guidance on how names of the metadata columns should be chosen.
When implementing a new dataset, it is strongly encouraged to reuse the names given here.
If no name for the attribute is given in this guide, please construct a name according to the general naming scheme.

## Naming scheme
All column names should follow the structure `CATEGORY_PARAMETER_UNIT`.
The `CATEGORY` defines which object the parameter describes.
The default categories are `trace`, `source` and `station`.
The `PARAMETER` describes the provided information, e.g., `latitude` or `longitude`.
The `UNIT` defines the unit, in which the information is provided.
It can be omitted if the parameter is unit-less.

Column names should be in snake case, i.e., lowercase and using underscores as separators. 

## Common parameter names
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
|trace_p_travel_sec|p_travel_sec||
|trace_s_arrival_sample|s_arrival_sample||
|trace_s_status|s_status|Pick.evaluationMode, pick.evaluationStatus|
|trace_s_weight|s_weight|Arrival.weight|
|trace_s_travel_sec|s_travel_sec||
|trace_back_azimuth_deg|back_azimuth_deg||
|trace_snr_db|snr_db||
|trace_coda_end_sample|coda_end_sample||
|station_network_code|network_code|Network.code|
|station_code|receiver_code|Station.code|
|station_type|receiver_type||
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
