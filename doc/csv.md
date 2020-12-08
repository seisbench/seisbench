The following is an attempt to map information from standard formats to the proposed csv data structure.

Station/network ID's in QuakeML:
```
network_code:                           WaveformStreamID.networkCode
receiver_code:                          WaveformStreamID.stationCode
```

Note that there is no station metadata schema in QuakeML. Metadata are referred to via `WaveformStreamID`. The corresponding FDSN standard would be FDSN StationXML, where there is a hierarchy with a `Station` being one of several children of `Network`.

```
network_code:                           Network.code
receiver_code:                          Station.code
receiver_type:                          model name? "broadband" or "strongmotion"?
receiver_latitude:                      Station.latitude
receiver_longitude:                     Station.longitude
receiver_elevation_m:                   Station.elevation
```

We also might need sensor depth. Not crrently but perhaps later on, if borehole data shall come into play. We definitely need sensor orientation information, which currently is missing. All this information is present and documented in FDSN StationXML.

In QuakeML, `Arrival` is the class that connects `Pick` to `Origin`.

```
p_arrival_sample: depends on the sampling interval, time window start time and will typically be computed on the fly.
p_status:                                Pick.evaluationMode, pick.evaluationStatus
p_weight:                                Arrival.weight if that is the same
p_travel_sec:                            Pick.time.value minus Origin.time.value
s_arrival_sample:                        see p_arrival_sample
s_status:                                Pick.evaluationMode, pick.evaluationStatus
s_weight:                                Arrival.weight if that is the same
```

Event informations as in QuakeML. Note that below, `Origin` is the object referenced in QuakeML by `Event.preferredOriginID`. Likewise, `Magnitude` is the object referenced in QuakeML by `Event.preferredMagnitudeID`.

```
source_id:                              Event.publicID
source_origin_time:                     Origin.time.value
source_origin_uncertainty_sec:          Origin.time.uncertainty
source_latitude:                        Origin.latitude.value
source_longitude:                       Origin.longitude.value
source_error_sec:                       Origin.quality.standardError
source_gap_deg:                         Origin.quality.azimuthalGap
source_horizontal_uncertainty_km:       Origin.originUncertainty.horizontalUncertainty / 1000
source_depth_km:                        Origin.depth.value / 1000
source_depth_uncertainty_km:            Origin.depth.uncertainty / 1000
source_magnitude:                       Magnitude.mag.value
source_magnitude_type:                  Magnitude.type
source_magnitude_author:                Magnitude.creationInfo.author
```

`source_mechanism_strike_dip_rake` is an unfortunate restriction to double-couple sources. IMHO it would be better to represent the source mechanism by either the six MT elements or the stress axes. From the latter we can also conveniently derive the class of focal mechanism.

```
source_distance_deg:                    Arrival.distance
source_distance_km:                     Arrival.distance * km/deg@Origin.latitude
back_azimuth_deg:                       derived parameter
snr_db:                                 no equivalence in QuakeML
coda_end_sample:                        no equivalence in QuakeML
trace_start_time:                       from MiniSEED
trace_category:                         ??
trace_name:                             ??
```

Note that in order to represent SNR, SeisComP uses an snr-type `Amplitude`.

