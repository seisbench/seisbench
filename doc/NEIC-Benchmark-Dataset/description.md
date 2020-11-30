# NEIC Benchmark dataset

This is an unofficial picker benchmark dataset compiled by the NEIC and provided to us as a tar file.

## Description

`global_pick_data.tar.gz` is a huge tar file containing a `Data`
subdirectory. I expected the content to be some monster HDF5 file
but it is not. Instead there are about 50 files in in NumPy-specific
`npy` format [1]:

```
saul@sec24c35:~/NEIC-Benchmark-Dataset/Data$ ls
P_0.npy   P_14.npy  P_5.npy  Pg_0.npy	Pg_4.npy  Pg_9.npy   Pn_13.npy	Pn_4.npy  Pn_9.npy  Sg_3.npy  Sn_3.npy
P_10.npy  P_1.npy   P_6.npy  Pg_10.npy	Pg_5.npy  Pn_0.npy   Pn_14.npy	Pn_5.npy  S_0.npy   Sg_4.npy  Sn_4.npy
P_11.npy  P_2.npy   P_7.npy  Pg_1.npy	Pg_6.npy  Pn_10.npy  Pn_1.npy	Pn_6.npy  Sg_0.npy  Sn_0.npy  Sn_5.npy
P_12.npy  P_3.npy   P_8.npy  Pg_2.npy	Pg_7.npy  Pn_11.npy  Pn_2.npy	Pn_7.npy  Sg_1.npy  Sn_1.npy  Sn_6.npy
P_13.npy  P_4.npy   P_9.npy  Pg_3.npy	Pg_8.npy  Pn_12.npy  Pn_3.npy	Pn_8.npy  Sg_2.npy  Sn_2.npy  Sn_7.npy
saul@sec24c35:~/NEIC-Benchmark-Dataset/Data$ ls
110247324	.
```

That's 110 GB uncompressed! All indivitual `npy` files are around 2 GB each.

On a side note, the `npy` format was developed as a simple HDF5
alternative for the NumPy ecosystem, according to [2].

There is no documentation of any kind but the data structure is rather simple and for the most part self explaining.

```
>>> import numpy
>>> arr=numpy.load("Pg_0.npy",allow_pickle=True)
>>> len(arr)
25000
>>> arr[0]
{'MetaData': 'GS KAN05 HHZ 01 Pg 2015-07-28T07:32:24.980000Z 200030R0 2.8 2015-07-28T07:32:04.300000Z 36.515 -98.957 1.05 55.4', 'WFData': array([[-34161.54633207, -34226.77126392, -34192.63113299, ...,
        -33939.97764346, -33657.23261635, -33864.76033195],
       [  2798.85275158,   3915.04735783,   3632.32918324, ...,
          1985.13703209,   1812.62408178,   1223.51817846],
       [  2777.62091827,   2819.73075529,   2772.58328941, ...,
          1220.30239045,   2196.8059435 ,   2542.36771801]])}
>>> arr[0].keys()
dict_keys(['MetaData', 'WFData'])
>>> arr[0]['MetaData']
'GS KAN05 HHZ 01 Pg 2015-07-28T07:32:24.980000Z 200030R0 2.8 2015-07-28T07:32:04.300000Z 36.515 -98.957 1.05 55.4'
>>> arr[1]['MetaData']
'HV JOKA HHZ -- Pg 2018-06-19T00:03:37.840000Z 1000EUZR 3.49 2018-06-19T00:03:31.960000Z 19.4005 -155.2653 0.25 82.3'
>>> arr[2]['MetaData']
'AK YAH BHZ -- Pg 2015-08-06T17:39:57.410000Z 10002ZVH 2.9 2015-08-06T17:39:44.000000Z 60.5026 -143.0562 0.66 102.0'
```

etc. So there is some metadata in the form of a simple one-liner.

```
>>> arr[0]['WFData']
array([[-34161.54633207, -34226.77126392, -34192.63113299, ...,
        -33939.97764346, -33657.23261635, -33864.76033195],
       [  2798.85275158,   3915.04735783,   3632.32918324, ...,
          1985.13703209,   1812.62408178,   1223.51817846],
       [  2777.62091827,   2819.73075529,   2772.58328941, ...,
          1220.30239045,   2196.8059435 ,   2542.36771801]])
array([-34161.54633207, -34226.77126392, -34192.63113299, ...,
       -33939.97764346, -33657.23261635, -33864.76033195])
>>> len(arr[0]['WFData'][0])
2400
```

Three components per entry, floating point numbers, so we apparently have gain corrected raw data in nanometers per second. It is not yet clear what the components are but that should be easy to figure out by comparing with data downloaded from a web service.

Also there seems to be no time window starting time specified.  Unless I missed something, we only have the pick times and the event origin times. Also there seems to be no indication what the sampling frequency is. Data snippets are always 2400 samples long. It appears that the time window is centered exactly around the pick times, with 1200 samples before and 1200 samples after that time. Otherwise it would not be possible to recover absolute timing. 2400 (1200+1200) samples is not a lot and corresponds to only 30 s lead time for 40 Hz data.

## Evaluation

Quite nice! Very pragmatic approach, but unfortunately entirely NumPy specific and therefore cannot be considered portable except in a Python/NumPy ecosystem. Probably easy handling and fast reading. A disadvantage is the dependency on a specific software, which would make it very hard to sell it as a "standard". What I also don't like is the difficulty to access individual data, e.g. for simple viewing in PQL or so. Plain MiniSEED files are handier here but also slower to read (I presume).

This NumPy format may well come into play as a "common denominator" intermediate format immediately before the actual processing.

What I don't like

* unspecified sampling frequency and time window starting time

* fixed number of samples leading to sampling-frequency dependent time windows

* rather short time windows exactly centered about the pick time

* data set not suitable for all kinds of seismic onsets, especially
  long-period onsets lacking short periods (not uncommon!)

* data set hardly suitable for other analyses like magnitudes

The unknown sampling frequency is interesting in the sense that the ML is then agnostic of absolute frequencies. This might prevent the ML from learning about certain kinds of noise that shows up prominently in rather narrow frequency bands, especially microseisms. But that might also have advantages (TBD). For instance, the evaluation may be simplified: time axis is always samples and the reference onset is always at sample number 1200. We then simply compare offset in samples of the ML-determined onset. Not sure if that is the motivation behind, though.



[1] [numpy.lib.format](https://numpy.org/devdocs/reference/generated/numpy.lib.format.html)

[2] [NEP 1 — A Simple File Format for NumPy Arrays](https://numpy.org/neps/nep-0001-npy-format.html)
