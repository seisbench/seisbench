import seisbench.data
import seisbench.util.region as region
import seisbench

import numpy as np
import pytest
import logging
from pathlib import Path
import h5py
import pandas as pd
from unittest.mock import patch


def test_get_dimension_mapping():
    # For legacy reasons test dimensions are called "ZNE".
    # However, component mapping is now handled by _get_component_mapping.

    # Test ordering and list/string format
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_dimension_mapping(
        "ZNE", "ZNE"
    )
    assert [1, 2, 0] == seisbench.data.WaveformDataset._get_dimension_mapping(
        "ZNE", "NEZ"
    )
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_dimension_mapping(
        ["Z", "N", "E"], "ZNE"
    )
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_dimension_mapping(
        "ZNE", ["Z", "N", "E"]
    )
    assert [0, 2, 1] == seisbench.data.WaveformDataset._get_dimension_mapping(
        ["Z", "E", "N"], ["Z", "N", "E"]
    )

    # Test failures
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_dimension_mapping("ZNE", "Z")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_dimension_mapping("ZNE", "ZZE")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_dimension_mapping("ZEZ", "ZNE")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_dimension_mapping("ZNE", "ZRT")


def test_get_component_mapping():
    # Strategy "pad"
    dummy = seisbench.data.DummyDataset(missing_components="pad")
    assert dummy._get_component_mapping("Z", "ZNE") == [0, 1, 1]
    assert dummy._get_component_mapping("ZE", "ZNE") == [0, 2, 1]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2, 3]

    # Strategy "copy"
    dummy = seisbench.data.DummyDataset(missing_components="copy")
    assert dummy._get_component_mapping("Z", "ZNE") == [0, 0, 0]
    assert dummy._get_component_mapping("N", "ZNE") == [0, 0, 0]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2, 0]

    # Strategy "ignore"
    dummy = seisbench.data.DummyDataset(missing_components="ignore")
    assert dummy._get_component_mapping("Z", "ZNE") == [0]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2]


def test_pad_packed_sequence():
    seq = [np.ones((5, 1)), np.ones((6, 3)), np.ones((1, 2)), np.ones((7, 2))]

    packed = seisbench.data.WaveformDataset._pad_packed_sequence(seq)

    assert packed.shape == (4, 7, 3)
    assert np.sum(packed == 1) == sum(x.size for x in seq)
    assert np.sum(packed == 0) == packed.size - sum(x.size for x in seq)


def test_preload():
    dummy = seisbench.data.DummyDataset(cache="trace")
    assert len(dummy._waveform_cache) == 0

    dummy.preload_waveforms()
    assert len(dummy._waveform_cache) == len(dummy)


def test_filter_and_cache_evict():
    # Block caching
    dummy = seisbench.data.DummyDataset(cache="full")
    dummy.preload_waveforms()
    blocks = set(dummy.metadata["trace_name"].apply(lambda x: x.split("$")[0]))
    assert len(dummy._waveform_cache) == len(blocks)

    mask = np.arange(len(dummy)) < len(dummy) / 2
    dummy.filter(mask)

    assert len(dummy) == np.sum(mask)  # Correct metadata length
    blocks = set(dummy.metadata["trace_name"].apply(lambda x: x.split("$")[0]))
    assert len(dummy._waveform_cache) == len(blocks)  # Correct cache eviction

    # Trace caching
    dummy = seisbench.data.DummyDataset(cache="trace")
    dummy.preload_waveforms()
    assert len(dummy._waveform_cache) == len(dummy)

    mask = np.arange(len(dummy)) < len(dummy) / 2
    dummy.filter(mask)

    assert len(dummy) == np.sum(mask)  # Correct metadata length
    assert len(dummy._waveform_cache) == len(dummy)  # Correct cache eviction


def test_region_filter():
    # Receiver + Circle domain
    dummy = seisbench.data.DummyDataset()

    lat = 0
    lon = np.linspace(-10, 10, len(dummy))

    dummy.metadata["station_latitude_deg"] = lat
    dummy.metadata["station_longitude_deg"] = lon

    domain = region.CircleDomain(0, 0, 1, 5)

    dummy.region_filter_receiver(domain)

    assert len(dummy) == np.sum(np.logical_and(1 < np.abs(lon), np.abs(lon) < 5))

    # Source + RectangleDomain
    dummy = seisbench.data.DummyDataset()

    np.random.seed(42)
    n = len(dummy)
    lat = np.random.random(n) * 40 - 20
    lon = np.random.random(n) * 40 - 20

    dummy.metadata["source_latitude_deg"] = lat
    dummy.metadata["source_longitude_deg"] = lon

    domain = region.RectangleDomain(
        minlatitude=-5, maxlatitude=5, minlongitude=10, maxlongitude=15
    )

    dummy.region_filter_source(domain)

    mask_lat = np.logical_and(-5 <= lat, lat <= 5)
    mask_lon = np.logical_and(10 <= lon, lon <= 15)

    assert len(dummy) == np.sum(np.logical_and(mask_lat, mask_lon))


def test_get_waveforms_dimensions():
    for cache in ["full", "trace", None]:
        dummy = seisbench.data.DummyDataset(cache=cache)

        waveforms = dummy.get_waveforms()
        assert waveforms.shape == (len(dummy), 3, 1200)

        dummy.component_order = "ZEN"
        waveforms_zen = dummy.get_waveforms()
        assert (waveforms[:, 1] == waveforms_zen[:, 2]).all()
        assert (waveforms[:, 2] == waveforms_zen[:, 1]).all()
        assert (waveforms[:, 0] == waveforms_zen[:, 0]).all()

        mask = np.arange(len(dummy)) < len(dummy) / 2
        assert dummy.get_waveforms(mask=mask).shape[0] == np.sum(mask)

        dummy.dimension_order = "CWN"
        waveforms = dummy.get_waveforms()

        assert waveforms.shape == (3, 1200, len(dummy))


def test_get_waveforms_select():
    for cache in ["full", "trace", None]:
        dummy = seisbench.data.DummyDataset(cache=cache)

        waveforms_full = dummy.get_waveforms()
        assert waveforms_full.shape == (len(dummy), 3, 1200)

        waveforms_ind = dummy.get_waveforms(idx=5)
        assert waveforms_ind.shape == (3, 1200)
        assert (waveforms_ind == waveforms_full[5]).all()

        waveforms_list = dummy.get_waveforms(idx=[10])
        assert waveforms_list.shape == (1, 3, 1200)
        assert (waveforms_list[0] == waveforms_full[10]).all()

        mask = np.zeros(len(dummy), dtype=bool)
        mask[15] = True
        waveforms_mask = dummy.get_waveforms(mask=mask)
        assert waveforms_mask.shape == (1, 3, 1200)
        assert (waveforms_mask[0] == waveforms_full[15]).all()


def test_preload_no_cache(caplog):
    with caplog.at_level(logging.WARNING):
        dummy = seisbench.data.DummyDataset(cache=None)
        dummy.preload_waveforms()

    assert "Skipping preload, as cache is disabled." in caplog.text


def test_writer(caplog, tmp_path: Path):
    # Test empty writer
    with seisbench.data.WaveformDataWriter(
        tmp_path / "writer_a" / "metadata.csv", tmp_path / "writer_a" / "waveforms.hdf5"
    ) as writer:
        pass

    assert (tmp_path / "writer_a").is_dir()  # Path exists
    assert not any((tmp_path / "writer_a").iterdir())  # Path is empty

    # Test correct write
    with seisbench.data.WaveformDataWriter(
        tmp_path / "writer_b" / "metadata.csv", tmp_path / "writer_b" / "waveforms.hdf5"
    ) as writer:
        trace = {"trace_name": "dummy", "split": 2}
        writer.add_trace(trace, np.zeros((3, 100)))

    assert (tmp_path / "writer_b").is_dir()  # Path exists
    assert (
        "No data format options specified" in caplog.text
    )  # Check warning data format
    assert (
        tmp_path / "writer_b" / "metadata.csv"
    ).is_file()  # Check metadata file exist
    assert (
        tmp_path / "writer_b" / "waveforms.hdf5"
    ).is_file()  # Check waveform file exist

    # Test with failing write
    with pytest.raises(Exception):
        with seisbench.data.WaveformDataWriter(
            tmp_path / "writer_c" / "metadata.csv.partial",
            tmp_path / "writer_c" / "waveforms.hdf5.partial",
        ) as writer:
            trace = {"trace_name": "dummy", "split": 2}
            writer.add_trace(trace, np.zeros((3, 100)))
            raise Exception("Dummy exception to test failure handling of writer")

    assert (tmp_path / "writer_c").is_dir()  # Path exists
    assert "Error in downloading dataset" in caplog.text  # Check error data writer
    assert (
        tmp_path / "writer_c" / "metadata.csv.partial"
    ).is_file()  # Check partial metadata file exist
    assert not (
        tmp_path / "writer_c" / "metadata.csv"
    ).is_file()  # Check metadata file exist
    assert (
        tmp_path / "writer_c" / "waveforms.hdf5.partial"
    ).is_file()  # Check partial waveform file exist
    assert not (
        tmp_path / "writer_c" / "waveforms.hdf5"
    ).is_file()  # Check waveform file exist


def test_available_chunks(caplog, tmp_path: Path):
    # Empty directory raises FileNotFoundError
    folder = tmp_path / "a"
    folder.mkdir(exist_ok=True, parents=True)
    with pytest.raises(FileNotFoundError):
        seisbench.data.WaveformDataset.available_chunks(folder)

    # Empty chunk file prints warning
    folder = tmp_path / "b"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "chunks", "w").close()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError):
            seisbench.data.WaveformDataset.available_chunks(folder)
    assert (
        "Found empty chunks file. Using chunk detection from file names." in caplog.text
    )

    # Unchunked dataset
    folder = tmp_path / "c"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "metadata.csv", "w").close()
    open(folder / "waveforms.hdf5", "w").close()
    chunks = seisbench.data.WaveformDataset.available_chunks(folder)
    assert chunks == [""]

    # Chunked dataset detected from chunkfile
    folder = tmp_path / "d"
    folder.mkdir(exist_ok=True, parents=True)
    with open(folder / "chunks", "w") as f:
        f.write("a\nb\nc\n")
    chunks = seisbench.data.WaveformDataset.available_chunks(folder)
    assert chunks == ["a", "b", "c"]

    # Chunked dataset detected from file names
    folder = tmp_path / "e"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "metadataa.csv", "w").close()
    open(folder / "waveformsa.hdf5", "w").close()
    open(folder / "metadatab.csv", "w").close()
    open(folder / "waveformsb.hdf5", "w").close()
    open(folder / "metadatac.csv", "w").close()
    open(folder / "waveformsc.hdf5", "w").close()
    chunks = seisbench.data.WaveformDataset.available_chunks(folder)
    assert chunks == ["a", "b", "c"]

    # Chunked dataset with inconsistent chunks
    folder = tmp_path / "f"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "metadataa.csv", "w").close()
    open(folder / "waveformsa.hdf5", "w").close()
    open(folder / "metadatab.csv", "w").close()
    open(folder / "waveformsc.hdf5", "w").close()
    with caplog.at_level(logging.WARNING):
        chunks = seisbench.data.WaveformDataset.available_chunks(folder)
    assert chunks == ["a"]
    assert "Found metadata but no waveforms for chunks" in caplog.text
    assert "Found waveforms but no metadata for chunks" in caplog.text


def test_chunked_loading():
    chunk0 = seisbench.data.ChunkedDummyDataset(chunks=["0"])
    chunk1 = seisbench.data.ChunkedDummyDataset(chunks=["1"])
    chunk01 = seisbench.data.ChunkedDummyDataset()

    assert len(chunk0) + len(chunk1) == len(chunk01)

    wv = chunk0.get_waveforms()
    assert wv.shape[0] == len(chunk0)
    wv = chunk1.get_waveforms()
    assert wv.shape[0] == len(chunk1)
    wv = chunk01.get_waveforms()
    assert wv.shape[0] == len(chunk01)


def test_download_dataset_chunk_arg(tmp_path):
    """
    Test ensures that datasets with/out chunking are accordingly called with/out chunks in _download_dataset
    """
    with patch(
        "seisbench.cache_root", tmp_path
    ):  # Ensure test does not modify SeisBench cache

        class MockDataset(seisbench.data.BenchmarkDataset):
            def __init__(self, **kwargs):
                super().__init__(citation="", **kwargs)

            def _download_dataset(self, writer):
                raise ValueError("Called without chunks")

        class ChunkedMockDataset(seisbench.data.BenchmarkDataset):
            def __init__(self, **kwargs):
                super().__init__(citation="", **kwargs)

            def _download_dataset(self, writer, chunk):
                raise ValueError("Called with chunks")

        # Note: This would raise a TypeError when called with the chunk parameter
        with pytest.raises(ValueError) as e:
            MockDataset()
        assert "Called without chunks" in str(e)

        # Note: This would raise a TypeError when called without the chunk parameter
        with pytest.raises(ValueError) as e:
            ChunkedMockDataset()
        assert "Called with chunks" in str(e)


def test_unify_sampling_rate(caplog):
    dummy = seisbench.data.DummyDataset()
    del dummy._metadata["trace_sampling_rate_hz"]
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert "Sampling rate not specified in data set." in caplog.text

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._data_format["sampling_rate"] = 40.0
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
        in caplog.text
    )

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    del dummy._metadata["trace_sampling_rate_hz"]
    dummy._metadata["trace_dt_s"] = 1 / 20.0
    dummy._data_format["sampling_rate"] = 40.0
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
        in caplog.text
    )

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 40.0
    dummy._metadata["trace_dt_s"] = 1 / 20.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Inconsistent sampling rates in metadata. Using values from 'trace_sampling_rate_hz'."
        in caplog.text
    )

    # Small deviations do not cause warnings
    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_dt_s"] = 1 / 20.00001
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_sampling_rate_hz"].values[:20] = np.nan
    dummy._metadata["trace_dt_s"] = 1 / 20.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_sampling_rate_hz"].values[:20] = np.nan
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert "Found some traces with undefined sampling rates." in caplog.text

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_sampling_rate_hz"].values[:20] = 40.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Data set contains mixed sampling rate, but no sampling rate was specified for the dataset."
        in caplog.text
    )

    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""


def test_unify_component_order(caplog):
    # Component order not specified
    dummy = seisbench.data.DummyDataset()
    del dummy._metadata["trace_component_order"]
    del dummy._data_format["component_order"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_component_order()
    assert "Component order not specified in data set." in caplog.text

    # Component order inconsistent
    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_component_order"] = "ZNE"
    dummy._metadata["trace_component_order"].values[10] = "ZEN"
    dummy._data_format["component_order"] = "ZNE"
    order = dummy._metadata["trace_component_order"].values.copy()
    with caplog.at_level(logging.WARNING):
        dummy._unify_component_order()
    assert (
        "Found inconsistent component orders between data format and metadata."
        in caplog.text
    )
    assert (dummy._metadata["trace_component_order"].values == order).all()

    # Component order only in data_format
    caplog.clear()
    dummy = seisbench.data.DummyDataset()
    del dummy._metadata["trace_component_order"]
    dummy._data_format["component_order"] = "ZNE"
    with caplog.at_level(logging.WARNING):
        dummy._unify_component_order()
    assert len(caplog.text) == 0
    assert (dummy._metadata["trace_component_order"] == "ZNE").all()


def test_resample():
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_sampling_rate_hz"].values[0] = np.nan

    # NaN sampling rate raises no error if sampling rate is None, but raises error otherwise
    dummy.get_waveforms(idx=0)
    with pytest.raises(ValueError):
        dummy.get_waveforms(idx=0, sampling_rate=20)

    # Incorrect dimension order raises an issue if actual resampling is required
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._data_format["dimension_order"] = dummy._data_format[
        "dimension_order"
    ].replace("W", "X")
    dummy.get_waveforms(idx=0, sampling_rate=20)
    with pytest.raises(ValueError):
        dummy.get_waveforms(idx=0, sampling_rate=40)

    # Correct cases have expected length
    dummy = seisbench.data.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    wv20 = dummy.get_waveforms(idx=0, sampling_rate=20)
    wv100 = dummy.get_waveforms(idx=0, sampling_rate=100)
    wv10 = dummy.get_waveforms(idx=0, sampling_rate=10)
    wv15 = dummy.get_waveforms(idx=0, sampling_rate=15)

    assert wv20.shape[0] == wv100.shape[0] == wv10.shape[0] == wv15.shape[0]
    assert wv20.shape[1] * 100 / 20 == wv100.shape[1]
    assert wv20.shape[1] * 10 / 20 == wv10.shape[1]
    assert wv20.shape[1] * 15 / 20 == wv15.shape[1]

    # Data set sampling rate is overwritten by function argument sampling_rate
    dummy.sampling_rate = 100
    wv100 = dummy.get_waveforms(idx=0)
    wv20 = dummy.get_waveforms(idx=0, sampling_rate=20)
    assert wv20.shape[0] == wv100.shape[0]
    assert wv20.shape[1] * 100 / 20 == wv100.shape[1]


def test_get_sample():
    # Checks that the parameters are correctly overwritten, when the sampling_rate is changed
    dummy = seisbench.data.DummyDataset()
    dummy.sampling_rate = None

    base_sampling_rate = dummy.metadata.iloc[0]["trace_sampling_rate_hz"]
    base_arrival_sample = 500
    dummy._metadata["trace_p_arrival_sample"] = base_arrival_sample
    dummy._metadata["trace_dt_s"] = 1 / dummy._metadata["trace_sampling_rate_hz"]

    # No resampling
    waveforms, metadata = dummy.get_sample(0, sampling_rate=None)
    assert waveforms.shape[-1] == metadata["trace_npts"]
    assert metadata["trace_sampling_rate_hz"] == base_sampling_rate
    assert metadata["trace_dt_s"] == 1.0 / base_sampling_rate
    assert metadata["trace_p_arrival_sample"] == base_arrival_sample

    # Different resampling rates
    for factor in [0.1, 0.5, 1, 2, 5]:
        waveforms, metadata = dummy.get_sample(
            0, sampling_rate=base_sampling_rate * factor
        )
        assert waveforms.shape[-1] == metadata["trace_npts"]
        assert metadata["trace_sampling_rate_hz"] == base_sampling_rate * factor
        assert metadata["trace_dt_s"] == 1.0 / (base_sampling_rate * factor)
        assert metadata["trace_p_arrival_sample"] == base_arrival_sample * factor

    # Sampling rate defined globally for data set
    factor = 0.5
    dummy.sampling_rate = base_sampling_rate * factor
    waveforms, metadata = dummy.get_sample(0)
    assert waveforms.shape[-1] == metadata["trace_npts"]
    assert metadata["trace_sampling_rate_hz"] == base_sampling_rate * factor
    assert metadata["trace_dt_s"] == 1.0 / (base_sampling_rate * factor)
    assert metadata["trace_p_arrival_sample"] == base_arrival_sample * factor


def test_load_waveform_data_with_sampling_rate():
    # Checks that preloading waveform data works with sampling rate specified
    dummy = seisbench.data.DummyDataset(cache="trace", sampling_rate=20)
    dummy.preload_waveforms()
    assert len(dummy._waveform_cache) == len(dummy.metadata)


def test_copy():
    dummy1 = seisbench.data.DummyDataset(cache="full")
    dummy1.preload_waveforms()
    dummy2 = dummy1.copy()

    # Metadata and waveforms are copied by value
    assert dummy1._waveform_cache is not dummy2._waveform_cache
    assert dummy1._metadata is not dummy2._metadata

    # Cache entries were copied by reference
    assert len(dummy1._waveform_cache) == len(dummy2._waveform_cache)
    for key in dummy1._waveform_cache.keys():
        assert dummy1._waveform_cache[key] is dummy2._waveform_cache[key]


def test_filter_inplace():
    dummy = seisbench.data.DummyDataset(cache="trace")
    dummy.preload_waveforms()
    org_len = len(dummy)
    mask = np.zeros(len(dummy), dtype=bool)
    mask[50:] = True

    dummy2 = dummy.filter(mask, inplace=False)
    # dummy was not modified inplace
    assert len(dummy) == org_len
    assert len(dummy._waveform_cache) == org_len
    # dummy2 has correct length and had correct cache eviction
    assert len(dummy2) == np.sum(mask)
    assert len(dummy2._waveform_cache) == np.sum(mask)

    dummy.filter(mask, inplace=True)
    # dummy was modified inplace
    assert len(dummy) == np.sum(mask)
    assert len(dummy._waveform_cache) == np.sum(mask)


def test_splitting():
    dummy = seisbench.data.DummyDataset()

    if "split" in dummy._metadata.columns:
        dummy._metadata.drop(columns="split", inplace=True)

    # Fails if no split is defined
    with pytest.raises(ValueError):
        dummy.train()

    # Test splitting works
    splits = 60 * ["train"] + 10 * ["dev"] + 30 * ["test"]
    dummy._metadata["split"] = splits
    train, dev, test = dummy.train_dev_test()
    assert len(train) == 60
    assert len(dev) == 10
    assert len(test) == 30


def test_bucketer_type(tmp_path: Path):
    data_path = tmp_path / "bucketer_type"
    writer = seisbench.data.WaveformDataWriter(
        data_path / "metadata.csv", data_path / "waveforms.hdf5"
    )

    with pytest.raises(TypeError):
        writer.bucketer = 10

    # Don't use a bucketer
    writer.bucketer = None

    # Use a Geometric bucketer
    writer.bucketer = seisbench.data.GeometricBucketer()

    with pytest.raises(ValueError):
        writer.bucket_size = 0

    writer.bucket_size = 1024


def test_geometric_bucketer():
    # Split is false
    bucketer = seisbench.data.GeometricBucketer(
        minbucket=100, factor=1.2, splits=False, track_channels=False, axis=-1
    )

    # Minimum bucket
    assert "0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # First bucket
    assert "1" == bucketer.get_bucket({}, np.ones((3, 101)))

    # Later bucket
    assert "10" == bucketer.get_bucket({}, np.ones((3, int(100 * 1.2 ** 9 + 1))))

    # Ignores split
    assert "0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))

    # Split is true
    bucketer = seisbench.data.GeometricBucketer(
        minbucket=100, factor=1.2, splits=True, track_channels=False, axis=-1
    )

    # Minimum bucket
    assert "train0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))

    # First bucket
    assert "dev1" == bucketer.get_bucket({"split": "dev"}, np.ones((3, 101)))

    # Later bucket
    assert "test10" == bucketer.get_bucket(
        {"split": "test"}, np.ones((3, int(100 * 1.2 ** 9 + 1)))
    )

    # Ignores missing split
    assert "0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # track_channels is true
    bucketer = seisbench.data.GeometricBucketer(
        minbucket=100, factor=1.2, splits=False, track_channels=True, axis=-1
    )

    # Minimum bucket
    assert "(3)_0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # First bucket
    assert "(1)_1" == bucketer.get_bucket({}, np.ones((1, 101)))

    # Later bucket
    assert "(3)_10" == bucketer.get_bucket({}, np.ones((3, int(100 * 1.2 ** 9 + 1))))

    # Ignores split
    assert "(3)_0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))


def test_pack_arrays():
    arrays = [np.random.rand(5, 2, 3), np.random.rand(4, 4, 2)]
    output, locations = seisbench.data.WaveformDataWriter._pack_arrays(arrays)

    assert output.shape == (2, 5, 4, 3)
    assert (output[0, :5, :2, :3] == arrays[0]).all()
    assert (output[1, :4, :4, :2] == arrays[1]).all()
    assert locations[0] == "0,:5,:2,:3"
    assert locations[1] == "1,:4,:4,:2"


def test_bucketer_cache(tmp_path: Path):
    data_path = tmp_path / "bucketer_cache"
    with seisbench.data.WaveformDataWriter(
        data_path / "metadata.csv", data_path / "waveforms.hdf5"
    ) as writer:
        writer.bucket_size = 10
        writer.bucketer.track_channels = False

        # Traces are kept in bucket
        for i in range(9):
            writer.add_trace({"split": "test"}, np.ones((3, 12)))

        assert len(writer._cache) == 1
        assert len(writer._cache["test0"]) == 9

        # Traces are kept in bucket
        for i in range(9):
            writer.add_trace({"split": "train"}, np.ones((3, 12)))

        assert len(writer._cache) == 2
        assert len(writer._cache["test0"]) == 9
        assert len(writer._cache["train0"]) == 9

        # Traces are written out
        writer.add_trace({"split": "train"}, np.ones((3, 12)))
        assert len(writer._cache["test0"]) == 9
        assert len(writer._cache["train0"]) == 0

        # Remaining traces are written out
        writer.flush_hdf5()
        assert len(writer._cache["test0"]) == 0

    # Inspect output files
    with h5py.File(data_path / "waveforms.hdf5", "r") as f:
        assert len(f["data"].keys()) == 2
        assert f["data/bucket0"].shape == (10, 3, 12)
        assert f["data/bucket1"].shape == (9, 3, 12)

    metadata = pd.read_csv(data_path / "metadata.csv")
    for trace_name in metadata["trace_name"].values:
        assert trace_name.startswith("bucket")
        assert trace_name[7] == "$"


def test_parse_location():
    x = np.random.rand(100, 90, 70)
    assert (x[0] == x[seisbench.data.WaveformDataset._parse_location("0")]).all()

    assert (x[:5] == x[seisbench.data.WaveformDataset._parse_location(":5")]).all()

    assert (x[1:] == x[seisbench.data.WaveformDataset._parse_location("1:")]).all()

    assert (x[3:10] == x[seisbench.data.WaveformDataset._parse_location("3:10")]).all()

    assert (x[-10] == x[seisbench.data.WaveformDataset._parse_location("-10")]).all()

    assert (
        x[1:99:3] == x[seisbench.data.WaveformDataset._parse_location("1:99:3")]
    ).all()

    assert (
        x[0, 1:10, :-5]
        == x[seisbench.data.WaveformDataset._parse_location("0, 1:10, :-5")]
    ).all()

    assert (
        x[0, 1, 2] == x[seisbench.data.WaveformDataset._parse_location("0,1,2")]
    ).all()


def test_writer_padding_reader_unpadding(tmp_path: Path):
    trace1 = np.random.rand(3, 50)
    trace2 = np.random.rand(3, 60)
    trace3 = np.random.rand(3, 200)
    trace4 = np.random.rand(3, 201)

    # Write output where padding actually occurs
    data_path = tmp_path / "padding_unpadding"
    with seisbench.data.WaveformDataWriter(
        data_path / "metadata.csv", data_path / "waveforms.hdf5"
    ) as writer:
        writer.data_format["component_order"] = "ZNE"
        writer.bucketer = seisbench.data.GeometricBucketer(minbucket=100, factor=1.2)

        writer.add_trace({}, trace1)
        writer.add_trace({}, trace2)
        writer.add_trace({}, trace3)
        writer.add_trace({}, trace4)

    # Inspect output files
    # Check that both values match and padding was removed again
    data = seisbench.data.WaveformDataset(data_path)

    assert (data.get_waveforms(0) == trace1).all()
    assert (data.get_waveforms(1) == trace2).all()
    assert (data.get_waveforms(2) == trace3).all()
    assert (data.get_waveforms(3) == trace4).all()


def test_get_waveforms_component_orders():
    dummy = seisbench.data.DummyDataset(component_order="ZNE", dimension_order="NCW")
    wv_org = dummy.get_waveforms(0)
    wv_org1 = dummy.get_waveforms(1)

    dummy.component_order = "ZNEH"
    dummy.missing_components = "pad"
    wv = dummy.get_waveforms(0)
    assert wv.shape[0] == 4
    assert (wv_org == wv[:3]).all()
    assert (wv[3] == 0).all()

    dummy.component_order = "ZNEH"
    dummy.missing_components = "copy"
    wv = dummy.get_waveforms(0)
    assert wv.shape[0] == 4
    assert (wv_org == wv[:3]).all()
    assert (wv[3] == wv_org[0]).all()

    dummy.component_order = "ZNEH"
    dummy.missing_components = "ignore"
    wv = dummy.get_waveforms(0)
    assert wv.shape[0] == 3
    assert (wv_org == wv).all()

    dummy._metadata["trace_component_order"].values[1] = "NEZ"
    dummy.component_order = "ZNE"
    dummy.missing_components = "ignore"

    wv = dummy.get_waveforms(0)
    assert wv.shape[0] == 3
    assert (wv_org == wv).all()

    wv = dummy.get_waveforms(1)
    assert wv.shape[0] == 3
    assert (wv_org1[[2, 0, 1]] == wv).all()


def test_get_waveform_component_order_mismatching():
    # Tests different strategies for mismatching traces
    dummy = seisbench.data.DummyDataset(component_order="ZNE", dimension_order="NCW")
    dummy._metadata["trace_component_order"].values[1] = "Z"

    dummy.missing_components = "pad"
    wv = dummy.get_waveforms([0, 1])
    assert wv.shape[1] == 3

    dummy.missing_components = "copy"
    wv = dummy.get_waveforms([0, 1])
    assert wv.shape[1] == 3

    dummy.missing_components = "ignore"
    with pytest.raises(ValueError) as e:
        dummy.get_waveforms([0, 1])
        assert "Requested traces with mixed number of components." in str(e)
