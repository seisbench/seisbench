import seisbench.data
import seisbench.util.region as region

import numpy as np
import pytest
import logging
from pathlib import Path


def test_get_order_mapping():
    # Test ordering and list/string format
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_order_mapping("ZNE", "ZNE")
    assert [1, 2, 0] == seisbench.data.WaveformDataset._get_order_mapping("ZNE", "NEZ")
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_order_mapping(
        ["Z", "N", "E"], "ZNE"
    )
    assert [0, 1, 2] == seisbench.data.WaveformDataset._get_order_mapping(
        "ZNE", ["Z", "N", "E"]
    )
    assert [0, 2, 1] == seisbench.data.WaveformDataset._get_order_mapping(
        ["Z", "E", "N"], ["Z", "N", "E"]
    )

    # Test failures
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_order_mapping("ZNE", "Z")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_order_mapping("ZNE", "ZZE")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_order_mapping("ZEZ", "ZNE")
    with pytest.raises(ValueError):
        seisbench.data.WaveformDataset._get_order_mapping("ZNE", "ZRT")


def test_pad_packed_sequence():
    seq = [np.ones((5, 1)), np.ones((6, 3)), np.ones((1, 2)), np.ones((7, 2))]

    packed = seisbench.data.WaveformDataset._pad_packed_sequence(seq)

    assert packed.shape == (4, 7, 3)
    assert np.sum(packed == 1) == sum(x.size for x in seq)
    assert np.sum(packed == 0) == packed.size - sum(x.size for x in seq)


def test_lazyload():
    dummy = seisbench.data.DummyDataset(lazyload=True, cache=True)
    assert len(dummy._waveform_cache) == 0

    dummy = seisbench.data.DummyDataset(lazyload=False, cache=True)
    assert len(dummy._waveform_cache) == len(dummy)


def test_filter_and_cache_evict():
    dummy = seisbench.data.DummyDataset(lazyload=False, cache=True)
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
    dummy = seisbench.data.DummyDataset()

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
    dummy = seisbench.data.DummyDataset()

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


def test_lazyload_cache(caplog):
    with caplog.at_level(logging.WARNING):
        seisbench.data.DummyDataset(lazyload=False, cache=False)
    assert "Skipping preloading of waveforms as cache is set to inactive" in caplog.text


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
    seisbench.remote_root = tmp_path  # Ensure test does not modify SeisBench cache

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
    dummy = seisbench.data.DummyDataset(cache=True, lazyload=False, sampling_rate=20)
    assert len(dummy._waveform_cache) == len(dummy.metadata)


def test_copy():
    dummy1 = seisbench.data.DummyDataset(cache=True, lazyload=False)
    dummy2 = dummy1.copy()

    # Metadata and waveforms are copied by value
    assert dummy1._waveform_cache is not dummy2._waveform_cache
    assert dummy1._metadata is not dummy2._metadata

    # Cache entries were copied by reference
    assert len(dummy1._waveform_cache) == len(dummy2._waveform_cache)
    for key in dummy1._waveform_cache.keys():
        assert dummy1._waveform_cache[key] is dummy2._waveform_cache[key]


def test_filter_inplace():
    dummy = seisbench.data.DummyDataset(cache=True, lazyload=False)
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
