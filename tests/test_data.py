import logging
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

import seisbench.data as sbd
import seisbench.util.region as region


def test_get_dimension_mapping():
    # For legacy reasons test dimensions are called "ZNE".
    # However, component mapping is now handled by _get_component_mapping.

    # Test ordering and list/string format
    assert [0, 1, 2] == sbd.WaveformDataset._get_dimension_mapping("ZNE", "ZNE")
    assert [1, 2, 0] == sbd.WaveformDataset._get_dimension_mapping("ZNE", "NEZ")
    assert [0, 1, 2] == sbd.WaveformDataset._get_dimension_mapping(
        ["Z", "N", "E"], "ZNE"
    )
    assert [0, 1, 2] == sbd.WaveformDataset._get_dimension_mapping(
        "ZNE", ["Z", "N", "E"]
    )
    assert [0, 2, 1] == sbd.WaveformDataset._get_dimension_mapping(
        ["Z", "E", "N"], ["Z", "N", "E"]
    )

    # Test failures
    with pytest.raises(ValueError):
        sbd.WaveformDataset._get_dimension_mapping("ZNE", "Z")
    with pytest.raises(ValueError):
        sbd.WaveformDataset._get_dimension_mapping("ZNE", "ZZE")
    with pytest.raises(ValueError):
        sbd.WaveformDataset._get_dimension_mapping("ZEZ", "ZNE")
    with pytest.raises(ValueError):
        sbd.WaveformDataset._get_dimension_mapping("ZNE", "ZRT")


def test_get_component_mapping():
    # Strategy "pad"
    dummy = sbd.DummyDataset(missing_components="pad")
    assert dummy._get_component_mapping("Z", "ZNE") == [0, 1, 1]
    assert dummy._get_component_mapping("ZE", "ZNE") == [0, 2, 1]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2, 3]

    # Strategy "copy"
    dummy = sbd.DummyDataset(missing_components="copy")
    assert dummy._get_component_mapping("Z", "ZNE") == [0, 0, 0]
    assert dummy._get_component_mapping("N", "ZNE") == [0, 0, 0]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2, 0]

    # Strategy "ignore"
    dummy = sbd.DummyDataset(missing_components="ignore")
    assert dummy._get_component_mapping("Z", "ZNE") == [0]
    assert dummy._get_component_mapping("ZNE", "ZNE") == [0, 1, 2]
    assert dummy._get_component_mapping("ENZ", "ZNE") == [2, 1, 0]
    assert dummy._get_component_mapping("ZNE", "ZNEH") == [0, 1, 2]


def test_preload():
    dummy = sbd.DummyDataset(cache="trace")
    assert len(dummy._waveform_cache) == 0

    dummy.preload_waveforms()
    assert len(dummy._waveform_cache[""]) == len(dummy)


def test_filter_and_cache_evict():
    # Block caching
    dummy = sbd.DummyDataset(cache="full")
    dummy.preload_waveforms()
    blocks = set(dummy.metadata["trace_name"].apply(lambda x: x.split("$")[0]))
    assert len(dummy._waveform_cache[""]) == len(blocks)

    mask = np.arange(len(dummy)) < len(dummy) / 2
    dummy.filter(mask)

    assert len(dummy) == np.sum(mask)  # Correct metadata length
    blocks = set(dummy.metadata["trace_name"].apply(lambda x: x.split("$")[0]))
    assert len(dummy._waveform_cache[""]) == len(blocks)  # Correct cache eviction

    # Trace caching
    dummy = sbd.DummyDataset(cache="trace")
    dummy.preload_waveforms()
    assert len(dummy._waveform_cache[""]) == len(dummy)

    mask = np.arange(len(dummy)) < len(dummy) / 2
    dummy.filter(mask)

    assert len(dummy) == np.sum(mask)  # Correct metadata length
    assert len(dummy._waveform_cache[""]) == len(dummy)  # Correct cache eviction


def test_region_filter():
    # Receiver + Circle domain
    dummy = sbd.DummyDataset()

    lat = 0
    lon = np.linspace(-10, 10, len(dummy))

    dummy.metadata["station_latitude_deg"] = lat
    dummy.metadata["station_longitude_deg"] = lon

    domain = region.CircleDomain(0, 0, 1, 5)

    dummy.region_filter_receiver(domain)

    assert len(dummy) == np.sum(np.logical_and(1 < np.abs(lon), np.abs(lon) < 5))

    # Source + RectangleDomain
    dummy = sbd.DummyDataset()

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
        dummy = sbd.DummyDataset(cache=cache)

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
        dummy = sbd.DummyDataset(cache=cache)

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
        dummy = sbd.DummyDataset(cache=None)
        dummy.preload_waveforms()

    assert "Skipping preload, as cache is disabled." in caplog.text


def test_writer(caplog, tmp_path: Path):
    # Test empty writer
    with sbd.WaveformDataWriter(
        tmp_path / "writer_a" / "metadata.csv", tmp_path / "writer_a" / "waveforms.hdf5"
    ):
        pass

    assert (tmp_path / "writer_a").is_dir()  # Path exists
    assert not any((tmp_path / "writer_a").iterdir())  # Path is empty

    # Test correct write
    with sbd.WaveformDataWriter(
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
        with sbd.WaveformDataWriter(
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
        sbd.WaveformDataset.available_chunks(folder)

    # Empty chunk file prints warning
    folder = tmp_path / "b"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "chunks", "w").close()
    with caplog.at_level(logging.WARNING):
        with pytest.raises(FileNotFoundError):
            sbd.WaveformDataset.available_chunks(folder)
    assert (
        "Found empty chunks file. Using chunk detection from file names." in caplog.text
    )

    # Unchunked dataset
    folder = tmp_path / "c"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "metadata.csv", "w").close()
    open(folder / "waveforms.hdf5", "w").close()
    chunks = sbd.WaveformDataset.available_chunks(folder)
    assert chunks == [""]

    # Chunked dataset detected from chunkfile
    folder = tmp_path / "d"
    folder.mkdir(exist_ok=True, parents=True)
    with open(folder / "chunks", "w") as f:
        f.write("a\nb\nc\n")
    chunks = sbd.WaveformDataset.available_chunks(folder)
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
    chunks = sbd.WaveformDataset.available_chunks(folder)
    assert chunks == ["a", "b", "c"]

    # Chunked dataset with inconsistent chunks
    folder = tmp_path / "f"
    folder.mkdir(exist_ok=True, parents=True)
    open(folder / "metadataa.csv", "w").close()
    open(folder / "waveformsa.hdf5", "w").close()
    open(folder / "metadatab.csv", "w").close()
    open(folder / "waveformsc.hdf5", "w").close()
    with caplog.at_level(logging.WARNING):
        chunks = sbd.WaveformDataset.available_chunks(folder)
    assert chunks == ["a"]
    assert "Found metadata but no waveforms for chunks" in caplog.text
    assert "Found waveforms but no metadata for chunks" in caplog.text


def test_chunked_loading():
    chunk0 = sbd.ChunkedDummyDataset(chunks=["0"])
    chunk1 = sbd.ChunkedDummyDataset(chunks=["1"])
    chunk01 = sbd.ChunkedDummyDataset()

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
        "seisbench.cache_data_root", tmp_path
    ):  # Ensure test does not modify SeisBench cache

        class MockDataset(sbd.WaveformBenchmarkDataset):
            def __init__(self, **kwargs):
                super().__init__(citation="", **kwargs)

            def _download_dataset(self, writer):
                raise ValueError("Called without chunks")

        class ChunkedMockDataset(sbd.WaveformBenchmarkDataset):
            def __init__(self, **kwargs):
                super().__init__(citation="", **kwargs)

            def _download_dataset(self, writer, chunk):
                raise ValueError("Called with chunks")

        # Note: This would raise a TypeError when called with the chunk parameter
        with pytest.raises(ValueError) as e:
            MockDataset(compile_from_source=True)
        assert "Called without chunks" in str(e)

        # Note: This would raise a TypeError when called without the chunk parameter
        with pytest.raises(ValueError) as e:
            ChunkedMockDataset(compile_from_source=True)
        assert "Called with chunks" in str(e)


def test_unify_sampling_rate(caplog):
    dummy = sbd.DummyDataset()
    del dummy._metadata["trace_sampling_rate_hz"]
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert "Sampling rate not specified in data set." in caplog.text

    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._data_format["sampling_rate"] = 40.0
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
        in caplog.text
    )

    dummy = sbd.DummyDataset()
    caplog.clear()
    del dummy._metadata["trace_sampling_rate_hz"]
    dummy._metadata["trace_dt_s"] = 1 / 20.0
    dummy._data_format["sampling_rate"] = 40.0
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Inconsistent sampling rates between metadata and data_format. Using values from metadata."
        in caplog.text
    )

    dummy = sbd.DummyDataset()
    caplog.clear()
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
    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata["trace_dt_s"] = 1 / 20.00001
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""

    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata.loc[:20, "trace_sampling_rate_hz"] = np.nan
    dummy._metadata["trace_dt_s"] = 1 / 20.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""

    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata.loc[:20, "trace_sampling_rate_hz"] = np.nan
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert "Found some traces with undefined sampling rates." in caplog.text

    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata.loc[:20, "trace_sampling_rate_hz"] = 40.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert (
        "Data set contains mixed sampling rate, but no sampling rate was specified for the dataset."
        in caplog.text
    )

    dummy = sbd.DummyDataset()
    caplog.clear()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    del dummy._data_format["sampling_rate"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_sampling_rate()
    assert caplog.text == ""


def test_unify_component_order(caplog):
    # Component order not specified
    dummy = sbd.DummyDataset()
    del dummy._metadata["trace_component_order"]
    del dummy._data_format["component_order"]
    with caplog.at_level(logging.WARNING):
        dummy._unify_component_order()
    assert "Component order not specified in data set." in caplog.text

    # Component order inconsistent
    dummy = sbd.DummyDataset()
    caplog.clear()
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
    dummy = sbd.DummyDataset()
    caplog.clear()
    del dummy._metadata["trace_component_order"]
    dummy._data_format["component_order"] = "ZNE"
    with caplog.at_level(logging.WARNING):
        dummy._unify_component_order()
    assert len(caplog.text) == 0
    assert (dummy._metadata["trace_component_order"] == "ZNE").all()


@pytest.mark.parametrize("zerophase_resample", [True, False])
def test_resample(zerophase_resample):
    dummy = sbd.DummyDataset(
        metadata_cache=False, zerophase_resample=zerophase_resample
    )
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._metadata.loc[0, "trace_sampling_rate_hz"] = np.nan

    # NaN sampling rate raises no error if sampling rate is None, but raises error otherwise
    dummy.get_waveforms(idx=0)
    with pytest.raises(ValueError):
        dummy.get_waveforms(idx=0, sampling_rate=20)

    # Incorrect dimension order raises an issue if actual resampling is required
    dummy = sbd.DummyDataset()
    dummy._metadata["trace_sampling_rate_hz"] = 20.0
    dummy._data_format["dimension_order"] = dummy._data_format[
        "dimension_order"
    ].replace("W", "X")
    dummy.get_waveforms(idx=0, sampling_rate=20)
    with pytest.raises(ValueError):
        dummy.get_waveforms(idx=0, sampling_rate=40)

    # Correct cases have expected length
    dummy = sbd.DummyDataset()
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


def test_resample_empty_trace(caplog):
    # Check that resampling does not crash for empty trace and issues a warning
    dummy = sbd.DummyDataset()
    waveform = np.ones((3, 0))
    with caplog.at_level(logging.INFO):
        dummy._resample(waveform, 19, 20)
    assert "Trying to resample empty trace, skipping resampling." in caplog.text


@pytest.mark.parametrize(
    "metadata_cache",
    [True, False],
)
def test_get_sample(metadata_cache):
    # Checks that the parameters are correctly overwritten, when the sampling_rate is changed
    dummy = sbd.DummyDataset(metadata_cache=metadata_cache)
    dummy.sampling_rate = None

    base_sampling_rate = dummy.metadata.iloc[0]["trace_sampling_rate_hz"]
    base_arrival_sample = 500
    dummy._metadata["trace_p_arrival_sample"] = base_arrival_sample
    dummy._metadata["trace_dt_s"] = 1 / dummy._metadata["trace_sampling_rate_hz"]
    dummy._rebuild_metadata_cache()

    # No resampling
    waveforms, metadata = dummy.get_sample(0, sampling_rate=None)
    assert waveforms.shape[-1] == metadata["trace_npts"]
    assert metadata["trace_sampling_rate_hz"] == base_sampling_rate
    assert metadata["trace_dt_s"] == 1.0 / base_sampling_rate
    assert metadata["trace_p_arrival_sample"] == base_arrival_sample
    assert np.all(waveforms == dummy.get_waveforms(0))

    # Different resampling rates
    for factor in [0.1, 0.5, 1, 2, 5]:
        waveforms, metadata = dummy.get_sample(
            0, sampling_rate=base_sampling_rate * factor
        )
        assert waveforms.shape[-1] == metadata["trace_npts"]
        assert metadata["trace_sampling_rate_hz"] == base_sampling_rate * factor
        assert metadata["trace_dt_s"] == 1.0 / (base_sampling_rate * factor)
        assert metadata["trace_p_arrival_sample"] == base_arrival_sample * factor
        assert np.all(
            waveforms
            == dummy.get_waveforms(0, sampling_rate=base_sampling_rate * factor)
        )

    # Sampling rate defined globally for data set
    factor = 0.5
    dummy.sampling_rate = base_sampling_rate * factor
    waveforms, metadata = dummy.get_sample(0)
    assert waveforms.shape[-1] == metadata["trace_npts"]
    assert metadata["trace_sampling_rate_hz"] == base_sampling_rate * factor
    assert metadata["trace_dt_s"] == 1.0 / (base_sampling_rate * factor)
    assert metadata["trace_p_arrival_sample"] == base_arrival_sample * factor
    assert np.all(waveforms == dummy.get_waveforms(0))


def test_load_waveform_data_with_sampling_rate():
    # Checks that preloading waveform data works with sampling rate specified
    dummy = sbd.DummyDataset(cache="trace", sampling_rate=20)
    dummy.preload_waveforms()
    assert len(dummy._waveform_cache[""]) == len(dummy.metadata)


def test_copy():
    dummy1 = sbd.DummyDataset(cache="full")
    dummy1.preload_waveforms()
    dummy2 = dummy1.copy()

    # Metadata and waveforms are copied by value
    assert dummy1._waveform_cache is not dummy2._waveform_cache
    for chunk in dummy1._waveform_cache.keys():
        assert dummy1._waveform_cache[chunk] is not dummy2._waveform_cache[chunk]
    assert dummy1._metadata is not dummy2._metadata

    # Cache entries were copied by reference
    assert len(dummy1._waveform_cache) == len(dummy2._waveform_cache)
    for chunk in dummy1._waveform_cache.keys():
        assert len(dummy1._waveform_cache[chunk]) == len(dummy2._waveform_cache[chunk])
        for key in dummy1._waveform_cache[chunk].keys():
            assert (
                dummy1._waveform_cache[chunk][key] is dummy2._waveform_cache[chunk][key]
            )


def test_filter_inplace():
    dummy = sbd.DummyDataset(cache="trace")
    dummy.preload_waveforms()
    org_len = len(dummy)
    mask = np.zeros(len(dummy), dtype=bool)
    mask[50:] = True

    dummy2 = dummy.filter(mask, inplace=False)
    # dummy was not modified inplace
    assert len(dummy) == org_len
    assert len(dummy._waveform_cache[""]) == org_len
    # dummy2 has correct length and had correct cache eviction
    assert len(dummy2) == np.sum(mask)
    assert len(dummy2._waveform_cache[""]) == np.sum(mask)

    dummy.filter(mask, inplace=True)
    # dummy was modified inplace
    assert len(dummy) == np.sum(mask)
    assert len(dummy._waveform_cache[""]) == np.sum(mask)


def test_splitting():
    dummy = sbd.DummyDataset()

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
    writer = sbd.WaveformDataWriter(
        data_path / "metadata.csv", data_path / "waveforms.hdf5"
    )

    with pytest.raises(TypeError):
        writer.bucketer = 10

    # Don't use a bucketer
    writer.bucketer = None

    # Use a Geometric bucketer
    writer.bucketer = sbd.GeometricBucketer()

    with pytest.raises(ValueError):
        writer.bucket_size = 0

    writer.bucket_size = 1024


def test_geometric_bucketer():
    # Split is false
    bucketer = sbd.GeometricBucketer(
        minbucket=100, factor=1.2, splits=False, track_channels=False, axis=-1
    )

    # Minimum bucket
    assert "0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # First bucket
    assert "1" == bucketer.get_bucket({}, np.ones((3, 101)))

    # Later bucket
    assert "10" == bucketer.get_bucket({}, np.ones((3, int(100 * 1.2**9 + 1))))

    # Ignores split
    assert "0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))

    # Split is true
    bucketer = sbd.GeometricBucketer(
        minbucket=100, factor=1.2, splits=True, track_channels=False, axis=-1
    )

    # Minimum bucket
    assert "train0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))

    # First bucket
    assert "dev1" == bucketer.get_bucket({"split": "dev"}, np.ones((3, 101)))

    # Later bucket
    assert "test10" == bucketer.get_bucket(
        {"split": "test"}, np.ones((3, int(100 * 1.2**9 + 1)))
    )

    # Ignores missing split
    assert "0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # track_channels is true
    bucketer = sbd.GeometricBucketer(
        minbucket=100, factor=1.2, splits=False, track_channels=True, axis=-1
    )

    # Minimum bucket
    assert "(3)_0" == bucketer.get_bucket({}, np.ones((3, 99)))

    # First bucket
    assert "(1)_1" == bucketer.get_bucket({}, np.ones((1, 101)))

    # Later bucket
    assert "(3)_10" == bucketer.get_bucket({}, np.ones((3, int(100 * 1.2**9 + 1))))

    # Ignores split
    assert "(3)_0" == bucketer.get_bucket({"split": "train"}, np.ones((3, 99)))


def test_pack_arrays():
    arrays = [np.random.rand(5, 2, 3), np.random.rand(4, 4, 2)]
    output, locations = sbd.WaveformDataWriter._pack_arrays(arrays)

    assert output.shape == (2, 5, 4, 3)
    assert (output[0, :5, :2, :3] == arrays[0]).all()
    assert (output[1, :4, :4, :2] == arrays[1]).all()
    assert locations[0] == "0,:5,:2,:3"
    assert locations[1] == "1,:4,:4,:2"


def test_bucketer_cache(tmp_path: Path):
    data_path = tmp_path / "bucketer_cache"
    with sbd.WaveformDataWriter(
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
    assert (x[0] == x[sbd.WaveformDataset._parse_location("0")]).all()

    assert (x[:5] == x[sbd.WaveformDataset._parse_location(":5")]).all()

    assert (x[1:] == x[sbd.WaveformDataset._parse_location("1:")]).all()

    assert (x[3:10] == x[sbd.WaveformDataset._parse_location("3:10")]).all()

    assert (x[-10] == x[sbd.WaveformDataset._parse_location("-10")]).all()

    assert (x[1:99:3] == x[sbd.WaveformDataset._parse_location("1:99:3")]).all()

    assert (
        x[0, 1:10, :-5] == x[sbd.WaveformDataset._parse_location("0, 1:10, :-5")]
    ).all()

    assert (x[0, 1, 2] == x[sbd.WaveformDataset._parse_location("0,1,2")]).all()


def test_writer_padding_reader_unpadding(tmp_path: Path):
    trace1 = np.random.rand(3, 50)
    trace2 = np.random.rand(3, 60)
    trace3 = np.random.rand(3, 200)
    trace4 = np.random.rand(3, 201)

    # Write output where padding actually occurs
    data_path = tmp_path / "padding_unpadding"
    with sbd.WaveformDataWriter(
        data_path / "metadata.csv", data_path / "waveforms.hdf5"
    ) as writer:
        writer.data_format["component_order"] = "ZNE"
        writer.bucketer = sbd.GeometricBucketer(minbucket=100, factor=1.2)

        writer.add_trace({}, trace1)
        writer.add_trace({}, trace2)
        writer.add_trace({}, trace3)
        writer.add_trace({}, trace4)

    # Inspect output files
    # Check that both values match and padding was removed again
    data = sbd.WaveformDataset(data_path)

    assert (data.get_waveforms(0) == trace1).all()
    assert (data.get_waveforms(1) == trace2).all()
    assert (data.get_waveforms(2) == trace3).all()
    assert (data.get_waveforms(3) == trace4).all()


def test_get_waveforms_component_orders():
    dummy = sbd.DummyDataset(component_order="ZNE", dimension_order="NCW")
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
    dummy._rebuild_metadata_cache()
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
    dummy = sbd.DummyDataset(component_order="ZNE", dimension_order="NCW")
    dummy._metadata["trace_component_order"].values[1] = "Z"
    dummy._rebuild_metadata_cache()

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


def test_multi_waveform_dataset_type():
    dummy = sbd.DummyDataset()

    # Works
    sbd.MultiWaveformDataset([dummy])

    # Not a list as input
    with pytest.raises(TypeError):
        sbd.MultiWaveformDataset(dummy)

    # List entry not a WaveformDataset
    with pytest.raises(TypeError):
        sbd.MultiWaveformDataset([dummy, 2])

    # Empty list
    with pytest.raises(ValueError):
        sbd.MultiWaveformDataset([])


def test_test_attribute_equal():
    datasets = [
        sbd.DummyDataset(cache=None),
        sbd.DummyDataset(cache=None),
    ]
    assert sbd.MultiWaveformDataset._test_attribute_equal(datasets, "cache")

    datasets = [
        sbd.DummyDataset(cache="full"),
        sbd.DummyDataset(cache=None),
    ]
    assert not sbd.MultiWaveformDataset._test_attribute_equal(datasets, "cache")


def test_multi_waveform_dataset_split_warning(caplog):
    datasets = [sbd.DummyDataset(), sbd.DummyDataset()]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Combining datasets with and without split." not in caplog.text

    caplog.clear()

    del datasets[0].metadata["split"]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Combining datasets with and without split." in caplog.text


def test_multi_waveform_dataset_cache_warning(caplog):
    datasets = [
        sbd.DummyDataset(cache=None),
        sbd.DummyDataset(cache=None),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent caching strategies." not in caplog.text

    caplog.clear()

    datasets = [
        sbd.DummyDataset(cache="full"),
        sbd.DummyDataset(cache=None),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent caching strategies." in caplog.text


def test_multi_waveform_dataset_sampling_rate(caplog):
    datasets = [
        sbd.DummyDataset(sampling_rate=100),
        sbd.DummyDataset(sampling_rate=100),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found mismatching sampling rates between datasets." not in caplog.text

    datasets = [
        sbd.DummyDataset(sampling_rate=100),
        sbd.DummyDataset(sampling_rate=200),
    ]
    with caplog.at_level(logging.WARNING):
        multi = sbd.MultiWaveformDataset(datasets)
    assert "Found mismatching sampling rates between datasets." in caplog.text
    assert all(x.sampling_rate is None for x in multi.datasets)


def test_multi_waveform_dataset_component_warning(caplog):
    datasets = [
        sbd.DummyDataset(component_order="ZNE"),
        sbd.DummyDataset(component_order="ZNE"),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent component orders." not in caplog.text

    datasets = [
        sbd.DummyDataset(component_order="ZNE"),
        sbd.DummyDataset(component_order="NEZ"),
    ]
    with caplog.at_level(logging.WARNING):
        multi = sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent component orders." in caplog.text
    assert all(x.component_order == "ZNE" for x in multi.datasets)


def test_multi_waveform_dataset_dimension_warning(caplog):
    datasets = [
        sbd.DummyDataset(dimension_order="NCW"),
        sbd.DummyDataset(dimension_order="NCW"),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent dimension orders." not in caplog.text

    datasets = [
        sbd.DummyDataset(dimension_order="NCW"),
        sbd.DummyDataset(dimension_order="NWC"),
    ]
    with caplog.at_level(logging.WARNING):
        multi = sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent dimension orders." in caplog.text
    assert all(x.dimension_order == "NCW" for x in multi.datasets)


def test_multi_waveform_dataset_missing_component_warning(caplog):
    datasets = [
        sbd.DummyDataset(missing_components="pad"),
        sbd.DummyDataset(missing_components="pad"),
    ]
    with caplog.at_level(logging.WARNING):
        sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent missing_components." not in caplog.text

    datasets = [
        sbd.DummyDataset(missing_components="pad"),
        sbd.DummyDataset(missing_components="copy"),
    ]
    with caplog.at_level(logging.WARNING):
        multi = sbd.MultiWaveformDataset(datasets)
    assert "Found inconsistent missing_components." in caplog.text
    assert all(x.dimension_order == "NCW" for x in multi.datasets)


def test_multi_waveform_dataset_resolve_idx():
    datasets = [sbd.DummyDataset(), sbd.DummyDataset()]
    multi = sbd.MultiWaveformDataset(datasets)

    assert multi._resolve_idx(0) == (0, 0)
    assert multi._resolve_idx(55) == (0, 55)
    assert multi._resolve_idx(-150) == (0, 50)
    assert multi._resolve_idx(100) == (1, 0)
    assert multi._resolve_idx(114) == (1, 14)
    assert multi._resolve_idx(-70) == (1, 30)

    with pytest.raises(IndexError):
        multi._resolve_idx(200)

    with pytest.raises(IndexError):
        multi._resolve_idx(-201)

    with pytest.raises(IndexError):
        multi._resolve_idx(205)

    with pytest.raises(IndexError):
        multi._resolve_idx(-205)


def test_multi_waveform_dataset_test_get_sample():
    datasets = [sbd.DummyDataset(), sbd.DummyDataset()]
    multi = sbd.MultiWaveformDataset(datasets)

    waveform1, metadata1 = datasets[0].get_sample(10)
    waveform2, metadata2 = multi.get_sample(10)

    assert (waveform1 == waveform2).all()
    assert metadata1 == metadata2


def test_multi_waveform_dataset_split_mask():
    np.random.seed(42)
    dummy = sbd.DummyDataset()
    multi = sbd.MultiWaveformDataset(list(dummy.train_dev_test()))

    mask = np.random.rand(100) > 0.5

    mask1, mask2, mask3 = multi._split_mask(mask)

    assert (mask1 == mask[:60]).all()
    assert (mask2 == mask[60:70]).all()
    assert (mask3 == mask[70:]).all()


def test_pad_pack_along_axis():
    np.random.seed(42)

    x = np.random.rand(5, 2, 3)
    y = np.random.rand(2, 3, 2)

    packed = sbd.MultiWaveformDataset._pad_pack_along_axis([x, y], axis=1)
    assert packed.shape == (5, 5, 3)
    assert (packed[:5, :2, :3] == x).all()
    assert (packed[:2, 2:5, :2] == y).all()


def test_multi_waveform_dataset_test_get_waveforms():
    datasets = [sbd.DummyDataset(), sbd.DummyDataset()]
    multi = sbd.MultiWaveformDataset(datasets)

    # Single idx
    waveform1 = datasets[0].get_waveforms(10)
    waveform2 = multi.get_waveforms(10)
    assert (waveform1 == waveform2).all()

    # Single idx
    waveform1 = datasets[1].get_waveforms(10)
    waveform2 = multi.get_waveforms(110)
    assert (waveform1 == waveform2).all()

    # Multiple idx
    waveform1 = datasets[0].get_waveforms([1, 10])
    waveform2 = datasets[1].get_waveforms(11)
    waveform3 = multi.get_waveforms([1, 10, 111])
    assert (waveform3[:2] == waveform1).all()
    assert (waveform3[2] == waveform2).all()

    # Mask - empty for one datasets
    mask = np.zeros(len(multi), dtype=bool)
    mask[10:20] = True
    waveform1 = datasets[0].get_waveforms(mask=mask[:100])
    waveform2 = multi.get_waveforms(mask=mask)
    assert (waveform1 == waveform2).all()

    # Mask - non-empty for both datasets
    mask = np.zeros(len(multi), dtype=bool)
    mask[10:20] = True
    mask[120:130] = True
    waveform1 = datasets[0].get_waveforms(mask=mask[:100])
    waveform2 = datasets[1].get_waveforms(mask=mask[100:])
    waveform3 = multi.get_waveforms(mask=mask)
    assert (waveform1 == waveform3[:10]).all()
    assert (waveform2 == waveform3[10:]).all()


def test_multi_waveform_dataset_filter():
    np.random.seed(42)
    dummy = sbd.DummyDataset()
    multi = sbd.MultiWaveformDataset(list(dummy.train_dev_test()))

    # Empty output
    mask = np.zeros(100, dtype=bool)
    filtered = multi.filter(mask, inplace=False)
    assert len(filtered) == 0
    assert len(multi) == 100

    # Random mask
    mask = np.random.rand(len(multi)) > 0.5
    filtered = multi.filter(mask, inplace=False)
    assert len(filtered) == np.sum(mask)
    assert len(multi) == 100

    # Random mask - Inplace
    mask = np.random.rand(len(multi)) > 0.5
    multi.filter(mask, inplace=True)
    assert len(multi) == np.sum(mask)


def test_waveform_dataset_add():
    dummy = sbd.DummyDataset()
    train, dev, test = dummy.train_dev_test()

    with pytest.raises(TypeError):
        _ = train + 1

    td = train + dev
    assert len(td.datasets) == 2
    assert len(td.datasets[0]) == len(train)
    assert len(td.datasets[1]) == len(dev)

    ttd = test + td
    assert len(ttd.datasets) == 3
    assert len(ttd.datasets[0]) == len(test)
    assert len(ttd.datasets[1]) == len(train)
    assert len(ttd.datasets[2]) == len(dev)

    tdt = td + test
    assert len(tdt.datasets) == 3
    assert len(tdt.datasets[0]) == len(train)
    assert len(tdt.datasets[1]) == len(dev)
    assert len(tdt.datasets[2]) == len(test)


def test_get_idx_from_trace_name():
    np.random.seed(42)

    dummy = sbd.DummyDataset()
    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[0]) == 0
    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[90]) == 90

    mask = np.random.rand(len(dummy)) > 0.5
    dummy.filter(mask, inplace=True)

    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[0]) == 0
    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[20]) == 20


def test_get_idx_from_trace_name_multi():
    # Test for MultiWaveformDataset
    np.random.seed(42)

    dummy_ = sbd.DummyDataset()
    train, dev, test = dummy_.train_dev_test()
    dummy = train + dev + test

    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[0]) == 0
    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[90]) == 90

    mask = np.random.rand(len(dummy)) > 0.5
    dummy.filter(mask, inplace=True)

    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[0]) == 0
    assert dummy.get_idx_from_trace_name(dummy["trace_name"].values[20]) == 20


def test_trace_name_to_idx_dict():
    dummy = sbd.DummyDataset()
    metadata = pd.DataFrame(
        [
            {"trace_name": "a", "trace_chunk": ""},
            {"trace_name": "b", "trace_chunk": ""},
            {"trace_name": "b", "trace_chunk": "1"},
        ]
    )
    dummy._metadata = metadata  # Overwrite metadata
    dummy._build_trace_name_to_idx_dict()
    assert not dummy._trace_identification_warning_issued
    assert len(dummy._trace_name_to_idx) == 4
    assert len(dummy._trace_name_to_idx["name"]) == 2  # 1 collision
    assert len(dummy._trace_name_to_idx["name_chunk"]) == 3  # 0 collisions
    assert len(dummy._trace_name_to_idx["name_dataset"]) == 0  # NA
    assert len(dummy._trace_name_to_idx["name_chunk_dataset"]) == 0  # NA

    # Only test unique answers
    assert dummy.get_idx_from_trace_name("a") == 0
    assert dummy.get_idx_from_trace_name("b", "") == 1
    with pytest.raises(KeyError):
        dummy.get_idx_from_trace_name("a", "1")

    metadata = pd.DataFrame(
        [
            {"trace_name": "a", "trace_chunk": "", "trace_dataset": "0"},
            {"trace_name": "b", "trace_chunk": "", "trace_dataset": "0"},
            {"trace_name": "b", "trace_chunk": "1", "trace_dataset": "0"},
            {"trace_name": "b", "trace_chunk": "1", "trace_dataset": "1"},
        ]
    )
    dummy._metadata = metadata  # Overwrite metadata
    dummy._build_trace_name_to_idx_dict()
    assert len(dummy._trace_name_to_idx) == 4
    assert len(dummy._trace_name_to_idx["name"]) == 2  # 2 collisions
    assert len(dummy._trace_name_to_idx["name_chunk"]) == 3  # 1 collision
    assert len(dummy._trace_name_to_idx["name_dataset"]) == 3  # 1 collision
    assert len(dummy._trace_name_to_idx["name_chunk_dataset"]) == 4  # 0 collision

    # Only test unique answers
    assert dummy.get_idx_from_trace_name("a") == 0
    assert dummy.get_idx_from_trace_name("b", dataset="1") == 3
    assert dummy.get_idx_from_trace_name("b", "1", "0") == 2
    with pytest.raises(KeyError):
        dummy.get_idx_from_trace_name("a", "1")


def test_sample_without_replacement():
    # Test random sampling without replacement
    np.random.seed(42)

    sel_idxs, rem_idxs = sbd.SCEDC._sample_without_replacement(np.arange(1000), 100)

    assert len(sel_idxs) == 100
    assert len(rem_idxs) == 900
    assert (
        sorted(list(sel_idxs) + list(rem_idxs)) == np.arange(1000).astype(list)
    ).all()


def test_waveform_data_write_without_trace_name(tmp_path):
    # Test that a single trace without a trace name does not crash the writer
    with sbd.WaveformDataWriter(
        tmp_path / "metadata.csv", tmp_path / "waveforms.hdf5"
    ) as writer:
        trace = {"source_id": "dummyevent"}
        writer.add_trace(trace, np.zeros((3, 100)))


def test_chunked_datasets_key_collision(tmp_path):
    with open(tmp_path / "chunks", "w") as f:
        f.write("0\n1")

    for i in range(2):
        metadata_path = tmp_path / f"metadata{i}.csv"
        waveforms_path = tmp_path / f"waveforms{i}.hdf5"
        with sbd.WaveformDataWriter(metadata_path, waveforms_path) as writer:
            writer.data_format = {
                "dimension_order": "CW",
                "component_order": "ZNE",
                "sampling_rate": 100,
            }
            writer.add_trace({"trace_name": "fixed_trace_name"}, np.ones((1000, 3)) * i)

    data = sbd.WaveformDataset(tmp_path, cache="full")

    assert (data.get_waveforms(0) == 0).all()
    assert (data.get_waveforms(1) == 1).all()


def test_verify_grouping():
    data = sbd.DummyDataset()

    data.grouping = None
    with pytest.raises(ValueError):
        data.get_group_idx_from_params("dummy")

    with pytest.raises(ValueError):
        data.get_group_size(0)

    with pytest.raises(ValueError):
        data.get_group_samples(0)

    with pytest.raises(ValueError):
        data.get_group_waveforms(0)


@pytest.fixture
def grouping_test_data():
    data = sbd.DummyDataset()

    trace_names = data._metadata["trace_name"].values[:5]

    data._metadata = pd.DataFrame(
        [
            {
                "event": "a",
                "station": "A",
                "part": 0,
                "trace_chunk": "",
            },
            {
                "event": "b",
                "station": "A",
                "part": 0,
                "trace_chunk": "",
            },
            {
                "event": "b",
                "station": "B",
                "part": 0,
                "trace_chunk": "",
            },
            {
                "event": "b",
                "station": "C",
                "part": 0,
                "trace_chunk": "",
            },
            {
                "event": "b",
                "station": "C",
                "part": 1,
                "trace_chunk": "",
            },
        ]
    )
    data._metadata["trace_sampling_rate_hz"] = np.ones(5) * 100
    data._metadata["trace_component_order"] = "ZNE"
    data._metadata["trace_name"] = trace_names
    return data


def test_get_group_size(grouping_test_data):
    data = grouping_test_data

    data.grouping = "event"
    assert len(data.groups) == 2
    assert data.get_group_size(data.get_group_idx_from_params("a")) == 1
    assert data.get_group_size(data.get_group_idx_from_params("b")) == 4

    data.grouping = ["event", "station"]
    assert len(data.groups) == 4
    assert data.get_group_size(data.get_group_idx_from_params(("a", "A"))) == 1
    assert data.get_group_size(data.get_group_idx_from_params(("b", "A"))) == 1
    assert data.get_group_size(data.get_group_idx_from_params(("b", "B"))) == 1
    assert data.get_group_size(data.get_group_idx_from_params(("b", "C"))) == 2


def test_get_group_sample_waveforms(grouping_test_data):
    data = grouping_test_data

    data.grouping = "event"
    idx = data.get_group_idx_from_params("b")

    waveforms, metadata = data.get_group_samples(idx)
    assert len(waveforms) == len(metadata["trace_name"]) == 4
    assert metadata["station"] == ["A", "B", "C", "C"]
    assert isinstance(waveforms, list)
    for wv in waveforms:
        assert wv.shape == (3, 1200)
    assert all(x == 1200 for x in metadata["trace_npts"])

    waveforms, metadata = data.get_group_samples(idx, sampling_rate=50)
    assert len(waveforms) == len(metadata["trace_name"]) == 4
    assert metadata["station"] == ["A", "B", "C", "C"]
    assert isinstance(waveforms, list)
    for wv in waveforms:
        assert wv.shape == (3, 600)
    assert all(x == 600 for x in metadata["trace_npts"])

    # Test trace independent resampling
    data.metadata["trace_sampling_rate_hz"] = [50, 100, 50, 50, 100]
    waveforms, metadata = data.get_group_samples(idx, sampling_rate=50)
    assert len(waveforms) == len(metadata["trace_name"]) == 4
    assert metadata["station"] == ["A", "B", "C", "C"]
    assert isinstance(waveforms, list)
    assert waveforms[0].shape == (3, 600)
    assert metadata["trace_npts"][0] == 600
    assert waveforms[1].shape == (3, 1200)
    assert metadata["trace_npts"][1] == 1200
    assert waveforms[2].shape == (3, 1200)
    assert metadata["trace_npts"][2] == 1200
    assert waveforms[3].shape == (3, 600)
    assert metadata["trace_npts"][3] == 600


def test_multiwaveformdataset_grouping():
    # Mostly checks that nothing crashes
    data = sbd.DummyDataset() + sbd.DummyDataset()

    data.grouping = "source_magnitude"

    assert len(data.groups) == 97
    assert data.get_group_size(0) == 2
    data.get_group_waveforms(0)
    data.get_group_samples(0)
    data.get_group_idx_from_params(data.metadata.iloc[0]["source_magnitude"])


def test_grouping_filter(grouping_test_data):
    data = grouping_test_data

    data.grouping = "event"
    mask = data.metadata["event"] == "b"
    data.filter(mask, inplace=True)
    assert len(data.groups) == 1


def test_chunk_validation():
    # Requesting a non-existent chunk should raise a ValueError
    sbd.ChunkedDummyDataset(chunks=["0"])  # Passes

    with pytest.raises(ValueError) as e:
        sbd.ChunkedDummyDataset(chunks=["not_a_chunk"])  # Fails

    assert "not_a_chunk" in str(e)


def test_grouping_chunked_dataset():
    data = sbd.ChunkedDummyDataset()
    data.grouping = "trace_name"

    # Check that actually the index was reset and the group keys are indices now
    for i in range(len(data)):
        assert any(i in group_keys for group_keys in data._groups_to_trace_idx.values())


def test_grouping_index_filter():
    data = sbd.DummyDataset()
    mask = np.ones(len(data), dtype=bool)
    mask[:10] = False  # Remove the first 10 entries

    data.grouping = "trace_name"

    data.filter(mask, inplace=True)
    # Check that actually the index was reset and the group keys are indices now
    for i in range(len(data)):
        assert any(i in group_keys for group_keys in data._groups_to_trace_idx.values())


def test_metadata_lookup():
    data = sbd.DummyDataset(metadata_cache=False)
    assert data._metadata_lookup is None

    data = sbd.DummyDataset(metadata_cache=True)
    assert len(data._metadata_lookup) == len(data.metadata)

    mask = np.zeros(len(data), dtype=bool)
    mask[:20] = True

    data.filter(mask, inplace=True)
    assert len(data._metadata_lookup) == len(data.metadata)

    for i in range(len(data)):
        meta_lookup = data._metadata_lookup[i]
        meta_direct = data.metadata.iloc[i].to_dict()

        for key in meta_lookup.keys():
            assert meta_direct[key] == meta_lookup[key]


def test_chunks_with_paths_cache():
    data = sbd.DummyDataset()

    data._chunks_with_paths_cache = None
    data._chunks_with_paths()
    assert data._chunks_with_paths_cache is not None


def test_transpose_single_waveform():
    data = sbd.DummyDataset()  # source dimension order = "CW"
    wv = np.ones((3, 1200))

    data.dimension_order = "NCW"
    assert data._transpose_single_waveform(wv).shape == (3, 1200)

    data.dimension_order = "NWC"
    assert data._transpose_single_waveform(wv).shape == (1200, 3)

    data.dimension_order = "WNC"
    assert data._transpose_single_waveform(wv).shape == (1200, 3)

    data.dimension_order = "CWN"
    assert data._transpose_single_waveform(wv).shape == (3, 1200)


@pytest.mark.parametrize("sampling_rate", [50, 100])
def test_get_group_sample_multi_waveform_dataset(sampling_rate):
    data = sbd.DummyDataset() + sbd.DummyDataset()

    grouping_key = np.arange(len(data))
    grouping_key[1] = grouping_key[0]
    grouping_key[-2] = grouping_key[0]
    grouping_key[-1] = grouping_key[0]
    data.metadata["grouping_key"] = grouping_key

    data.grouping = "grouping_key"

    waveforms, metadata = data.get_group_samples(0, sampling_rate=sampling_rate)

    assert isinstance(waveforms, list)
    assert len(waveforms) == 4

    # As we modify only a copy of the metadata, these will be missing in the single samples
    del metadata["grouping_key"]
    del metadata["trace_dataset"]

    for i, target_idx in enumerate([0, 1, -2, -1]):
        target_waveforms, target_metadata = data.get_sample(
            target_idx, sampling_rate=sampling_rate
        )

        assert target_metadata == {k: v[i] for k, v in metadata.items()}
        assert np.all(waveforms[i] == target_waveforms)


def test_abstract_benchmark_dataset_download_logic(tmp_path):
    # The mix-in is required to consume the `chunks` argument
    class MixinDataset:
        def __init__(self, chunks):
            pass

    class UnchunkedBenchmarkDataset(sbd.AbstractBenchmarkDataset, MixinDataset):
        _files = ["abc.csv", "def.csv", "ghi.csv"]

        def __init__(self):
            super().__init__(repository_lookup=True)

        @classmethod
        def available_chunks(cls, *args, **kwargs):
            return [""]

    class ChunkedBenchmarkDataset(sbd.AbstractBenchmarkDataset, MixinDataset):
        _files = ["abc_$CHUNK.csv", "def_$CHUNK.csv", "ghi_$CHUNK.csv"]

        def __init__(self):
            super().__init__(repository_lookup=True)

        @classmethod
        def available_chunks(cls, *args, **kwargs):
            return ["1", "2", "3"]

    with patch(
        "seisbench.cache_data_root", tmp_path
    ):  # Ensure test does not modify SeisBench cache

        def write_files(files, *args, **kwargs):
            for file in files:
                with open(file, "w"):
                    pass

        with patch("seisbench.util.callback_if_uncached", side_effect=write_files):
            unchunked = UnchunkedBenchmarkDataset()

        for file in ["abc", "def", "ghi"]:
            assert (unchunked.path / f"{file}.csv").is_file()

        with patch("seisbench.util.callback_if_uncached", side_effect=write_files):
            unchunked = ChunkedBenchmarkDataset()

            for file in ["abc", "def", "ghi"]:
                for chunk in ["1", "2", "3"]:
                    assert (unchunked.path / f"{file}_{chunk}.csv").is_file()
