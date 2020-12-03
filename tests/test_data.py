import seisbench.data
import seisbench.util.region as region

import numpy as np
import pytest
import logging


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

    dummy.metadata["receiver_latitude"] = lat
    dummy.metadata["receiver_longitude"] = lon

    domain = region.CircleDomain(0, 0, 1, 5)

    dummy.region_filter_receiver(domain)

    assert len(dummy) == np.sum(np.logical_and(1 < np.abs(lon), np.abs(lon) < 5))

    # Source + RectangleDomain
    dummy = seisbench.data.DummyDataset()

    np.random.seed(42)
    n = len(dummy)
    lat = np.random.random(n) * 40 - 20
    lon = np.random.random(n) * 40 - 20

    dummy.metadata["source_latitude"] = lat
    dummy.metadata["source_longitude"] = lon

    domain = region.RectangleDomain(
        minlatitude=-5, maxlatitude=5, minlongitude=10, maxlongitude=15
    )

    dummy.region_filter_source(domain)

    mask_lat = np.logical_and(-5 <= lat, lat <= 5)
    mask_lon = np.logical_and(10 <= lon, lon <= 15)

    assert len(dummy) == np.sum(np.logical_and(mask_lat, mask_lon))


def test_get_waveforms():
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


def test_lazyload_cache(caplog):
    with caplog.at_level(logging.WARNING):
        seisbench.data.DummyDataset(lazyload=False, cache=False)
    assert "Skipping preloading of waveforms as cache is set to inactive" in caplog.text
