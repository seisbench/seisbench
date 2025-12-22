import datetime
import pytest
import numpy as np
import pandas as pd
import h5py

import seisbench.data as sbd


def test_das_data_writer_validate_metadata(tmp_path):
    with sbd.DASDataWriter(tmp_path / "example_a", strict=True) as writer:
        valid_example = {
            "record_sampling_rate_hz": 100,
            "record_channel_spacing_m": 10,
            "record_start_time": datetime.datetime.now(),
            "some_key": 123,
        }

        invalid_example = {
            "record_channel_spacing_m": 10,
            "record_start_time": datetime.datetime.now(),
            "some_key": 123,
        }

        writer._validate_metadata_entry(valid_example)
        with pytest.raises(ValueError):
            writer._validate_metadata_entry(invalid_example)

        writer.strict = False
        writer._validate_metadata_entry(valid_example)
        writer._validate_metadata_entry(invalid_example)


def test_das_data_writer_validate_annotations(tmp_path):
    with sbd.DASDataWriter(tmp_path / "example_b", strict=True) as writer:
        data = np.ones((100, 200))
        annotations_valid = {
            "P": np.random.rand(200) * 100,
            "S": np.random.rand(200) * 100,
        }
        annotations_invalid = {
            "P": np.random.rand(200) * 100,
            "S": np.random.rand(201) * 100,
        }

        writer._validate_annotations(data, annotations_valid)
        with pytest.raises(ValueError):
            writer._validate_annotations(data, annotations_invalid)


def test_das_data_writer(tmp_path):
    metadata = {
        "record_sampling_rate_hz": 100,
        "record_channel_spacing_m": 10,
        "record_start_time": datetime.datetime.now(),
        "some_key": 123,
    }
    data = np.ones((100, 200))
    annotations = {"P": np.random.rand(200) * 100, "S": np.random.rand(200) * 100}

    chunk = "0"
    with sbd.DASDataWriter(tmp_path / "example_c", chunk=chunk) as writer:
        writer.add_record(metadata, data, annotations)

    assert (
        tmp_path / "example_c" / sbd.DASDataset.DATA_FILE.replace("$CHUNK", chunk)
    ).is_file()
    assert (
        tmp_path / "example_c" / sbd.DASDataset.METADATA_FILE.replace("$CHUNK", chunk)
    ).is_file()

    metadata = pd.read_parquet(
        tmp_path / "example_c" / sbd.DASDataset.METADATA_FILE.replace("$CHUNK", chunk)
    )
    assert len(metadata) == 1

    with h5py.File(
        tmp_path / "example_c" / sbd.DASDataset.DATA_FILE.replace("$CHUNK", chunk), "r"
    ) as f:
        assert "records/record_0/record" in f
        assert "records/record_0/annotations/P" in f
        assert "records/record_0/annotations/S" in f


def test_das_dataset_available_chunks(tmp_path):
    # Read chunks file
    example1 = tmp_path / "example_1"
    example1.mkdir()
    with open(example1 / "chunks", "w") as f:
        f.write("0\n1\n")

    assert sbd.DASDataset.available_chunks(example1) == ["0", "1"]

    # Empty chunks file
    example2 = tmp_path / "example_2"
    example2.mkdir()
    with open(example2 / "chunks", "w") as f:
        f.write("")

    assert sbd.DASDataset.available_chunks(example2) == [""]

    # Parse from files
    example3 = tmp_path / "example_3"
    example3.mkdir()
    files = [
        sbd.DASDataset.METADATA_FILE.replace("$CHUNK", chunk) for chunk in ["0", "1"]
    ]
    files += [sbd.DASDataset.DATA_FILE.replace("$CHUNK", chunk) for chunk in ["0", "1"]]

    for file in files:
        open(example3 / file, "w").close()

    assert sbd.DASDataset.available_chunks(example3) == ["0", "1"]

    # Mismatching, ignore extra files
    example4 = tmp_path / "example_4"
    example4.mkdir()
    files = [
        sbd.DASDataset.METADATA_FILE.replace("$CHUNK", chunk) for chunk in ["0", "1"]
    ]
    files += [
        sbd.DASDataset.DATA_FILE.replace("$CHUNK", chunk) for chunk in ["0", "1", "2"]
    ]

    for file in files:
        open(example4 / file, "w").close()

    assert sbd.DASDataset.available_chunks(example4) == ["0", "1"]


@pytest.mark.parametrize("record_virtual", [True, False])
@pytest.mark.parametrize("annotations_virtual", [True, False])
def test_das_dataset_get_sample(record_virtual: bool, annotations_virtual: bool):
    data = sbd.RandomDASDataset()
    for i in range(len(data)):
        metadata, record, annotations = data.get_sample(
            i, record_virtual=record_virtual, annotations_virtual=annotations_virtual
        )
        assert record.ndim == 2
        assert isinstance(record, np.ndarray) ^ record_virtual
        for key, value in annotations.items():
            assert value.ndim == 1
            assert value.shape[0] == record.shape[1]
            assert isinstance(value, np.ndarray) ^ annotations_virtual

        assert all(
            key in metadata
            for key in [
                "record_sampling_rate_hz",
                "record_channel_spacing_m",
                "record_start_time",
            ]
        )


def test_das_dataset_chunks():
    # Load all chunks
    data = sbd.RandomDASDataset(chunks=None)
    assert len(data) == 10

    # Load only one chunk
    data = sbd.RandomDASDataset(chunks=["1"])
    assert len(data) == 5

    # Chunk does not exist
    with pytest.raises(ValueError):
        sbd.RandomDASDataset(chunks=["2"])


def test_das_dataset_copy():
    data = sbd.RandomDASDataset()
    data.get_sample(0)
    assert len(data._data_pointers) > 0

    data_copy = data.copy()
    assert data._metadata is not data_copy._metadata  # Metadata was deep copied
    assert len(data_copy._data_pointers) == 0  # Pointers were reset


def test_das_dataset_filter():
    data = sbd.RandomDASDataset()
    data._metadata["extra_column"] = np.arange(10)
    data.filter(data.metadata["extra_column"] % 2 == 0, inplace=True)
    assert all(data.metadata["extra_column"] == [0, 2, 4, 6, 8])

    data = sbd.RandomDASDataset()
    data._metadata["extra_column"] = np.arange(10)
    data_filtered = data.filter(data.metadata["extra_column"] % 2 == 0, inplace=False)
    assert len(data) == 10
    assert all(data_filtered.metadata["extra_column"] == [0, 2, 4, 6, 8])


def test_multi_data_dataset_add():
    data_a = sbd.RandomDASDataset(chunks=["0"])
    data_b = sbd.RandomDASDataset(chunks=["1"])

    data = data_a + data_b
    assert isinstance(data, sbd.MultiDASDataset)
    assert len(data) == len(data_a) + len(data_b)

    assert (
        data.get_sample(0, record_virtual=False)[0]
        == data_a.get_sample(0, record_virtual=False)[0]
    )
    assert (
        data.get_sample(0, record_virtual=False)[1]
        == data_a.get_sample(0, record_virtual=False)[1]
    ).all()
    assert (
        data.get_sample(len(data_a), record_virtual=False)[0]
        == data_b.get_sample(0, record_virtual=False)[0]
    )
    assert (
        data.get_sample(len(data_a), record_virtual=False)[1]
        == data_b.get_sample(0, record_virtual=False)[1]
    ).all()

    data2 = data + data_a
    assert (
        data2.get_sample(len(data), record_virtual=False)[0]
        == data_a.get_sample(0, record_virtual=False)[0]
    )
    assert (
        data2.get_sample(len(data), record_virtual=False)[1]
        == data_a.get_sample(0, record_virtual=False)[1]
    ).all()

    data3 = data + data
    assert (
        data3.get_sample(len(data) + len(data_a), record_virtual=False)[0]
        == data_b.get_sample(0, record_virtual=False)[0]
    )
    assert (
        data3.get_sample(len(data) + len(data_a), record_virtual=False)[1]
        == data_b.get_sample(0, record_virtual=False)[1]
    ).all()


def test_multi_data_dataset_filter():
    data_a = sbd.RandomDASDataset(chunks=["0"])
    data_b = sbd.RandomDASDataset(chunks=["1"])
    data = sbd.MultiDASDataset([data_a, data_b])

    data.filter(np.arange(len(data)) % 2 == 0, inplace=True)
    assert (
        data.get_sample(0, record_virtual=False)[0]
        == data_a.get_sample(0, record_virtual=False)[0]
    )
    assert (
        data.get_sample(0, record_virtual=False)[1]
        == data_a.get_sample(0, record_virtual=False)[1]
    ).all()
    assert (
        data.get_sample(3, record_virtual=False)[0]
        == data_b.get_sample(1, record_virtual=False)[0]
    )
    assert (
        data.get_sample(3, record_virtual=False)[1]
        == data_b.get_sample(1, record_virtual=False)[1]
    ).all()
