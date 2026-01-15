import shutil
import tempfile
from pathlib import Path

import pytest

import seisbench
import seisbench.data as sbd

DATA_DIR = Path(__file__).parent / "examples"


@pytest.fixture(scope="session")
def data_dir() -> Path:
    return DATA_DIR


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """
    Create SeisBench cache and provide files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        seisbench.cache_data_root = tmpdir

        dummy_dataset_path = sbd.DummyDataset._path_internal()
        shutil.copytree(DATA_DIR / "dummydataset", dummy_dataset_path)

        chunked_dummy_dataset_path = sbd.ChunkedDummyDataset._path_internal()
        shutil.copytree(DATA_DIR / "chunkeddummydataset", chunked_dummy_dataset_path)

        yield
