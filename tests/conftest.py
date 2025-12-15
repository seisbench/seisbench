import pytest
import shutil
import seisbench
import seisbench.data as sbd
import tempfile


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """
    Create SeisBench cache and provide files
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        seisbench.cache_data_root = tmpdir

        dummy_dataset_path = sbd.DummyDataset._path_internal()
        shutil.copytree("./tests/examples/dummydataset", dummy_dataset_path)

        chunked_dummy_dataset_path = sbd.ChunkedDummyDataset._path_internal()
        shutil.copytree(
            "./tests/examples/chunkeddummydataset", chunked_dummy_dataset_path
        )

        yield
