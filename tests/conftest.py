import pytest
import shutil
import seisbench.data as sbd


@pytest.fixture(scope="session", autouse=True)
def setup_data():
    """
    Ensures that the DummyDataset is available in the SeisBench cache.
    """
    dummy_dataset_path = sbd.DummyDataset._path_internal()
    if not dummy_dataset_path.is_dir():
        dummy_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree("./tests/examples/DummyDataset", dummy_dataset_path)

    chunked_dummy_dataset_path = sbd.ChunkedDummyDataset._path_internal()
    if not chunked_dummy_dataset_path.is_dir():
        chunked_dummy_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            "./tests/examples/ChunkedDummyDataset", chunked_dummy_dataset_path
        )
