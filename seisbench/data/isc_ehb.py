from .base import BenchmarkDataset


class ISC_EHB_DepthPhases(BenchmarkDataset):
    """
    Dataset of depth phase picks from the
    `ISC-EHB bulletin <http://www.isc.ac.uk/isc-ehb/>`_.
    """

    def __init__(self, **kwargs):
        citation = (
            "Münchmeyer, J., Saul, J. & Tilmann, F. (2023) "
            "Learning the deep and the shallow: Deep learning "
            "based depth phase picking and earthquake depth estimation."
            "Seismological Research Letters (in revision)."
        )

        self._write_chunk_file()

        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    @staticmethod
    def available_chunks(*args, **kwargs):
        return [str(x) for x in range(1987, 2019)]

    def _write_chunk_file(self):
        """
        Write out the chunk file

        :return: None
        """
        chunks_path = self.path / "chunks"

        if chunks_path.is_file():
            return

        chunks = self.available_chunks()
        chunks_str = "\n".join(chunks) + "\n"

        self.path.mkdir(exist_ok=True, parents=True)
        with open(chunks_path, "w") as f:
            f.write(chunks_str)

    def _download_dataset(self, *args, **kwargs):
        raise NotImplementedError(
            "Downloading from source is not implemented. "
            "However, this dataset is available in the SeisBench repository. "
            "Please verify you can access the repository at: "
            f"{self._remote_path()}."
        )
