import seisbench
from .base import BenchmarkDataset


class SCEDC(BenchmarkDataset):
    """
    SCEDC waveform archive.
    With entire catalog > 500Gb, utilizes SeisBench chunk
    reading to keep operations manageable.
    """

    def __init__(self, **kwargs):
        citation = (
            "SCEDC (2013): Southern California Earthquake Center."
            "doi:10.7909/C3WD3xH1"
        )
        super().__init__(citation=citation, repository_lookup=True, **kwargs)

    def _download_dataset(self, writer, chunk, basepath=None, **kwargs):
        download_instructions = "SCEDC dataset currently unavailable for download at this point in dev. process."

        basepath = self.path

        if not basepath.exists():
            raise ValueError(
                "No cached version of SCEDC found. " + download_instructions
            )

        chunks_path = basepath.path / "chunks"
        if not chunks_path.is_file():
            basepath.path.mkdir(exist_ok=True, parents=True)
            with open(chunks_path, "w") as f:
                f.write("\n".join(["_b{:02d}".format(i) for i in range(2)]))
