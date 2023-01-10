import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


def main(input_path):
    np.random.seed(505)

    input_path = Path(input_path)
    print("Loading metadata")
    metadata = pd.read_csv(input_path / "metadata.csv")

    # Allows up to ten experiments while avoiding overlap of the blocks to avoid advantages from caching.
    n_part = int(0.1 * len(metadata))

    # Random access unsorted
    print("Random access unsorted")
    metadata_exp = metadata.iloc[:n_part]
    traces = metadata_exp["trace_name"].values.copy()
    np.random.shuffle(traces)
    traces = traces[:30000]
    test_list = []
    with h5py.File(input_path / "waveforms.hdf5") as h5_file:
        for trace in tqdm(traces):
            if "$" in trace:
                bucket, arrloc = trace.split("$")
                arr_index = int(arrloc.split(",")[0])
                test_list.append(np.mean(h5_file["data"][bucket][arr_index, :, :]))
            else:
                test_list.append(np.mean(h5_file["data"][trace]))

    # Random access sorted - Not implemented

    # Sequential access - Not implemented


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmarks read performance for a dataset. Only very simple and slightly incomplete version."
    )
    parser.add_argument("input", type=str, help="Path to input dataset")
    args = parser.parse_args()

    main(args.input)
