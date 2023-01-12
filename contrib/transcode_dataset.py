import argparse
import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import seisbench.data as sbd


def transcode(data_in, data_out):
    dataset_args = {"missing_components": "ignore"}
    try:
        # Check if dataset is available in SeisBench
        dataset = sbd.__getattribute__(data_in)(**dataset_args)
    except AttributeError:
        # Otherwise interpret data_in as path
        dataset = sbd.WaveformDataset(data_in, **dataset_args)

    # Fix inconsistent component order entries
    if (
        len(dataset["trace_component_order"].unique()) > 1
        and "component_order" in dataset._data_format
    ):
        print("Removing inconsistent component order from data format.")
        del dataset._data_format["component_order"]

    component_order_mapping = get_component_order_mapping(dataset)

    if "component_order" in dataset._data_format:
        dataset._data_format["component_order"] = component_order_mapping[
            dataset._data_format["component_order"]
        ]

    if len(dataset.chunks) > 0:
        metadata_path = dataset.path / f"metadata{dataset.chunks[0]}.csv"
    else:
        metadata_path = dataset.path / "metadata.csv"

    # Load original columns to avoid writing auxiliary columns
    columns = set(pd.read_csv(metadata_path, index_col=False, nrows=0).columns.tolist())

    if os.path.exists(data_out):
        raise ValueError("Output path must not exist.")

    data_out = Path(data_out)
    with sbd.WaveformDataWriter(
        data_out / "metadata.csv", data_out / "waveforms.hdf5"
    ) as writer:
        writer.data_format = dataset.data_format

        for i in tqdm(range(len(dataset))):
            waveform, metadata = dataset.get_sample(idx=i)
            metadata = {
                key: val for key, val in metadata.items() if key in columns
            }  # Filter metadata
            if "trace_component_order" in metadata:
                metadata["trace_component_order"] = component_order_mapping[
                    metadata["trace_component_order"]
                ]
            writer.add_trace(metadata, waveform)


def get_component_order_mapping(dataset):
    """
    Calculates for each input component order the actual order that will be returned by the dataset.
    Assumes missing_components == "ignore".

    :param dataset:
    :return:
    """
    assert dataset.missing_components == "ignore"

    mapping = {}
    target_order = dataset.component_order
    for order in dataset["trace_component_order"].unique():
        new_order = "".join([x for x in target_order if x in order])
        if len(order) != len(new_order):
            raise ValueError(
                f"Writing order {new_order} in format {target_order} will lead to data loss."
            )

        mapping[order] = new_order

    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcode a WaveformDataset into a WaveformDataset with block structure. "
        "Always uses the default setting for the block structure. "
        "The current implementation will read chunked dataset, "
        "but will write all data in one chunk."
    )
    parser.add_argument(
        "input", type=str, help="Path to input dataset of SeisBench dataset name"
    )
    parser.add_argument(
        "output", type=str, help="Path for output dataset. Must not exist."
    )
    args = parser.parse_args()

    transcode(args.input, args.output)
