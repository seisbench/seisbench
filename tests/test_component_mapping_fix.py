import math
import pathlib

import seisbench
import seisbench.data
import numpy as np
import pytest


def test_component_mapping_fix_for_implicit_int_casts(tmp_path: pathlib.Path):

    # The following implicit calls of seisbench.data.WaveformDataset._get_component_mapping() no longer raise

    # Example values to combine
    check_co_vals = ["ZNE", "Z12", "12", "Z12", "Z12H"]
    for source_co in check_co_vals:
        for target_co in check_co_vals:
            data_path = tmp_path / "writer_a"
            with seisbench.data.WaveformDataWriter(
                data_path / "metadata.csv", data_path / "waveforms.hdf5"
            ) as writer:
                trace = {"trace_name": "dummy"}
                writer.add_trace(trace, np.zeros((3, 100)))
                writer.data_format["component_order"] = source_co
            # Seisbench will call _get_component_mapping() in order to calculate possibly missing components
            seisbench.data.WaveformDataset(data_path, component_order=target_co)

    # None of these test values should should be a problem
    for test_val in [1, 12, "1", "12", "12H", "Z12H"]:
        data_path = tmp_path / "writer_b"
        with seisbench.data.WaveformDataWriter(
            data_path / "metadata.csv", data_path / "waveforms.hdf5"
        ) as writer:
            # A trace with component order different to the one passed to the dataset init call
            trace = {"trace_name": "dummy", "trace_component_order": test_val}
            writer.add_trace(trace, np.zeros((3, 100)))
        # Seisbench will check each trace for missing components compared with the passed component order
        seisbench.data.WaveformDataset(data_path, component_order="Z12")

    # Missing trace_component_order entries should raise
    for test_val in ["", math.nan, np.nan]:
        data_path = tmp_path / "writer_c"
        with seisbench.data.WaveformDataWriter(
            data_path / "metadata.csv", data_path / "waveforms.hdf5"
        ) as writer:
            trace = {"trace_name": "dummy", "trace_component_order": test_val}
            writer.add_trace(trace, np.zeros((3, 100)))
        with pytest.raises(ValueError):
            seisbench.data.WaveformDataset(data_path, component_order="Z12")
