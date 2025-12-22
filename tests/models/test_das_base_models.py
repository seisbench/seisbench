import numpy as np
import pytest
import xdas
from xdas import Coordinate, DataArray
from scipy.signal import resample_poly, sosfilt_zi, sosfilt

import seisbench.models as sbm


def test_get_n_patches():
    patching_structure = sbm.PatchingStructure(
        in_samples=500,
        in_channels=400,
        out_samples=500,
        out_channels=400,
        overlap_samples=0,
        overlap_channels=0,
        range_samples=(0, 500),
        range_channels=(0, 400),
    )

    def create_data(n_samples, n_channels):
        channel_coords = Coordinate(
            {
                "tie_indices": [0, n_channels - 1],
                "tie_values": [0, 1e4],
            }
        )
        time_coords = Coordinate(
            {
                "tie_indices": [0, n_samples - 1],
                "tie_values": [
                    np.datetime64("2023-01-01T00:00:00", "us"),
                    np.datetime64("2023-01-01T00:00:30", "us"),
                ],
            }
        )
        return DataArray(
            data=np.random.rand(n_samples, n_channels),
            coords={"time": time_coords, "channel": channel_coords},
        )

    data = create_data(3000, 2000)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (6, 5)
    data = create_data(3001, 2001)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (7, 6)
    data = create_data(2999, 1999)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (6, 5)

    patching_structure = sbm.PatchingStructure(
        in_samples=600,
        in_channels=500,
        out_samples=500,
        out_channels=400,
        overlap_samples=100,
        overlap_channels=100,
        range_samples=(0, 500),
        range_channels=(0, 400),
    )

    data = create_data(3000, 2000)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (6, 5)
    data = create_data(3001, 2001)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (6, 5)
    data = create_data(2999, 1999)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (6, 5)
    data = create_data(600, 500)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (1, 1)
    data = create_data(601, 501)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (2, 2)
    data = create_data(1099, 500)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (2, 1)
    data = create_data(1100, 500)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (2, 1)
    data = create_data(1101, 500)
    assert sbm.VirtualTransformedDataArray(
        data, patching_structure
    )._get_n_patches() == (3, 1)


class DemoModel(sbm.DASModel):
    def __init__(self):
        patching_structure = sbm.PatchingStructure(
            in_samples=100,
            in_channels=90,
            out_samples=100,
            out_channels=90,
            range_samples=(0, 100),
            range_channels=(0, 90),
        )
        super().__init__(
            patching_structure=patching_structure,
            annotate_keys=["x"],
        )

    def forward(self, x, argdict=None):
        return {"x": x}


class DemoCallback(sbm.DASAnnotateCallback):
    def __init__(self):
        self.results = []

    def handle_patch(self, annotations, in_coords, out_coords):
        self.results.append((annotations, in_coords, out_coords))

    def get_results_dict(self):
        return {"results": self.results}

    def setup(self, data, patching_structure, annotate_keys):
        self.results = []


@pytest.mark.parametrize("virtual_data", [True, False])
def test_das_model_annotate(tmp_path, virtual_data):
    model = DemoModel()
    callback = DemoCallback()

    n_samples = 500
    n_channels = 300

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    da = DataArray(data=data, coords={"time": time_coords, "channel": channel_coords})
    if virtual_data:
        # Write and read back to make sure the data array is virtual
        da.to_netcdf(tmp_path / "data.nc")
        da = xdas.open_dataarray(tmp_path / "data.nc")

    model.annotate(da, callback, overlap_samples=0.5, overlap_channels=0.5)

    assert len(callback.results) == 54
    for pred, input_coords, output_coords in callback.results:
        assert set(pred.keys()) == set(model.annotate_keys)
        assert pred["x"].shape == (100, 90)
        assert input_coords == output_coords


@pytest.mark.parametrize("axis", ["time", "channel"])
def test_validate_data(axis):
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=90,
        out_samples=100,
        out_channels=90,
        range_samples=(0, 100),
        range_channels=(0, 90),
        overlap_channels=0,
        overlap_samples=0,
    )

    # Too small data
    n_samples = 1000
    n_channels = 1000
    if axis == "time":
        n_samples = 10
    else:
        n_channels = 10
    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )
    da = DataArray(data=data, coords={"time": time_coords, "channel": channel_coords})

    with pytest.raises(ValueError) as execinfo:
        sbm.VirtualTransformedDataArray(da, patching_structure)
    assert "too small" in str(execinfo.value)

    # Transposed component order
    n_samples = 1000
    n_channels = 1000
    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )
    da = DataArray(data=data, coords={"channel": channel_coords, "time": time_coords})
    assert sbm.VirtualTransformedDataArray(da, patching_structure).transpose

    # Wrong type of coordinates
    n_samples = 1000
    n_channels = 1000
    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )
    da = DataArray(data=data, coords={"time": time_coords, "channel": channel_coords})

    da.coords[axis] = xdas.ScalarCoordinate(0)
    with pytest.raises(ValueError) as execinfo:
        sbm.VirtualTransformedDataArray(da, patching_structure)
    assert "Coordinates type" in str(execinfo.value)

    # Valid dense coordinates
    da.coords["time"] = time_coords
    da.coords["channel"] = channel_coords
    da.coords[axis] = xdas.DenseCoordinate(
        data=da.coords[axis].to_dataarray().data, dim=axis
    )
    assert sbm.VirtualTransformedDataArray(da, patching_structure).transpose is False

    wrong_coords = np.linspace(0, 1, len(da.coords[axis]))
    wrong_coords[0] = -1
    da.coords[axis] = xdas.DenseCoordinate(data=wrong_coords, dim=axis)
    with pytest.raises(ValueError) as execinfo:
        sbm.VirtualTransformedDataArray(da, patching_structure)
    assert "not uniformly spaced" in str(execinfo.value)

    # Valid interpolated coordinates after interpolation
    da.coords["time"] = time_coords
    da.coords["channel"] = channel_coords
    da.coords[axis] = xdas.InterpCoordinate(
        data={
            "tie_indices": [0, 50, len(da.coords[axis]) - 1],
            "tie_values": [
                da.coords[axis].get_value(0),
                da.coords[axis].get_value(50),
                da.coords[axis].get_value(len(da.coords[axis]) - 1),
            ],
        },
        dim=axis,
    )
    sbm.VirtualTransformedDataArray(da, patching_structure)

    # Invalid interpolated coordinates after interpolation
    da.coords["time"] = time_coords
    da.coords["channel"] = channel_coords
    da.coords[axis] = xdas.InterpCoordinate(
        data={
            "tie_indices": [0, 50, len(da.coords[axis]) - 1],
            "tie_values": [
                da.coords[axis].get_value(0),
                da.coords[axis].get_value(51),
                da.coords[axis].get_value(len(da.coords[axis]) - 1),
            ],
        },
        dim=axis,
    )
    with pytest.raises(ValueError) as execinfo:
        sbm.VirtualTransformedDataArray(da, patching_structure)
    assert "not uniformly spaced" in str(execinfo.value)


def test_transform_patch_coordinates():
    # 1:1 mapping, no boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=90,
        out_samples=100,
        out_channels=90,
        range_samples=(0, 100),
        range_channels=(0, 90),
    )

    in_coords = sbm.PatchCoordinate(
        sample=100,
        channel=300,
        w_sample=patching_structure.in_samples,
        w_channel=patching_structure.in_channels,
    )
    out_coords = sbm.DASModel._transform_patch_coordinates(
        in_coords, patching_structure
    )

    assert out_coords == in_coords

    # 1:1 mapping, truncated boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=50,
        out_samples=80,
        out_channels=28,
        range_samples=(10, 90),
        range_channels=(11, 39),
    )
    in_coords = sbm.PatchCoordinate(
        sample=100,
        channel=300,
        w_sample=patching_structure.in_samples,
        w_channel=patching_structure.in_channels,
    )
    out_coords = sbm.DASModel._transform_patch_coordinates(
        in_coords, patching_structure
    )

    assert np.allclose(out_coords, [100, 300, 80, 28])

    # scaled mapping, no boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=50,
        out_samples=60,
        out_channels=25,
        range_samples=(0, 100),
        range_channels=(0, 50),
    )
    in_coords = sbm.PatchCoordinate(
        sample=100,
        channel=300,
        w_sample=patching_structure.in_samples,
        w_channel=patching_structure.in_channels,
    )
    out_coords = sbm.DASModel._transform_patch_coordinates(
        in_coords, patching_structure
    )

    assert np.allclose(out_coords, [60, 150, 60, 25])

    # scaled mapping, truncated boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=120,
        in_channels=70,
        out_samples=60,
        out_channels=25,
        range_samples=(10, 110),
        range_channels=(10, 60),
    )
    in_coords = sbm.PatchCoordinate(
        sample=110,
        channel=300,
        w_sample=patching_structure.in_samples,
        w_channel=patching_structure.in_channels,
    )
    out_coords = sbm.DASModel._transform_patch_coordinates(
        in_coords, patching_structure
    )

    assert np.allclose(out_coords, [66, 150, 60, 25])


@pytest.mark.parametrize("dense_coordinates", [False, True])
def test_calc_output_shape_and_coordinates(dense_coordinates):
    def time_close(t0, t1):
        return abs(t0 - t1) < np.timedelta64(3, "us")

    n_samples = 1000
    n_channels = 500
    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    if dense_coordinates:
        channel_coords = xdas.DenseCoordinate(
            data=np.linspace(0, 1e4, n_channels), dim="channel"
        )
        t0 = np.datetime64("2023-01-01T00:00:00", "us")
        t1 = np.datetime64("2023-01-01T00:00:30", "us")
        td = t1 - t0
        time_coords = xdas.DenseCoordinate(
            data=t0 + td * np.linspace(0, 1, n_samples),
            dim="time",
        )
    else:
        channel_coords = Coordinate(
            {
                "tie_indices": [0, n_channels - 1],
                "tie_values": [0, 1e4],
            }
        )
        time_coords = Coordinate(
            {
                "tie_indices": [0, n_samples - 1],
                "tie_values": [
                    np.datetime64("2023-01-01T00:00:00", "us"),
                    np.datetime64("2023-01-01T00:00:30", "us"),
                ],
            }
        )
    da_base = DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    # 1:1 mapping, no boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=50,
        out_samples=100,
        out_channels=50,
        range_samples=(0, 100),
        range_channels=(0, 50),
        overlap_samples=0,
        overlap_channels=0,
    )
    da = sbm.VirtualTransformedDataArray(da_base, patching_structure=patching_structure)
    output_shape, output_coords = sbm.DASModel.calc_output_shape_and_coordinates(
        da, patching_structure
    )
    assert output_shape == da.shape
    assert time_close(output_coords["time"][0].data, time_coords[0].data)
    assert time_close(output_coords["time"][-1].data, time_coords[-1].data)
    assert np.isclose(output_coords["channel"][0].data, channel_coords[0].data)
    assert np.isclose(output_coords["channel"][-1].data, channel_coords[-1].data)

    # 1:1 mapping, truncated boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=50,
        out_samples=80,
        out_channels=28,
        range_samples=(10, 90),
        range_channels=(11, 39),
        overlap_samples=0,
        overlap_channels=0,
    )
    da = sbm.VirtualTransformedDataArray(da_base, patching_structure=patching_structure)
    output_shape, output_coords = sbm.DASModel.calc_output_shape_and_coordinates(
        da, patching_structure
    )
    assert output_shape == (980, 478)
    assert time_close(output_coords["time"][0].data, time_coords[10].data)
    assert time_close(output_coords["time"][-1].data, time_coords[-11].data)
    assert np.isclose(output_coords["channel"][0].data, channel_coords[11].data)
    assert np.isclose(output_coords["channel"][-1].data, channel_coords[-12].data)

    # scaled mapping, no boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=80,
        in_channels=50,
        out_samples=79,
        out_channels=25,
        range_samples=(0, 80),
        range_channels=(0, 50),
        overlap_samples=0,
        overlap_channels=0,
    )
    da = sbm.VirtualTransformedDataArray(da_base, patching_structure=patching_structure)
    output_shape, output_coords = sbm.DASModel.calc_output_shape_and_coordinates(
        da, patching_structure
    )
    assert output_shape == (988, 250)
    assert time_close(output_coords["time"][0].data, time_coords[0].data)
    assert time_close(output_coords["time"][-1].data, time_coords[-1].data)
    assert np.isclose(output_coords["channel"][0].data, channel_coords[0].data)
    assert np.isclose(output_coords["channel"][-1].data, channel_coords[-1].data)

    # scaled mapping, truncated boundaries
    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        in_channels=50,
        out_samples=79,
        out_channels=14,
        range_samples=(10, 90),
        range_channels=(11, 39),
        overlap_samples=0,
        overlap_channels=0,
    )
    da = sbm.VirtualTransformedDataArray(da_base, patching_structure=patching_structure)
    output_shape, output_coords = sbm.DASModel.calc_output_shape_and_coordinates(
        da, patching_structure
    )
    assert output_shape == (968, 239)
    assert time_close(output_coords["time"][0].data, time_coords[10].data)
    assert time_close(output_coords["time"][-1].data, time_coords[-11].data)
    assert np.isclose(output_coords["channel"][0].data, channel_coords[11].data)
    assert np.isclose(output_coords["channel"][-1].data, channel_coords[-12].data)


@pytest.mark.parametrize("sample_length_factor", [1, 1.1, 4.5])
@pytest.mark.parametrize("channel_length_factor", [1, 1.1, 4.5])
@pytest.mark.parametrize("stacking", ["max", "avg"])
@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_inmemory_collection_callback(
    sample_length_factor, channel_length_factor, stacking, overlap
):
    model = DemoModel()
    callback = sbm.InMemoryCollectionCallback(stacking=stacking)

    # Make sure we get both an exact match along each axis and a larger example
    n_samples = int(model.patching_structure.in_samples * sample_length_factor)
    n_channels = int(model.patching_structure.in_channels * channel_length_factor)

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    model.annotate(da, callback, overlap_samples=overlap, overlap_channels=overlap)

    assert np.allclose(da.data, callback.annotations["x"].data)


@pytest.mark.parametrize("stacking", ["max", "avg"])
@pytest.mark.parametrize("overlap", [0.0, 0.5])
def test_writer_callback(tmp_path, stacking, overlap):
    model = DemoModel()
    callback = sbm.WriterCallback(tmp_path)

    n_samples = 500
    n_channels = 300

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    model.annotate(da, callback, overlap_samples=overlap, overlap_channels=overlap)
    annotations = xdas.open_mfdataarray(str(tmp_path) + "/x/*")

    assert np.allclose(da.data, annotations.data)


# Heavily parametrized to cover different parameter combinations
@pytest.mark.parametrize("resample_samples", [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize("resample_channels", [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)])
@pytest.mark.parametrize("overlap_samples", [0, 42, 50])
@pytest.mark.parametrize("overlap_channels", [0, 42, 50])
def test_load_and_resample_patch(
    resample_samples, resample_channels, overlap_samples, overlap_channels
):
    n_samples = 500
    n_channels = 300

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        out_samples=100,
        in_channels=100,
        out_channels=100,
        range_samples=(0, 99),
        range_channels=(0, 99),
        overlap_samples=overlap_samples,
        overlap_channels=overlap_channels,
    )

    virtual = sbm.VirtualTransformedDataArray(
        da,
        patching_structure,
        resample_samples=resample_samples,
        resample_channels=resample_channels,
    )

    # Analog resample
    resampled = resample_poly(
        resample_poly(data, *virtual.resample_samples, axis=0),
        *virtual.resample_channels,
        axis=1,
    )

    n_samples, n_channels = virtual._get_n_patches()
    for idx_samples in range(n_samples):
        for idx_channels in range(n_channels):
            coord = virtual._patch_coords(idx_samples, idx_channels)
            patch = virtual._load_and_resample_patch(coord)

            assert np.allclose(
                patch,
                resampled[
                    coord.sample_int : coord.sample_int + coord.w_sample,
                    coord.channel_int : coord.channel_int + coord.w_channel,
                ],
            )


@pytest.mark.parametrize("transpose", [True, False])
def test_virtual_data_transpose(transpose):
    n_samples = 500
    n_channels = 300

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    if transpose:
        da = xdas.DataArray(
            data=data.T, coords={"channel": channel_coords, "time": time_coords}
        )
    else:
        da = xdas.DataArray(
            data=data, coords={"time": time_coords, "channel": channel_coords}
        )

    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        out_samples=100,
        in_channels=100,
        out_channels=100,
        range_samples=(0, 99),
        range_channels=(0, 99),
        overlap_samples=0,
        overlap_channels=0,
    )

    virtual = sbm.VirtualTransformedDataArray(
        da, patching_structure, resample_samples=(1, 1), resample_channels=(1, 1)
    )
    assert virtual.transpose == transpose

    n_samples, n_channels = virtual._get_n_patches()
    for idx_samples in range(n_samples):
        for idx_channels in range(n_channels):
            coord = virtual._patch_coords(idx_samples, idx_channels)
            patch = virtual._load_and_resample_patch(coord)

            # Note that because of the transpose the user output is identical no matter
            assert np.allclose(
                patch,
                data[
                    coord.sample_int : coord.sample_int + coord.w_sample,
                    coord.channel_int : coord.channel_int + coord.w_channel,
                ],
            )


def test_get_resample_ratios():
    class DemoModel(sbm.DASModel):
        def __init__(self):
            super().__init__(
                dt_range=(1, 1.1),
                dx_range=(2, 2.1),
            )

        def forward(self, x):
            return {"x": x}

    model = DemoModel()

    n_samples = 2
    n_channels = 2

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:01", "us"),
            ],
        }
    )
    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    assert model.get_resample_ratios(da, channel_coord_name="channel") == (
        (1, 1),
        (1, 2),
    )

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 6.14],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:00.70000", "us"),
            ],
        }
    )
    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    assert model.get_resample_ratios(da, channel_coord_name="channel") == (
        (2, 3),
        (3, 1),
    )


def test_virtual_data_filter():
    n_samples = 1000
    n_channels = 300

    data = np.random.rand(n_samples, n_channels).astype(np.float32)
    channel_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_channels - 1],
            "tie_values": [0, 1e4],
        }
    )
    time_coords = xdas.Coordinate(
        {
            "tie_indices": [0, n_samples - 1],
            "tie_values": [
                np.datetime64("2023-01-01T00:00:00", "us"),
                np.datetime64("2023-01-01T00:00:30", "us"),
            ],
        }
    )

    da = xdas.DataArray(
        data=data, coords={"time": time_coords, "channel": channel_coords}
    )

    patching_structure = sbm.PatchingStructure(
        in_samples=100,
        out_samples=100,
        in_channels=100,
        out_channels=100,
        range_samples=(0, 99),
        range_channels=(0, 99),
        overlap_samples=50,
        overlap_channels=50,
    )

    virtual = sbm.VirtualTransformedDataArray(
        da,
        patching_structure,
        filter_samples=("butter", {"N": 4, "Wn": [1, 10], "btype": "bandpass"}),
    )

    sos = virtual.filter_sos
    zi = sosfilt_zi(sos)[:, :, None] * data[0]

    # Filter whole data matrix at once
    filtered, _ = sosfilt(sos, data, zi=zi, axis=0)

    n_samples, n_channels = virtual._get_n_patches()
    for idx_samples in range(n_samples):
        for idx_channels in range(n_channels):
            coord = virtual._patch_coords(idx_samples, idx_channels)
            patch = virtual._load_and_resample_patch(coord)
            patch = virtual._filter_patch(patch, idx_samples, idx_channels)

            assert patch.dtype == data.dtype

            # Validate that patch-wise filtering is identical to filtering the whole data matrix
            assert np.allclose(
                patch,
                filtered[
                    coord.sample_int : coord.sample_int + coord.w_sample,
                    coord.channel_int : coord.channel_int + coord.w_channel,
                ],
            )
