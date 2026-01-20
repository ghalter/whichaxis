import numpy as np
import pytest

from tests.utils import arr
from whichaxis import NamedArray


def test_shape_matches_dims(arr):
    assert arr.data.shape == (2, 3, 4)
    assert arr.dims == ["time", "lat", "lon"]


def test_invalid_coord_length():
    with pytest.raises(ValueError):
        NamedArray(
            data=np.zeros((2, 3)),
            dims=["x", "y"],
            coords={"x": np.arange(2), "y": np.arange(4)},
        )


def test_integer_index_drops_dim(arr):
    out = arr[0]
    assert out.dims == ["lat", "lon"]
    assert out.data.shape == (3, 4)


def test_slice_keeps_dim(arr):
    out = arr[:, 1:]
    assert out.dims == ["time", "lat", "lon"]
    assert out.data.shape == (2, 2, 4)


def test_ellipsis(arr):
    out = arr[..., 2]
    assert out.dims == ["time", "lat"]
    assert out.data.shape == (2, 3)


def test_fancy_indexing(arr):
    out = arr[:, [0, 2]]
    assert out.dims == ["time", "lat", "lon"]
    assert out.data.shape == (2, 2, 4)
    np.testing.assert_array_equal(out.coords["lat"], [10, 30])


def test_isel_scalar(arr):
    out = arr.isel(time=1)
    assert out.dims == ["lat", "lon"]
    assert out.coords["lat"].size == 3


def test_isel_slice(arr):
    out = arr.isel(lat=slice(0, 2))
    assert out.coords["lat"].tolist() == [10, 20]


def test_sel_scalar(arr):
    out = arr.sel(time=2020)
    assert out.dims == ["lat", "lon"]


def test_sel_list(arr):
    out = arr.sel(lat=[10, 30])
    assert out.coords["lat"].tolist() == [10, 30]


def test_sel_missing_raises(arr):
    with pytest.raises(KeyError):
        arr.sel(time=1999)


def test_reduce_dim(arr):
    out = arr.mean(dim="time")
    assert out.dims == ["lat", "lon"]
    assert out.data.shape == (3, 4)


def test_reduce_multi_dim(arr):
    out = arr.sum(dim=["lat", "lon"])
    assert out.dims == ["time"]


def test_keepdims(arr):
    out = arr.max(dim="lat", keepdims=True)
    assert out.dims == ["time", "lat", "lon"]
    assert out.data.shape[1] == 1


def test_numpy_max(arr):
    out = np.max(arr, axis=0)
    assert out.dims == ["lat", "lon"]


def test_array_max(arr):
    out = arr.max(dim="time")
    assert out.dims == ["lat", "lon"]


def test_numpy_axis(arr):
    out = np.mean(arr, axis=0)
    assert out.dims == ["lat", "lon"]


def test_transpose_by_name(arr):
    out = arr.transpose(["lon", "lat", "time"])
    assert out.dims == ["lon", "lat", "time"]
    assert out.data.shape == (4, 3, 2)


def test_transpose_by_index(arr):
    out = arr.transpose([2, 1, 0])
    assert out.dims == ["lon", "lat", "time"]


def test_transpose_mixed_raises(arr):
    with pytest.raises(TypeError):
        arr.transpose(["time", 1])


def test_metadata_preserved(arr):
    out = arr.mean(dim="time")
    assert out.meta_data == arr.meta_data


def test_sel_scalar_vs_list(arr):
    out_scalar = arr.sel(time=2020)
    out_list = arr.sel(time=[2020])

    assert out_scalar.dims == ["lat", "lon"]
    assert out_list.dims == ["time", "lat", "lon"]
    assert out_list.coords["time"].size == 1
