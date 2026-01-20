import numpy as np

from whichaxis import NamedArray

from tests.utils import arr

def test_from_xarray_roundtrip(arr):
    xr = arr.to_xarray()
    back = NamedArray.from_xarray(xr)

    assert back.dims == arr.dims
    np.testing.assert_array_equal(back.data, arr.data)
    for d in arr.dims:
        np.testing.assert_array_equal(back.coords[d], arr.coords[d])


def test_xarray_attrs_preserved(arr):
    xr = arr.to_xarray()
    assert xr.attrs == arr.meta_data
