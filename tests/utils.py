import numpy as np
import pytest

from whichaxis import NamedArray


@pytest.fixture
def arr():
    data = np.arange(2 * 3 * 4).reshape(2, 3, 4)
    return NamedArray(
        data=data,
        dims=["time", "lat", "lon"],
        coords={
            "time": np.array([2020, 2021]),
            "lat": np.array([10, 20, 30]),
            "lon": np.array([1, 2, 3, 4]),
        },
        meta_data={"unit": "test"},
    )
