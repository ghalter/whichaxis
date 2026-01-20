from tests.utils import arr


def test_quantile_extender(arr):
    out = arr.quantile([0.25, 0.75], dim="time")

    assert out.dims == ["quantile", "lat", "lon"]
    assert out.coords["quantile"].tolist() == [0.25, 0.75]
    assert out.data.shape == (2, 3, 4)


def test_percentile_extender(arr):
    out = arr.percentile([10, 90], dim="lat")

    assert out.dims == ["percentile", "time", "lon"]
    assert out.coords["percentile"].tolist() == [10, 90]


def test_rolling_creates_window_dim(arr):
    out = arr.rolling(dim="time", window=2)

    assert out.dims == ["time", "window", "lat", "lon"]
    assert out.coords["window"].tolist() == [0, 1]

    # time dimension shrinks by window-1
    assert out.data.shape == (1, 2, 3, 4)
