import numpy as np

from whichaxis import NamedArray

if __name__ == "__main__":
    # Create a small array with named axes
    arr = NamedArray(
        data=np.array([[1, 2, 3], [4, 5, 6]]),
        dims=["y", "x"],
        coords={
            "y": ["a", "b"],
            "x": [10, 20, 30],
        },
    )

    print("Original array:")
    print(arr)
    print("-" * 40 + "\n\n")

    # NumPy-style indexing, names follow automatically
    print("Indexing arr[0]:")
    print(arr[0])
    print("-" * 40 + "\n\n")

    # Explicit positional indexing by name
    print("isel(y=1):")
    print(arr.isel(y=1))
    print("-" * 40 + "\n\n")

    # Reduction by dimension name
    print("Mean over x:")
    print(arr.mean(dim="x"))
    print("-" * 40 + "\n\n")

    # Computing quantiles
    print("Three quantiles over y:")
    print(arr.quantile([0.25, 0.5, 0.75], dim="y"))
    print("-" * 40 + "\n\n")

    # Convert to xarray
    print("Convert to xarray:")
    xr_arr = arr.to_xarray()
    print(xr_arr)
    print("-" * 40 + "\n\n")

    # Convert to pandas DataFrame
    print("Convert to pandas DataFrame:")
    df = arr.to_xarray().to_dataframe(name="value").reset_index()
    print(df)
    print("-" * 40 + "\n\n")

    # Rolling window example
    print("Rolling window of size 2 over x:")
    rolled = arr.rolling(dim="x", window=3).mean(dim="window")
    print(rolled)

