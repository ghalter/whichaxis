from typing import Dict, Hashable, Iterable, List

import numpy as np
import xarray as xr
from numpy.lib._stride_tricks_impl import sliding_window_view

from whichaxis.reducers import REDUCERS, _HANDLED_FUNCTIONS, _make_reducer_method


# -----------------------------------------------------------------------------
# NamedArray
# -----------------------------------------------------------------------------
class NamedArray:
    def __init__(
            self,
            data: np.ndarray,
            coords: Dict[Hashable, np.ndarray],
            dims: Iterable[Hashable],
            meta_data: dict | None = None,
    ):
        self.data: np.ndarray = np.asarray(data)
        self.coords: Dict[Hashable, np.ndarray] = dict(coords)
        self.dims: List[Hashable] = list(dims)
        self.meta_data: dict | None = dict(meta_data) if meta_data is not None else None

        # Enforce all invariants up front
        self._validate()

    # -------------------------------------------------------------------------
    # xarray interop
    # -------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, ds: xr.DataArray, meta_data: dict | None = None):
        """
        Converts an xarray DataArray into a NamedArray object.
        This is a boundary operation: semantics stop here.
        """
        coords = {k: np.asarray(ds.coords[k].values) for k in ds.dims}
        meta_data = meta_data if meta_data is not None else dict(ds.attrs)
        return cls(np.asarray(ds.data), coords, ds.dims, meta_data)

    def to_xarray(self):
        """
        Converts this object into an xarray DataArray.
        """
        return xr.DataArray(
            self.data,
            coords={k: v for k, v in self.coords.items()},
            dims=self.dims,
            attrs=self.meta_data or {},
        )

    # -------------------------------------------------------------------------
    # Axis helpers
    # -------------------------------------------------------------------------

    def index_of(self, dim: Hashable) -> int:
        """Return axis index of a dimension name."""
        return self.dims.index(dim)

    def axes(self, dims: Iterable[Hashable]) -> tuple[int, ...]:
        """Translate dimension names to axis indices."""
        return tuple(self.index_of(d) for d in dims)

    def transpose(self, dims: Iterable[str | int]) -> "NamedArray":
        """
        Transpose the array to the given dimension order.
        """
        if not (
                all(isinstance(d, (str, np.str_)) for d in dims)
                or all(isinstance(d, int) for d in dims)):
            raise TypeError("Transpose dims must be all names or all indices")

        if all(isinstance(d, int) for d in dims):
            dims = [self.dims[d] for d in dims]

        if not set(dims) == set(self.dims):
            raise ValueError("Transpose dims must match existing dims")
        return NamedArray(
            np.transpose(self.data, np.array([self.index_of(d) for d in dims])),
            self.coords,
            dims,
            self.meta_data,
        )

    def _normalize_dims(self, dim):
        """
        Normalize dim argument to a tuple of axis indices.
        """
        if dim is None:
            return None
        if isinstance(dim, (list, tuple)):
            return tuple(self.index_of(d) for d in dim)
        return (self.index_of(dim),)

    def _drop_axes(self, axes):
        """
        Drop axes after a reduction and return new dims + coords.
        """
        axes = set(axes)
        new_dims = [d for i, d in enumerate(self.dims) if i not in axes]
        new_coords = {d: self.coords[d] for d in new_dims}
        return new_dims, new_coords

    # -------------------------------------------------------------------------
    # Selecting Data
    # -------------------------------------------------------------------------
    def isel(self, **indexers) -> "NamedArray":
        """
        Positional indexing by dimension name.
        Equivalent to NumPy-style indexing, but explicit.
        """
        index = [slice(None)] * self.data.ndim

        for dim, idx in indexers.items():
            axis = self.index_of(dim)
            index[axis] = idx

        return self[tuple(index)]

    def sel(self, **indexers) -> "NamedArray":
        """
        Label-based selection by coordinate value.
        Exact matches only.

        Scalar labels drop the dimension.
        Multiple labels keep the dimension.
        """
        isel_indexers = {}

        for dim, value in indexers.items():
            coord = self.coords[dim]

            if isinstance(value, (list, tuple, np.ndarray)):
                idx = np.nonzero(np.isin(coord, value))[0]
                if idx.size == 0:
                    raise KeyError(f"Values {value!r} not found in coord '{dim}'")
                isel_indexers[dim] = idx
            else:
                idx = np.nonzero(coord == value)[0]
                if idx.size == 0:
                    raise KeyError(f"Value {value!r} not found in coord '{dim}'")
                # 🔑 scalar → drop dimension
                isel_indexers[dim] = int(idx[0])

        return self.isel(**isel_indexers)

    # -------------------------------------------------------------------------
    # NumPy-style indexing
    # -------------------------------------------------------------------------

    def __getitem__(self, index):
        """
        Pure NumPy indexing.
        Dims drop exactly when NumPy drops them.
        Coords follow mechanically.
        """
        data = self.data[index]

        # Normalize index to tuple
        if not isinstance(index, tuple):
            index = (index,)

        # --- Expand Ellipsis explicitly ---
        expanded = []
        for idx in index:
            if idx is Ellipsis:
                remaining = len(self.dims) - (len(index) - 1)
                expanded.extend([slice(None)] * remaining)
            else:
                expanded.append(idx)

        new_dims = []
        new_coords = {}

        dim_i = 0
        for idx in expanded:
            dim = self.dims[dim_i]

            if isinstance(idx, int):
                # Integer indexing drops the dimension
                dim_i += 1
                continue

            # slice / array / list → dimension survives
            new_dims.append(dim)
            new_coords[dim] = self.coords[dim][idx]
            dim_i += 1

        for j in range(dim_i, len(self.dims)):
            dim = self.dims[j]
            new_dims.append(dim)
            new_coords[dim] = self.coords[dim]

        return NamedArray(data, new_coords, new_dims, self.meta_data)

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def _validate(self):
        """
        Enforce all structural invariants.
        Fail fast and loudly.
        """
        if self.data.ndim != len(self.dims):
            raise ValueError(
                f"Data has {self.data.ndim} dims, "
                f"but {len(self.dims)} dim names were given"
            )

        if set(self.dims) != set(self.coords):
            raise ValueError(
                "Each dim must have exactly one coordinate array"
            )

        for i, d in enumerate(self.dims):
            coord = self.coords[d]
            if coord.ndim != 1:
                raise ValueError(f"Coord '{d}' must be 1D")
            if coord.shape[0] != self.data.shape[i]:
                raise ValueError(
                    f"Coord '{d}' length {coord.shape[0]} "
                    f"does not match data axis {i} ({self.data.shape[i]})"
                )

    # -------------------------------------------------------------------------
    # Generic reducer (used by all reductions)
    # -------------------------------------------------------------------------

    def _reduce(self, fn, dim=None, keepdims=False):
        """
        Generic reduction helper.
        NumPy does the math; we only manage axes and metadata.
        """
        axes = self._normalize_dims(dim)
        data = fn(self.data, axis=axes, keepdims=keepdims)

        if axes is None:
            return NamedArray(data, {}, [], self.meta_data)

        if keepdims:
            # shrink coords for reduced axes to length 1
            new_coords = dict(self.coords)
            for ax in axes:
                dim_name = self.dims[ax]
                new_coords[dim_name] = new_coords[dim_name][:1]

            return NamedArray(data, new_coords, self.dims, self.meta_data)

        new_dims, new_coords = self._drop_axes(axes)
        return NamedArray(data, new_coords, new_dims, self.meta_data)

    # -------------------------------------------------------------------------
    # NumPy protocol hooks
    # -------------------------------------------------------------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Elementwise NumPy ufunc support.
        No alignment. No broadcasting by name.
        """
        if method != "__call__":
            return NotImplemented

        arrays = [
            x.data if isinstance(x, NamedArray) else x
            for x in inputs
        ]

        result = ufunc(*arrays, **kwargs)
        return NamedArray(result, self.coords, self.dims, self.meta_data)

    def __array_function__(self, func, types, args, kwargs):
        """
        Intercept a very small whitelist of NumPy functions.
        Everything else falls back to NumPy.
        """
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented
        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    # -----------------------------------------------------------------------------
    # Extenders (dimension-creating operations)
    # -----------------------------------------------------------------------------

    def quantile(self, q, dim):
        q = np.atleast_1d(q)
        axes = self._normalize_dims(dim)
        data = np.quantile(self.data, q, axis=axes)

        new_dims, new_coords = self._drop_axes(axes)

        return NamedArray(
            data=data,
            dims=["quantile", *new_dims],
            coords={"quantile": q, **new_coords},
            meta_data=self.meta_data,
        )

    def percentile(self, p, dim):
        p = np.atleast_1d(p)
        axes = self._normalize_dims(dim)
        data = np.percentile(self.data, p, axis=axes)

        new_dims, new_coords = self._drop_axes(axes)

        return NamedArray(
            data=data,
            dims=["percentile", *new_dims],
            coords={"percentile": p, **new_coords},
            meta_data=self.meta_data,
        )

    from numpy.lib.stride_tricks import sliding_window_view
    import numpy as np

    def rolling(self, dim, window):
        axis = self.index_of(dim)

        if window < 1:
            raise ValueError("window must be >= 1")
        if window > self.data.shape[axis]:
            raise ValueError("window larger than dimension length")

        # NumPy creates window axis at the end
        data = sliding_window_view(self.data, window, axis=axis)

        # Move window axis to axis+1
        window_axis = data.ndim - 1
        data = np.moveaxis(data, window_axis, axis + 1)

        # Build dims
        new_dims = list(self.dims)
        new_dims.insert(axis + 1, "window")

        # Build coords
        new_coords = {}
        for i, d in enumerate(self.dims):
            coord = self.coords[d]
            if i == axis:
                new_coords[d] = coord[: data.shape[axis]]
            else:
                new_coords[d] = coord

        new_coords["window"] = np.arange(window)

        return NamedArray(
            data=data,
            dims=new_dims,
            coords=new_coords,
            meta_data=self.meta_data,
        )


def _make_numpy_wrapper(method_name):
    """
    Create a NumPy-level wrapper (np.max(arr, dim=...), etc).
    """

    def wrapper(*args, **kwargs):
        arr = args[0]
        if not isinstance(arr, NamedArray):
            return NotImplemented

        dim = kwargs.pop("dim", None)
        axis = kwargs.pop("axis", None)
        keepdims = kwargs.pop("keepdims", False)

        if kwargs:
            raise TypeError(f"Unsupported kwargs: {list(kwargs)}")

        if dim is not None:
            return getattr(arr, method_name)(dim=dim, keepdims=keepdims)

        if axis is not None:
            dims = (
                [arr.dims[a] for a in axis]
                if isinstance(axis, (list, tuple))
                else arr.dims[axis]
            )
            return getattr(arr, method_name)(dim=dims, keepdims=keepdims)

        return getattr(arr, method_name)(keepdims=keepdims)

    return wrapper


# Install reducers once, at import time
for op in REDUCERS:
    setattr(NamedArray, op.name, _make_reducer_method(op.np_func))
    _HANDLED_FUNCTIONS[op.np_func] = _make_numpy_wrapper(op.name)
