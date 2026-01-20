from typing import Dict, Hashable, Iterable, List

import numpy as np
import xarray as xr
from numpy.lib._stride_tricks_impl import sliding_window_view

from whichaxis.reducers import REDUCERS


# -----------------------------------------------------------------------------
# NamedArray
# -----------------------------------------------------------------------------
class NamedArray:
    """
    NumPy array with named axes and explicit coordinates.

    A NamedArray wraps a NumPy ndarray and associates each axis with:
    - a dimension name
    - a 1D coordinate array of matching length

    The goal is to preserve axis meaning in hot compute paths without
    introducing alignment, broadcasting, or lazy semantics.

    Attributes:
        data (np.ndarray): Underlying NumPy array.
        dims (list[Hashable]): Names of each axis.
        coords (dict[Hashable, np.ndarray]): Coordinate arrays for each axis.
        meta_data (dict | None): Optional metadata passed through operations.
    """

    def __init__(
            self,
            data: np.ndarray,
            coords: Dict[Hashable, np.ndarray],
            dims: Iterable[Hashable],
            meta_data: dict | None = None,
    ):
        self.data: np.ndarray = np.asarray(data)
        self.coords: Dict[Hashable, np.ndarray] = {k: np.asarray(v) for k, v in coords.items()}
        self.dims: List[Hashable] = list(dims)
        self.meta_data: dict | None = dict(meta_data) if meta_data is not None else None

        # Enforce all invariants up front
        self._validate()

    # -------------------------------------------------------------------------
    # xarray interop
    # -------------------------------------------------------------------------

    @classmethod
    def from_xarray(cls, da: xr.DataArray, meta_data: dict | None = None):
        """
        Construct a NamedArray from an xarray DataArray.
    
        Args:
            da (xr.DataArray): Source xarray DataArray.
            meta_data (dict | None): Optional metadata. If not provided,
                ``ds.attrs`` is copied.
    
        Returns:
            NamedArray: A new NamedArray with identical data, dims, and coords.
        """
        coords = {k: np.asarray(da.coords[k].values) for k in da.dims}
        meta_data = meta_data if meta_data is not None else dict(da.attrs)
        return cls(np.asarray(da.data), coords, da.dims, meta_data)

    def to_xarray(self):
        """
        Convert this NamedArray into an xarray DataArray.

        Returns:
            xr.DataArray: Equivalent xarray DataArray.
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
        """
        Return the axis index corresponding to a dimension name.

        Args:
            dim (Hashable): Dimension name.

        Returns:
            int: Axis index.

        Raises:
            ValueError: If the dimension name is not present.
        """
        return self.dims.index(dim)

    def axes(self, dims: Iterable[Hashable]) -> tuple[int, ...]:
        """
        Translate dimension names to axis indices.

        Args:
            dims (Iterable[Hashable]): Dimension names.

        Returns:
            tuple[int, ...]: Corresponding axis indices.
        """
        return tuple(self.index_of(d) for d in dims)

    def transpose(self, dims: Iterable[str | int]) -> "NamedArray":
        """
        Reorder axes by dimension name or axis index.

        Args:
            dims (Iterable[str | int]): New axis order, specified either
                entirely by dimension names or entirely by axis indices.

        Returns:
            NamedArray: Transposed array.

        Raises:
            TypeError: If names and indices are mixed.
            ValueError: If the provided dimensions do not match existing dims.
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
        """ Normalize dim argument to a tuple of axis indices."""
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

        This is equivalent to NumPy indexing, but explicit about which
        dimension is being indexed.

        Scalar indices drop the dimension.
        Slice or array indices keep the dimension.

        Args:
            **indexers: Mapping from dimension name to index, slice,
                or index array.

        Returns:
            NamedArray: Indexed result.
        """
        index = [slice(None)] * self.data.ndim

        for dim, idx in indexers.items():
            axis = self.index_of(dim)
            index[axis] = idx

        return self[tuple(index)]

    def sel(self, **indexers) -> "NamedArray":
        """
        Label-based selection using coordinate values.

        Matches exact coordinate values only.
        No interpolation or fuzzy matching is performed.

        Scalar values drop the dimension.
        List or array values keep the dimension.

        Args:
            **indexers: Mapping from dimension name to coordinate value
                or list of values.

        Returns:
            NamedArray: Selected result.

        Raises:
            KeyError: If a value is not found in the coordinate array.
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
        NumPy-style indexing.

        Indexing follows NumPy rules exactly.
        Dimension names and coordinates are updated mechanically.

        If NumPy drops an axis, the corresponding dimension name is dropped.

        Args:
            index: Any valid NumPy index expression.

        Returns:
            NamedArray: Indexed result.
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
        Apply a NumPy reduction along one or more dimensions.

        Args:
            fn (callable): NumPy reduction function (e.g. np.mean).
            dim (Hashable | Iterable[Hashable] | None): Dimension(s) to reduce.
                If None, reduce over all dimensions.
            keepdims (bool): Whether to keep reduced dimensions with length 1.

        Returns:
            NamedArray: Reduced result.
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

    def max(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Maximum over one or more dimensions."""
        return self._reduce(np.max, dim=dim, keepdims=keepdims)

    def min(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Minimum over one or more dimensions."""
        return self._reduce(np.min, dim=dim, keepdims=keepdims)

    def sum(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Sum over one or more dimensions."""
        return self._reduce(np.sum, dim=dim, keepdims=keepdims)

    def mean(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Mean over one or more dimensions."""
        return self._reduce(np.mean, dim=dim, keepdims=keepdims)

    def prod(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Product over one or more dimensions."""
        return self._reduce(np.prod, dim=dim, keepdims=keepdims)

    def any(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Logical OR over one or more dimensions."""
        return self._reduce(np.any, dim=dim, keepdims=keepdims)

    def all(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Logical AND over one or more dimensions."""
        return self._reduce(np.all, dim=dim, keepdims=keepdims)

        # -------------------------------------------------------------------------
        # NaN-aware reducers
        # -------------------------------------------------------------------------

    def nanmax(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware maximum."""
        return self._reduce(np.nanmax, dim=dim, keepdims=keepdims)

    def nanmin(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware minimum."""
        return self._reduce(np.nanmin, dim=dim, keepdims=keepdims)

    def nansum(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware sum."""
        return self._reduce(np.nansum, dim=dim, keepdims=keepdims)

    def nanmean(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware mean."""
        return self._reduce(np.nanmean, dim=dim, keepdims=keepdims)

        # -------------------------------------------------------------------------
        # Statistical reducers (safe, shape-preserving)
        # -------------------------------------------------------------------------

    def std(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Standard deviation."""
        return self._reduce(np.std, dim=dim, keepdims=keepdims)

    def var(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Variance."""
        return self._reduce(np.var, dim=dim, keepdims=keepdims)

    def median(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Median."""
        return self._reduce(np.median, dim=dim, keepdims=keepdims)

    def nanstd(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware standard deviation."""
        return self._reduce(np.nanstd, dim=dim, keepdims=keepdims)

    def nanvar(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware variance."""
        return self._reduce(np.nanvar, dim=dim, keepdims=keepdims)

    def nanmedian(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """NaN-aware median."""
        return self._reduce(np.nanmedian, dim=dim, keepdims=keepdims)

    def ptp(self, dim=None, keepdims: bool = False) -> "NamedArray":
        """Peak-to-peak (max - min)."""
        return self._reduce(np.ptp, dim=dim, keepdims=keepdims)

    # -----------------------------------------------------------------------------
    # Extenders (dimension-creating operations)
    # -----------------------------------------------------------------------------

    def quantile(self, q, dim):
        """
        Compute quantiles along a dimension and create a new ``quantile`` axis.

        Args:
            q (float | array-like): Quantile or quantiles in [0, 1].
            dim (Hashable): Dimension along which to compute quantiles.

        Returns:
            NamedArray: Result with a new ``quantile`` dimension.
        """
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
        """
        Compute percentiles along a dimension and create a new ``percentile`` axis.

        Args:
            p (float | array-like): Percentile or percentiles in [0, 100].
            dim (Hashable): Dimension along which to compute percentiles.

        Returns:
            NamedArray: Result with a new ``percentile`` dimension.
        """
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

    def rolling(self, dim, window):
        """
        Create rolling windows along a dimension.

        A new dimension called ``window`` is inserted directly after
        the rolling dimension.

        The underlying data is a view created via ``sliding_window_view``.

        Args:
            dim (Hashable): Dimension along which to create windows.
            window (int): Window size. Must be >= 1 and <= dimension length.

        Returns:
            NamedArray: Array with an added ``window`` dimension.

        Raises:
            ValueError: If window size is invalid.
        """
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

    def __repr__(self) -> str:
        dims = ", ".join(f"{d}:{s}" for d, s in zip(self.dims, self.data.shape))
        return (
            f"NamedArray({dims}, dtype={self.data.dtype})"
        )

    def __str__(self) -> str:
        def _format_bytes(nbytes: int) -> str:
            if nbytes < 1024:
                return f"{nbytes} B"
            if nbytes < 1024 ** 2:
                return f"{nbytes / 1024:.2f} KB"
            if nbytes < 1024 ** 3:
                return f"{nbytes / 1024 ** 2:.2f} MB"
            return f"{nbytes / 1024 ** 3:.2f} GB"

        size = _format_bytes(self.data.nbytes)
        lines = []

        # Header
        lines.append(f"NamedArray ({size})")
        lines.append(f"  dims: {self.dims}")
        lines.append(f"  shape: {self.data.shape}")
        lines.append(f"  dtype: {self.data.dtype}")

        # Coordinates
        lines.append("  coords:")
        for d in self.dims:
            coord = self.coords[d]
            preview = np.array2string(
                coord[:5],
                separator=", ",
                threshold=5,
            )
            suffix = " ..." if coord.size > 5 else ""
            lines.append(f"    {d}: {preview}{suffix}")

        # Data preview
        lines.append("  data:")
        preview = np.array2string(
            self.data,
            max_line_width=80,
            threshold=10,
        )
        lines.append(f"    {preview}")

        return "\n".join(lines)


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
_HANDLED_FUNCTIONS = {}
for op in REDUCERS:
    _HANDLED_FUNCTIONS[op.np_func] = _make_numpy_wrapper(op.name)
