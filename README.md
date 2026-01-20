# whichaxis

**NumPy, but you don’t forget which axis is which.**

Have you been lucky enough to buy RAM before 2026?
Flabbergasted because you called `xarray.load()` twice and `apply_ufunc` is *still* not faster?
So you dropped down to NumPy — and immediately forgot whether `axis=1` was `lat` or `lon` before you finished writing
the second line of code?

**Don’t worry. I got you.**

`whichaxis` gives you **NumPy-speed arrays with named axes** for fast in-memory computation, where axes have names and
performance remains at NumPy level.

```python
import numpy as np
from whichaxis import NamedArray

arr = NamedArray(
    data=np.random.rand(2, 3, 4),
    dims=["time", "lat", "lon"],
    coords={
        "time": np.array([2020, 2021]),
        "lat": np.array([10, 20, 30]),
        "lon": np.array([1, 2, 3, 4]),
    },
)

max = arr.max(dim="lat")

```

Simple!

---

## What is `whichaxis`?

* A **thin wrapper** around NumPy arrays
* Naive, half by design, half by ignorance
* Axes have **names**
* Coordinates are **kept**, not interpreted
* All math happens in **NumPy**
* Designed for **hot compute paths**

---

## What it is not

* Not xarray
* Not lazy
* Not distributed
* Not clever
* Not here to save you from bad math

---

## Design Philosophy

`whichaxis` is built on a radical idea:

> **Remembering which axis is which is underrated!**

I believe:

* Computers are very good at adding numbers.
* Humans are very bad at remembering whether `axis=1` was `lat` or `lon`.
* Often, xarray is the go-to tool for labeled arrays.
* But sometimes, all you really want is NumPy speed **with named axes**.

`whichaxis` exists for the moment when:

* your data is already in memory,
* you already know what you’re doing,
* and you just want NumPy to go fast **without gaslighting you about axes**.

If an operation needs:

* alignment,
* broadcasting by coordinate values,
* deferred execution,
* graph rewriting,
* or a PhD in semantic array theory,

then this is not the library for you. Way smarter people have built those tools already.

---

## Non-Goals (Read Carefully)

`whichaxis` will **never**:

* Automatically align data
  *(If two arrays disagree, that’s your problem.)*

* Broadcast by dimension names
  *(Axes are named, not psychic.)*

* Be lazy, chunked, streamed, distributed, or “optimized later”
  *(It is fast **now** or it does not exist.)*

* Replace xarray
  *(I like xarray. I just don’t want its cleverness when I have a clever day of my own)*

* Save you from confusing `lat` and `lon` in your **math**
  *(Only from confusing them in your **code**.)*

* Grow a plugin system, expression engine, or DSL
  *(This is not a lifestyle choice.)*

---

## The Contract

* Axes have names.
* Names map **1-to-1** to NumPy axes.
* Coordinates are metadata, not alignment keys.
* NumPy does the math.
* Nothing happens behind your back.

If you violate the contract, `whichaxis` will not fix it for you.
It will simply do exactly what you asked, very fast.

---

## A Short, Practical Tutorial

This section walks through the core concepts in 10 minutes.

### 1. Creating a `NamedArray`

A `NamedArray` is just:

* a NumPy array
* plus dimension names
* plus 1D coordinate arrays (same length as each axis)

```python
import numpy as np
from whichaxis import NamedArray

data = np.random.rand(2, 3, 4)

arr = NamedArray(
    data=data,
    dims=["time", "lat", "lon"],
    coords={
        "time": np.array([2020, 2021]),
        "lat": np.array([10, 20, 30]),
        "lon": np.array([1, 2, 3, 4]),
    },
)
```

Pretty? Maybe not. Explicit? Definitely!

---

### 2. Indexing (NumPy rules, named results)

Indexing behaves **exactly like NumPy**, but dimension names follow automatically.

```python
arr[0]  # drops "time"
arr[:, 1:]  # keeps all dims
arr[..., 2]  # drops "lon"
arr[:, [0, 2]]  # fancy indexing works
```

Example:

```python
out = arr[0]
print(out.dims)
# ['lat', 'lon']
```

If NumPy drops an axis, `whichaxis` drops the name. No surprises.

---

### 3. `isel`: positional indexing by name

Use `isel` when you want to be explicit.

```python
arr.isel(time=0)
arr.isel(lat=slice(0, 2))
arr.isel(time=[0, 1])
```

Scalar indices drop the dimension:

```python
arr.isel(time=0).dims
# ['lat', 'lon']
```

List/array indices keep the dimension:

```python
arr.isel(time=[0, 1]).dims
# ['time', 'lat', 'lon']
```

---

### 4. `sel`: label-based indexing

`sel` matches **exact coordinate values**.

```python
arr.sel(time=2020)
arr.sel(lat=[10, 30])
```

Rules:

* scalar → drops the dimension
* list/array → keeps the dimension
* no fuzzy matching, no interpolation

```python
arr.sel(time=2020).dims
# ['lat', 'lon']

arr.sel(time=[2020]).dims
# ['time', 'lat', 'lon']
```

---

### 5. Reductions with named dimensions

You can apply the basic NumPy reductions by **dimension name**.

```python
arr.mean(dim="time")
arr.max(dim=["lat", "lon"])
arr.sum(dim="lon", keepdims=True)
```

You never touch `axis=…`.

Internally this is just NumPy:

```python
np.max(arr, axis=0)  # also works
np.mean(arr, dim="time")  # Does not work, use arr.mean(dim="time")
```

---

### 6. NumPy ufuncs and arithmetic

Elementwise operations “just work”.

```python
np.sqrt(arr)
arr + 10
arr * arr
```

Rules:

* dimensions must match exactly
* no broadcasting by name
* no alignment

If shapes don’t match, you get an error — immediately.

---

### 7. Transposing by name (or index)

```python
arr.transpose(["lon", "lat", "time"])
arr.transpose([2, 1, 0])
```

Mixing names and indices is not allowed.

---

### 8. Interop with xarray (explicit boundary)

Convert **in or out**, nothing in between.

```python
xr = arr.to_xarray()
back = NamedArray.from_xarray(xr)
```
---
That instinct is *absolutely correct*.
The README is already clear about philosophy — repeating constraints here just adds noise.

Here’s a **clean, minimal, example-only section** that fits the tone and stays elegant.

You can drop this in verbatim.

---

### 9. Rolling windows

Create sliding windows along a dimension.

```python
out = arr.rolling(dim="time", window=3)
```

This adds a new dimension called `"window"`:

```python
print(out.dims)
# ['time', 'window', 'lat', 'lon']

print(out.coords["window"])
# [0, 1, 2]
```

You can then reduce over the window dimension:

```python
out.mean(dim="window")
out.max(dim="window")
```

---

### 10. Quantiles

Compute quantiles and keep them as a named dimension.

```python
out = arr.quantile([0.1, 0.5, 0.9], dim="time")
```

```python
print(out.dims)
# ['quantile', 'lat', 'lon']

print(out.coords["quantile"])
# [0.1, 0.5, 0.9]
```

---

### 11. Percentiles

Percentiles behave the same way, just in percent.

```python
out = arr.percentile([5, 50, 95], dim="time")
```

```python
print(out.dims)
# ['percentile', 'lat', 'lon']

print(out.coords["percentile"])
# [5, 50, 95]
```


### 12. What to do when this is not enough

If you need:

* alignment
* interpolation
* rolling with boundary logic
* resampling
* labeled broadcasting

Do it in:

* **xarray** (preferred)
* or **NumPy** directly

Then come back to `NamedArray` when things are clean and hot.

---

## NumPy Compatibility

`whichaxis` integrates with NumPy via:

* `__array_ufunc__` → elementwise ops (`+`, `*`, `np.sin`, …)
* `__array_function__` → selected NumPy APIs

The math stays in C.
Only the semantics are wrapped.

### Who is this for?

People who:

* already know NumPy and probably xarray,
* know exactly what they want,
* have already paid the semantic tax,
* and are done pretending that remembering axis order is “part of the fun”.

---

## When to Use / When to Run Away

**Use `whichaxis` when:**

* data is already in memory
* performance matters
* you want NumPy, not a framework

**Run away when:**

* you need alignment
* you need broadcasting by labels
* you don’t fully trust your data yet

---

## Final Words

`whichaxis` is intentionally boring.
Boring code is fast, readable, and correct.

If you want magic, use xarray.
If you are a NumPy wizard doing crazy things, use NumPy directly.
If you just want fast arrays with named axes, `whichaxis` is here for you.
