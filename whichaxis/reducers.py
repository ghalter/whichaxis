from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReducerOp:
    name: str
    np_func: callable


REDUCERS = [
    ReducerOp("max", np.max),
    ReducerOp("min", np.min),
    ReducerOp("sum", np.sum),
    ReducerOp("mean", np.mean),
    ReducerOp("prod", np.prod),
    ReducerOp("any", np.any),
    ReducerOp("all", np.all),

    # NaN-aware
    ReducerOp("nanmax", np.nanmax),
    ReducerOp("nanmin", np.nanmin),
    ReducerOp("nansum", np.nansum),
    ReducerOp("nanmean", np.nanmean),

    # Statistical (safe)
    ReducerOp("std", np.std),
    ReducerOp("var", np.var),
    ReducerOp("median", np.median),
    ReducerOp("nanstd", np.nanstd),
    ReducerOp("nanvar", np.nanvar),
    ReducerOp("nanmedian", np.nanmedian),
    ReducerOp("ptp", np.ptp),
]

