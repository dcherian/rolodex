from __future__ import annotations

import copy
import datetime
import enum
import itertools
import operator
from collections.abc import Hashable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr
from xarray import Index, IndexSelResult
from xarray.core.indexing import merge_sel_results
from xarray.indexes import PandasIndex

Timestamp = str | datetime.datetime | pd.Timestamp | np.datetime64
Timedelta = str | datetime.timedelta | np.timedelta64  # TODO: pd.DateOffset also?


def create_lazy_valid_time_variable(
    *, reference_time: xr.DataArray, period: xr.DataArray
) -> xr.DataArray:
    # TODO: work to make this public API upstream
    from xarray.coding.variables import lazy_elemwise_func

    assert reference_time.ndim == 1
    assert period.ndim == 1

    shape = (reference_time.size, period.size)
    time_broadcast = np.broadcast_to(reference_time.data[:, np.newaxis], shape)
    step_broadcast = np.broadcast_to(period.data[np.newaxis, :], shape)

    valid_time = lazy_elemwise_func(
        time_broadcast, partial(operator.add, step_broadcast), dtype=reference_time.dtype
    )
    return xr.DataArray(
        valid_time, dims=(*reference_time.dims, *period.dims), attrs={"standard_name": "time"}
    )


####################
####  Indexer types
# reference: https://www.unidata.ucar.edu/presentations/caron/FmrcPoster.pdf


class Model(enum.StrEnum):
    HRRR = enum.auto()


@dataclass(init=False)
class ModelRun:
    """
    The complete results for a single run is a model run dataset.

    Parameters
    ----------
    time: Timestamp-like
        Initialization time for model run
    """

    time: pd.Timestamp

    def __init__(self, time: Timestamp):
        self.time = pd.Timestamp(time)

    def get_indexer(
        self, model: Model | None, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex
    ) -> tuple[int, slice]:
        time_idxr = time_index.get_loc(self.time)
        period_idxr = slice(None)

        if model is Model.HRRR and self.time.hour % 6 != 0:
            period_idxr = slice(19)

        return time_idxr, period_idxr


@dataclass(init=False)
class ConstantOffset:
    """
    A constant offset dataset is created from all the data that have the same offset time.
    Offset here refers to a variable usually named `step` or `lead` or with CF standard name
    `forecast_period`.
    """

    step: pd.Timedelta

    def __init__(self, step: Timedelta):
        self.step = pd.Timedelta(step)

    def get_indexer(
        self, model: Model | None, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex
    ) -> tuple[slice | np.ndarray, int]:
        time_idxr = slice(None)
        (period_idxr,) = period_index.get_indexer([self.step])

        if model is Model.HRRR and self.step.total_seconds() / 3600 > 18:
            model_mask = np.ones(time_index.shape, dtype=bool)
            model_mask[time_index.hour % 6 != 0] = False
            time_idxr = np.arange(time_index.size)[model_mask]

        return time_idxr, period_idxr


@dataclass(init=False)
class ConstantForecast:
    """
    A constant forecast dataset is created from all the data that have the same forecast/valid time.

    Parameters
    ----------
    time: Timestamp-like

    """

    time: pd.Timestamp

    def __init__(self, time: Timestamp):
        self.time = pd.Timestamp(time)

    def get_indexer(
        self, model: Model | None, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        target = self.time
        max_timedelta = period_index[-1]

        # earliest timestep we can start at
        earliest = target - max_timedelta
        left = time_index.get_slice_bound(earliest, side="left")

        # latest we can get
        right = time_index.get_slice_bound(target, side="right")

        needed_times = time_index[slice(left, right)]
        needed_steps = target - needed_times  # type: ignore
        # print(needed_times, needed_steps)

        needed_time_idxs = np.arange(left, right)
        needed_step_idxs = period_index.get_indexer(needed_steps)
        model_mask = np.ones(needed_time_idxs.shape, dtype=bool)

        # TODO: refactor this out
        if model is Model.HRRR:
            model_mask[(needed_times.hour % 6 != 0) & (needed_steps > pd.to_timedelta("18h"))] = (
                False
            )

        # It's possible we don't have the right step.
        # If pandas doesn't find an exact match it returns -1.
        mask = needed_step_idxs != -1

        needed_step_idxs = needed_step_idxs[model_mask & mask]
        needed_time_idxs = needed_time_idxs[model_mask & mask]

        assert needed_step_idxs.size == needed_time_idxs.size

        return needed_time_idxs, needed_step_idxs


@dataclass
class BestEstimate:
    """
    For each forecast time in the collection, the best estimate for that hour is used to create a
    best estimate dataset, which covers the entire time range of the collection.
    """

    # To restrict the start of the slice,
    # just use the standard `.sel(time=slice(since, None))`
    # TODO: `since` could be a timedelta relative to `asof`.
    # since: pd.Timestamp | None = None
    asof: pd.Timestamp | None = None
    # Start at this step.
    offset: int = 0

    def __post_init__(self):
        if self.asof is not None and self.asof < self.since:
            raise ValueError(
                "Can't request best estimate since {since=!r} "
                "which is earlier than requested {asof=!r}"
            )

    def get_indexer(
        self, model: Model | None, time_index: pd.DatetimeIndex, period_index: pd.TimedeltaIndex
    ) -> tuple[np.ndarray, np.ndarray]:
        if period_index[0] != pd.Timedelta(0):
            raise ValueError(
                "Can't make a best estimate dataset if forecast_period doesn't start at 0."
            )

        # TODO: consolidate the get_indexer lookup
        # if self.since is None:
        first_index = 0
        # else:
        # (first_index,) = time_index.get_indexer([self.since])
        last_index = time_index.size - 1 if self.asof is None else time_index.get_loc(self.asof)

        # TODO: refactor to a Model dataclass that does this filtering appropriately.
        if model is Model.HRRR and time_index[last_index].hour % 6 != 0:
            nsteps = 19
        else:
            nsteps = period_index.size

        time_diff = np.diff(time_index)
        # assume that the period differences are constant
        period_diff = period_index[1] - period_index[0]
        
        n_best_steps_per_forecast = (time_diff / period_diff).astype(int) 

        needed_time_idxrs = np.concatenate(
            [
                np.repeat(np.arange(len(time_index)- 1), n_best_steps_per_forecast)
            ] + [
                np.repeat(last_index, nsteps - self.offset),
            ]
        )

        # assume that there are enough steps to fulfill the requested offset
        needed_step_idxrs = np.concatenate(
            [
                np.arange(self.offset, n_best_steps_per_forecast[i] + self.offset)
                for i in range(len(time_index) - 1)
            ] + [
                np.arange(self.offset, nsteps)
            ]
        )

        return needed_time_idxrs, needed_step_idxrs


@dataclass
class Indexes:
    reference_time: PandasIndex
    period: PandasIndex

    def get_names(self) -> [Hashable, Hashable]:
        # TODO: is this dependable?
        return {"reference_time": self.reference_time.index.name, "period": self.period.index.name}


class ForecastIndex(Index):
    """
    An Xarray custom Index that allows indexing a forecast data-cube with
    `forecast_reference_time` (commonly `init`) and `forecast_period`
    (commonly `step` or `lead`) dimensions as _Forecast Model Run Collections_.


    Examples
    --------
    To do FMRC-style indexing, you'll need to first add a "valid time" variable.

    >>> from rolodex.forecast import (
    ...     BestEstimate,
    ...     ConstantForecast,
    ...     ConstantOffset,
    ...     ForecastIndex,
    ...     ModelRun,
    ...     Model,
    ...     create_lazy_valid_time_variable,
    ... )
    >>> ds.coords["valid_time"] = create_lazy_valid_time_variable(
    ...     reference_time=ds.time, period=ds.step
    ... )

    Create the new index where `time` is the `forecast_reference_time` dimension,
    and `step` is the `forecast_period` dimension.
    >>> newds = ds.drop_indexes(["time", "step"]).set_xindex(
        ["time", "step", "valid_time"], ForecastIndex, model=Model.HRRR
        )
    >>> newds

    Use `valid_time` to indicate FMRC-style indexing

    >>> newds.sel(valid_time=BestEstimate())

    >>> newds.sel(valid_time=ConstantForecast("2024-05-20"))

    >>> newds.sel(valid_time=ConstantOffset("32h"))

    >>> newds.sel(valid_time=ModelRun("2024-05-20 13:00"))
    """

    def __init__(self, variables: Indexes, valid_time_name: str, model: Model | None = None):
        self._indexes = variables

        assert isinstance(valid_time_name, str)
        self.valid_time_name = valid_time_name
        self.model = model

        # We use "reference_time", "period" as internal references.
        self.names = variables.get_names()

    @classmethod
    def from_variables(cls, variables, options):
        """
        Must be created from three variables:
        1. A dummy scalar `forecast` variable.
        2. A variable with the CF attribute`standard_name: "forecast_reference_time"`.
        3. A variable with the CF attribute`standard_name: "forecast_period"`.
        """
        assert len(variables) == 3

        indexes = {}
        valid_time_name: str | None = None
        for k in ["forecast_reference_time", "forecast_period"]:
            for name, var in variables.items():
                std_name = var.attrs.get("standard_name", None)
                if k == std_name:
                    indexes[k.removeprefix("forecast_")] = PandasIndex.from_variables(
                        {name: var}, options={}
                    )
                elif var.ndim == 2:
                    valid_time_name = name

        if valid_time_name is None:
            raise ValueError("Could not detect the 2D 'valid time' variable.")

        assert isinstance(valid_time_name, str)

        if "reference_time" not in indexes:
            raise ValueError(
                "No array with attribute `standard_name: 'forecast_reference_time'` found."
            )
        if "period" not in indexes:
            raise ValueError("No array with attribute `standard_name: 'forecast_period'` found.")

        return cls(Indexes(**indexes), valid_time_name=valid_time_name, **options)

    def sel(self, labels, **kwargs):
        """
        Allows three kinds of indexing
        1. Along the dummy "forecast" variable: enable specialized methods using
           ConstantOffset, ModelRun, ConstantForecast, BestEstimate
        2. Along the `forecast_reference_time` dimension, identical to ModelRun
        3. Along the `forecast_period` dimension, identical to ConstantOffset

        You cannot mix (1) with (2) or (3), but (2) and (3) can be combined in a single statement.
        """
        if self.valid_time_name in labels and len(labels) != 1:
            raise ValueError(
                f"Indexing along {self.valid_time_name!r} cannot be combined with "
                f"indexing along {tuple(self.names)!r}"
            )

        time_name, period_name = self.names["reference_time"], self.names["period"]

        # This allows normal `.sel` along `time_name` and `period_name` to work
        if time_name in labels or period_name in labels:
            if self.valid_time_name in labels:
                raise ValueError(
                    f"Selecting along {time_name!r} or {period_name!r} cannot "
                    f"be combined with FMRC-style indexing along {self.valid_time_name!r}."
                )
            time_index, period_index = self._indexes.reference_time, self._indexes.period
            new_indexes = copy.deepcopy(self._indexes)
            results = []
            if time_name in labels:
                result = time_index.sel({time_name: labels[time_name]}, **kwargs)
                results.append(result)
                idxr = result.dim_indexers[time_name]
                new_indexes.reference_time = new_indexes.reference_time[idxr]
            if period_name in labels:
                result = period_index.sel({period_name: labels[period_name]}, **kwargs)
                results.append(result)
                idxr = result.dim_indexers[period_name]
                new_indexes.period = new_indexes.period[idxr]
            new_index = type(self)(
                new_indexes, valid_time_name=self.valid_time_name, model=self.model
            )
            results.append(
                IndexSelResult(
                    {},
                    indexes={k: new_index for k in [self.valid_time_name, time_name, period_name]},
                )
            )
            return merge_sel_results(results)

        assert len(labels) == 1
        assert next(iter(labels.keys())) == self.valid_time_name

        label = next(iter(labels.values()))
        if not isinstance(label, ConstantOffset | ModelRun | ConstantForecast | BestEstimate):
            if isinstance(label, list | tuple | np.ndarray):
                raise ValueError(
                    f"Along {self.valid_time_name!r}, only indexing with scalars, or one of the FMRC-style indexer objects is allowed."
                )
            label = ConstantForecast(label)

        if TYPE_CHECKING:
            assert isinstance(label, ConstantOffset | ModelRun | ConstantForecast | BestEstimate)

        time_index: pd.DatetimeIndex = self._indexes.reference_time.index  # type: ignore[assignment]
        period_index: pd.TimedeltaIndex = self._indexes.period.index  # type: ignore[assignment]

        time_idxr, period_idxr = label.get_indexer(self.model, time_index, period_index)

        indexer, indexes, variables = {}, {}, {}
        match label:
            case ConstantOffset():
                indexer[time_name] = time_idxr
                indexer[period_name] = period_idxr
                indexes[time_name] = self._indexes.reference_time[time_idxr]
                valid_time_dim = time_name

            case ModelRun():
                indexer[time_name] = time_idxr
                indexer[period_name] = period_idxr
                indexes[period_name] = self._indexes.period[period_idxr]
                valid_time_dim = period_name

            case ConstantForecast():
                indexer = {
                    time_name: xr.Variable(time_name, time_idxr),
                    period_name: xr.Variable(time_name, period_idxr),
                }
                indexes[time_name] = self._indexes.reference_time[time_idxr]
                variables["valid_time"] = xr.Variable((), label.time, {"standard_name": "time"})

            case BestEstimate():
                indexer = {
                    time_name: xr.Variable("valid_time", time_idxr),
                    period_name: xr.Variable("valid_time", period_idxr),
                }
                valid_time_dim = "valid_time"

            case _:
                raise ValueError(f"Invalid indexer type {type(label)} for label: {label}")

        if not isinstance(label, ConstantForecast):
            valid_time = time_index[time_idxr] + period_index[period_idxr]
            variables["valid_time"] = xr.Variable(
                valid_time_dim, data=valid_time, attrs={"standard_name": "time"}
            )
            indexes["valid_time"] = PandasIndex(valid_time, dim=valid_time_dim)

        return IndexSelResult(dim_indexers=indexer, indexes=indexes, variables=variables)

    def __repr__(self):
        string = (
            "<ForecastIndex along ["
            + ", ".join(itertools.chain((self.valid_time_name,), self.names.values()))
            + "]>"
        )
        return string
