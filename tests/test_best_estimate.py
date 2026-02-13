import numpy as np
import pandas as pd
import pytest
import xarray as xr

from rolodex.forecast import (
    BestEstimate,
    ForecastIndex,
    create_lazy_valid_time_variable,
)


def _make_dataset(time_index, n_steps, n_x=2):
    """Helper to create a forecast dataset with a ForecastIndex."""
    step = pd.to_timedelta(np.arange(n_steps), unit="h")

    ds = xr.Dataset()
    ds["foo"] = (("x", "time", "step"), np.ones((n_x, len(time_index), n_steps)))
    ds["time"] = ("time", time_index, {"standard_name": "forecast_reference_time"})
    ds["step"] = ("step", step, {"standard_name": "forecast_period"})
    ds.coords["valid_time"] = create_lazy_valid_time_variable(
        reference_time=ds.time, period=ds.step
    )
    return ds.drop_indexes(["time", "step"]).set_xindex(
        ["time", "step", "valid_time"], ForecastIndex
    )


def test_best_estimate_with_time_gap_longer_than_step_range():
    """BestEstimate should not fail when a gap in 'time' exceeds the 'step' duration."""
    # 5 hourly runs, then a 72-hour gap, then 5 more hourly runs
    # step only covers 49 hours, so the gap is larger than the step range
    times = pd.DatetimeIndex(
        pd.date_range("2023-01-01", periods=5, freq="h").tolist()
        + pd.date_range("2023-01-04", periods=5, freq="h").tolist()
    )
    ds = _make_dataset(times, n_steps=49)

    subset = ds.sel(valid_time=BestEstimate())
    assert "valid_time" in subset.dims
    assert not subset.foo.isnull().any().item()


def test_best_estimate_contiguous():
    """BestEstimate works on a normal contiguous hourly time series."""
    times = pd.date_range("2023-01-01", periods=48, freq="h")
    ds = _make_dataset(times, n_steps=49)

    subset = ds.sel(valid_time=BestEstimate())
    assert "valid_time" in subset.dims
    assert not subset.foo.isnull().any().item()


def test_best_estimate_single_large_gap():
    """BestEstimate with a gap much larger than the step range (e.g. 1 week)."""
    times = pd.DatetimeIndex(
        pd.date_range("2023-01-01", periods=3, freq="h").tolist()
        + pd.date_range("2023-01-08", periods=3, freq="h").tolist()
    )
    ds = _make_dataset(times, n_steps=24)

    subset = ds.sel(valid_time=BestEstimate())
    assert "valid_time" in subset.dims
    assert not subset.foo.isnull().any().item()
