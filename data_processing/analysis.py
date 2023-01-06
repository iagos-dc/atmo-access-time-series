import toolz
import pandas as pd
import xarray as xr
from log.log import log_exectime


@log_exectime
def gaussian_mean_and_std(da_by_var, aggregation_period, min_sample_size=1):
    series_resampled_by_var = toolz.valmap(
        lambda da: da.to_series().resample(rule=aggregation_period),
        da_by_var
    )
    series_mean = [series_resampled.mean().rename(v) for v, series_resampled in series_resampled_by_var.items()]
    mean = pd.concat(series_mean, axis='columns', join='outer')

    series_std = [series_resampled.std(ddof=1).rename(v) for v, series_resampled in series_resampled_by_var.items()]
    std = pd.concat(series_std, axis='columns', join='outer')

    series_count = [series_resampled.count().rename(v) for v, series_resampled in series_resampled_by_var.items()]
    count = pd.concat(series_count, axis='columns', join='outer')

    return mean.where(count >= min_sample_size), std.where(count >= min_sample_size), count


@log_exectime
def percentiles(da_by_var, aggregation_period, p, min_sample_size=1):
    q = [x / 100. for x in p]
    da_resampled_by_var = toolz.valmap(
        lambda da: da.resample({'time': aggregation_period}),
        da_by_var
    )
    darrays_quantiles = [da_resampled.quantile(q).rename(v) for v, da_resampled in da_resampled_by_var.items()]
    quantiles = xr.merge(darrays_quantiles, compat='override').reset_coords(drop=True)

    darrays_count = [da_resampled.count().rename(v) for v, da_resampled in da_resampled_by_var.items()]
    count = xr.merge(darrays_count, compat='override').reset_coords(drop=True)

    return quantiles.where(count >= min_sample_size), count


@log_exectime
def gaussian_mean_and_std_old(da_by_var, aggregation_period, min_sample_size=1):
    da_resampled_by_var = toolz.valmap(
        lambda da: da.resample({'time': aggregation_period}),
        da_by_var
    )
    darrays_mean = [da_resampled.mean().rename(v) for v, da_resampled in da_resampled_by_var.items()]
    # TODO: this is a temporary patch for the issue of conflicting 'layer' coordinates
    # df = xr.Dataset(da_mean).reset_coords(drop=True).to_dataframe()
    mean = xr.merge(darrays_mean, compat='override').reset_coords(drop=True)

    darrays_std = [da_resampled.std(ddof=1).rename(v) for v, da_resampled in da_resampled_by_var.items()]
    std = xr.merge(darrays_std, compat='override').reset_coords(drop=True)

    darrays_count = [da_resampled.count().rename(v) for v, da_resampled in da_resampled_by_var.items()]
    count = xr.merge(darrays_count, compat='override').reset_coords(drop=True)

    return mean.where(count >= min_sample_size), std.where(count >= min_sample_size), count


@log_exectime
def percentiles_old(da_by_var, aggregation_period, p, min_sample_size=1):
    q = [x / 100. for x in p]
    da_resampled_by_var = toolz.valmap(
        lambda da: da.resample({'time': aggregation_period}),
        da_by_var
    )
    darrays_quantiles = [da_resampled.quantile(q).rename(v) for v, da_resampled in da_resampled_by_var.items()]
    quantiles = xr.merge(darrays_quantiles, compat='override').reset_coords(drop=True)

    darrays_count = [da_resampled.count().rename(v) for v, da_resampled in da_resampled_by_var.items()]
    count = xr.merge(darrays_count, compat='override').reset_coords(drop=True)

    return quantiles.where(count >= min_sample_size), count
