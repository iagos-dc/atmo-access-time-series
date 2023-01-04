import toolz
import xarray as xr


def gaussian_mean_and_std(da_by_var, aggregation_period, calc_std=False, calc_count=False):
    da_resampled_by_var = toolz.valmap(
        lambda da: da.resample({'time': aggregation_period}),
        da_by_var
    )
    darrays_mean = [da_resampled.mean().rename(v) for v, da_resampled in da_resampled_by_var.items()]
    # TODO: this is a temporary patch for the issue of conflicting 'layer' coordinates
    # df = xr.Dataset(da_mean).reset_coords(drop=True).to_dataframe()
    mean = xr.merge(darrays_mean, compat='override').reset_coords(drop=True)

    if calc_std:
        darrays_std = [da_resampled.std(ddof=1).rename(v) for v, da_resampled in da_resampled_by_var.items()]
        std = xr.merge(darrays_std, compat='override').reset_coords(drop=True)
    else:
        std = None

    if calc_count:
        darrays_count = [da_resampled.count().rename(v) for v, da_resampled in da_resampled_by_var.items()]
        count = xr.merge(darrays_count, compat='override').reset_coords(drop=True)
    else:
        count = None

    return mean, std, count


def percentiles(da_by_var, aggregation_period, p, calc_count=False):
    q = [x / 100. for x in p]
    da_resampled_by_var = toolz.valmap(
        lambda da: da.resample({'time': aggregation_period}),
        da_by_var
    )
    darrays_quantiles = [da_resampled.quantile(q).rename(v) for v, da_resampled in da_resampled_by_var.items()]
    quantiles = xr.merge(darrays_quantiles, compat='override').reset_coords(drop=True)

    if calc_count:
        darrays_count = [da_resampled.count().rename(v) for v, da_resampled in da_resampled_by_var.items()]
        count = xr.merge(darrays_count, compat='override').reset_coords(drop=True)
    else:
        count = None

    return quantiles, count
