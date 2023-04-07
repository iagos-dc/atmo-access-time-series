import functools
import toolz
from . import analysis
import xarray as xr


def broadcast_dict(func):
    @functools.wraps(func)
    def func_wrapper(ds, *args, **kwargs):
        if isinstance(ds, dict):
            res = toolz.valmap(lambda _ds: func(_ds, *args, **kwargs), ds)
        else:
            res = func(ds, *args, **kwargs)
        return res
    return func_wrapper


@broadcast_dict
def mean_and_std(da, aggregation_period, min_sample_size=1):
    mean, std, count = analysis.gaussian_mean_and_std(da, aggregation_period, min_sample_size=min_sample_size)
    var_label = da.name
    return xr.Dataset({
        f'{var_label}_mean': mean,
        f'{var_label}_std': std,
        f'{var_label}_count': count,
    }).assign_attrs(da.attrs)


@broadcast_dict
def percentiles(da, aggregation_period, p, min_sample_size=1):
    quantiles_by_p, count = analysis.percentiles(da, aggregation_period, p, min_sample_size=min_sample_size)
    var_label = da.name
    return xr.Dataset(
        {f'{var_label}_p{p}': quantiles for p, quantiles in quantiles_by_p.items()}
    ).assign_attrs(da.attrs)
