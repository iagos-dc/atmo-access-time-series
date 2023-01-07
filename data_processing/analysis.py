import pandas as pd
# from log.log import log_exectime


# @log_exectime
def gaussian_mean_and_std(da, aggregation_period, min_sample_size=1):
    series_resampled = da.to_series().resample(rule=aggregation_period)
    count = series_resampled.count()
    mean = series_resampled.mean()
    std = series_resampled.std(ddof=1)
    return mean.where(count >= min_sample_size), std.where(count >= min_sample_size), count


# @log_exectime
def percentiles(da, aggregation_period, p, min_sample_size=1):
    q = [x / 100. for x in p]
    series_resampled = da.to_series().resample(rule=aggregation_period)
    count = series_resampled.count()
    if len(q) > 0:
        quantiles = series_resampled.quantile(q=q).unstack()
    else:
        quantiles = pd.DataFrame(index=count.index)
    quantiles = quantiles.where(count >= min_sample_size)
    quantiles_by_p = {_p: quantiles[_q] for _p, _q in zip(p, q)}
    return quantiles_by_p, count
