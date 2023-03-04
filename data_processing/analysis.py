import numpy as np
import pandas as pd
# from log.log import log_exectime


def aggregate(da, aggregation_period, aggregation_function, min_sample_size=1):
    series_resampled = da.to_series().resample(rule=aggregation_period)
    count = series_resampled.count()
    return aggregation_function(series_resampled).where(count >= min_sample_size)


# @log_exectime
def gaussian_mean_and_std(da, aggregation_period, min_sample_size=1):
    series_resampled = da.to_series().resample(rule=aggregation_period)
    count = series_resampled.count()
    mean = series_resampled.mean()
    std = series_resampled.std(ddof=1)
    return (
        mean.where(count >= min_sample_size),
        std.where(count >= min_sample_size), # / np.sqrt(count),
        count
    )


def gaussian_mean_and_std_by_rolling_window(da, window, min_sample_size=1):
    window = pd.Timedelta(window)
    series_rolled = da.to_series().rolling(window, min_periods=min_sample_size)  # center=True, closed='both'
    count = series_rolled.count()
    mean = series_rolled.mean()
    std = series_rolled.std(ddof=1) # / np.sqrt(count)
    return mean, std, count


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


def percentiles_by_rolling_window(da, window, p, min_sample_size=1):
    window = pd.Timedelta(window)
    q = [x / 100. for x in p]
    series_rolled = da.to_series().rolling(window, min_periods=min_sample_size)  # center=True, closed='both'
    count = series_rolled.count()
    quantiles_by_p = {_p: series_rolled.quantile(_q) for _p, _q in zip(p, q)}
    return quantiles_by_p, count


def linear_regression(df, x_var, y_var):
    no_result = (np.nan, ) * 3
    df = df[[x_var, y_var]].dropna()
    if len(df) <= 1:
        return no_result

    mx, my = df.mean()
    cov = df.cov(ddof=0).values
    cxy = cov[0, 1]
    vx, vy = np.diag(cov)
    if abs(cxy) < 1e-10 and vy < 1e-10:
        return no_result

    a = cxy / vx
    b = my - a * mx
    r2 = cxy / vx * cxy / vy

    return a, b, r2


def theil_sen_slope(series, subsampling=3000, deseasonalize=True):
    series = series.dropna()
    if len(series) <= 1:
        raise ValueError(f'cannot estimate slope from {len(series)} value(s); series={series}')

    if subsampling and len(series) > subsampling:
        series = series.sample(n=subsampling)

    x = series.index.values
    y = series.values

    def cast_to_f8(z):
        if z.dtype.kind == 'M':
            z = z - z[0]
        if z.dtype.kind == 'm':
            z = z - z.mean()
            return z.astype('m8[s]').astype('f8'), np.timedelta64(1, 's')
        else:
            try:
                return z.astype('f8'), 1.
            except Exception as e:
                raise ValueError(f'cannot cast array of dtype={z.dtype} to f8 dtype; {e}')

    x, x_unit = cast_to_f8(x)
    y, y_unit = cast_to_f8(y)

    dx = x[:, np.newaxis] - x
    dy = y[:, np.newaxis] - y
    slope = dy[dx > 0] / dx[dx > 0]

    n = len(x)
    N = n * (n - 1) / 2
    dq = 1.96 * np.sqrt(N * (2 * n + 5)) / 3
    q_low = (N / 2 - dq) / N
    q_high = (N / 2 + dq) / N
    a, ci0, ci1 = np.nanpercentile(slope, [50, 100 * q_low, 100 * q_high])
    b = np.median(y - a * x)

    return (a, b), (ci0, ci1), (x_unit, y_unit)
