def subsampling(series, n):
    if len(series) > n:
        series = series.sample(n=n, replace=False).sort_index()
    return series
