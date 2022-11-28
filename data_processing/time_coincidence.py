import numpy as np
import pandas as pd
import xarray as xr
import numba

import data_processing


@numba.njit
def _get_max_ind_of_a_leq_b(a, b):
    """
    For each index i in the array b, find the last index j in the array a, such that a[j] <= b[i];
    if it does not exist, j == -1.
    :param a: numpy 1d-array, non-decreasing
    :param b: numpy 1d-array, non-decreasing
    :return: numpy array of int8 of size of the array b
    """
    m = len(a)
    n = len(b)
    ind = np.empty(shape=(n, ), dtype='i8')
    j = 0
    for i in range(n):
        while j < m and a[j] <= b[i]:
            j += 1
        ind[i] = j - 1
    return ind


@numba.njit
def _get_min_ind_of_a_geq_b(a, b):
    """
    For each index i in the array b, find the first index j in the array a, such that a[j] >= b[i];
    if it does not exist, j == len(a).
    :param a: numpy 1d-array, non-decreasing
    :param b: numpy 1d-array, non-decreasing
    :return: numpy array of int8 of size of the array b
    """
    m = len(a)
    n = len(b)
    ind = np.empty(shape=(n, ), dtype='i8')
    j = m - 1
    for i in range(n - 1, -1, -1):
        while j >= 0 and a[j] >= b[i]:
            j -= 1
        ind[i] = j + 1
    return ind


def interpolate_mask_1d(mask_da, target_coord, tolerance, source_dim=None):
    """
    For a given 1-dim boolean array mask_da with coordinates, find boolean mask array with coordinates given by
    target_coord which has True at coordinate value x iff mask_da has at least one True value for a coordinate
    in the interval [x - tolerance, x + tolerance].
    :param mask_da: 1-dim bool xarray DataArray with dimensional coordinates of numerical or numpy.datetime64 dtype
    :param target_coord: 1-dim xarray DataArray or numpy array of numbers or numpy.datetime64
    :param tolerance: number or numpy.timedelta64
    :param source_dim: str, optional; name of mask_da dimensional coordinates
    :return: bool array compatible with target_coord
    """
    # check validity of input
    if len(mask_da.shape) != 1:
        raise ValueError(f'mask_da must be 1-dimensional; mask_da.shape={mask_da.shape}')
    if len(target_coord.shape) != 1:
        raise ValueError(f'target_coord must be 1-dimensional; target_coord.shape={target_coord.shape}')
    if mask_da.dtype != bool:
        raise ValueError(f'mask_da must be of bool dtype; mask_da.dtype={mask_da.dtype}')

    # infer dimension label of mask_da, if not given
    if source_dim is None:
        source_dim, = mask_da.dims
    source_coord = mask_da[source_dim]

    # extract numpy arrays
    source_coord_values = source_coord.values if isinstance(source_coord, xr.DataArray) else np.asanyarray(source_coord)
    target_coord_values = target_coord.values if isinstance(target_coord, xr.DataArray) else np.asanyarray(target_coord)

    # for each coordinate x in target_coord find a low and a high index in mask_da which correspond to the endpoints
    # of the interval [x - tolerance, x + tolerance]
    lo_i = _get_min_ind_of_a_geq_b(source_coord_values, target_coord_values - tolerance)
    hi_i = _get_max_ind_of_a_leq_b(source_coord_values, target_coord_values + tolerance)

    # find a boolean mask array along target_coord
    mask_cumsum = np.cumsum(mask_da.values)
    mask_cumsum_suffix = [mask_cumsum[-1], 0] if len(mask_cumsum) > 0 else [0]
    mask_cumsum_with_guards = np.concatenate((mask_cumsum, mask_cumsum_suffix))
    mask_in_target = mask_cumsum_with_guards[lo_i] < mask_cumsum_with_guards[hi_i]

    # provide the result in an appropriate form (xarray DataArray, if it was so for target_coord)
    if isinstance(target_coord, xr.DataArray):
        target_dim, = target_coord.dims
        mask_in_target = xr.DataArray(mask_in_target, coords={target_dim: target_coord}, dims=[target_dim])
    return mask_in_target


def filter_dataset(da_by_varlabel, rng_by_varlabel, ignore_time=False, tolerance=np.timedelta64(1, 'D')):
    def get_cond_conjunction(conds, target_coords):
        cond_conjunction = True
        for cond in conds:
            cond_conjunction &= interpolate_mask_1d(cond, target_coords, tolerance, source_dim='time')
        return cond_conjunction

    def get_var_range(rng):
        _min, _max = rng
        if isinstance(_min, str):
            _min = pd.Timestamp(_min).to_datetime64()
        if isinstance(_max, str):
            _max = pd.Timestamp(_max).to_datetime64()
        return _min, _max

    cond_by_varlabel = {}
    for v, da in da_by_varlabel.items():
        cond = True
        if v in rng_by_varlabel:
            _min, _max = get_var_range(rng_by_varlabel[v])
            if _min is not None:
                cond &= (da >= _min)
            if _max is not None:
                cond &= (da <= _max)

        if cond is not True:
            cond_by_varlabel[v] = cond

    ds_filtered = {}
    if not ignore_time:
        if 'time' in rng_by_varlabel:
            _t_min, _t_max = get_var_range(rng_by_varlabel['time'])
        else:
            _t_min, _t_max = None, None

    for v, da in da_by_varlabel.items():
        conds = [cond for v_other, cond in cond_by_varlabel.items() if v_other != v]
        cond = get_cond_conjunction(conds, da['time'])
        if ignore_time:
            cond_for_v = cond_by_varlabel.get(v)
            if cond_for_v is not None:
                cond &= cond_for_v
        else:
            if _t_min is not None:
                cond &= da['time'] >= _t_min
            if _t_max is not None:
                cond &= da['time'] <= _t_max
        ds_filtered[v] = da.where(cond, drop=False)

    return ds_filtered
