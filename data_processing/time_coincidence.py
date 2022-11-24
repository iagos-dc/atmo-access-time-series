import numpy as np
import xarray as xr
import numba


@numba.njit
def _get_max_ind_of_a_leq_b(a, b):
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
    if len(mask_da.shape) != 1:
        raise ValueError(f'mask_da must be 1-dimensional; mask_da.shape={mask_da.shape}')
    if len(target_coord.shape) != 1:
        raise ValueError(f'target_coord must be 1-dimensional; target_coord.shape={target_coord.shape}')
    if mask_da.dtype != bool:
        raise ValueError(f'mask_da must be of bool dtype; mask_da.dtype={mask_da.dtype}')

    if source_dim is None:
        source_dim, = mask_da.dims
    source_coord = mask_da[source_dim]

    source_coord_values = source_coord.values if isinstance(source_coord, xr.DataArray) else np.asanyarray(source_coord)
    target_coord_values = target_coord.values if isinstance(target_coord, xr.DataArray) else np.asanyarray(target_coord)

    lo_i = _get_min_ind_of_a_geq_b(source_coord_values, target_coord_values - tolerance)
    hi_i = _get_max_ind_of_a_leq_b(source_coord_values, target_coord_values + tolerance)

    mask_cumsum = np.cumsum(mask_da.values)
    mask_cumsum_with_guards = np.concatenate((mask_cumsum, [mask_cumsum[-1]], [0]))
    mask_in_target = mask_cumsum_with_guards[lo_i] < mask_cumsum_with_guards[hi_i]
    if isinstance(target_coord, xr.DataArray):
        target_dim, = target_coord.dims
        mask_in_target = xr.DataArray(mask_in_target, coords={target_dim: target_coord}, dims=[target_dim])
    return mask_in_target
