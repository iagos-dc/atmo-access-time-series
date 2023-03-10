import numpy as np
import pandas as pd
import numba


def subsampling_old(series, n):
    if len(series) > n:
        series = series.sample(n=n, replace=False).sort_index()
    return series


@numba.njit
def _compress_block_of_nans(isnotna):
    """
    Based on the boolean array isnotna indicating not N/A values, masks out the prefix block of nan's and from any
    subsequent block of nan's only a single nan remains.
    :param isnotna: 1-d boolean array
    :return: mask array like isnotna
    """
    m = len(isnotna)
    mask = np.empty_like(isnotna)

    # if array is of size 1 and has nan, keep this nan:
    if m == 1:
        mask[0] = True
        return mask

    block_of_nans = True
    for i in range(m):
        if isnotna[i]:
            if block_of_nans:
                block_of_nans = False
            mask[i] = True
        else:
            if block_of_nans:
                mask[i] = False
            else:
                block_of_nans = True
                mask[i] = True

    return mask


@numba.njit
def _compress_block_of_nans_and_apply_mask(isnotna, mask_for_notna):
    """
    Based on the boolean array isnotna indicating not N/A values, masks out the prefix block of nan's and from any
    subsequent block of nan's only a single nan remains. Additionally, all not nan's are masked according to
    mask_for_notna boolean array.
    :param isnotna: 1-d boolean array
    :param mask_for_notna: 1-d boolean array of size of isnotna.sum()
    :return: mask array like isnotna
    """
    m = len(isnotna)
    mask = np.empty_like(isnotna)

    # if array is of size 1 and has nan, keep this nan:
    if m == 1:
        if isnotna[0]:
            mask[0] = mask_for_notna[0]
        else:
            mask[0] = True
        return mask

    block_of_nans = True
    j = 0
    for i in range(m):
        if isnotna[i]:
            if block_of_nans:
                block_of_nans = False
            mask[i] = mask_for_notna[j]
            j += 1
        else:
            if block_of_nans:
                mask[i] = False
            else:
                block_of_nans = True
                mask[i] = True

    return mask


def get_subsampling_mask(isnotna, n):
    m = isnotna.sum()
    if m > n:
        random_mask = np.random.rand(m) < n / m
        mask = _compress_block_of_nans_and_apply_mask(isnotna, random_mask)
    else:
        mask = _compress_block_of_nans(isnotna)
    return mask


def subsampling(series, n):
    isnotna = ~np.isnan(series.values)
    mask = get_subsampling_mask(isnotna, n)
    return series[mask]
