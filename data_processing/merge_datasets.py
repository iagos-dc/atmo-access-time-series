import numpy as np
import pandas as pd
import xarray as xr
from log.log import logger
from . import metadata


def dim_coords_sorted_and_unique(ds, dims, warnings=None):
    """
    Make dimensional coordinates strictly increasing by sorting and removing duplicates and NA
    :param ds: xarray Dataset or DataArray
    :param dims: str or list of str; label(s) of dimensional coordinates of the dataset
    :param warnings: a list or None; if a list, it can be extended with warning messages
    :return: xarray Dataset or DataArray
    """
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    for dim in dims:
        dim_notnull_mask = ds[dim].notnull()
        if dim_notnull_mask.any():
            ds = ds.where(dim_notnull_mask, drop=True)
        if not ds.indexes[dim].is_monotonic_increasing:
            ds = ds.sortby(dim)
            warning_msg = f'coordinates of the dimension {dim} in {ds.xrx.short_dataset_repr()} ' \
                          f'are not increasing and hence has been sorted'
            if warnings is not None:
                warnings.append(warning_msg)
            else:
                logger().warning(warning_msg)
        if not ds.indexes[dim].is_unique:
            t = ds[dim]
            dt = t.diff(dim)
            duplicated_t = dt.where(dt <= xr.zeros_like(dt), drop=True)[dim]
            warning_msg = f'coordinates of the dimension {dim} in {ds} are not unique; ' \
                          f'there are {len(duplicated_t)} duplicates: {duplicated_t}'
            if warnings is not None:
                warnings.append(warning_msg)
            else:
                logger().warning(warning_msg)
            mask = t.isin(duplicated_t)
            ds = ds.where(~mask, drop=True)
    return ds


def interpolate(ds, coords, subset=None, interp_nondim_coords=False, method='linear', kwargs=None, warnings=None):
    """
    Interpolate a dataset to new coordinates. This is a wrapper around xarray.Dataset.interp.
    :param ds: xarray Dataset or DataArray; a dataset to interpolate
    :param coords: Mapping from dimension names to the new coordinates. New coordinate can be an scalar,
    array-like or DataArray. If DataArrays are passed as new coordinates, their dimensions are used for the broadcasting.
    Missing values are propagated (or skipped???)
    :param subset: str or list of str; default None; list of variables to be interpolated;
    if None then interpolate all variables; for interpolating or not coordinates see 'interp_nondim_coords' switch;
    use subset=[] and interp_nondim_coords=True to interpolate non-dimensional coordinates only
    :param interp_nondim_coords: bool, default False; whether to interpolate non-dimensional coordinates
    :param method: see xarray.Dataset.interp
    :param kwargs: passed to xarray.Dataset.interp
    :param warnings: a list of warning messages
    :return: xarray Dataset or DataArray
    """
    dims = list(coords.keys())
    # manage possible problems with time index of the dataset fp_ds from Flexpart
    ds = dim_coords_sorted_and_unique(ds, dims, warnings=warnings)

    # prepare dataset to interpolate
    ds_to_interpolate = ds[dims + list(subset)] if subset is not None else ds
    if not interp_nondim_coords:
        ds_to_interpolate = ds_to_interpolate.reset_coords(drop=True)

    # do interpolation; note: if coord has a NA value, interp propagates it towards an interpolated dataset
    return ds_to_interpolate.interp(coords, assume_sorted=True, method=method, kwargs=kwargs)


def merge_datasets_old(dss):
    freqs_tmin_tmax = [
        (da.attrs['_aats_freq'], da['time'].values.min(), da['time'].values.max())
        for ri, da_by_var in dss for da in da_by_var.values()
    ]
    freqs, tmin, tmax = zip(*freqs_tmin_tmax)
    freq = min(pd.Timedelta(int(f[:-1]), unit=f[-1]) if f[-1] != 'M' else pd.Timedelta(30, unit='D') for f in freqs)
    t_min = min(tmin)
    t_max = max(tmax)
    t = pd.date_range(t_min, t_max, freq=freq)

    da_by_ri_var = {}
    for ri, da_by_var in dss:
        for v, da in da_by_var.items():
            da_interpolated = interpolate(da, {'time': t})
            v_ri = f'{v}_{ri}'
            da_interpolated.name = v_ri
            da_by_ri_var[v_ri] = da_interpolated  # TODO: can cause override the dictionary da_by_ri_var...; apply xr.concat in such a case?

    return xr.merge(da_by_ri_var.values())


class _Tree:
    def __init__(self, children=None):
        self.leafs = []
        self.children = dict(children) if children is not None else {}

    def add_path(self, labels, leaf):
        if len(labels) == 0:
            self.leafs.append(leaf)
        else:
            label, *other_labels = labels
            subtree = self.children.setdefault(label, _Tree())
            subtree.add_path(other_labels, leaf)

    def _get_paths(self, path, result):
        if len(self.children) == 0:
            result[path] = self.leafs
        else:
            for label, child in self.children.items():
                child._get_paths(path + (label,), result)

    def get_paths(self):
        result = {}
        self._get_paths(tuple(), result)
        return result

    def _get_compressed_paths(self, path, result, after_level, level):
        if len(self.children) == 0:
            result[path] = self.leafs
        elif len(self.children) == 1 and level >= after_level:
            child, = self.children.values()
            child._get_compressed_paths(path, result, after_level, level + 1)
        else:
            for label, child in self.children.items():
                child._get_compressed_paths(path + (label,), result, after_level, level + 1)

    def get_compressed_paths(self, after_level=0):
        result = {}
        self._get_compressed_paths(tuple(), result, after_level, 0)
        return result


def integrate_datasets(dss):
    tree_of_var_ids = _Tree()
    print(f'integrate_datasets with len(dss)={len(dss)}')
    for ri, selector, md, ds in dss:
        print(f'ri={ri}, selector={selector}, md={md}')
        for v, da in ds.data_vars.items():
            da = da.copy()
            da['time'].attrs = {
                'standard_name': 'time',
                'long_name': 'time',
            }
            attrs = dict(ds.attrs)
            attrs.update(da.attrs)
            da.attrs = attrs
            da.name = '???'
            da_md = metadata.da_attr_to_metadata_dict(da)

            freq = attrs.get('_aats_freq', '0s')
            if freq != '0s':
                if freq != '1M':
                    freq_as_timedelta = pd.Timedelta(freq)
                    freq_resol = freq_as_timedelta.resolution_string
                    mul = freq_as_timedelta // pd.Timedelta(f'1{freq_resol}')
                    freq_str = f'{mul}{freq_resol}'
                else:
                    freq_str = '1M'
            else:
                freq_str = ''

            v_ri_freq = f'{v}_{ri}_{freq_str}' if freq_str else f'{v}_{ri}'
            var_id = (v_ri_freq, da_md[metadata.CITY_OR_STATION_NAME], attrs.get('sampling_height'), selector, )
            var_id = tuple(map(lambda i: str(i) if i is not None else '', var_id))  # ensure all parts of var_id are strings
            tree_of_var_ids.add_path(var_id, da)

    das_by_var_id = tree_of_var_ids.get_compressed_paths(after_level=1)  # we want to keep v_ri_freq in the var_id
    da_by_var_id = {}
    for var_id, das in das_by_var_id.items():
        if len(das) == 1:
            da, = das
            da.name = '_'.join(var_id)
            da_by_var_id[da.name] = da
        elif len(das) > 1:
            logger().info(f'var_id={var_id}: concatenate {len(das)} DataArrays:')
            for da in das:
                logger().info(f'{da}')
            try:
                da = xr.concat(das, dim='time')
                print(f'got da={da}')
                print(f'with time={da.time}')
            except Exception as e:
                logger().exception(
                    f'var_id={var_id}: concatenate of {len(das)} DataArrays failed; '
                    f'fall back to present them separately by add an No. to their var_id', exc_info=e
                )
                _var_id = '_'.join(var_id)
                for i, da in enumerate(das):
                    da.name = f'{_var_id}_{i}'
                    da_by_var_id[da.name] = da
                continue

            da.name = '_'.join(var_id)
            da_by_var_id[da.name] = da
        else:
            raise RuntimeError(f'got an empty list of DataArrays for var_id={var_id}')

    return da_by_var_id


def merge_datasets(dss):
    """

    :param dss: list of (string indicating RI, xarray.Dataset)
    :return:
    """
    freqs_tmin_tmax = [
        (da.attrs['_aats_freq'], da['time'].values.min(), da['time'].values.max())
        for ri, ds in dss for v, da in ds.data_vars.items()
    ]
    freqs, tmin, tmax = zip(*freqs_tmin_tmax)
    freqs = [
        pd.Timedelta(int(f[:-1]), unit=f[-1]) if f[-1] != 'M' else pd.Timedelta(30 * int(f[:-1]), unit='D')
        for f in freqs
    ]
    freq = min(f for f in freqs if f > pd.Timedelta(0))
    t_min = min(tmin)
    t_max = max(tmax)
    t = pd.date_range(t_min, t_max, freq=freq)

    da_by_ri_var = {}
    for ri, ds in dss:
        for v, da in ds.data_vars.items():
            da_interpolated = interpolate(da, {'time': t}, method='nearest')
            da_interpolated['time'].attrs = {
                'standard_name': 'time',
                'long_name': 'time',
            }
            v_ri = f'{v}_{ri}'
            da_interpolated.name = v_ri
            attrs = dict(ds.attrs)
            attrs.update(da.attrs)
            da_interpolated.attrs = attrs
            da_by_ri_var[v_ri] = da_interpolated  # TODO: can cause override the dictionary da_by_ri_var...; apply xr.concat in such a case?

    ds = xr.merge(da_by_ri_var.values())
    # clear global attributes
    ds.attrs = {}
    return ds
