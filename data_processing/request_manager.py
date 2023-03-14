import pkg_resources
import abc
import itertools
import functools
import hashlib
import json
import diskcache
import time
import icoscp.cpb.dobj
import pandas as pd

import data_access.common
from log import logger

import data_access
from .merge_datasets import merge_datasets, integrate_datasets
from .filtering import filter_dataset


_CACHE_URL = str(data_access.common.CACHE_DIR / 'cache.tmp')

_RESULT_EXPIRE = 3600 * 12
_IN_PROGRESS_EXPIRE = 30
_FAIL_EXPIRE = 30


# see: https://grantjenks.com/docs/diskcache/tutorial.html
_cache_map = diskcache.Cache(_CACHE_URL)


def _get_hashable(obj):
    """
    Provides a hashable version of obj, if possible:
      list -> tuple,
      set -> tuple of sorted elems,
      dict -> tuple of pairs (k, v), where keys are sorted
      Request -> Request.get_hashable(obj)
    :param obj: a Python object
    :return: a hashable version of obj
    """
    if isinstance(obj, list):
        return tuple(_get_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(_get_hashable(item) for item in sorted(obj)),
    elif isinstance(obj, dict):
        return tuple((k, _get_hashable(obj[k])) for k in sorted(obj))
    elif isinstance(obj, Request):
        return obj.get_hashable()
    else:
        return obj


class _NoResultYet:
    def __str__(self):
        return '<NoResultYet>'


class _ResultInProgress:
    def __str__(self):
        return '<ResultInProgress>'


class _FailResult:
    def __init__(self, exc):
        self.exc = exc


def cache_setdefault(req):
    assert isinstance(req, Request)

    i = req.deterministic_hash()
    with _cache_map.transact():
        while True:
            req_in_cache, res = _cache_map.get(i, default=(None, None))
            if req_in_cache is None:
                no_result_yet = _NoResultYet()
                result_in_progress = _ResultInProgress()
                _cache_map.set(i, (req, result_in_progress), expire=_IN_PROGRESS_EXPIRE)
                # logger().info(f'_cache_map[{i}] = ({str(req)}, {str(result_in_progress)})')
                return i, no_result_yet
            elif req_in_cache == req:
                return i, res
            else:
                logger().warning(f'deterministic_hash collision: req={str(req)} and req_in_map={str(req_in_cache)} '
                                 f'have the same hash={i}')
                i = hex(int(i, 16) + 1)[2:]


def request_cache(custom_expire=-1):
    def _request_cache(func):
        @functools.wraps(func)
        def _(req):
            active_waiting = 0
            res = _ResultInProgress()
            while isinstance(res, _ResultInProgress):
                i, res = cache_setdefault(req)
                if isinstance(res, _NoResultYet):
                    expire = _FAIL_EXPIRE
                    try:
                        res = func(req)
                        expire = custom_expire if custom_expire != -1 else _RESULT_EXPIRE
                        return res
                    except Exception as e:
                        res = _FailResult(e)
                        raise
                    finally:
                        _cache_map.set(i, (req, res), expire=expire)
                elif isinstance(res, _FailResult):
                    raise res.exc
                elif isinstance(res, _ResultInProgress):
                    active_waiting += 1
                    logger().info(f'active_waiting={active_waiting} for i={i}, req={req}')
                    time.sleep(0.5)
                else:
                    if active_waiting > 0:
                        logger().info(f'got result after active_waiting={active_waiting} for i={i}, req={req}')
                    return res

        return _
    return _request_cache


class Request(abc.ABC):
    """
    Represents a web-service request with internal caching mechanism based on action and args, excluding aux_args
    """
    _REQUEST_KEYS = {'action', 'args', 'aux_args'}

    @abc.abstractmethod
    def execute(self):
        pass

    @abc.abstractmethod
    def get_hashable(self):
        pass

    @abc.abstractmethod
    def to_dict(self):
        pass

    @classmethod
    @abc.abstractmethod
    def from_dict(cls, d):
        pass

    @classmethod
    def from_json(cls, js):
        d = json.loads(js)
        return cls.from_dict(d)

    def deterministic_hash(self):
        return hashlib.sha256(bytes(str(self.get_hashable()), encoding='utf-8')).hexdigest()

    def __hash__(self):
        return hash(self.get_hashable())

    def __eq__(self, other):
        return self.get_hashable() == other.get_hashable()

    @request_cache()
    def compute(self):
        return self.execute()

    def __str__(self):
        return str(self.to_dict())


class GetICOSDatasetTitleRequest(Request):
    def __init__(self, dobj):
        self.dobj = dobj

    def execute(self):
        print(f'execute {str(self)}')
        return icoscp.cpb.dobj.Dobj(self.dobj).meta['references']['title']

    @request_cache(custom_expire=None)
    def compute(self):
        return self.execute()

    def get_hashable(self):
        return 'get_ICOS_dataset_title', _get_hashable(self.dobj)

    def to_dict(self):
        return dict(
            _action='get_ICOS_dataset_title',
            dobj=self.dobj,
        )

    @classmethod
    def from_dict(cls, d):
        try:
            dobj = d['dobj']
        except KeyError:
            raise ValueError(f'bad GetICOSDatasetTitleRequest: d={str(d)}')
        return GetICOSDatasetTitleRequest(dobj)


class ReadDataRequest(Request):
    def __init__(self, ri, url, ds_metadata, selector=None):
        self.ri = ri
        self.url = url
        self.ds_metadata = dict(ds_metadata)
        self.selector = selector

    def execute(self):
        print(f'execute {str(self)}')
        return data_access.read_dataset(self.ri, self.url, self.ds_metadata, selector=self.selector)

    def get_hashable(self):
        return 'read_dataset', _get_hashable(self.ri), _get_hashable(self.url), _get_hashable(self.selector)

    def to_dict(self):
        return dict(
            _action='read_dataset',
            ri=self.ri,
            url=self.url,
            ds_metadata=self.ds_metadata,
            selector=self.selector,
        )

    @classmethod
    def from_dict(cls, d):
        try:
            ri = d['ri']
            url = d['url']
            ds_metadata = d['ds_metadata']
            selector = d['selector']
        except KeyError:
            raise ValueError(f'bad ReadDataRequest: d={str(d)}')
        return ReadDataRequest(ri, url, ds_metadata, selector=selector)


class IntegrateDatasetsRequest(Request):
    def __init__(self, read_dataset_requests):
        self.read_dataset_requests = read_dataset_requests

    def execute(self):
        # print(f'execute {str(self)}')
        dss = [
            (
                read_dataset_request.ri,
                read_dataset_request.selector,
                read_dataset_request.ds_metadata,
                read_dataset_request.compute()
            )
            for read_dataset_request in self.read_dataset_requests
        ]
        return integrate_datasets(dss)

    def get_hashable(self):
        rs = sorted(read_dataset_request.get_hashable() for read_dataset_request in self.read_dataset_requests)
        return ('integrate_datasets', ) + tuple(rs)

    def to_dict(self):
        return dict(
            _action='integrate_datasets',
            read_dataset_requests=tuple(
                read_dataset_request.to_dict() for read_dataset_request in self.read_dataset_requests
            ),
        )

    @classmethod
    def from_dict(cls, d):
        try:
            read_dataset_requests_as_dict = d['read_dataset_requests']
        except KeyError:
            raise ValueError(f'bad IntegrateDatasetsRequest: d={str(d)}')
        return IntegrateDatasetsRequest(tuple(
            request_from_dict(read_dataset_request_as_dict)
            for read_dataset_request_as_dict in read_dataset_requests_as_dict
        ))


class FilterDataRequest(Request):
    def __init__(
            self,
            integrate_datasets_request,
            rng_by_varlabel,
            cross_filtering,
            cross_filtering_time_coincidence_dt
    ):
        self.integrate_datasets_request = integrate_datasets_request
        self.rng_by_varlabel = rng_by_varlabel
        self.cross_filtering = cross_filtering
        self.cross_filtering_time_coincidence_dt = cross_filtering_time_coincidence_dt

    def execute(self):
        # print(f'execute {str(self)}')
        ds = self.integrate_datasets_request.compute()

        ds_filtered_by_var = filter_dataset(
            ds,
            self.rng_by_varlabel,
            cross_filtering=self.cross_filtering,
            tolerance=self.cross_filtering_time_coincidence_dt,
            filter_data_request=True,
        )

        return ds_filtered_by_var

    def get_hashable(self):
        if self.cross_filtering_time_coincidence_dt is not None:
            cross_filtering_time_coincidence_as_str = str(pd.Timedelta(self.cross_filtering_time_coincidence_dt))
        else:
            cross_filtering_time_coincidence_as_str = None
        return (
            'filter_data',
            self.integrate_datasets_request.get_hashable(),
            _get_hashable(self.rng_by_varlabel),
            _get_hashable(self.cross_filtering),
            _get_hashable(cross_filtering_time_coincidence_as_str),
        )


    def to_dict(self):
        if self.cross_filtering_time_coincidence_dt is not None:
            cross_filtering_time_coincidence_as_str = str(pd.Timedelta(self.cross_filtering_time_coincidence_dt))
        else:
            cross_filtering_time_coincidence_as_str = None
        return dict(
            _action='filter_data',
            integrate_datasets_request=self.integrate_datasets_request.to_dict(),
            rng_by_varlabel=self.rng_by_varlabel,
            cross_filtering=self.cross_filtering,
            cross_filtering_time_coincidence_as_str=cross_filtering_time_coincidence_as_str,
        )

    @classmethod
    def from_dict(cls, d):
        try:
            integrate_datasets_request_as_dict = d['integrate_datasets_request']
            rng_by_varlabel = d['rng_by_varlabel']
            cross_filtering = d['cross_filtering']
            cross_filtering_time_coincidence_as_str = d['cross_filtering_time_coincidence_as_str']
        except KeyError:
            raise ValueError(f'bad FilterDataRequest: d={str(d)}')
        if cross_filtering_time_coincidence_as_str is not None:
            cross_filtering_time_coincidence_dt = pd.Timedelta(cross_filtering_time_coincidence_as_str).to_timedelta64()
        else:
            cross_filtering_time_coincidence_dt = None
        return FilterDataRequest(
            request_from_dict(integrate_datasets_request_as_dict),
            rng_by_varlabel,
            cross_filtering,
            cross_filtering_time_coincidence_dt
        )


class MergeDatasetsRequest(Request):
    def __init__(self, filter_dataset_request):
        self.filter_dataset_request = filter_dataset_request

    def execute(self):
        # print(f'execute {str(self)}')
        da_by_varlabel = self.filter_dataset_request.compute()
        return merge_datasets(da_by_varlabel)

    def get_hashable(self):
        return 'merge_datasets', self.filter_dataset_request.get_hashable()

    def to_dict(self):
        return dict(
            _action='merge_datasets',
            filter_dataset_request=self.filter_dataset_request.to_dict(),
        )

    @classmethod
    def from_dict(cls, d):
        try:
            filter_dataset_request_as_dict = d['filter_dataset_request']
        except KeyError:
            raise ValueError(f'bad MergeDatasetsRequest: d={str(d)}')
        return MergeDatasetsRequest(request_from_dict(filter_dataset_request_as_dict))


def request_from_dict(d):
    if not isinstance(d, dict):
        raise ValueError(f'd must be a dict; type(d)={str(type(d))}; d={str(d)}')
    try:
        action = d['_action']
    except KeyError:
        raise ValueError(f'd does not represent a request; d={str(d)}')
    if action == 'get_ICOS_dataset_title':
        return GetICOSDatasetTitleRequest.from_dict(d)
    elif action == 'read_dataset':
        return ReadDataRequest.from_dict(d)
    elif action == 'merge_datasets':
        return MergeDatasetsRequest.from_dict(d)
    elif action == 'integrate_datasets':
        return IntegrateDatasetsRequest.from_dict(d)
    elif action == 'filter_data':
        return FilterDataRequest.from_json(d)
    else:
        raise NotImplementedError(f'd={d}')


def request_from_json(js):
    d = json.loads(js)
    return request_from_dict(d)
