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


def _get_max_id(m):
    return max(itertools.chain([0], (int(k, base=16) for k in m.keys())))


_REQUESTS_CACHE_URL = str(data_access.common.CACHE_DIR / 'requests.tmp')
_RESULTS_CACHE_URL = str(data_access.common.CACHE_DIR / 'results.tmp')

# see: https://grantjenks.com/docs/diskcache/tutorial.html
req_map = diskcache.Cache(_REQUESTS_CACHE_URL)
res_map = diskcache.Cache(_RESULTS_CACHE_URL)


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


#@functools.lru_cache(maxsize=10)
def get_request_id(req):
    """
    Retrieve an id for a request req already stored in req_map or assign a new one
    :param req: a hashable object
    :return: (str, bool): (id, is_id_new)
    """
    i = req.deterministic_hash()
    while True:
        try:
            req_in_map = req_map[i]
        except KeyError:
            return i, True
        if req_in_map == req:
            return i, False
        else:
            logger().warning(f'deterministic_hash collision: req={str(req)} and req_in_map={str(req_in_map)} '
                             f'have the same hash={i}')
            i = hex(int(i, 16) + 1)[2:]


# TODO: check if it is proper solution; cannot use a transaction (diskcache.Cache.transact())?
def request_cache(func):
    @functools.wraps(func)
    def _(req):
        try:
            i = None
            with diskcache.Lock(req_map, '_lock'):
                i, is_id_new = get_request_id(req)
                if is_id_new:
                    print(f'req_map[{i}] = {str(req)}')
                    req_map[i] = req
            if is_id_new:
                # print(f'prepare to execute req={str(req)}')
                res = func(req)
                res_map[i] = res
            else:
                while i not in res_map:
                    time.sleep(0.1)
                res = res_map[i]
                # print(f'retrieved req={str(req)}')
            return res
        except Exception as e:
            print(repr(ValueError(f'req={str(req)}')))
            if i is not None:
                req_map.pop(i)
                res_map.pop(i)
            raise ValueError(f'req={str(req)}') from e
    return _


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

    @request_cache
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


class MergeDatasetsRequest(Request):
    def __init__(self, read_dataset_requests):
        self.read_dataset_requests = read_dataset_requests

    def execute(self):
        print(f'execute {str(self)}')
        dss = [
            (read_dataset_request.ri, read_dataset_request.compute())
            for read_dataset_request in self.read_dataset_requests
        ]
        return merge_datasets(dss)

    def get_hashable(self):
        rs = sorted(read_dataset_request.get_hashable() for read_dataset_request in self.read_dataset_requests)
        return ('merge_datasets', ) + tuple(rs)

    def to_dict(self):
        return dict(
            _action='merge_datasets',
            read_dataset_requests=tuple(
                read_dataset_request.to_dict() for read_dataset_request in self.read_dataset_requests
            ),
        )

    @classmethod
    def from_dict(cls, d):
        try:
            read_dataset_requests_as_dict = d['read_dataset_requests']
        except KeyError:
            raise ValueError(f'bad MergeDatasetsRequest: d={str(d)}')
        return MergeDatasetsRequest(tuple(
            request_from_dict(read_dataset_request_as_dict)
            for read_dataset_request_as_dict in read_dataset_requests_as_dict
        ))


class IntegrateDatasetsRequest(Request):
    def __init__(self, read_dataset_requests):
        self.read_dataset_requests = read_dataset_requests

    def execute(self):
        print(f'execute {str(self)}')
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
        print(f'execute {str(self)}')
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
