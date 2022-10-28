import abc
import itertools
import functools
import hashlib
import json
from mmappickle import mmapdict
import icoscp.cpb.dobj

from log import logger

import data_access
from .merge_datasets import merge_datasets


def _get_max_id(m):
    return max(itertools.chain([0], (int(k, base=16) for k in m.keys())))


# TODO: switch into diskcache ??? https://grantjenks.com/docs/diskcache/tutorial.html
_REQUESTS_MMAPDICT_URL = '/home/wolp/PycharmProjects/atmo-access-time-series/data_access/cache/requests.tmp'
_RESULTS_MMAPDICT_URL = '/home/wolp/PycharmProjects/atmo-access-time-series/data_access/cache/results.tmp'

req_map = mmapdict(_REQUESTS_MMAPDICT_URL)
res_map = mmapdict(_RESULTS_MMAPDICT_URL)


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


@functools.lru_cache(maxsize=10)
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


# TODO: make it thread/multiprocess safe
def request_cache(func):
    @functools.wraps(func)
    def _(req):
        try:
            i, is_id_new = get_request_id(req)
            if not is_id_new:
                res = res_map[i]
            else:
                res = func(req)
                req_map[i] = req
                res_map[i] = res
            return res
        except Exception as e:
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
        # print(f'execute {str(self)}')
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
            raise ValueError(f'bad ReadDataRequest: d={str(d)}')
        return MergeDatasetsRequest(tuple(
            request_from_dict(read_dataset_request_as_dict)
            for read_dataset_request_as_dict in read_dataset_requests_as_dict
        ))


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
    else:
        raise NotImplementedError(f'd={d}')
