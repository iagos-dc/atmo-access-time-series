import abc
import itertools
import functools
from mmappickle import mmapdict

import data_access
from .merge_datasets import merge_datasets


def _get_max_id(m):
    return max(itertools.chain([0], (int(k, base=16) for k in m.keys())))


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
        hash(obj)
        return obj


@functools.lru_cache(maxsize=10)
def get_request_id(req):
    """
    Retrieve an id for a request req already stored in req_map or assign a new one
    :param req: a hashable object
    :return: (str, bool): (id, is_id_new)
    """
    i = hash(req)
    i_str = str(i)
    while True:
        try:
            req_in_map = req_map[i_str]
        except KeyError:
            return i_str, True
        if req_in_map == req:
            return i_str, False
        else:
            i += 1
            i_str = str(i)


# TODO: make it thread/multiprocess safe
def request_cache(func):
    @functools.wraps(func)
    def _(req):
        try:
            i, is_id_new = get_request_id(req)
            if i in res_map:
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

    def __hash__(self):
        return hash(self.get_hashable())

    def __eq__(self, other):
        return self.get_hashable() == other.get_hashable()

    @request_cache
    def compute(self):
        print(f'compute {str(self)}')
        # args_computed = {k: v.compute() for k, v in self.args}
        # aux_args_computed = {k: v.compute() for k, v in self.aux_args.items()}
        return self.execute() #**args_computed, **aux_args_computed)

    def __str__(self):
        return str(self.to_dict())

    # @staticmethod
    # def from_obj(obj):
    #     """
    #     Constructs an instance of Request class
    #     :param obj: a tree structure of request(s); if a node of the tree is a dictionary with keys
    #     'action', 'args' and 'aux_args', then it is understood as a request; otherwise, it is interpreted as a value
    #     :return: an instance of Request class
    #     """
    #     if isinstance(obj, dict) and set(obj) == Request._REQUEST_KEYS:
    #         args = obj['args']
    #         return NodeRequest(
    #             obj['action'],
    #             tuple((k, Request.from_obj(args[k])) for k in sorted(args)),
    #             {k: Request.from_obj(v) for k, v in obj['aux_args'].items()}
    #         )
    #     else:
    #         return LeafRequest(obj)


class ReadDataRequest(Request):
    def __init__(self, ri, url, ds_metadata, selector=None):
        self.ri = ri
        self.url = url
        self.ds_metadata = dict(ds_metadata)
        self.selector = selector

        # super().__init__(
        #     'read_dataset',
        #     args=dict(ri=self.ri, url=self.url, selector=self.selector),
        #     aux_args=dict(ds_metadata=self.ds_metadata)
        # )

    def execute(self):
        print(f'execute {str(self)}')
        return data_access.read_dataset(self.ri, self.url, self.ds_metadata, selector=self.selector)

    def get_hashable(self):
        return ('read_dataset', _get_hashable(self.ri), _get_hashable(self.url), _get_hashable(self.selector))

    def to_dict(self):
        return dict(
            _action='read_dataset',
            ri=self.ri,
            url=self.url,
            ds_metadata=self.ds_metadata,
            selector=self.selector,
        )

    @staticmethod
    def from_dict(d):
        try:
            ri = d['ri']
            url = d['url']
            ds_metadata = d['ds_metadata']
            selector = d['selector']
        except KeyError:
            raise ValueError(f'bad ReadDataRequest: d={str(d)}')
        return ReadDataRequest(ri, url, ds_metadata, selector=selector)

    # def __str__(self):
    #     args_str = [f'{k}={v}' for k, v in self.args]
    #     args_str = ', '.join(args_str)
    #     aux_args_str = [f'{k}={v}' for k, v in self.aux_args.items()]
    #     aux_args_str = ', '.join(aux_args_str)
    #     return f'{self.action}({args_str}; {aux_args_str})' if len(aux_args_str) > 0 else f'{self.action}({args_str})'


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

    @staticmethod
    def from_dict(d):
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
    if action == 'read_dataset':
        return ReadDataRequest.from_dict(d)
    elif action == 'merge_datasets':
        return MergeDatasetsRequest.from_dict(d)
    else:
        raise NotImplementedError(f'd={d}')


# def request(action, args=None, aux_args=None):
#     if args is None:
#         args = {}
#     if aux_args is None:
#         aux_args = {}
#     return Request.from_obj({
#         'action': action,
#         'args': args,
#         'aux_args': aux_args
#     })
#
#
# class NodeRequest(Request):
#     def __init__(self, action, args, aux_args):
#         self.action = action
#         self.args = toolz.valmap(lambda v: v if isinstance(v, Request) else LeafRequest(v), args)
#         self.aux_args = toolz.valmap(lambda v: v if isinstance(v, Request) else LeafRequest(v), aux_args)
#
#     def __hash__(self):
#         return hash((self.action, self.args))
#
#     def __eq__(self, other):
#         return self.action == other.action and self.args == other.args
#
#     @request_cache
#     def compute(self):
#         print(f'compute {self}')
#         args_computed = {k: v.compute() for k, v in self.args}
#         aux_args_computed = {k: v.compute() for k, v in self.aux_args.items()}
#         return self.execute(**args_computed, **aux_args_computed)
#
#     def __str__(self):
#         args_str = [f'{k}={v}' for k, v in self.args]
#         args_str = ', '.join(args_str)
#         aux_args_str = [f'{k}={v}' for k, v in self.aux_args.items()]
#         aux_args_str = ', '.join(aux_args_str)
#         return f'{self.action}({args_str}; {aux_args_str})' if len(aux_args_str) > 0 else f'{self.action}({args_str})'
#
#
# class LeafRequest(Request):
#     def __init__(self, obj):
#         self.obj = obj
#         self.__hash__()  # to check upfront if the request is hashable
#
#     @functools.cached_property
#     def hashable_obj(self):
#         return _get_hashable(self.obj)
#
#     def __hash__(self):
#         return hash(self.hashable_obj)
#
#     def __eq__(self, other):
#         return self.hashable_obj == other.hashable_obj
#
#     def execute(self):
#         return self.obj
#
#     def compute(self):
#         return self.execute()
#
#     def __str__(self):
#         return str(self.obj)
