import functools


def check_param_compatibility(p1, p2):
    if isinstance(p1, (list, tuple)) and isinstance(p2, (list, tuple)):
        return len(p1) == len(p2)
    if isinstance(p1, dict) and isinstance(p2, dict):
        return len(p1) == len(p2) and set(p1) == set(p2)
    return False


def broadcast(*args_and_kwargs_list):
    def _broadcast(func):
        @functools.wraps(func)
        def _broadcast_wrapper(*args, **kwargs):
            args_list = list(args)
            def get_param(param):
                if isinstance(param, int):
                    return args_list[param]
                elif isinstance(param, str):
                    return kwargs[param]
                else:
                    raise ValueError(f'param must be int or str, got {param} of type {type(param)}')

            def set_param(param, value):
                if isinstance(param, int):
                    args_list[param] = value
                elif isinstance(param, str):
                    kwargs[param] = value
                else:
                    raise ValueError(f'param must be int or str, got {param} of type {type(param)}')

            def substitude_params(params_with_values):
                for p, v in params_with_values:
                    set_param(p, v)

            if len(args_and_kwargs_list) != 1:
                raise NotImplementedError
            args_and_kwargs_list_0, = args_and_kwargs_list
            args_and_kwargs_list_0 = list(set(args_and_kwargs_list_0))
            if len(args_and_kwargs_list_0) == 0:
                raise ValueError('list of args/kwargs to broadcast must be non-empty')

            # check if all params along which broadcast is to be done have the same structure
            params_value = [get_param(param) for param in args_and_kwargs_list_0]
            first_param, *other_params = args_and_kwargs_list_0
            first_param_value, *other_params_value = params_value
            for other_param, other_param_value in zip(other_params, other_params_value):
                if not check_param_compatibility(first_param_value, other_param_value):
                    raise ValueError(f'params {first_param} and {other_param} are not compatible: '
                                     f'{first_param}={first_param_value}, {other_param}={other_param_value}')

            # perform broadcasting
            if isinstance(first_param_value, (list, tuple)):
                results = []
                for vs in zip(*params_value):
                    substitude_params(zip(args_and_kwargs_list_0, vs))
                    res = func(*args_list, **kwargs)
                    results.append(res)
            elif isinstance(first_param_value, dict):
                results = {}
                for k in first_param_value:
                    vs = (d[k] for d in params_value)
                    substitude_params(zip(args_and_kwargs_list_0, vs))
                    res = func(*args_list, **kwargs)
                    results[k] = res
            else:
                raise ValueError(f'invalid type of {first_param_value} for broadcasting: '
                                 f'{type(first_param_value)}; must be list, tuple or dict')
            return results
        return _broadcast_wrapper
    return _broadcast


if __name__ == '__main__':
    def f(a, b, c=None, d=None):
        r = a + b
        if c is not None:
            r *= c
        if d is not None:
            r **= d
        return r

    res = broadcast([0, 1, 'd'])(f)({'a': 1, 'b': 2}, {'b': 4, 'a': 3}, c=5, d={'a': 2, 'b': 1})
    print(res)

    res = broadcast([0, 1, 'd'])(f)([1, 2], [3, 4], c=5, d=[2, 1])
    print(res)
