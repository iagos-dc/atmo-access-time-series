def any_is_None(*items):
    return any(map(lambda item: item is None, items))


def ensurelist(obj):
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    elif isinstance(obj, dict):
        return list(obj.values())
    else:
        return [obj]
