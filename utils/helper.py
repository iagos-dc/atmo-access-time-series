def any_is_None(*items):
    return any(map(lambda item: item is None, items))
