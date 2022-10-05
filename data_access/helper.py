import functools
import toolz


def many2many_to_dictOfList(relation_pairs, keep_set=False):
    mapping = {}
    for k, v in relation_pairs:
        mapping.setdefault(k, set()).add(v)
    if keep_set:
        return mapping
    else:
        return toolz.valmap(list, mapping)


def many2manyLists_to_dictOfList(relation_pairs, keep_set=False):
    mapping = {}
    for k, v in relation_pairs:
        mapping.setdefault(k, set()).update(v)
    if keep_set:
        return mapping
    else:
        return toolz.valmap(list, mapping)


def image_of_dict(dom, dic):
    dom_in_dic = set(dic.keys()).intersection(dom)
    return set(dic[k] for k in dom_in_dic)


def image_of_dictOfLists(dom, dic):
    return functools.reduce(lambda img, x: img.union(dic.get(x, [])), dom, set())


def getattr_or_keyval(obj, attrs=None, keys=None):
    if obj is None:
        return None
    if attrs is None:
        attrs = []
    if keys is None:
        keys = []
    for attr in attrs:
        try:
            obj = getattr(obj, attr, None)
        except AttributeError:
            return None
    for key in keys:
        try:
            obj = obj[key]
        except KeyError:
            return None
    return obj


def getattrs(obj, *attrs):
    if obj is None:
        return None
    for attr in attrs:
        try:
            obj = getattr(obj, attr, None)
        except AttributeError:
            return None
    return obj


def getkeyvals(obj, *keys):
    if obj is None or not isinstance(obj, dict):
        return None
    for key in keys:
        try:
            obj = obj[key]
        except KeyError:
            return None
    return obj
