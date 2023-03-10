import dash


def get_value_by_aio_id(aio_id, ids, values):
    for i, v in zip(ids, values):
        if i['aio_id'] == aio_id:
            return v
    raise ValueError(f'cannot find aio_id={aio_id} among ids={ids}')


def set_value_by_aio_id(aio_id, ids, value):
    return [value if i['aio_id'] == aio_id else dash.no_update for i in ids]
