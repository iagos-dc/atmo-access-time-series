PERSISTENCE_TYPE = 'session'


def get_dash_persistence_kwargs(persistence_id=None, persistence_type=PERSISTENCE_TYPE):
    if persistence_id is None:
        return {}
    return {
        'persistence': persistence_id,
        'persistence_type': persistence_type,
    }
