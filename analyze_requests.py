import pathlib
import pandas as pd
import diskcache

import config


_REQUESTS_DIR = pathlib.PurePath(config.APP_CACHE_DIR) / 'requests.tmp'


def analyze_requests(requests_dir=_REQUESTS_DIR):
    requests = diskcache.Deque(directory=requests_dir)
    df1 = pd.DataFrame.from_records(list(requests), columns=['request', 'time'])
    df1['day'] = df1['time'].dt.floor('D')
    df2 = pd.DataFrame.from_records(df1['request'].map(lambda req: req.to_dict()))
    return pd.concat([df1, df2], axis='columns')
