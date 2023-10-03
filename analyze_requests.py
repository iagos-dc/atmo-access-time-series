import pkg_resources
import pandas as pd
import diskcache


_REQUEST_DIR = pkg_resources.resource_filename('data_access', 'cache')


def analyze_requests(requests_dir=_REQUEST_DIR):
    requests = diskcache.Deque(directory=requests_dir)
    df1 = pd.DataFrame.from_records(list(requests), columns=['req', 'time'])
    df1['day'] = df1['time'].dt.floor('D')
    df2 = pd.DataFrame.from_records(df1['req'].map(lambda req: req.to_dict()))
    return pd.concat([df1, df2])
