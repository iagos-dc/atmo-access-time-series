import pkg_resources
import pathlib
import json
import pandas as pd
import xarray as xr


data_path = pathlib.Path(pkg_resources.resource_filename('data_access', 'resources'))
md = []
for url in data_path.glob('*/*.nc'):
    rel_url = url.relative_to(data_path)
    with xr.open_dataset(url) as ds:
        ds_md = {
            'title': ds.attrs['title'],
            'urls': str(rel_url),
        }

        if 'CO_mean' in ds:
            v = 'Carbon Monoxide'
        elif 'O3_mean' in ds:
            v = 'Ozone'
        else:
            continue
        ds_md['ecv_variables'] = [v]

        t0, t1 = ds.time.min(), ds.time.max()
        time_fmt = '%Y-%m-%dT%H:%M:%SZ'
        ds_md['time_period'] = [pd.Timestamp(t.values, tz='UTC').strftime(time_fmt) for t in (t0, t1)]

        ds_md['platform_id'] = ds.attrs['IATA_code']
        ds_md['RI'] = 'IAGOS'

        ds_md['vars'] = list(ds)
        ds_md['layer'] = list(ds.layer.values.tolist())

        for k in ['longitude', 'latitude']:
            ds_md[k] = ds.attrs[k]
    md.append(ds_md)

catalogue_url = pkg_resources.resource_filename('data_access', 'resources/catalogue.json')
with open(catalogue_url, 'w') as f:
    json.dump(md, f, indent=2)
