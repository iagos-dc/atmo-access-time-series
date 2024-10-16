import requests
import json
import pandas as pd
import xarray as xr

from . import CACHE_DIR


MAPPING_ECV2ACTRIS = {
    # TODO: temporary patch for performance reasons (there are many very short datasets with AOP variables)
    # 'Aerosol Optical Properties': ['aerosol.absorption.coefficient', 'aerosol.backscatter.coefficient', 'aerosol.backscatter.coefficient.hemispheric', 'aerosol.backscatter.ratio', 'aerosol.depolarisation.coefficient', 'aerosol.depolarisation.ratio', 'aerosol.extinction.coefficient', 'aerosol.extinction.ratio', 'aerosol.extinction.to.backscatter.ratio', 'aerosol.optical.depth', 'aerosol.optical.depth.550', 'aerosol.rayleigh.backscatter', 'aerosol.scattering.coefficient', 'volume.depolarization.ratio', 'cloud.condensation.nuclei.number.concentration'],
    'Aerosol Chemical Properties': ['elemental.carbon', 'organic.carbon.concentration', 'organic.mass.concentration', 'total.carbon.concentration'],
    'Aerosol Physical Properties': ['particle.number.concentration', 'particle.number.size.distribution', 'pm10.concentration', 'pm1.concentration', 'pm2.5.concentration', 'pm2.5-&gt;pm10.concentration'],
}


def get_list_platforms():

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    actris_variable_list = ['elemental.carbon', 'organic.carbon.concentration', 'organic.mass.concentration', 'total.carbon.concentration', 'aerosol.absorption.coefficient', 'aerosol.backscatter.coefficient.hemispheric',
                            'aerosol.scattering.coefficient', 'particle.number.concentration', 'particle.number.size.distribution', 'pm10.concentration', 'pm1.concentration', 'pm2.5.concentration', 'pm2.5-&gt;pm10.concentration']

    data = '{"where":{"argument":{"type":"content","sub-type":"attribute_type","value":' + \
        str(actris_variable_list) + \
        ',"case-sensitive":false,"and":{"argument":{"type":"temporal_extent","comparison-operator":"overlap","value":["1970-01-01T00:00:00","2020-01-01T00:00:00"]}}}}}'

    response = requests.post(
        'https://prod-actris-md.nilu.no/Metadata/query',
        headers=headers,
        data=data)

    stations_demonstrator = []
    unique_identifiers = []

    for ds in response.json():

        if ds['md_data_identification']['station']['wmo_region'] == 'Europe':

            if ds['md_data_identification']['station']['identifier'] in unique_identifiers:
                pass
            else:
                unique_identifiers.append(
                    ds['md_data_identification']['station']['identifier'])

                stations_demonstrator.append(
                    {
                        'short_name': ds['md_data_identification']['station']['identifier'],
                        'latitude': ds['md_data_identification']['station']['lat'],
                        'longitude': ds['md_data_identification']['station']['lon'],
                        'long_name': ds['md_data_identification']['station']['name'],
                        'URI': 'https://prod-actris-md.nilu.no/Stations/{0}'.format(ds['md_data_identification']['station']['identifier']),
                        'altitude': ds['md_data_identification']['station']['alt']})
        else:
            pass

    return stations_demonstrator


def get_list_variables():

    response = requests.get(
        'https://prod-actris-md.nilu.no/ContentInformation/attributes')

    variables_demonstrator = []

    for v in response.json():
        for k, var_list in MAPPING_ECV2ACTRIS.items():
            if k == 'Cloud Properties' and v['attribute_type'] in var_list:
                variables_demonstrator.append(
                    {'variable_name': v['attribute_type'], 'ECV_name': ['Cloud Properties']})
            elif k == 'Aerosol Optical Properties' and v['attribute_type'] in var_list:
                variables_demonstrator.append(
                    {'variable_name': v['attribute_type'], 'ECV_name': ['Aerosol Optical Properties']})
            elif k == 'Aerosol Chemical Properties' and v['attribute_type'] in var_list:
                variables_demonstrator.append(
                    {'variable_name': v['attribute_type'], 'ECV_name': ['Aerosol Chemical Properties']})
            elif k == 'Aerosol Physical Properties' and v['attribute_type'] in var_list:
                variables_demonstrator.append(
                    {'variable_name': v['attribute_type'], 'ECV_name': ['Aerosol Physical Properties']})
            elif k == 'Precursors' and v['attribute_type'] in var_list:
                variables_demonstrator.append(
                    {'variable_name': v['attribute_type'], 'ECV_name': ['Precursors']})
            else:
                pass

    return variables_demonstrator


_all_datasets = None    # for caching


def query_datasets(variables=None, temporal_extent=None, spatial_extent=None):
    def check_arguments(variables, temporal_extent, spatial_extent):
        if not variables:
            variables = list(MAPPING_ECV2ACTRIS)
        if not isinstance(variables, (tuple, list)):
            raise ValueError(f'variables must be a list; got {variables}')

        if not temporal_extent:
            start_time = '1900-01-01T00:00:00'
            end_time = '2100-01-01T00:00:00'
            temporal_extent = [start_time, end_time]
        if not isinstance(temporal_extent, (tuple, list)) or len(temporal_extent) != 2:
            raise ValueError(f'temporal_extent must be a list of length 2; got {temporal_extent}')

        temporal_extent = [pd.Timestamp(t).strftime('%Y-%m-%dT%H:%M:%S') for t in temporal_extent]

        if not spatial_extent:
            lon0, lat0, lon1, lat1 = -180, -90, 180, 90
            spatial_extent = [lon0, lat0, lon1, lat1]
        if not isinstance(spatial_extent, (tuple, list)) or len(spatial_extent) != 4:
            raise ValueError(f'spatial_extent must be a list of length 4; got {spatial_extent}')

        return variables, temporal_extent, spatial_extent

    global _all_datasets
    if _all_datasets is None:
        cache_path = CACHE_DIR / 'actris_datasets.json'
        try:
            with open(cache_path, 'r') as f:
                _all_datasets = json.load(f)
        except FileNotFoundError:  # TODO: other exceptions?
            all_variables, global_temporal_extent, global_spatial_extent = check_arguments(None, None, None)
            _all_datasets = _query_datasets(all_variables, global_temporal_extent, global_spatial_extent)
            with open(cache_path, 'w') as f:
                json.dump(_all_datasets, f)

    variables, temporal_extent, spatial_extent = check_arguments(variables, temporal_extent, spatial_extent)
    variables = set(variables)
    lon0, lat0, lon1, lat1 = spatial_extent
    start_time, end_time = [pd.Timestamp(t, tz='UTC') for t in temporal_extent]

    datasets = []
    for ds in _all_datasets:
        if lon0 <= ds['lon'] <= lon1 and lat0 <= ds['lat'] <= lat1 and not variables.isdisjoint(ds['ecv_variables']):
            ds_t0, ds_t1 = [pd.Timestamp(t) for t in ds['time_period']]
            if not (end_time < ds_t0 or ds_t1 < start_time):
                datasets.append({k: v for k, v in ds.items() if k not in ['lon', 'lat']})
    return datasets


def _query_datasets(variables, temporal_extent, spatial_extent):

    # try:

    actris_variable_list = []

    for v in variables:
        if v in MAPPING_ECV2ACTRIS:
            actris_variable_list.extend(MAPPING_ECV2ACTRIS[v])

    start_time, end_time = temporal_extent[0], temporal_extent[1]
    #temporal_extent = [start_time, end_time]
    lon0, lat0, lon1, lat1 = spatial_extent[0], spatial_extent[1], spatial_extent[2], spatial_extent[3],
    #spatial_extent = [lon0, lat0, lon1, lat1]

    dataset_endpoints = []

    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = '{"where":{"argument":{"type":"content","sub-type":"attribute_type","value":' + \
        str(actris_variable_list) + \
        ',"case-sensitive":false,"and":{"argument":{"type":"temporal_extent","comparison-operator":"overlap","value":["' + \
        temporal_extent[0] + '","' + temporal_extent[1] + '"]}}}}}'

    response = requests.post(
        'https://prod-actris-md.nilu.no/Metadata/query',
        headers=headers,
        data=data)

    for ds in response.json():

        # filter urls by data provider.

        lat_point, lon_point = ds['md_data_identification']['station']['lat'], ds['md_data_identification']['station']['lon']

        if lon0 <= lon_point <= lon1 and lat0 <= lat_point <= lat1:
            local_filename = ds['md_distribution_information'][0]['dataset_url'].split('/')[-1]

            if ds['md_metadata']['provider_id'] == 14:

                opendap_url = 'http://thredds.nilu.no/thredds/dodsC/ebas/{0}'.format(
                    local_filename)
            else:
                opendap_url = None

            attribute_descriptions = ds['md_content_information']['attribute_descriptions']

            ecv_vars = []

            # TODO: temporary patch for performance reasons (there are many very short datasets with AOP variables)
            # if any(x in MAPPING_ECV2ACTRIS['Aerosol Optical Properties'] for x in attribute_descriptions):
            #    ecv_vars.append('Aerosol Optical Properties')
            # else:
            #    pass

            if any(x in MAPPING_ECV2ACTRIS['Aerosol Chemical Properties'] for x in attribute_descriptions):
                ecv_vars.append('Aerosol Chemical Properties')
            else:
                pass

            if any(x in MAPPING_ECV2ACTRIS['Aerosol Physical Properties'] for x in attribute_descriptions):
                ecv_vars.append('Aerosol Physical Properties')
            else:
                pass

            # generate dataset_metadata dict

            ds_title = 'Ground based observations of {0} (matrix: {1}) using {2} at {3}: {4} -> {5}'.format(','.join(ds['md_content_information']['attribute_descriptions']), ds['md_actris_specific']['matrix'], ','.join(
                ds['md_actris_specific']['instrument_type']), ds['md_data_identification']['station']['name'], ds['ex_temporal_extent']['time_period_begin'], ds['ex_temporal_extent']['time_period_end'])
            dataset_metadata = {'title': ds_title, 'urls': [{'url': opendap_url, 'type': 'opendap'}, {'url': ds['md_distribution_information'][0]['dataset_url'], 'type':'data_file'}], 'ecv_variables': ecv_vars, 'time_period': [
                ds['ex_temporal_extent']['time_period_begin'], ds['ex_temporal_extent']['time_period_end']], 'platform_id': ds['md_data_identification']['station']['identifier']}
            dataset_metadata.update({'lon': lon_point, 'lat': lat_point})
            dataset_endpoints.append(dataset_metadata)

        else:
            pass

    return dataset_endpoints

    # except BaseException:
    #    return "Variables must be one of the following: 'Aerosol Optical Properties','Aerosol Chemical Properties','Aerosol Physical Properties'"


def read_dataset(url, variables=None):

    # TODO: temporary patch; to be removed ???
    if variables is None:
        return xr.load_dataset(url)

    # For InSitu specific variables
    actris2insitu = {'particle_number_size_distribution': 'particle.number.size.distribution',
                     'aerosol_absorption_coefficient': 'aerosol.absorption.coefficient',
                     'aerosol_light_backscattering_coefficient': 'aerosol.backscatter.coefficient.hemispheric',
                     'aerosol_light_scattering_coefficient': 'aerosol.scattering.coefficient',
                     'cloud_condensation_nuclei_number_concentration': 'cloud.condensation.nuclei.number.concentration',
                     'particle_number_concentration': 'particle.number.concentration',
                     'elemental_carbon': 'elemental.carbon',
                     'organic_carbon': 'organic.carbon.concentration',
                     'organic_mass': 'organic.mass.concentration',
                     'particle_number_concentration': 'particle.number.concentration',
                     'pm1_mass': 'pm1.concentration',
                     'pm10_mass': 'pm10.concentration',
                     'pm25_mass': 'pm2.5.concentration',
                     'pm10_pm25_mass': 'pm2.5-&gt;pm10.concentration',
                     'total_carbon': 'total.carbon.concentration',
                     'aerosol_optical_depth': 'aerosol.optical.depth'
                     }

    # For ARES specific variables
    actris2ares = {'backscatter': 'aerosol.backscatter.coefficient',
                   'particledepolarization': 'aerosol.depolarisation.ratio',
                   'extinction': 'aerosol.extinction.coefficient',
                   'lidarratio': 'aerosol.extinction.to.backscatter.ratio',
                   'volumedepolarization': 'volume.depolarization.ratio'
                   }

    varlist_tmp = []
    for k, v in MAPPING_ECV2ACTRIS.items():
        if k in variables:
            varlist_tmp.extend(v)

    actris_varlist = []
    for k, v in actris2insitu.items():
        if v in varlist_tmp:
            actris_varlist.append(k)

    for k, v in actris2ares.items():
        if v in varlist_tmp:
            actris_varlist.append(k)

    try:
        with xr.open_dataset(url) as ds:
            var_list = []
            for varname, da in ds.data_vars.items():
                if 'metadata' in varname or 'time' in varname or '_qc' in varname:
                    pass
                elif any(var in varname for var in actris_varlist):
                    var_list.append(varname)
                else:
                    pass
            ds_filtered = ds[var_list].compute()
        return ds_filtered

    except Exception:
        return None
