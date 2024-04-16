import io
import requests
from requests.exceptions import HTTPError
import xarray as xr  

REST_URL_PATH = "https://services.iagos-data.fr/prod/v2.0/"
REST_URL_STATIONS = REST_URL_PATH + "airports/l3"
REST_URL_REGIONS = REST_URL_PATH + "regions"
REST_URL_VARIABLES = REST_URL_PATH + "parameters/public"
REST_URL_SEARCH = REST_URL_PATH + "l3/search?codes="
REST_URL_DOWNLOAD = REST_URL_PATH + "l3/loadNetcdfFile"

STATIC_PARAMETERS = ["latitude", "longitude", "air_pressure", "barometric_altitude"]
MAPPING_ECV_IAGOS = {
    "Temperature (near surface)": [ "air_temperature" ],
    "Water Vapour (surface)": [ "mole_fraction_of_water_vapor_in_air", "relative_humidity" ],
    "Temperature (upper-air)": [ "air_temperature" ],
    "Water Vapour (upper air)": [ "mole_fraction_of_water_vapor_in_air, relative_humidity" ],
    "Cloud Properties": [ "number_concentration_of_cloud_liquid_water_particles_in_air" ],
    "Wind speed and direction (upper-air)": [ "wind_speed, wind_from_direction" ],
    "Carbon Dioxide": [ "mole_fraction_of_carbon_dioxide_in_air" ],
    "Methane": [ "mole_fraction_of_methane_in_air" ],
    "Ozone": [ "mole_fraction_of_ozone_in_air" ],
    "Carbon Monoxide": [ "mole_fraction_of_carbon_monoxide_in_air" ],
    "NO2": [ "mole_fraction_of_nitrogen_dioxide_in_air" ]
}
MAPPING_CF_IAGOS = {
    "air_temperature": "air_temp",
    "mole_fraction_of_water_vapor_in_air": "H2O_gas",
    "relative_humidity": "RHL",
    "mole_fraction_of_methane_in_air": "CH4",
    "number_concentration_of_cloud_liquid_water_particles_in_air": "cloud",
    "mole_fraction_of_carbon_monoxide_in_air": "CO",
    "mole_fraction_of_carbon_dioxide_in_air": "CO2",
    "mole_fraction_of_nitrogen_dioxide_in_air": "NO2",
    "mole_fraction_of_ozone_in_air": "O3"
}


def reverse_mapping(mapping):
    ret = {}
    for key, values in mapping.items():
        for value in values:
            if value not in ret:
                ret[value] = []
            ret[value].append(key)
    return ret


MAPPING_IAGOS_ECV = reverse_mapping(MAPPING_ECV_IAGOS)

        
def get_list_platforms():
    try:
        response = requests.get(REST_URL_STATIONS)
        response.raise_for_status()
        return response.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

        
def get_list_regions():
    try:
        response = requests.get(REST_URL_REGIONS)
        response.raise_for_status()
        return response.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


def get_list_variables():
    try:
        response = requests.get(REST_URL_VARIABLES)
        response.raise_for_status()
        jsonResponse = response.json()
        ret = []
        done = []
        for item in jsonResponse:
            if(item['cf_standard_name'] in MAPPING_IAGOS_ECV and item['cf_standard_name'] not in done):
                variable = { 'variable_name': item['cf_standard_name'], 'ECV_name': MAPPING_IAGOS_ECV[item['cf_standard_name']] }
                done.append(item['cf_standard_name'])
                ret.append(variable)
        return ret    
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

        
def query_datasets(variables_list, temporal_extent, spatial_extent): 
    pass


def query_datasets_stations(codes): 
    try:
        url = REST_URL_SEARCH + codes
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


def read_dataset(dataset_id, variables_list):
    try:
        request_url = REST_URL_DOWNLOAD + "?fileId=" + dataset_id.replace("#", "%23")
        response = requests.get(request_url)
        response.raise_for_status()
        with io.BytesIO(response.content) as buf:
            with xr.open_dataset(buf) as ds:
                varlist = []
                for varname, da in ds.data_vars.items():
                    if 'standard_name' in da.attrs and (da.attrs['standard_name'] in variables_list or da.attrs['standard_name'] in STATIC_PARAMETERS):
                        varlist.append(varname)
                return ds[varlist].load()
    except Exception as e:
        raise RuntimeError(f'Reading the IAGOS dataset failed: {dataset_id}') from e


if __name__ == "__main__":
    # print(get_list_platforms())
    # print(get_list_regions())
    # print(get_list_variables())
    # print(query_datasets_stations("FRA,NAt"))
    for dataset in query_datasets_stations("FRA,NAt"):
        for url in dataset['urls']:
            if url['type'] == "LANDING_PAGE":
                array = read_dataset(url['url'], ['mole_fraction_of_carbon_monoxide_in_air', 'mole_fraction_of_ozone_in_air' ])
                print(array['CO_mean'][0:10])
                
