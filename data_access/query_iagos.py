import requests
from requests.exceptions import HTTPError
import xarray as xr  

REST_URL_PATH = "https://services.iagos-data.fr/prod/v2.0/"
REST_URL_STATIONS = "https://api.sedoo.fr/aeris-catalogue-prod/metadata/3228ee61-f23e-452d-a06b-ffe060f3b728"
REST_URL_REGIONS = "https://api.sedoo.fr/aeris-catalogue-prod/metadata/77642149-2427-491e-b3a5-79b09cca241a"
REST_URL_VARIABLES = REST_URL_PATH + "parameters/public"
REST_URL_SEARCH = REST_URL_PATH + "l3/searchNetcdfFilenames"
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
        jsonResponse = response.json()
        ret = []
        for airport in jsonResponse['spatialExtents']:
            station = { 'short_name': airport['name'].split(", ")[0], 'long_name': airport['name'].split(", ")[1],
                       'longitude': airport['area']['longitude'], 'latitude': airport['area']['latitude'], 'altitude': airport['area']['altitude']  }
            ret.append(station)
        return ret    
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')

        
def get_list_regions():
    try:
        response = requests.get(REST_URL_REGIONS)
        response.raise_for_status()
        jsonResponse = response.json()
        ret = []
        for region in jsonResponse['spatialExtents']:
            station = { 'short_name': region['name'], 'long_name': region['description'],
                       'longitude_min': region['area']['westLongitude'], 'longitude_max': region['area']['eastLongitude'],
                       'latitude_min': region['area']['southLatitude'], 'latitude__max': region['area']['northLatitude']   }
            ret.append(station)
        return ret     
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
        url = REST_URL_SEARCH + "?codes=" + codes
        response = requests.get(url)
        response.raise_for_status()
        jsonResponse = response.json()
        ret = []
        for item in jsonResponse:
            ret.append(item)
        return ret    
    except HTTPError as http_err:
        print(f'HTTP error occurred: {http_err}')
    except Exception as err:
        print(f'Other error occurred: {err}')


def read_dataset(dataset_id, variables_list, temporal_extent, spatial_extent):
    results = requests.get(REST_URL_DOWNLOAD + "?fileId=" + dataset_id.replace("#", "%23"))
    with open('/tmp/fic.nc', 'wb') as f:
        f.write(results.content)
    ds = xr.open_dataset('/tmp/fic.nc', decode_times=False)
    varlist = [ ]
    for varname, da in ds.data_vars.items():
        if 'standard_name' in da.attrs and (da.attrs['standard_name'] in variables_list or da.attrs['standard_name'] in STATIC_PARAMETERS):
            varlist.append(varname)
    ds = ds[varlist] 
    return ds


if __name__ == "__main__":
    print(get_list_platforms())
    print(get_list_regions())
    print(get_list_variables())
    print(query_datasets_stations('FRA,NAt'))
    for dataset in query_datasets_stations('FRA,NAt'):
        array = read_dataset(dataset, ['mole_fraction_of_carbon_monoxide_in_air', 'mole_fraction_of_ozone_in_air' ], None, None)
        print(array['CO_mean'][0:10])

