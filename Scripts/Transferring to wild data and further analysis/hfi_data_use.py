#!/usr/bin/python3
"""
Filename: hfi_data_use.py
Author: Eva Reinhardt
Date: 2024-02-10
Version: 1.0
Description: extracting the mean, min, max and std of hfi data per individual

"""

import json
from collections.abc import Iterable

import numpy as np

import xarray as xr
import shapely.geometry as shgeo


from rasterio import features
from affine import Affine



# where to find th used data
HFI_PATH = '/home/eva/Schreibtisch/Master/NeuerVersuch'
HFI_FILE = HFI_PATH+"\ml_hfi_v1_2019.nc"
HFI_FILE_CROP = HFI_PATH+"\ml_hfi_v1_2019_crop.nc"

POLY_PATH = "C:\git-home\eva-racoon2\Scripts"
POLY_FILE = POLY_PATH+"\hull_polygons.geojson"



def transform_from_latlon(lat, lon) -> Affine:
    """
    create affine transform from lat/lon to array-indices
    @param lat: first elements (at least 2) of latitude-dimension
    @param lon: first elements (at least 2) of longitude-dimension
    @return: affine-transform lat/lon -> indices
    """
    trans = Affine.translation(lon[0], lat[0])
    scale = Affine.scale(lon[1] - lon[0], lat[1] - lat[0])
    return trans * scale

def mask_data( shape, data:np.ndarray, transform:Affine) -> np.ma.MaskedArray:
    """
    mask a Numpy-Array with shape
    @param shape:   single shape or list of shapes
    @param data:    Numpy-Array to mask
    @param transform:   transform for shape coordinates in lat(lon to array indices
    @return:    MaskedArray
    """
    if not isinstance(shape, Iterable):
        shape = [shape]     # make shape iterable
    mask = features.geometry_mask(geometries=shape,
                               out_shape=data.shape,
                               transform=transform,
                               all_touched=True)
    return np.ma.masked_array(data, mask)

if __name__ == "__main__":
    # load data
    hfi = xr.open_dataset(HFI_FILE)
    with open(POLY_FILE, "r") as poly_file:
        poly_data = json.load(poly_file)
    poly_count = len(poly_data['features'])
    
    print('Polygons found:', poly_count)
    
    poly_list = []
    logger_list   = []
    

    
    # create shapely geometries from data in json file
    for i in range(poly_count):
        polygon_coords = poly_data["features"][i]["geometry"]["coordinates"][0]
        pc_array = np.array(polygon_coords)
        # swap columns
        pc_array[:, [0, 1]] = pc_array[:, [1, 0]]
        
        polygon = shgeo.Polygon(pc_array)
        # append to lists
        poly_list.append(polygon)
        logger_list.append(poly_data["features"][i]['properties']['logger'])
    
    # create bounding box for all polygons
    max_lon = -180
    min_lon =  180
    max_lat = -90
    min_lat =  90
    for i in range(poly_count):
        (minX, minY, maxX, maxY) = poly_list[i].bounds    # X -> lon Y -> lat
        max_lon = max( max_lon, maxX)
        min_lon = min( min_lon, minX)
        max_lat = max( max_lat, maxY)
        min_lat = min( min_lat, minY)

    # print min-max for lat/lon for all polygones
    print('Lat : ', min_lat, '-', max_lat)
    print('Lon : ', min_lon, '-', max_lon)
    
    
    # get HFI-data cropped to research area
    hfi_crop = hfi.sel(indexers={'lat': slice(min_lat, max_lat), 'lon': slice(min_lon, max_lon)}).copy()
    hfi = None      # free mem !
    
    hfi_crop.to_netcdf(path = HFI_FILE_CROP)
    
    # hfi_shape = hfi_crop.__xarray_dataarray_variable__.shape
    hfi_trans = transform_from_latlon(hfi_crop.coords['lat'], hfi_crop.coords['lon'])
    hfi_data  = hfi_crop.__xarray_dataarray_variable__
    
    hfi_values = {}
    for i in range(poly_count):
        masked = mask_data(poly_list[i], hfi_data, hfi_trans)
        
        values = {'min':masked.min(),
                  'max':masked.max(),
                  'mean':masked.mean(),
                  'std':masked.std()}
        hfi_values[logger_list[i]] = values
        print(logger_list[i],values)
    
        

    
'''
Polygons found: 33
Lat :  51.4763269 - 53.3739447
Lon :  9.950392 - 13.8255092
1631 {'min': 0.2213534116744995, 'max': 0.7292912006378174, 'mean': 0.4932973631998388, 'std': 0.11040261406671228}
1628 {'min': 0.2061307728290558, 'max': 0.4830079674720764, 'mean': 0.3607120297171853, 'std': 0.080379598159198}
1630 {'min': 0.24863570928573608, 'max': 0.4703472852706909, 'mean': 0.3276277631521225, 'std': 0.06845484204779931}
1634 {'min': 0.2061307728290558, 'max': 0.4110027551651001, 'mean': 0.31135842433342564, 'std': 0.06331227121506051}
1636 {'min': 0.12486141920089722, 'max': 0.4464835822582245, 'mean': 0.2639136784004443, 'std': 0.08598744724993891}
1637 {'min': 0.2061307728290558, 'max': 0.3709324598312378, 'mean': 0.2894990305105845, 'std': 0.05096056283136639}
1638 {'min': 0.1763254702091217, 'max': 0.40239217877388, 'mean': 0.2791478857398033, 'std': 0.072632795696179}
1633 {'min': 0.22338202595710754, 'max': 0.4411213994026184, 'mean': 0.36205437034368515, 'std': 0.06711649981450711}
5133 {'min': 0.7514729499816895, 'max': 0.8494600057601929, 'mean': 0.8043683916330338, 'std': 0.03501176991454929}
5038 {'min': 0.7632853388786316, 'max': 0.917446494102478, 'mean': 0.8664542837784841, 'std': 0.03235247996228332}
5115 {'min': nan, 'max': nan, 'mean': nan, 'std': masked}
5116 {'min': 0.7514729499816895, 'max': 0.8494600057601929, 'mean': 0.8043683916330338, 'std': 0.03501176991454929}
5031 {'min': 0.7632853388786316, 'max': 0.9015777111053467, 'mean': 0.8523741886019707, 'std': 0.04187503262243441}
5034 {'min': 0.7875505685806274, 'max': 0.8922655582427979, 'mean': 0.8567233383655548, 'std': 0.033422981310735206}
5035 {'min': nan, 'max': nan, 'mean': nan, 'std': masked}
5134 {'min': 0.806998610496521, 'max': 0.892788827419281, 'mean': 0.8462088227272033, 'std': 0.026610762239084765}
5135 {'min': 0.8127000331878662, 'max': 0.8589399456977844, 'mean': 0.8330323425206271, 'std': 0.016403667275390037}
5136 {'min': 0.8247408866882324, 'max': 0.9057974815368652, 'mean': 0.8770999014377594, 'std': 0.02060874193927822}
5599 {'min': 0.8388617634773254, 'max': 0.8785666227340698, 'mean': 0.8543563485145569, 'std': 0.013511045448231462}
5602 {'min': 0.8569999933242798, 'max': 0.885715663433075, 'mean': 0.8673025965690613, 'std': 0.013050725204331357}
5603 {'min': 0.8405996561050415, 'max': 0.900370717048645, 'mean': 0.8739191144704819, 'std': 0.021714739857359056}
5604 {'min': 0.8300896286964417, 'max': 0.900370717048645, 'mean': 0.8714051577779982, 'std': 0.02585040060146218}
5605 {'min': 0.8300896286964417, 'max': 0.900370717048645, 'mean': 0.8615213433901469, 'std': 0.026414107493966967}
5606 {'min': 0.7097309827804565, 'max': 0.8558489084243774, 'mean': 0.7834844986597697, 'std': 0.04515331551537424}
5607 {'min': 0.8127000331878662, 'max': 0.8965068459510803, 'mean': 0.8422188392052283, 'std': 0.026344172910641006}
5036 {'min': 0.7907008528709412, 'max': 0.9084402322769165, 'mean': 0.8711031031608581, 'std': 0.027136420202112956}
5128 {'min': 0.33732688426971436, 'max': 0.711503267288208, 'mean': 0.5244150757789612, 'std': 0.18708819150924683}
5126 {'min': 0.33732688426971436, 'max': 0.711503267288208, 'mean': 0.5863197644551595, 'std': 0.17606544776018515}
5124 {'min': 0.26603755354881287, 'max': 0.7220220565795898, 'mean': 0.5987957545689174, 'std': 0.16140878079620782}
5129 {'min': 0.33732688426971436, 'max': 0.711503267288208, 'mean': 0.5863197644551595, 'std': 0.17606544776018515}
5123 {'min': 0.33732688426971436, 'max': 0.33732688426971436, 'mean': 0.33732688426971436, 'std': 0.0}
5125 {'min': 0.2777557373046875, 'max': 0.8596065044403076, 'mean': 0.6148701806863149, 'std': 0.18776897944891086}
5131 {'min': 0.2769567668437958, 'max': 0.6984824538230896, 'mean': 0.505779979750514, 'std': 0.1485573647063913}
'''