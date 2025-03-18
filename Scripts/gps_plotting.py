#!/usr/bin/python3

"""
Filename: gps_plotting.py
Author: Eva Reinhardt
Date: 2024-12-09
Version: 1.0
Description: Plotting gps points with respective behavior prediction denoted by the color.
Different layers denote different days.
Additionally, the areas that are roamed by one individual are given as an extra layer per html file as well as a file
showing all areas per data set.

Functions in this file:

calculate_direction(): calculates the direction of the arrow between two consecutive GPS points

calculate_distance(): calculates distance between two consecutive GPS points

create_arrow_icon(): creates the arrow icon that is displayed between two consecutive points
"""
import raccoon_acc_setup.variables_simplefunctions as sim_func
import pandas as pd
import os
import re
from folium import Element
import math
from scipy.spatial import ConvexHull
import numpy as np
import folium
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import rasterio
from matplotlib.colors import Normalize
import h5py


def calculate_direction(start, end):
    """
    function to calculate the arrow's direction
    @param start: start point (coordinates)
    @param end: end point (coordinates)
    @return: direction in degrees
    """
    lat1, lon1 = start
    lat2, lon2 = end

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)

    direction = math.atan2(y, x)
    direction = math.degrees(direction)

    direction = (direction + 360) % 360

    return direction


def calculate_distance(start, end):
    """
    function to calculate distance between two points
    @param start: start point (coordinates)
    @param end: end point (coordinates)
    @return: distance in meters
    """
    lat1, lon1 = start
    lat2, lon2 = end

    R = 6371000

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def create_arrow_icon(angle):
    """
    function to create the arrow icon that can be used in folium
    @param angle: angle of line between points
    @return: arrow icon
    """
    adjusted_angle = (angle - 90) % 360
    return folium.DivIcon(html=f'''
        <div style="font-size: 24px; 
                    transform: rotate({adjusted_angle}deg); 
                    transform-origin: center center;">&#x2192;</div>
    ''')


if __name__ == "__main__":
    filepaths_class = [sim_func.IMPORT_PATH_CLASS + f for f in os.listdir(sim_func.IMPORT_PATH_CLASS) if
                       os.path.isfile(os.path.join(sim_func.IMPORT_PATH_CLASS, f)) and '.csv' in f and
                       'predictions_mw_layered' in f]
    filepaths_caros = [f for f in filepaths_class if 'Caro S' in f]
    filepaths_carow = [f for f in filepaths_class if 'Caro W' in f]
    filepaths_katti = [f for f in filepaths_class if 'Katti' in f]

    filepaths = {'Caro S': filepaths_caros,
                 'Caro W': filepaths_carow,
                 'Katti': filepaths_katti}

    all_polygons = {}

    option = "all"
    if option == "all":

        bounds_all = []
        all_areas = []
        for name, fps in filepaths.items():
            all_areas_dataset = []
            means_lat = []
            means_lon = []
            for filepath in fps:
                logger = match = re.search(r"\d{4}", filepath)[0]
                filename_output = sim_func.EXPORT_PATH_HTML + name + str(logger) + '_gps.html'
                color_mapping = sim_func.COLOR_MAPPING_HTML

                gps_data = pd.read_csv(filepath, sep=',')
                gps_data['datetime'] = pd.to_datetime(gps_data['datetime'], format='mixed')
                gps_data['datetime_raccoon'] = gps_data['datetime'] + pd.Timedelta(hours=12)
                gps_data_1 = gps_data.dropna(subset='pred_incl_unkn')
                gps_data_1 = gps_data_1.dropna(subset='location-long')
                if gps_data_1.empty:
                    gps_only = gps_data.dropna(subset='location-long')[['location-long', 'location-lat', 'datetime']]
                    pred_only = gps_data.dropna(subset='pred_incl_unkn')
                    pred_only = pred_only.drop(['location-long', 'location-lat'], axis=1)
                    gps_data = pd.merge_asof(
                        gps_only.sort_values("datetime"),
                        pred_only.sort_values("datetime"),
                        on="datetime",
                        # tolerance=pd.Timedelta(seconds=2),
                        direction="nearest"
                    )
                    gps_data = gps_data.dropna(subset='pred_incl_unkn')
                    gps_data = gps_data.dropna(subset='location-long')

                else:
                    gps_data = gps_data_1
                gps_data = gps_data.drop_duplicates(['datetime'], keep='last')
                gps_data = gps_data.loc[gps_data['location-lat'] != 0]
                gps_data['date'] = gps_data['datetime_raccoon'].dt.date
                gps_data['time'] = gps_data['datetime'].dt.time
                gps_data = gps_data.dropna(subset='location-long')
                gps_data = gps_data.reset_index(drop=True)
                center_lat = gps_data['location-lat'].mean()
                center_lon = gps_data['location-long'].mean()

                bounds = [[gps_data['location-lat'].min(), gps_data['location-long'].min()],
                          [gps_data['location-lat'].max(), gps_data['location-long'].max()]]

                m = folium.Map(location=[center_lat, center_lon], zoom_start=14, control_scale=True)
                m.fit_bounds(bounds)

                print(gps_data[gps_data['pred_incl_unkn'] == 'resting'])

                grouped = gps_data.groupby('date')

                for (date), group in grouped:
                    fg = folium.FeatureGroup(name=str(date), show=False).add_to(m)

                    coordinates = list(zip(group['location-lat'], group['location-long']))

                    for start, end in zip(coordinates[:-1], coordinates[1:]):

                        midpoint = [(start[0] + end[0]) / 2, (start[1] + end[1]) / 2]

                        distance = calculate_distance(start, end)

                        if distance >= 50:
                            bearing = calculate_direction(start, end)

                            folium.Marker(
                                location=midpoint,
                                icon=create_arrow_icon(bearing),
                                icon_size=(30, 30),
                                icon_anchor=(15, 15),
                            ).add_to(fg)

                        folium.PolyLine(
                            locations=[start, end],
                            color='grey',
                            weight=8,
                            opacity=0.6
                        ).add_to(fg)

                    for i, row in group.iterrows():
                        folium.CircleMarker(
                            location=[row['location-lat'], row['location-long']],
                            popup=folium.Popup(str(row['time']), parse_html=True, max_width="100%"),
                            radius=5,
                            color=color_mapping[row['pred_incl_unkn']],
                            fill=True,
                            fillColor=color_mapping[row['pred_incl_unkn']],
                            fillOpacity=1
                        ).add_to(fg)

                legend_html = '''
                <div style="
                    position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; 
                    border:2px solid grey; 
                    z-index:9999; 
                    font-size:14px;">
                    <b>Legend</b><br>
                    <i style="background: #f9bc08; width: 12px; height: 12px; display: inline-block;"></i> Resting<br>
                    <i style="background: #4e7496; width: 12px; height: 12px; display: inline-block;"></i> Intermediate Energy<br>
                    <i style="background: #a2cffe; width: 12px; height: 12px; display: inline-block;"></i> Exploring<br>
                    <i style="background: #751973; width: 12px; height: 12px; display: inline-block;"></i> Walking<br>
                    <i style="background: #c875c4; width: 12px; height: 12px; display: inline-block;"></i> Climbing<br>
                    <i style="background: #7b002c; width: 12px; height: 12px; display: inline-block;"></i> High Energy<br>
                    <i style="background: #FF0000; width: 12px; height: 12px; display: inline-block;"></i> Unknown
                </div>,
                '''

                all_points = list(zip(gps_data["location-lat"], gps_data["location-long"]))

                points_array = np.column_stack((gps_data["location-lat"].values, gps_data["location-long"].values))

                hull = ConvexHull(points_array)
                hull_polygon = points_array[hull.vertices]
                hull_polygon = np.vstack([hull_polygon, hull_polygon[0]])

                fg = folium.FeatureGroup(name='complete area', show=False).add_to(m)
                folium.Polygon(
                    locations=hull_polygon,
                    color="blue",
                    fill=True,
                    fill_opacity=0.2,
                    popup="Covered Area"
                ).add_to(fg)

                all_areas_dataset.append(hull_polygon)
                means_lat.append(center_lat)
                means_lon.append(center_lon)

                all_polygons[logger] = hull_polygon

                legend = Element(legend_html)
                m.get_root().html.add_child(legend)

                folium.LayerControl().add_to(m)
                m.save(filename_output)

            all_coords = np.concatenate(all_areas_dataset)

            min_lat, max_lat = np.min(all_coords[:, 0]), np.max(all_coords[:, 0])
            min_lon, max_lon = np.min(all_coords[:, 1]), np.max(all_coords[:, 1])

            bounds_folium = pd.DataFrame([[min_lat, min_lon, max_lat, max_lon]],
                                         columns=["min_lat", "min_lon", "max_lat", "max_lon"])

            tiff_file = "/home/eva/Schreibtisch/Master/NeuerVersuch/human_footprint_germany.tif"
            with rasterio.open(tiff_file) as src:
                img = src.read(1)

                transform = src.transform
                min_lon, min_lat, max_lon, max_lat = src.bounds
                max_lat = max_lat - 0.09
                min_lat = min_lat - 0.09

            img = np.nan_to_num(img, nan=0)
            img_min, img_max = np.min(img), np.max(img)
            img_norm = (img - img_min) / (img_max - img_min)

            colormap = plt.get_cmap("viridis")
            norm = Normalize(vmin=0, vmax=1)
            rgb_image = (colormap(norm(img_norm))[:, :, :3] * 255).astype(np.uint8)
            rgb_image = np.squeeze(rgb_image)
            rgb_image = rgb_image[:, :, :3]
            rgb_image = np.flipud(rgb_image)

            bounds_ger = [[max_lat, max_lon], [min_lat, min_lon]]

            map_dataset = folium.Map(
                location=[sum(means_lat) / len(means_lat), sum(means_lon) / len(means_lon)],
                zoom_start=14,
                control_scale=True
            )
            map_dataset.add_child(folium.raster_layers.ImageOverlay(rgb_image, opacity=.7,
                                                                    bounds=bounds_ger))
            for polygon in all_areas_dataset:
                folium.Polygon(
                    locations=polygon,
                    color="red",
                    fill=True,
                    fill_opacity=0.2,
                    popup="Combined Area"
                ).add_to(map_dataset)

            map_dataset.save(sim_func.EXPORT_PATH_HTML + name + '_gps.html')

            bounds_all.append(bounds_folium)

        bounds_all = pd.concat(bounds_all)
        bounds_all.to_csv("areaspername.csv")

        f = open('polygons.txt', "w")

        for k, v in all_polygons.items():
            f.write(str(k) + ': ' + ','.join(v) + '\n\n')

        f.close()

    elif option == "hfi":
        dataset_areas = pd.read_csv("areaspername.csv")

        min_lat_com = dataset_areas['min_lat'].min() - 0.5
        min_lon_com = dataset_areas['min_lon'].min() - 0.5
        max_lat_com = dataset_areas['max_lat'].max() + 0.5
        max_lon_com = dataset_areas['max_lon'].max() + 0.5
        dataset_areas.index = ['Caro S', 'Caro W', 'Katti']

        height_ratio = dataset_areas['max_lat'] - dataset_areas['min_lat']
        height_ratio = height_ratio.to_list()
