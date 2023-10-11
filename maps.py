#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:38:59 2023

@author: dimitris
"""
import os
import osmnx as ox
import folium
import requests
import polyline
import pandas as pd
import numpy as np
from itertools import product
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment


class PostOfficeRouter:

    def __init__(self, place_name, A_coords):
        self.place_name = place_name
        self.A_coords = A_coords
        self.locations = self.get_post_office_locations()
        self.cost_matrix, self.info_root = self.create_cost_matrix()

    def get_post_office_locations(self):
        tags = {'amenity': 'post_office'}
        geometries = ox.geometries_from_place(self.place_name, tags)
        df = pd.DataFrame(geometries)
        coords = df['geometry'].apply(lambda polygon: [polygon.centroid.y, polygon.centroid.x])
        locations = []
        for i, row in df.iterrows():
            if isinstance(row['name'], str) and row['name'] != 'nan':
                name = row['name']
                lat, lon = coords[i]
                locations.append({'name': name, 'latitude': lat, 'longitude': lon})
        return locations

    def get_route(self, start_coords, end_coords):
        pickup_lat, pickup_lon = start_coords
        dropoff_lat, dropoff_lon = end_coords
        loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
        url = "http://router.project-osrm.org/route/v1/driving/"
        r = requests.get(url + loc)
        if r.status_code != 200:
            return {}
        res = r.json()
        routes = polyline.decode(res['routes'][0]['geometry'])
        return {
            'route': routes,
            'start_point': [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]],
            'end_point': [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]],
            'distance': res['routes'][0]['distance'],
            'duration': res['routes'][0]['duration'],
            'weight': res['routes'][0]['weight']
        }

    def create_cost_matrix(self):
        coords = [(loc['latitude'], loc['longitude']) for loc in self.locations]
        coords_with_A = [self.A_coords] + coords
        cost_matrix = np.zeros((len(coords) + 1, len(coords) + 1))
        info_root = np.empty((len(coords) + 1, len(coords) + 1), dtype=object)
        for i, j in product(range(len(coords) + 1), range(len(coords) + 1)):
            if i >= j:
                start_coords = coords_with_A[i]
                end_coords = coords_with_A[j]
                info = self.get_route(start_coords, end_coords)
                info_root[i][j] = info
                cost_matrix[i][j] = info['weight']
        return cost_matrix, info_root

    def create_clusters(self, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.cost_matrix[1:, 1:])
        labels = kmeans.labels_
        clusters = []
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0] + 1
            clusters.append(cluster_indices)
        return clusters

    def find_optimal_transport(self, subcost_matrix):
        row_ind, col_ind = linear_sum_assignment(subcost_matrix)
        return row_ind, col_ind

    def compute_subcost_matrix(self, sub_list):
        indices = np.append([0], sub_list)
        return self.cost_matrix[indices][:, indices]

    def construct_routes(self, row_ind, col_ind):
        transport_plan = np.zeros((len(row_ind), len(row_ind)))
        transport_plan[row_ind, col_ind] = 1
        routes = []
        for src_index, dest_index in zip(row_ind, col_ind):
            if src_index == 0:
                routes.append([0, dest_index])
            else:
                routes[-1].append(dest_index)
        for route in routes:
            route.append(0)
        return routes

    def display_map(self, path):
        m = folium.Map(location=self.A_coords, zoom_start=13)
        for route, cl in zip(path, range(len(path))):
            route = route[0]
            locations = []
            for i, j in zip(route[:-1], route[1:]):
                info = self.info_root[max(i, j)][min(i, j)]
                if i > j:
                    locations += info['route']
                else:
                    locations += reversed(info['route'])
            colors = ['blue', 'red', 'black', 'green', 'yellow']
            folium.PolyLine(locations, color=colors[cl], weight=2.5, opacity=1).add_to(m)
        return m

# Usage:
router = PostOfficeRouter('Vaud, Switzerland', [46.633367, 6.540033])
clusters = router.create_clusters(4)

path = []
for cluster_indices in clusters:
    subcost_matrix = router.compute_subcost_matrix(cluster_indices)
    row_ind, col_ind = router.find_optimal_transport(subcost_matrix)
    path.append(router.construct_routes(row_ind, col_ind))

map_display = router.display_map(path)
map_display.save("map.html")

# Display the map
