#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 09:38:59 2023

@author: dimitris
"""
import osmnx as ox
import folium
import osrm
from shapely.geometry import LineString
import os
import requests
import polyline
import pandas as pd
import tracemalloc
import numpy as np
from itertools import product
from sklearn.cluster import KMeans

def get_post_office_locations(place_name):
    # Define the tags to filter for
    tags = {'amenity': 'post_office'}

    # Download the points of interest data using OSMnx
    geometries = ox.geometries_from_place(place_name, tags)

    # Extract the latitude and longitude columns
    df = pd.DataFrame(geometries)
    coords = df['geometry'].apply(lambda polygon: [polygon.centroid.y, polygon.centroid.x])

    # Create a list of dictionaries with the name and coordinates of each post office
    locations = []
    for i, row in df.iterrows():
        if isinstance(row['name'], str) and row['name'] != 'nan':
            name = row['name']
            lat, lon = coords[i]
            locations.append({'name': name, 'latitude': lat, 'longitude': lon})

    return locations

place_name = 'Vaud, Switzerland'
locations = get_post_office_locations(place_name)
print(locations)

#Define a starting point starting point
A_coords= [46.633367, 6.540033] # Poste CH SA - Base Distribution 1310 Daillens

# start_coords=[46.4586359, 6.8495566]
# end_coords =[46.4519109, 6.2907229]
def get_route(start_coords,end_coords):
    pickup_lat=start_coords[0]
    pickup_lon=start_coords[1]
    dropoff_lat=end_coords[0]
    dropoff_lon=end_coords[1]
    loc = "{},{};{},{}".format(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)
    url = "http://router.project-osrm.org/route/v1/driving/"
    r = requests.get(url + loc) 
    if r.status_code!= 200:
        return {}
  
    res = r.json()   
    routes = polyline.decode(res['routes'][0]['geometry'])
    start_point = [res['waypoints'][0]['location'][1], res['waypoints'][0]['location'][0]]
    end_point = [res['waypoints'][1]['location'][1], res['waypoints'][1]['location'][0]]
    distance = res['routes'][0]['distance']
    duration = res['routes'][0]['duration']
    weight =res['routes'][0]['weight']
    out = {'route':routes,
           'start_point':start_point,
           'end_point':end_point,
           'distance':distance,
           'duration':duration,
           'weight':weight
          }

    return out


def create_cost_matrix(x, A_coords, get_route):
    # create a list of coordinates
    coords = [(loc['latitude'], loc['longitude']) for loc in x]

    # add the starting point to the coordinates list and create a matrix of zeros
    coords_with_A = [A_coords] + coords
    cost_matrix = np.zeros((len(coords) + 1, len(coords) + 1))
    info_root =  np.empty((len(coords) + 1,  len(coords) + 1), dtype=object)

    # calculate the cost between each pair of locations, including the starting point
    for i, j in product(range(len(coords) + 1), range(len(coords) + 1)):
        if i>=j :
            start_coords = coords_with_A[i]
            end_coords = coords_with_A[j]
            info=get_route(start_coords, end_coords)
            info_root[i][j] =info
            print(i,j)
            cost_matrix[i][j] = info['weight']  # time or distance, depending on the cost metric 

    return cost_matrix,info_root


cost_matrix,info_root =create_cost_matrix(locations, A_coords, get_route)
# with open('cost_matrix.npy', 'wb') as f:
#     np.save(f, cost_matrix)
    
# with open('info_root.npy', 'wb') as f:
#     np.save(f, info_root)

with open('cost_matrix.npy', 'rb') as f:
    cost_matrix = np.load(f)
    
cost_matrix=(cost_matrix + np.transpose(cost_matrix)) 
    
from sklearn.cluster import KMeans


def create_clusters(cost_matrix, num_clusters):
    # use K-means clustering to group the locations into N clusters with similar costs
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(cost_matrix[1:, 1:])
    labels = kmeans.labels_

    # create a list of indices for each cluster
    clusters = []
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0] + 1
        clusters.append(cluster_indices)

    return clusters    

clusters=create_clusters(cost_matrix, 4)

from scipy.optimize import linear_sum_assignment

def find_optimal_transport(cost_matrix):
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    transport_plan = np.zeros_like(cost_matrix)
    transport_plan[row_ind, col_ind] = 1
    return transport_plan


def compute_subcost_matrix(sub_list, cost_matrix):
    indices = np.append([0],sub_list)
    subcost_matrix = cost_matrix[sub_list]
    return subcost_matrix

path=[]
for cluster_indices in clusters:
    subcost_matrix=compute_subcost_matrix(cluster_indices, cost_matrix)
    # print(cluster_indices)
    transport_plan=find_optimal_transport(subcost_matrix)
    #print(np.size(transport_plan))
    path.append(construct_routes(transport_plan, 0))
    
def construct_routes(transport_plan, start_index):
    routes = []
    src_indices, dest_indices = transport_plan.nonzero()
    for src_index, dest_index in zip(src_indices, dest_indices):
        if src_index == 0:
            routes.append([start_index, dest_index])
        else:
            routes[-1].append(dest_index)

    # add the last location (the starting point) to each route
    for route in routes:
        route.append(start_index)

    return routes



def get_map(route):
    
    m = folium.Map(location=[(route['start_point'][0] + route['end_point'][0])/2, 
                             (route['start_point'][1] + route['end_point'][1])/2], 
                   zoom_start=13)

    folium.PolyLine(
        route['route'],
        weight=8,
        color='blue',
        opacity=0.6
    ).add_to(m)

    folium.Marker(
        location=route['start_point'],
        icon=folium.Icon(icon='play', color='green')
    ).add_to(m)

    folium.Marker(
        location=route['end_point'],
        icon=folium.Icon(icon='stop', color='red')
    ).add_to(m)

    return m



def get_map(path,A_coords):

    # Create a folium map centered on the starting point
    m = folium.Map(location=A_coords, zoom_start=13)

    # Add polylines and markers for each route
    for route,cl in zip(path,range(len(path))):
        # Create a list of locations for the polyline by concatenating the start point, intermediate points, and end point
        k=0
        route=route[0]
        for i,j in zip(route[:len(route)-1],route[1:]):
            if k==0:
                infos=info_root[j][i]['route']
                locations= infos[::-1]
                k=1
                # print(i,j)
                # print(info_root[j][i]['route'])
            elif  i>j :
                locations=locations+  info_root[i][j]['route']
                # print(i,j)
                # print( info_root[i][j]['route'])
            else : 
                infos=info_root[j][i]['route']
                locations=locations+  infos[::-1]
                # print(j,i)
                # print(info_root[j][i]['route'])
            print(i,j,len(locations))
                
        
        color_line=['blue','red','black','green','yellow']
        # Add a polyline representing the route to the map
        folium.PolyLine(
            locations,
            weight=8,
            color=color_line[cl],
            opacity=0.6
        ).add_to(m)

        # Add a marker for the start point with a green "play" icon
        folium.Marker(
            location=A_coords,
            icon=folium.Icon(icon='play', color='green')
        ).add_to(m)

        # Add markers for all intermediate points in the route
        for point in route[1:len(route)-1]:
            folium.Marker(
                location=info_root[point][point]['start_point'],
                icon=folium.Icon(icon='stop', color=color_line[cl])
            ).add_to(m)

        # Add a marker for the end point with a red "stop" icon
        folium.Marker(
            location=A_coords,
            icon=folium.Icon(icon='stop', color='red')
        ).add_to(m)

    return m


m= get_map(path,A_coords)
m.save("map.html")

# Define the start and end coordinates
start_coords = [46.525226, 6.633761] # Vevey
end_coords = [46.515153, 6.536695]# Lausanne


route=get_route(start_coords,end_coords)

m=get_map(route)
m.save("map.html")



# Define place and network type
place = 'Vaud, Switzerland'
network_type = 'drive'

# Load or save graph from/to file
graph_file = 'vaud.graphml'
if os.path.exists(graph_file):
    G = ox.load_graphml(graph_file)
else:
    G = ox.graph_from_place(place, network_type=network_type)
    ox.save_graphml(G, graph_file)

# Convert the network to a GeoDataFrame and reproject it to EPSG:4326
gdf = ox.graph_to_gdfs(G, nodes=False, edges=True)
gdf = gdf.to_crs(epsg=4326)

# Create a folium map object
m = folium.Map(location=[46.8, 8.2], zoom_start=10)

# Add the street network to the map
folium.GeoJson(gdf).add_to(m)

# Calculate the route using the OSRM API
client = osrm.Client(host="http://router.project-osrm.org")
geometry = client.route(coordinates=[start_coords, end_coords])['routes'][0]['geometry']
linestring = LineString([list(reversed(coord)) for coord in geometry['coordinates']])

# Add the route to the map
route_coords = [[coord[1], coord[0]] for coord in linestring.coords]
folium.PolyLine(locations=route_coords, color='red', weight=4).add_to(m)

# Display the map
