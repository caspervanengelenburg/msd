from itertools import combinations
import numpy as np
from shapely import wkt
from shapely.geometry import Polygon
import torch
import networkx as nx


def polygon_to_list(polygon: Polygon) -> list:
    """Converts a polygon into a list of coordinates."""
    return list(zip(*polygon.exterior.coords.xy))


def polygon_to_array(polygon: Polygon) -> np.array:
    """Converts a polygon into a numpy array."""
    return np.array(polygon_to_list(polygon))


def extract_access_graph(geoms, geoms_type, classes, id):
    """Extracts the access graph from a set of geometries."""

    # Sets the mapping
    mapping = {cat: index for index, cat in enumerate(classes)}

    # Initializes and separate areas (and types), doors, and entrance doors
    areas, areas_type, doors, entrance_doors = [], [], [], []

    # Makes sure to only have {Zone1, Zone2, Zone3, Zone4} in areas
    end = 4 if 'Zone1' in classes else 9
    for geom, geom_type in zip(geoms, geoms_type):
        if geom_type == 'Door':
            doors.append(geom)
        elif geom_type == 'Entrance Door':
            entrance_doors.append(geom)
        elif geom_type in classes[:end]:
            areas.append(geom)
            areas_type.append(geom_type)
        else: continue  # walls are omitted

    # Accumulate nodes
    area_nodes = {}
    for key, (area, area_type) in enumerate(zip(areas, areas_type)):

        # Zoning (input) graph node attributes
        if 'Zone1' in classes:
            area_nodes[key] = {
                'zoning_type': mapping[area_type]
            }
        # Full (output) graph attributes
        else:
            area_nodes[key] = {
                'geometry': polygon_to_list(area),
                'room_type': mapping[area_type],
                'centroid': torch.tensor(np.array([area.centroid.x, area.centroid.y]))
            }

    # Accumulate edges
    edges = []
    for (i, v1), (j, v2) in combinations(enumerate(areas), 2):

        # Option 1: PASSAGE (direct access := no wall in between)
        if v1.distance(v2) < 0.04:
            edges.append([i, j, {'connectivity': 'passage'}])

        # Option 2: DOOR
        else:
            for door in doors:
                if door.distance(v1) < 0.05 and door.distance(v2) < 0.05:
                    # Adds the geometry of the door as well (slightly different from paper)
                    edges.append([i, j, {'connectivity': 'door'}])  #, 'door_geometry': polygon_to_list(door)}])
                else: continue

        # Option 3: FRONT DOOR
        for entrance_door in entrance_doors:
            if entrance_door.distance(v1) < 0.05 and entrance_door.distance(v2) < 0.05:
                # Adds the geometry of the door as well (slightly different from paper)
                edges.append([i, j, {'connectivity': 'entrance'}])  #, 'door_geometry': polygon_to_list(entrance_door)}])
            else: continue

    # Defines the graph
    G = nx.Graph()
    G.graph["ID"] = id  # Give the floor ID as graph attribute
    G.add_nodes_from([(u, v) for u, v in area_nodes.items()])
    G.add_edges_from(edges)

    return G


def get_geometries_from_id(df, floor_id, column='zoning'):

    """
    Extracting geometry information from particular floor ID.
    """

    df_floor = df[(df.floor_id == floor_id)].reset_index(drop=True)
    df_floor.geom = df_floor.geom.apply(wkt.loads)

    geoms, geoms_type = zip(*df_floor[["geom", column]].values)

    return geoms, geoms_type