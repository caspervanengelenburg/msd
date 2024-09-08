import networkx as nx
import numpy as np

from shapely import geometry



def normalize_graph_out(graph: nx.Graph, bounds: np.ndarray):

    for i, node in graph.nodes(data=True):
        geometry_normalized = (np.array(node["geometry"]) - bounds[0]) / (bounds[1] - bounds[0]) * 512
        # node["geometry_og"] = node["geometry"]
        del node["geometry"]

        node["centroid_normalized"] = (np.array(node["centroid"]) - bounds[0]) / (bounds[1] - bounds[0]) * 512
        node["centroid_og"] = node["centroid"]
        del node["centroid"]


        # node["polygon"] = geometry.Polygon(node["geometry"])
        node["polygon"] = geometry.Polygon(geometry_normalized)

    # TODO: autoalign based on the full out image

    return graph


def make_minimum_rotated_rect(graph: nx.Graph):
    for i, node in graph.nodes(data=True):
        node["mrr"] = node["polygon"].minimum_rotated_rectangle

    return graph

def turn_mrr_into_house(graph: nx.Graph):
    """Turn the minimum rounded rectangle into house gan datareader house format"""
    
    for i, node in graph.nodes(data=True):

        corners = []
        for corner in node["mrr"].exterior.coords[:-1]:
            corners.append(corner)
        
        assert len(corners) == 4

        node["house_room"] = [np.array(corners), node["room_type"]]
        
    return graph


def simplify_room_polygon(polygon: geometry.Polygon):
    return polygon.simplify(tolerance=0.1)


def turn_polygon_into_house(graph: nx.Graph):
    """Turn the original room polygons into house gan datareader house format"""
    
    for i, node in graph.nodes(data=True):

        polygon = node["polygon"]

        # Some polygons had unnecessary corners
        # A tolerance of 0.1 will "most likely" keep the shape the same
        polygon = simplify_room_polygon(polygon)

        corners = []
        for corner in polygon.exterior.coords[:-1]:
            corners.append(corner)
        
        assert len(corners) >= 3, f"A polygon must have at least 3 corners, but this one has {len(corners)} corners"

        node["house_room"] = [np.array(corners), node["room_type"]]
        
    return graph

def process_graph_out(graph: nx.Graph, bounds: np.ndarray, img_out: np.ndarray, make_mrr=True):

    graph = graph.copy()

    graph = normalize_graph_out(graph, bounds)

    if make_mrr:
        graph = make_minimum_rotated_rect(graph)
        graph = turn_mrr_into_house(graph)
    else:
        graph = turn_polygon_into_house(graph)

    return graph


def make_graph_edges_like_hgdr(graph: nx.Graph):
    graph_edges_like_hgdr = []

    for u in graph.nodes():
        for v in graph.nodes():
            if u == v:
                continue
            
            if not graph.has_edge(u, v):
                graph_edges_like_hgdr.append([u, -1, v])
            else:
                graph_edges_like_hgdr.append([u, 1, v])
    
    return graph_edges_like_hgdr


def housegandatareader_like_process_subgraph(graph: nx.Graph, bounds: np.ndarray, img_out: np.ndarray=None, make_mrr=True):
    graph = process_graph_out(graph, bounds, img_out, make_mrr=make_mrr)

    house = []
    room_indices = []

    for i, node in graph.nodes(data=True):
        house.append(node["house_room"])
        room_indices.append(i)

    graph_edges_like_hgdr = make_graph_edges_like_hgdr(graph)

    graph_edges = {}

    for u, v, data in graph.edges(data=True):
        connectivity = data["connectivity"]

        graph_edges[(u, v)] = connectivity
        graph_edges[(v, u)] = connectivity
    
    return {
        "house": house,
        "room_indices": room_indices,
        "graph_edges": graph_edges,
        "graph_edges_like_hgdr": graph_edges_like_hgdr    
    }, graph


def process_graph_pred(graph: nx.Graph, number_of_corners: int):
    """Adds house_room corners filled with zeros"""

    assert number_of_corners >= 3, f"Number of corners must be at least 3, but is {number_of_corners}"

    graph = graph.copy()

    for i, node in graph.nodes(data=True):

        # Room type should have been predicted from the zoning type before
        node["house_room"] = [np.zeros((number_of_corners, 2)), node["room_type"]]

    return graph


def process_graph_pred_n_corners(graph: nx.Graph):
    """Adds house_room corners filled with zeros"""

    graph = graph.copy()

    for i, node in graph.nodes(data=True):

        node_num_corners = node["n_corners"]

        assert node_num_corners >= 3, f"Number of corners must be at least 3, but is {node_num_corners}"

        node_num_corners = int(node_num_corners)

        # Room type should have been predicted from the zoning type before
        node["house_room"] = [np.zeros((node_num_corners, 2)), node["room_type"]]

    return graph


def housegandatareader_like_process_subgraph_test_sample(graph: nx.Graph, override_num_corners=None):
    if override_num_corners is not None:
        graph = process_graph_pred(graph, number_of_corners=override_num_corners)
    else:
        graph = process_graph_pred_n_corners(graph)

    house = []
    room_indices = []

    for i, node in graph.nodes(data=True):
        house.append(node["house_room"])
        room_indices.append(i)

    graph_edges_like_hgdr = make_graph_edges_like_hgdr(graph)

    graph_edges = {}

    for u, v, data in graph.edges(data=True):
        connectivity = data["connectivity"]

        graph_edges[(u, v)] = connectivity
        graph_edges[(v, u)] = connectivity
    
    return {
        "house": house,
        "room_indices": room_indices,
        "graph_edges": graph_edges,
        "graph_edges_like_hgdr": graph_edges_like_hgdr    
    }, graph
