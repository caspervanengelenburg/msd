from itertools import pairwise
import typing
import numpy as np

from shapely import geometry

from skimage.util import invert

from skimage.morphology import medial_axis, skeletonize, thin


# Needs the github version of sknw: pip install git+https://github.com/Image-Py/sknw (used at commit 68cf723)
# (Even thouhg the pypi version number is the same, the pypi version doesn't work with ring=True)
import sknw


def process_structure_into_graph(structural_img: np.ndarray):
    structural_img_inv = invert(structural_img)


    skeleton = thin(structural_img_inv)

    # Convert the skeleton image to a graph
    graph = sknw.build_sknw(skeleton, multi=False, ring=True)

    return graph


def extract_walls_from_graph(graph) -> typing.List[geometry.LineString]:

    walls = []

    for (s,e) in graph.edges():

        length = graph[s][e]['weight']
        if length < 20:
            continue
        
        ps = graph[s][e]['pts']

        geom = geometry.LineString(ps)

        geom = geom.simplify(5)

        walls.append(geom)
    
    return walls


def process_structure(structural_img: np.ndarray):

    graph = process_structure_into_graph(structural_img)


    return extract_walls_from_graph(graph), graph


def extract_walls_from_graph_raw(graph) -> typing.List[geometry.LineString]:

    walls = []

    for (s,e) in graph.edges():
        
        ps = graph[s][e]['pts']

        geom = geometry.LineString(ps)

        walls.append(geom)
    
    return walls


def process_structure_alternative_method(structural_img: np.ndarray, lower_bound_wall_length=15, longest_length_factor=0.1):
    from topojson.core.dedup import Dedup

    _, walls_graph = process_structure(structural_img)
    walls = extract_walls_from_graph_raw(walls_graph)

    # Simplify with 2 pixel tolerance
    walls = geometry.MultiLineString(walls).simplify(2, preserve_topology=True)

    linestrings = Dedup(walls)
    linestrings = linestrings.to_dict()["linestrings"]
    linestrings = geometry.MultiLineString(linestrings)

    single_line_walls = turn_walls_into_single_line_walls(linestrings.geoms)

    longest_wall_length = max(single_line_walls, key=lambda x: x.length).length

    single_line_walls = [wall for wall in single_line_walls if wall.length > max(lower_bound_wall_length, longest_wall_length * longest_length_factor)]

    single_line_walls = sorted(single_line_walls, key=lambda x: x.length, reverse=True)

    return single_line_walls




def turn_walls_into_single_line_walls(walls: typing.List[geometry.LineString]):
    return [geometry.LineString(line) for line in turn_walls_into_single_lines(walls)]

def turn_walls_into_single_lines(walls: typing.List[geometry.LineString]):

    single_line_walls = []

    for wall in walls:
        assert isinstance(wall, geometry.LineString), f"Expected LineString, got {type(wall)}: {wall}"

        for pair in pairwise(wall.coords):
            single_line_walls.append(np.stack(pair))

    assert len(single_line_walls) >= len(walls)

    return single_line_walls
