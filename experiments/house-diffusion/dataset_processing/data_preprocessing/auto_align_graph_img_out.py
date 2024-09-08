import networkx as nx
import numpy as np


from shapely import ops, geometry, affinity
from rasterio import features

def compute_outline_from_graph(graph: nx.Graph, img_out):
    polys = [poly for _, poly in graph.nodes("polygon")]

    outline = ops.unary_union(polys)

    return outline


def compute_outline_from_img(img_out):

    polys = []

    for shape, value in features.shapes(img_out, mask=img_out < 13):
        # print(shape, value)

        poly = geometry.shape(shape)

        polys.append(poly)
    
    outline = ops.unary_union(polys)

    return outline




def compute_offset(graph: nx.Graph, img_out: np.ndarray):

    outline_graph = compute_outline_from_graph(graph, img_out)

    outline_img = compute_outline_from_img(img_out)

    MAX_OFFSET = 500
    OFFSET_STEP = 5

    offset_candidates = [[0, offset] for offset in range(-MAX_OFFSET,  MAX_OFFSET, OFFSET_STEP)]
    offset_candidates += [[offset, 0] for offset in range(-MAX_OFFSET,  MAX_OFFSET, OFFSET_STEP)]

    x1, y1, x2, y2 = outline_graph.bounds

    bounds = np.array([[x1, y1], [x2, y2]])


    alligned_areas = []

    assert img_out.shape[0] == img_out.shape[1], "Assumed square image for bounds checking"

    for offset in offset_candidates:

        # Skip computing if shape would be outside of image
        offset_bounds = bounds + offset
        if offset_bounds.min() < 0 or offset_bounds.max() > img_out.shape[0]:
            continue


        offset_outline_graph = affinity.translate(outline_graph, offset[0], offset[1])
        
        area = offset_outline_graph.intersection(outline_img).area

        alligned_areas.append((area, offset))
    
    if len(alligned_areas) == 0:
        raise ValueError("No offset candidates found!")

    # Get the offset with the largest area
    alligned_areas.sort(key=lambda x: x[0], reverse=True)
    
    offset = alligned_areas[0][1]


    return offset


def auto_align(graph: nx.Graph, img_out: np.ndarray):
    offset = compute_offset(graph, img_out)

    for _, data in graph.nodes(data=True):
        data["polygon"] = affinity.translate(data["polygon"], offset[0], offset[1])

        data["centroid_normalized"] = data["centroid_normalized"] + offset

    return graph, offset