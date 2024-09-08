"""Class for plotting Minimum Rounded Rectangles (MRR) of the Modified Swiss Dwellings (MDS) data set."""

from rasterio.features import shapes
import numpy as np
from shapely import geometry, ops


from .drawing import FPImage


def compute_structural_shape(structural_img: np.ndarray):
    """Extract shape of the structural element for plotting purposes."""
    polygons = []

    for shape, val in shapes(structural_img, mask=structural_img==0):
        polygons.append(geometry.shape(shape))

    structural_shape = geometry.MultiPolygon(polygons)

    return structural_shape

def cut_structural(polygon, structural_shape) -> geometry.Polygon:

    poly_cuts = polygon - structural_shape

    # Get the largest cut, continue until we have a polygon
    while not isinstance(poly_cuts, geometry.Polygon):
        poly_cuts = sorted(poly_cuts.geoms, key=lambda x: x.area, reverse=True)

        poly_cuts = poly_cuts[0]

    assert isinstance(poly_cuts, geometry.Polygon)
    
    return poly_cuts


def plot_minimum_rounded_rect_polygons_refined_by_structure_mask(room_corners, room_types, struct_img: np.ndarray):

    structural_geometry = compute_structural_shape(struct_img)

    room_polygons = {key: geometry.Polygon(value) for key, value in room_corners.items()}

    # Use the structural shape to refine the minimum rounded rectangle polygons
    room_polygons = {key: cut_structural(poly, structural_geometry) for key, poly in room_polygons.items()}
    
    # Sort room_polygons by area in reverse order
    room_polygons = sorted(room_polygons.items(), key=lambda x: x[1].area, reverse=True)

    fp_image = FPImage(background=13)

    outline = ops.unary_union([poly for _, poly in room_polygons]).buffer(5).buffer(-5)

    if isinstance(outline, geometry.Polygon):
        fp_image.draw_polygon(np.array(outline.exterior.coords), value=9)

    for key, polygon in room_polygons:        
        room_type = room_types[key]
        
        coords = np.array(polygon.exterior.coords)

        fp_image.draw_polygon(coords, value=room_type)

        # Draw room "walls"
        fp_image.draw_polygon_outline(coords, value=9, thickness=2)

    # Draw the outline (turned off for now)
    # fp_image.draw_polygon_outline(np.array(outline.exterior.coords), value=9, thickness=2)

    # Set the structure to 9
    selection = np.where(struct_img == 0)
    fp_image.img[selection] = 9

    return fp_image.img


def plot_minimum_rounded_rect_polygons_without_structure_mask(room_corners, room_colors, draw_outline=True):

    room_polygons = {key: geometry.Polygon(value) for key, value in room_corners.items()}
    
    # Sort room_polygons by area in reverse order
    room_polygons = sorted(room_polygons.items(), key=lambda x: x[1].area, reverse=True)

    fp_image = FPImage(background=13)

    if draw_outline:
        outline = ops.unary_union([poly for _, poly in room_polygons]).buffer(5).buffer(-5)

        fp_image.draw_polygon(np.array(outline.exterior.coords), value=9)

    for key, polygon in room_polygons:        
        room_color = room_colors[key]
        
        coords = np.array(polygon.exterior.coords)

        fp_image.draw_polygon(coords, value=room_color)

        # Draw room "walls"
        fp_image.draw_polygon_outline(coords, value=9, thickness=3)

    # Draw the outline (turned off for now)
    # fp_image.draw_polygon_outline(np.array(outline.exterior.coords), value=9, thickness=2)

    return fp_image.img




# def plot_mds_housediffusion(corners: np.ndarray, room_index: typing.List[np.ndarray], room_types: typing.List[np.ndarray], struct_img: np.ndarray):
#     """
    
#     Usage example:
    
#     result_img = plot_mds_housediffusion(house_dict["corners"], house_dict["room_indices"], house_dict["room_types"], sample.structural_img)
#     plt.imshow(result_img, cmap=CMAP_ROOMTYPE)

#     Can also be used with mIoU to compute a score between the plotted image and the ground truth one

#     :param corners: an N x 2 array of all the corners of all rooms
#     """
#     room_corners, room_types = group_corners_by_room(corners, room_index, room_types)

#     if struct_img is None:
#         return plot_minimum_rounded_rect_polygons_without_structure_mask(room_corners, room_types)
#     else:
#         return plot_minimum_rounded_rect_polygons_refined_by_structure_mask(room_corners, room_types, struct_img)
