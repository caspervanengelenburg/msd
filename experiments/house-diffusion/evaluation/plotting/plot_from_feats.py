import typing
import numpy as np
from collections import defaultdict

import numpy as np

from .plot_msd_mrr import plot_minimum_rounded_rect_polygons_refined_by_structure_mask, plot_minimum_rounded_rect_polygons_without_structure_mask

def decode_one_hot(one_hot: np.ndarray):
    return np.argmax(one_hot, axis=-1)

def decode_from_batch(sample, model_kwargs, batch_index, timestep=None):

    if timestep is None:
        timestep = -1

    padding_mask = model_kwargs["src_key_padding_mask"][batch_index] == 0

    # Model output is between -1 and 1, so we need to scale it to 0 to 512
    corners = (sample[timestep][batch_index][padding_mask].numpy() / 2 + 0.5) * 512


    room_indices = model_kwargs["room_indices"][batch_index][padding_mask]
    room_types = model_kwargs["room_types"][batch_index][padding_mask]
    corner_indices = model_kwargs["corner_indices"][batch_index][padding_mask]
    connections = model_kwargs["connections"][batch_index][padding_mask]

    room_indices = decode_one_hot(room_indices).numpy()
    room_types = decode_one_hot(room_types).numpy()
    corner_indices = decode_one_hot(corner_indices).numpy()
    
    res = {
        "corners": corners,

        "room_indices": room_indices,
        "room_types": room_types,
        "corner_indices": corner_indices,

        "connections": connections,
    }

    if "struct_corners_a" in model_kwargs:
        struct_corners_a = (model_kwargs["struct_corners_a"][batch_index].numpy() / 2 + 0.5) * 512
        struct_corners_b = (model_kwargs["struct_corners_b"][batch_index].numpy() / 2 + 0.5) * 512

        structure_active_mask = (model_kwargs["structural_mask"][batch_index] == 0).any(axis=0)

        res.update({
            "struct_corners_a": struct_corners_a,
            "struct_corners_b": struct_corners_b,

            "structure_active_mask": structure_active_mask,

            "structural_mask": model_kwargs["structural_mask"][batch_index].numpy()
        })

    if "id" in model_kwargs:
        id_ = model_kwargs["id"][batch_index]

        res["id"] = id_
    
    return res



def group_corners_by_room(corners, room_index, room_types):
    corners_by_room = defaultdict(list)
    room_to_roomtype = {}

    for corner, room_index, room_type in zip(corners, room_index, room_types):
                
        corners_by_room[room_index].append(corner)

        if room_index in room_to_roomtype:
            assert room_to_roomtype[room_index] == room_type
        else:
            room_to_roomtype[room_index] = room_type
    
    return corners_by_room, room_to_roomtype


def prepare_for_plotting(corners, room_indices, room_types):
    corners_by_room, room_type_dict = group_corners_by_room(corners, room_indices, room_types)
    corners_by_room = {k: np.array(v) for k, v in corners_by_room.items()}

    return corners_by_room, room_type_dict

# def plot_from_batch(sample, model_kwargs, batch_index, time_step=None, draw_outline=False, structural_img=None):
#     geom_dict = decode_from_batch(sample, model_kwargs, batch_index, timestep=time_step)

#     corners_by_room, room_type_dict = prepare_for_plotting(geom_dict["corners"], geom_dict["room_indices"], geom_dict["room_types"])

#     if structural_img is None:
#         return plot_minimum_rounded_rect_polygons_without_structure_mask(corners_by_room, room_type_dict, draw_outline=draw_outline)
#     else:
#         return plot_minimum_rounded_rect_polygons_refined_by_structure_mask(corners_by_room, room_type_dict, structural_img)


def draw_from_batch(sample, model_kwargs, batch_index, time_step=-1, draw_outline=False, structural_img=None, color_by_index_instead_of_type=False):
    geom_dict = decode_from_batch(sample, model_kwargs, batch_index, timestep=time_step)

    if color_by_index_instead_of_type:
        corners_by_room, room_color_dict = prepare_for_plotting(geom_dict["corners"], geom_dict["room_indices"], geom_dict["room_indices"])
    else:
        corners_by_room, room_color_dict = prepare_for_plotting(geom_dict["corners"], geom_dict["room_indices"], geom_dict["room_types"])

    if structural_img is None:
        return plot_minimum_rounded_rect_polygons_without_structure_mask(corners_by_room, room_color_dict, draw_outline=draw_outline)
    else:
        try:
            return plot_minimum_rounded_rect_polygons_refined_by_structure_mask(corners_by_room, room_color_dict, structural_img)
        except:
            return plot_minimum_rounded_rect_polygons_without_structure_mask(corners_by_room, room_color_dict, draw_outline=draw_outline)



from matplotlib.cm import get_cmap
import matplotlib.colors as mcolors

COLORS_ROOMTYPE = ['#1f77b4',
                      '#e6550d',
                      '#fd8d3c',
                      '#fdae6b',
                      '#fdd0a2',
                      '#72246c',
                      '#5254a3',
                      '#6b6ecf',
                      '#2ca02c',
                      '#000000',
                      '#1f77b4',
                      '#98df8a',
                      '#d62728',
                      "#ffffff"]

COLOR_MAP_ROOMTYPE = mcolors.ListedColormap(COLORS_ROOMTYPE)
CMAP_ROOMTYPE = get_cmap(COLOR_MAP_ROOMTYPE)


def plot_from_batch(sample, model_kwargs, batch_index, time_step=-1, draw_outline=False, structural_img=None, ax=None, plot_struct_lines=True, color_by_index_instead_of_type=False):
    img = draw_from_batch(sample, model_kwargs, batch_index, time_step, draw_outline, structural_img, color_by_index_instead_of_type=color_by_index_instead_of_type)

    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()

    if color_by_index_instead_of_type:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=CMAP_ROOMTYPE)

    if plot_struct_lines:
        geom_dict = decode_from_batch(sample, model_kwargs, batch_index, timestep=time_step)

        if "struct_corners_a" not in geom_dict:
            return

        # print(geom_dict["structure_active_mask"])

        for pointa, pointb, is_active in zip(geom_dict["struct_corners_a"].T, geom_dict["struct_corners_b"].T, geom_dict["structure_active_mask"]):
            if is_active:
                line = np.stack([pointa.T, pointb.T])

                ax.plot(line[:, 0], line[:, 1], color="red")
