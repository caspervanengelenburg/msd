import pickle
import numpy as np


def save_pickle(object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(object, f)
    f.close()


def load_pickle(filename):
    with open(filename, 'rb') as f:
        object = pickle.load(f)
        f.close()
    return object


def colorize_floorplan(img, classes, cmap):

    """
    Colorizes an integer-valued image (multi-class segmentation mask)
    based on a pre-defined cmap colorset.
    """

    h, w = np.shape(img)
    img_c = (np.ones((h, w, 3)) * 255).astype(int)
    for cat in classes:
        color = np.array(cmap(cat))[:3] * 255
        img_c[img == cat, :] = (color).astype(int)

    return img_c


def find_floor_boundary(polygons):
    min_x, min_y, max_x, max_y = 100, 100, -100, -100

    for polygon in polygons:

        bounds = polygon.bounds

        if min_x < bounds[0]:
            min_x = bounds[0]
        if min_y < bounds[1]:
            min_y = bounds[1]
        if max_x > bounds[2]:
            max_x = bounds[2]
        if max_y > bounds[3]:
            max_y = bounds[3]

    return min_x, min_y, max_x, max_y