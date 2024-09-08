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