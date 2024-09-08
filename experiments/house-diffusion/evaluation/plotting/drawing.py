import numpy as np

from skimage import draw

import cv2

class FPImage:

    def __init__(self, height=512, width=512, background=0) -> None:
        self.height = height
        self.width = width

        self.img = np.zeros((height, width), dtype=np.uint8)

        self.img[:] = background
    
    def draw_polygon(self, polygon: np.ndarray, value):
        rr, cc = draw.polygon(polygon[:, 1], polygon[:, 0], shape=self.img.shape)
        self.img[rr, cc] = value
    
    def draw_polygon_outline(self, polygon: np.ndarray, value, thickness):
        polygon = polygon.astype(np.int32)

        cv2.polylines(self.img, [polygon], True, value, thickness=thickness)

