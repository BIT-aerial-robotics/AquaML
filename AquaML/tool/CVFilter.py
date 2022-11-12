import cv2 as cv
import numpy as np


class PencilFilter:
    def __init__(self,
                 dilatation_size=2,
                 dilation_shape=cv.MORPH_ELLIPSE):
        self.dilatation_size = dilatation_size
        self.dilation_shape = dilation_shape  # cv.MORPH_ELLIPSE cv.MORPH_RECT cv.MORPH_CROSS

    def apply(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        dialted = self.dilatation(gray_img.copy())

        gray_img_32_bit = gray_img.copy().astype(np.uint32)
        dialted_my = ((gray_img_32_bit * 255) / dialted).astype(np.uint8)
        penciled = np.where(np.isnan(dialted_my), 255, dialted_my).astype(np.uint8)
        # penciled_rgb = cv.cvtColor(penciled, cv.COLOR_GRAY2RGB)

        return penciled

    def dilatation(self, img):
        element = cv.getStructuringElement(self.dilation_shape,
                                           (2 * self.dilatation_size + 1, 2 * self.dilatation_size + 1),
                                           (self.dilatation_size, self.dilatation_size))
        dilatation_dst = cv.dilate(img, element)
        return dilatation_dst