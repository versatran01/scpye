from __future__ import (print_function, absolute_import, division)

import cv2
import numpy as np

from scpye.image_transformer import ImageTransformer, split_label
from scpye.exception import FeatureNotSupportedError


class FeatureTransformer(ImageTransformer):
    @staticmethod
    def stack_list(func):
        """
        Decorator that stack the output if input is a list
        Currently only handles class member function
        :param func:
        """

        def func_wrapper(self, X, y=None):
            if isinstance(X, list):
                return np.vstack([func(self, each_X) for each_X in X])
            else:
                return func(self, X)

        return func_wrapper


class CspaceTransformer(FeatureTransformer):
    def __init__(self, cspace):
        self.cspace = cspace
        # self.image = None

    def cspace_transform(self, src):
        """
        :param src: bgr image
        :return: bgr image in other colorspace
        """
        if np.ndim(src) == 2:
            src = np.expand_dims(src, 1)

        if self.cspace == "hsv":
            des = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        elif self.cspace == "lab":
            des = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        elif self.cspace == "bgr":
            des = src
        else:
            raise FeatureNotSupportedError(self.cspace)

        return np.squeeze(des)

    @FeatureTransformer.stack_list
    def transform(self, X, y=None):
        """
        :param X: tuple of bgr image and mask
        :type X: MaskedData
        :param y:
        :return: masked image with transformed colorspace
        """
        bgr, mask = X.data, X.mask

        # Testing case, mask is 2D
        if np.ndim(mask) == 2:
            Xt = self.cspace_transform(bgr[mask])
            image = np.zeros_like(bgr)
            image[mask] = Xt
            # if y is None:
            #     self.image = image
        # Training case, mask is 3D
        else:
            neg, pos = split_label(mask)
            Xt_neg = self.cspace_transform(bgr[neg])
            Xt_pos = self.cspace_transform(bgr[pos])
            Xt = np.vstack((Xt_neg, Xt_pos))

        # Need to change to float to suppress later warnings
        return np.array(Xt, np.float64)


def xy_from_array(m):
    """
    Get locations of non-zero pixels in array
    :param m: array
    :type m: numpy.ndarray
    :return: n x 2 matrix of [x, y]
    :rtype: numpy.ndarray
    """
    assert np.ndim(m) == 2
    r, c = np.where(m)
    return np.transpose(np.vstack((r, c)))


class MaskLocator(FeatureTransformer):
    @FeatureTransformer.stack_list
    def transform(self, X, y=None):
        mask = X.mask

        if np.ndim(mask) == 2:
            Xt = xy_from_array(mask)
        else:
            neg, pos = split_label(mask)
            xy_neg = xy_from_array(neg)
            xy_pos = xy_from_array(pos)
            Xt = np.vstack((xy_neg, xy_pos))
        # Change to float to suppress warning
        return np.array(Xt, np.float64)


class PatchCreator(FeatureTransformer):
    def __init__(self, size):
        self.size = size

    @FeatureTransformer.stack_list
    def transform(self, X, y=None):
        bgr, mask = X.data, X.mask

        if np.ndim(mask) == 2:
            pass
        else:
            pass
