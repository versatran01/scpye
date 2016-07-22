from __future__ import (print_function, absolute_import, division)

import cv2
import numpy as np

from sklearn.feature_extraction.image import extract_patches_2d

from scpye.image_transformer import ImageTransformer, split_label, MaskedData
from scpye.exception import FeatureNotSupportedError


def stack_list(func):
    """
    Decorator that stacks the output if input is a list
    Currently only handles class member function
    :param func:
    """

    def func_wrapper(self, X, y=None):
        if isinstance(X, list):
            return np.vstack([func(self, each_X) for each_X in X])
        else:
            return func(self, X)

    return func_wrapper


class FeatureTransformer(ImageTransformer):
    @stack_list
    def transform(self, X, y=None):
        """
        :param X:
        :type X: MaskedData
        :param y:
        :return:
        """
        if np.ndim(X.mask) == 2:
            Xt = self._transform_mask(X)
        else:
            Xt = self._transform_labels(X)

        return np.array(Xt, np.float64)

    def _transform_mask(self, X):
        return X

    def _transform_labels(self, X):
        return X


class CspaceTransformer(FeatureTransformer):
    def __init__(self, cspace):
        self.cspace = cspace

    def cspace_transform(self, src):
        """
        :param src: bgr image
        :return: bgr image in other colorspace
        """
        # src is a nx3 vector, we need it to be nx1x3 to work with cvtColor
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

        # squeeze des back from nx1x3 to nx3
        return np.squeeze(des)

    def _transform_mask(self, X):
        bgr, mask = X.data, X.mask
        Xt = self.cspace_transform(bgr[mask])
        return Xt

    def _transform_labels(self, X):
        bgr, mask = X.data, X.mask
        neg, pos = split_label(mask)
        Xt_neg = self.cspace_transform(bgr[neg])
        Xt_pos = self.cspace_transform(bgr[pos])
        Xt = np.vstack((Xt_neg, Xt_pos))
        return Xt


class MaskLocator(FeatureTransformer):
    @staticmethod
    def where_row_col(m):
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

    def _transform_mask(self, X):
        Xt = self.where_row_col(X.mask)
        return Xt

    def _transform_labels(self, X):
        neg, pos = split_label(X.mask)
        xy_neg = self.where_row_col(neg)
        xy_pos = self.where_row_col(pos)
        Xt = np.vstack((xy_neg, xy_pos))
        return Xt


class PatchCreator(FeatureTransformer):
    def __init__(self, size):
        self.size = size

    @stack_list
    def transform(self, X, y=None):
        bgr, mask = X.data, X.mask

        if np.ndim(mask) == 2:
            pass
        else:
            pass
