from __future__ import (absolute_import, division, print_function)

import cv2
import numpy as np
from functools import partial
from collections import namedtuple

from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import maximum_filter

from scpye.bounding_box import extract_bbox

__all__ = ['ImageRotator', 'ImageCropper', 'ImageResizer', 'DarkRemover',
           'CspaceTransformer', 'MaskLocator', 'StandardScaler']

MaskedData = namedtuple('MaskedData', ['X', 'm'])


class ImageTransformer(BaseEstimator, TransformerMixin):
    @staticmethod
    def forward_list_input(func):
        """
        Decorator that output a list if input is list
        Currently only handles class member function
        :param func:
        """

        def func_wrapper(self, X, y=None):
            if y is None:
                if isinstance(X, list):
                    return [func(self, _X) for _X in X]
                else:
                    return func(self, X)
            else:
                # Hopefully the input of y won't be a list of None
                if isinstance(X, list):
                    assert isinstance(y, list) and len(X) == len(y)
                    Xts = []
                    yts = []
                    for _X, _y in zip(X, y):
                        Xt, yt = func(self, _X, _y)
                        Xts.append(Xt)
                        yts.append(yt)
                    return Xts, yts
                else:
                    return func(self, X, y)

        return func_wrapper

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            # fit method of arity 1 (unsupervised transformation)
            return self.fit(X, **fit_params).transform(X)
        else:
            # fit method of arity 2 (supervised transformation)
            return self.fit(X, y, **fit_params).transform(X, y)

    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None):
        assert False


class ImageRotator(ImageTransformer):
    def __init__(self, ccw=-1):
        """
        :param ccw: number of counter-clockwise rotations
        :type ccw: int
        """
        self.ccw = ccw

    @ImageTransformer.forward_list_input
    def transform(self, X, y=None):
        """
        :param X: image
        :param y: label
        :return: rotated image and label
        """
        func = partial(np.rot90, k=self.ccw)

        Xt = func(X)
        if y is None:
            return Xt
        else:
            yt = func(y)
            return Xt, yt


class ImageCropper(ImageTransformer):
    def __init__(self, bbox=None):
        self.bbox = bbox

    @ImageTransformer.forward_list_input
    def transform(self, X, y=None):
        """
        :param X: image
        :param y: label
        :return: region of image and label
        """
        func = partial(extract_bbox, bbox=self.bbox)

        Xt = func(X)
        if y is None:
            return Xt
        else:
            yt = func(y)
            return Xt, yt


class ImageResizer(ImageTransformer):
    def __init__(self, k=0.5):
        self.k = k

    @ImageTransformer.forward_list_input
    def transform(self, X, y=None):
        """
        :param X: image
        :param y: label
        :return: resized image
        """
        func = partial(cv2.resize, dsize=None, fx=self.k, fy=self.k,
                       interpolation=cv2.INTER_NEAREST)

        Xt = func(cv2.GaussianBlur(X, (5, 5), 1))
        if y is None:
            return Xt
        else:
            yt = func(y)
            return Xt, yt


def split_label01(label):
    """
    :param label:
    :return: split label
    :rtype: numpy.ndarray
    """
    assert np.ndim(label) == 3 and np.size(label, axis=-1) == 2
    return label[:, :, 0] > 0, label[:, :, 1] > 0


class DarkRemover(ImageTransformer):
    def __init__(self, v_min=25):
        """
        :param v_min: minimum value in v channel
        :type v_min: int
        """
        assert 0 < v_min < 255
        self.bgr = None
        self.hsv = None
        self.mask = None
        self.label = None
        self.v_min = v_min

    @ImageTransformer.forward_list_input
    def transform(self, X, y=None):
        """
        :param X: image
        :param y: label
        :return: a tuple of bgr image and mask
        """
        self.bgr = X
        self.hsv = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
        self.mask = self.hsv[:, :, -1] > self.v_min
        self.hsv[self.mask == 0] = 0

        if y is None:
            return MaskedData(X=X, m=self.mask)

        neg, pos = split_label01(y)

        neg_mask = self.mask & neg
        pos_mask = self.mask & pos

        y_pos = np.ones(np.count_nonzero(pos_mask))
        y_neg = np.zeros(np.count_nonzero(neg_mask))

        self.label = np.dstack((neg_mask, pos_mask))
        yt = np.hstack((y_neg, y_pos))
        return MaskedData(X=X, m=self.label), yt


class FeatureTransformer(ImageTransformer):
    @staticmethod
    def stack_list_input(func):
        """
        Decorator that stack the output if input is a list
        Currently only handles class member function
        :param func:
        """

        def func_wrapper(self, X, y=None):
            if isinstance(X, list):
                return np.vstack([func(self, _X) for _X in X])
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

        if self.cspace == 'hsv':
            des = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        elif self.cspace == 'lab':
            des = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        elif self.cspace == 'bgr':
            des = src
        else:
            raise ValueError("{0} not supported".format(self.cspace))

        return np.squeeze(des)

    @FeatureTransformer.stack_list_input
    def transform(self, X, y=None):
        """
        :param X: tuple of bgr image and mask
        :type X: MaskedData
        :param y:
        :return: masked image with transformed colorspace
        """
        bgr = X.X
        mask = X.m
        if np.ndim(mask) == 2:
            Xt = self.cspace_transform(bgr[mask])
            image = np.zeros_like(bgr)
            image[mask] = Xt
            # if y is None:
            #     self.image = image
        else:
            neg, pos = split_label01(mask)
            Xt_neg = self.cspace_transform(bgr[neg])
            Xt_pos = self.cspace_transform(bgr[pos])
            Xt = np.vstack((Xt_neg, Xt_pos))
        # Need to change to float to suppress later warnings
        return np.array(Xt, np.float64)


def xy_from_array(m):
    """
    :param m: array
    :type m: numpy.ndarray
    :return: n x 2 matrix of [x, y]
    :rtype: numpy.ndarray
    """
    assert np.ndim(m) == 2
    r, c = np.where(m)
    return np.transpose(np.vstack((r, c)))


class MaskLocator(FeatureTransformer):
    @FeatureTransformer.stack_list_input
    def transform(self, X, y=None):
        mask = X.m
        if np.ndim(mask) == 2:
            Xt = xy_from_array(mask)
        else:
            neg, pos = split_label01(mask)
            xy_neg = xy_from_array(neg)
            xy_pos = xy_from_array(pos)
            Xt = np.vstack((xy_neg, xy_pos))
        # Change to float to suppress warning
        return np.array(Xt, np.float64)
