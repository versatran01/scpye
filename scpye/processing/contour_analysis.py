from __future__ import (print_function, absolute_import, division)

import cv2
import numpy as np

"""
http://docs.opencv.org/trunk/d3/d05/tutorial_py_table_of_contents_contours.html#gsc.tab=0
"""

blob_dtype = [('bbox', np.int, 4), ('prop', np.float, 4)]


def analyze_contours_bw(bw, min_area=4):
    """
    Same as matlab regionprops but implemented in opencv
    Prefer using this than skimage's regionprops because this return a numpy
    recarray that is very compact
    :param bw: binary image
    :param min_area:
    :return: a structured array of blobs
    """
    contours = find_contours(bw)
    return analyze_contours(contours, min_area=min_area)


def analyze_contours(contours, min_area):
    """
    :param contours:
    :param min_area:
    :return:
    """
    blobs, cntrs = [], []
    for cntr in contours:
        cntr_area = contour_area(cntr)
        if cntr_area >= min_area:
            bbox = contour_bounding_rect(cntr)
            aspect = bounding_rect_aspect(bbox)
            extent = contour_extent(cntr, cntr_area=cntr_area, bbox=bbox)
            solidity = contour_solidity(cntr, cntr_area)

            # Assemble to recarray
            blob = np.array((bbox, (cntr_area, aspect, extent, solidity)),
                            dtype=blob_dtype)
            blobs.append(blob)
            cntrs.append(cntr)

    blobs = np.array(blobs)
    return blobs, cntrs


def find_contours(bw):
    """
    :param bw: binary image
    :return: a list of contours
    """
    _, cntrs, _ = cv2.findContours(bw.copy(),
                                   mode=cv2.RETR_EXTERNAL,
                                   method=cv2.CHAIN_APPROX_NONE)
    return cntrs


def contour_area(cntr):
    """
    :param cntr:
    :return: contour area
    """
    return cv2.contourArea(cntr)


def contour_bounding_rect(cntr):
    """
    :param cntr:
    :return: bounding box [x, y, w, h]
    """
    return np.array(cv2.boundingRect(cntr))


def bounding_rect_aspect(bbox):
    """
    :param bbox: bounding box
    :return: aspect ratio
    """
    _, _, w, h = bbox
    aspect = float(w) / h if float(w) > h else h / w
    return aspect


def contour_extent(cntr, cntr_area=None, bbox=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :param bbox: bounding box
    :return: extent
    """
    if cntr_area is None:
        cntr_area = contour_area(cntr)

    if bbox is None:
        bbox = contour_bounding_rect(cntr)

    _, _, w, h = bbox
    rect_area = w * h
    extent = float(cntr_area) / rect_area
    return extent


def contour_solidity(cntr, cntr_area=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :return: solidity
    """
    if cntr_area is None:
        cntr_area = cv2.contourArea(cntr)

    hull = cv2.convexHull(cntr)
    hull_area = cv2.contourArea(hull)
    solidity = float(cntr_area) / hull_area
    return solidity


def contour_equi_diameter(cntr, cntr_area=None):
    """
    :param cntr:
    :param cntr_area: contour area
    :return: equivalent diameter
    """
    if cntr_area is None:
        cntr_area = contour_area(cntr)

    equi_diameter = np.sqrt(4 * cntr_area / np.pi)
    return equi_diameter


def moment_centroid(mmt):
    """
    Centroid of moment
    :param mmt: moment
    :return: (x, y)
    """
    return np.array((mmt['m10'] / mmt['m00'], mmt['m01'] / mmt['m00']))
