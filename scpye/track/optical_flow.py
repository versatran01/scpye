from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np


def points_inside_image(points, image, b=0):
    """
    Check if point is inside image with a certain margin
    :param points:
    :param image:
    :param b: distance from border
    :return:
    """
    h, w = np.shape(image)
    px = points[:, :, 0]
    py = points[:, :, 1]
    return (px >= b) & (px < w - b) & (py >= b) & (py < h - b)


def _make_cv_points(points):
    """
    Prepare points for opencv, convert dtype to np.float32 and add extra
    dimension
    :param points:
    :return:
    """
    points = np.array(points)
    if points.dtype is not np.float32:
        points = np.array(points, dtype=np.float32)
    if np.ndim(points) == 2:
        points = np.expand_dims(points, 1)
    return points


def calc_optical_flow(gray1, gray2, points1, points2, win_size, max_level):
    """
    Thin wrapper around opencv's calcOpticalFlowPyrLK
    :param gray1: previous image
    :param gray2: current image
    :param points1: points in previous image
    :param points2: points in current image
    :param win_size: window size
    :param max_level: max pyramid level
    :return: (points1, points2, status)
    """
    points1 = _make_cv_points(points1)
    points2 = _make_cv_points(points2)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    klt_params = dict(winSize=(win_size, win_size),
                      maxLevel=max_level,
                      flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
                      criteria=criteria)

    points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1,
                                                  points2, **klt_params)

    # is_inside = points_inside_image(points2, gray2)

    # status = (status == 1) & is_inside

    return np.squeeze(points2), np.squeeze(status)


def calc_average_flow(points1, points2, status):
    """
    Calculate average optical flow
    :param points1:
    :param points2:
    :param status:
    :return:
    """
    points1 = np.squeeze(np.array(points1))
    points2 = np.squeeze(np.array(points2))
    status = np.squeeze(np.array(status) > 0)
    flows = points2 - points1
    valid_flows = flows[status]

    return np.mean(valid_flows, axis=0)
