from __future__ import print_function, division, absolute_import

import numpy as np

__all__ = ['OverlapRatio', 'extract_bbox', 'bbox_center',
           'bbox_distsq']


class OverlapRatio:
    """
    Min: intersection area / min(bbox1 area, bbox2 area)
    Union: intersection area / union(bbox1 area, bbox2 area)
    """
    Union, Min = range(2)

    def __init__(self):
        pass


def extract_bbox(image, bbox, copy=False):
    """
    Extract region of image defined by bbox
    :param image: image
    :param bbox: bbox
    :return: region of image
    :rtype: numpy.ndarray
    """
    x, y, w, h = bbox
    if copy:
        return np.array(image[y:y + h, x:x + w, ...], copy=True)
    else:
        return image[y:y + h, x:x + w, ...]


def bbox_center(bbox):
    """
    Center of a bounding box
    :param bbox: bbox
    :type bbox: numpy.ndarray
    :return: center of bbox
    :rtype: numpy.ndarray
    """
    x, y, w, h = bbox
    return np.array([x + w / 2, y + h / 2])


def bbox_distsq(bbox1, bbox2):
    """
    Squared distance between bbox1 and bbox2
    :param bbox1: bbox
    :type bbox1: numpy.ndarray
    :param bbox2: bbox
    :type bbox2: numpy.ndarray
    :return: squared distance between center of bbox
    :rtype: float
    """
    cx1, cy1 = bbox_center(bbox1)
    cx2, cy2 = bbox_center(bbox2)
    return (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2


def bbox_intersect(bbox1, bbox2):
    """
    Whether two bboxes intersect or not
    :param bbox1: bbox
    :type bbox1: numpy.ndarray
    :param bbox2: bbox
    :type bbox2: numpy.ndarray
    :return: True if two bboxes intersect
    :rtype: bool
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    dx = abs((x1 + w1 / 2) - (x2 + w2 / 2))
    dy = abs((y1 + h1 / 2) - (y2 + h2 / 2))
    return (dx * 2 < (w1 + w2)) and (dy * 2 < (h1 + h2))


def bbox_intersect_area(bbox1, bbox2):
    """
    Intersection area of two bboxes
    :param bbox1: bbox
    :type bbox1: numpy.ndarray
    :param bbox2: bbox
    :return: intersection area
    :rtype: float
    """
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    w = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    h = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    return w * h


def bbox_overlap_ratio(bbox1, bbox2, ratio_type=OverlapRatio.Union):
    """
    Overlap ratio of two bboxes
    :param bbox1: bbox
    :param bbox2: bbox
    :param ratio_type:
    :return: overlap ratio
    :rtype: float
    """
    if not bbox_intersect(bbox1, bbox2):
        return 0.0

    # intersection area is a * b, where a is width and b is height
    area_intersection = bbox_intersect_area(bbox1, bbox2)
    area_b1 = bbox_area(bbox1)
    area_b2 = bbox_area(bbox2)
    if ratio_type == OverlapRatio.Union:
        area_union = area_b1 + area_b2 - area_intersection
        return area_intersection / area_union
    elif ratio_type == OverlapRatio.Min:
        area_min = min(area_b1, area_b2)
        return area_intersection / area_min


def bbox_area(bbox):
    """
    Area of bbox
    :param bbox: bbox
    :return: bbox area
    """
    return bbox[-1] * bbox[-2]


def bbox_distsq_area_ratio(bbox1, bbox2):
    dist_sq = bbox_distsq(bbox1, bbox2)
    area = bbox_area(bbox1) + bbox_area(bbox2)
    return dist_sq / area


def bboxes_overlap_ratio(bboxes1, bboxes2, ratio_type=OverlapRatio.Union):
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)
    n1 = len(bboxes1)
    n2 = len(bboxes2)
    assert n1 > 0 and n2 > 0

    R = np.zeros((n1, n2))
    for i1, b1 in enumerate(bboxes1):
        for i2, b2 in enumerate(bboxes2):
            R[i1, i2] = bbox_overlap_ratio(b1, b2, ratio_type)
    return R


def bboxes_assignment_cost(bboxes1, bboxes2):
    bboxes1 = np.atleast_2d(bboxes1)
    bboxes2 = np.atleast_2d(bboxes2)
    n1 = len(bboxes1)
    n2 = len(bboxes2)
    assert n1 > 0 and n2 > 0

    C = np.zeros((n1, n2))
    for i1, b1 in enumerate(bboxes1):
        for i2, b2 in enumerate(bboxes2):
            overlap_ratio = bbox_overlap_ratio(b1, b2)
            overlap_cost = 1 - overlap_ratio
            distance2_area_cost = bbox_distsq_area_ratio(b1, b2)
            C[i1, i2] = overlap_cost + distance2_area_cost
    return C
