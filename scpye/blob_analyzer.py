from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scpye.bounding_box import bbox_area, extract_bbox
from scpye.image_processing import (clean_bw, fill_bw, uint8_from_bw)
from scpye.contour_analysis import (analyze_contours_bw, find_contours)


class BlobAnalyzer(object):
    fruit_dtype = [('bbox', np.int, 4), ('num', np.int, 1)]

    def __init__(self, min_area=5, ksize=3, iters=2, do_split=False):
        """
        :param min_area: minimum area to be consider a blob
        :param do_split: whether to split big blob to smaller ones or not
        :return:
        """
        self.min_cntr_area = min_area
        self.ksize = ksize
        self.iters = iters
        self.split = do_split
        self.area_thresh = 0

    def analyze(self, bgr, bw):
        """
        :param bgr: color image
        :param bw: mask
        :return: (fruits, bw)
        """
        # Clean original bw
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bw_clean = self.clean(bw)
        gray[bw_clean == 0] = 0

        blobs, cntrs, bw_filled = self.extract(bw_clean)
        # areas = blobs['prop'][:, 0]
        # self.area_thresh = np.mean(areas)
        # TODO: min_area = (min_distance + 2) ** 2
        # fruits = [self.split_blob(blob, gray) for blob in blobs]
        # fruits = np.vstack(fruits)
        # return fruits, bw_clean
        return blobs, cntrs, gray, bw_filled

    def clean(self, bw):
        """
        Clean noise from binary image
        :param bw: binary image
        :return: cleaned binary image
        """
        bw = uint8_from_bw(bw)
        bw_clean = clean_bw(bw, ksize=self.ksize, iters=self.iters)
        return bw_clean

    def extract(self, bw):
        """
        Extract blobs and contours from binary image
        :param bw: binary image
        :return: blobs, contours, filled binary image
        """
        blobs, cntrs = analyze_contours_bw(bw, self.min_cntr_area)
        bw_filled = fill_bw(bw, cntrs)
        return blobs, cntrs, bw_filled

        # def split(self, blob, gray, min_aspect=1.4, max_extent=0.5):
        #     """
        #     :param blob:
        #     :param gray:
        #     :param min_aspect:
        #     :param max_extent:
        #     :return:
        #     """
        #     bbox = blob['bbox']
        #     v_bbox = extract_bbox(gray, bbox)
        #
        #     min_dist = min(np.sqrt(bbox_area(bbox)) / 5, 10)
        #     area, aspect, extent = blob['prop']
        #     if area > self.area_thresh and \
        #             (aspect > min_aspect or extent < max_extent):
        #         points = find_local_maximas(v_bbox, min_distance=min_dist)
        #
        #         if points is None:
        #             return np.hstack((bbox, 1))
        #
        #         num = len(points)
        #         if self.split:
        #             return self.split_local_max_points(points, bbox, area)
        #         else:
        #             return np.hstack((bbox, num))
        #     else:
        #         return np.hstack((bbox, 1))

        # @staticmethod
        # def split_local_max_points(points, bbox, area):
        #     """
        #     Split local max points into multiple fruits [bbox, num]
        #     :param points:
        #     :param bbox:
        #     :param area:
        #     :return:
        #     """
        #     xb, yb, _, _ = bbox
        #     num = len(points)
        #     a = np.sqrt(area / num)
        #     fruits = []
        #     for pt in points:
        #         x, y = pt
        #         fruit = np.array([x + xb - a / 2, y + yb - a / 2, a, a, 1])
        #         fruits.append(fruit)
        #     return np.vstack(fruits)


def find_local_maximas(image, min_distance=5):
    """
    Find points of local maximas from gray scale image
    :param image:
    :param min_distance:
    :return:
    """
    image_max = ndi.maximum_filter(image, size=3, mode='constant')
    local_max = peak_local_max(image_max, min_distance=min_distance,
                               indices=False, exclude_border=False)
    local_max = uint8_from_bw(local_max)
    points = local_max_points(local_max)
    return points


def local_max_points(bw):
    """
    :param bw:
    :return:
    """
    cntrs = find_contours(bw)

    if len(cntrs) == 0:
        return None

    points = []
    for cnt in cntrs:
        mmt = cv2.moments(cnt)
        cntr_area = cv2.contourArea(cnt)
        if cntr_area > 0:
            points.append(moment_centroid(mmt))

    if len(points) == 0:
        return None
    else:
        points = np.array(points)
        return points

# def label_blob_watershed(bbox, bw, v, k=5.5, return_num=False):
#     """
#     :param bbox: bounding box
#     :param bw: binary image
#     :param v: gray scale image
#     :param k: magic number
#     :param return_num: return number of labels
#     :return:
#     """
#     min_dist = np.sqrt(bbox_area(bbox)) / k
#
#     v_bbox = extract_bbox(v, bbox, copy=True)
#     bw_bbox = extract_bbox(bw, bbox, copy=True)
#     v_bbox[bw_bbox == 0] = 0
#     dist = ndi.distance_transform_edt(bw_bbox) * 2
#     dist += v_bbox
#
#     local_max = peak_local_max(dist, indices=False, min_distance=min_dist,
#                                labels=bw_bbox)
#     if np.count_nonzero(local_max) > 1:
#         markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
#         label = watershed(-dist, markers, mask=bw_bbox)
#     else:
#         label = bw_bbox
#     if return_num:
#         return label, np.max(label)
#     else:
#         return label
