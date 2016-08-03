from __future__ import (print_function, division, absolute_import)

import logging

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scpye.track.bounding_box import extract_bbox
from scpye.improc.image_processing import (fill_bw, scale_array, u8_from_bw)
from scpye.improc.contour_analysis import contour_bounding_rect


class BlobAnalyzer(object):
    def __init__(self, max_cntr_area=100, max_aspect=1.3, min_extent=0.62,
                 min_solidity=0.90, gauss_filter_sigma=2,
                 max_filter_size=4, gray_edt_ratio=2, min_distance=5,
                 exclude_border=True):
        # Parameters for extracting single blob
        self.max_cntr_area = max_cntr_area
        self.max_aspect = max_aspect
        self.min_extent = min_extent
        self.min_solidity = min_solidity

        # Parameters for splitting multi blob
        self.gauss_filter_sigma = gauss_filter_sigma
        self.max_filter_size = max_filter_size
        self.gray_edt_ratio = gray_edt_ratio
        self.gray_max = 100
        self.min_distance = min_distance
        self.exclude_border = exclude_border

        self.logger = logging.getLogger(__name__)
        # Drawing
        self.single_bboxes = None
        self.multi_bboxes = None

    def analyze(self, bgr, region_props):
        """
        :param bgr: color image
        :param region_props: region props
        :return: fruits
        """
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # Get potential multi-props
        fruits, multi_rprops = self.extract_multi(region_props)

        self.single_bboxes = np.array(fruits)
        self.logger.debug("single bboxes: {}".format(len(self.single_bboxes)))

        # Split them to single bbox and add to fruits
        split_fruits = self.split_multi(gray, multi_rprops)
        fruits.extend(split_fruits)

        self.multi_bboxes = np.array(split_fruits)

        self.logger.debug("multi bboxes: {}".format(len(self.multi_bboxes)))
        self.logger.info("fruits: {}".format(len(fruits)))

        return np.array(fruits)

    def is_single_blob(self, blob_prop):
        """
        Check if this blob is a single blob
        :param blob_prop:
        :return:
        """
        area, aspect, extent, solidity = blob_prop

        return area < self.max_cntr_area \
               or (extent > self.min_extent and aspect < self.max_aspect) \
               or solidity > self.min_solidity

    def extract_multi(self, region_props):
        """
        Extract potential multi-blobs
        :param region_props:
        :return: list of potential multi-blobs
        """
        multi_rprops = []
        single_bboxes = []
        for rprop in region_props:
            blob, cntr = rprop.blob, rprop.cntr
            bbox, prop = blob['bbox'], blob['prop']

            if self.is_single_blob(prop):
                single_bboxes.append(bbox)
            else:
                multi_rprops.append(rprop)

        return single_bboxes, multi_rprops

    def split_multi(self, gray, region_props):
        """
        Split potential multi-blobs into separate bounding boxes
        :param gray:
        :param region_props:
        """
        split_bboxes = []

        for rprop in region_props:
            bbox = rprop.blob['bbox']
            gray_bbox = extract_gray(gray, rprop)

            # calculate distance measure for watershed
            dist_max, markers, n_peaks = self.prepare_watershed(gray_bbox)

            if n_peaks < 2:
                split_bboxes.append(bbox)
            else:
                labels = watershed(-dist_max, markers, mask=gray_bbox)
                global_bboxes = bboxes_from_labels(labels, n_peaks, bbox)
                split_bboxes.extend(global_bboxes)

        return split_bboxes

    def prepare_watershed(self, gray):
        """
        Prepare for watershed
        :param gray:
        :return:
        """
        gray_blur = ndi.gaussian_filter(gray, self.gauss_filter_sigma)
        # gray will be converted to binary when performing edt
        euclid_dist = ndi.distance_transform_edt(gray)
        dist = scale_array(gray_blur, val=self.gray_max)
        # combination of intensity and distance transform
        dist += scale_array(euclid_dist,
                            val=self.gray_max / self.gray_edt_ratio)

        dist_max = ndi.maximum_filter(dist, size=self.max_filter_size,
                                      mode='constant')
        local_max = peak_local_max(dist_max, min_distance=self.min_distance,
                                   indices=False,
                                   exclude_border=self.exclude_border)
        markers, n_peaks = ndi.label(local_max)

        return dist_max, markers, n_peaks


def bboxes_from_labels(labels, n_peaks, bbox):
    """
    Extract bboxes in labels and convert to global bboxes
    :param labels:
    :param n_peaks:
    :param bbox:
    :return:
    """
    global_bboxes = []

    for i in range(n_peaks):
        label_i1 = u8_from_bw(labels == i + 1)  # 0 is background
        local_bbox = contour_bounding_rect(label_i1)
        # shift bbox from local to global
        local_bbox[:2] += bbox[:2]
        global_bboxes.append(local_bbox)

    return global_bboxes


def extract_gray(gray, rprop):
    """
    Extract gray and binary image from rprop
    :param gray:
    :param rprop:
    :return: gray bbox
    """
    bbox = rprop.blob['bbox']
    cntr = rprop.cntr
    gray_bbox = extract_bbox(gray, bbox, copy=True)
    # redraw contour so that we don't accidentally grab pixels from other blobs
    # and because cntr is global, we need to draw it onto full image
    bw_cntr = fill_bw(gray, [cntr])
    bw_bbox = extract_bbox(bw_cntr, bbox)
    gray_bbox[bw_bbox == 0] = 0

    return gray_bbox
