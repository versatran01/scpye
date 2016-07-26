from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import watershed

from scpye.tracking.bounding_box import extract_bbox
from scpye.processing.image_processing import (fill_bw, scale_array, u8_from_bw)
from scpye.processing.contour_analysis import contour_bounding_rect


class BlobAnalyzer(object):
    # fruit_dtype = [('bbox', np.int, 4), ('num', np.int, 1)]

    def __init__(self, max_cntr_area=100, max_aspect=1.3, min_extent=0.62,
                 min_solidity=0.90, gauss_filter_sigma=2,
                 max_filter_size=4, gray_edt_ratio=2, min_dist=5,
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
        self.min_dist = min_dist
        self.exclude_border = exclude_border

        self.fruits = None

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

    def analyze(self, bgr, bw, region_props):
        """
        :param bgr: color image
        :param bw: mask
        :param region_props: region props
        :return: fruits
        """
        self.fruits = []
        # Clean original bw
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # gray[bw == 0] = 0

        # Get potential multi-props
        multi_rprops = self.extract_multi(region_props)
        # Split them to single bbox and add to fruits
        self.split_multi(gray, bw, multi_rprops)

        return self.fruits

    def extract_multi(self, region_props):
        """
        Extract single bbox from props
        :param region_props:
        :return:
        """
        multi_rprops = []
        for rprop in region_props:
            blob, cntr = rprop.blob, rprop.cntr
            if self.is_single_blob(blob['prop']):
                self.fruits.append(blob['bbox'])
            else:
                multi_rprops.append(rprop)

        return multi_rprops

    def split_multi(self, gray, bw, region_props):
        """
        Split potential multi-blobs into separate bounding boxes
        :param gray:
        :param region_props:
        :return:
        """
        for rprop in region_props:
            blob, cntr = rprop.blob, rprop.cntr
            bbox = blob['bbox']

            gray_bbox = extract_bbox(gray, bbox, copy=True)
            # redraw contour so that we don't accidentally grab pixels from
            #  other blobs
            bw_cntr = fill_bw(bw, [cntr])
            bw_bbox = extract_bbox(bw_cntr, bbox)
            gray_bbox[bw_bbox == 0] = 0

            # calculate distance measure for watershed
            gray_blur = ndi.gaussian_filter(gray_bbox, self.gauss_filter_sigma)
            euclid_dist = ndi.distance_transform_edt(gray_bbox)
            dist = scale_array(gray_blur, val=self.gray_max) + \
                   scale_array(euclid_dist,
                               val=self.gray_max / self.gray_edt_ratio)
            dist_max = ndi.maximum_filter(dist, size=self.max_filter_size,
                                          mode='constant')
            local_max = peak_local_max(dist_max, min_distance=self.min_dist,
                                       indices=False,
                                       exclude_border=self.exclude_border)
            markers, n_peak = ndi.label(local_max)

            if n_peak < 2:
                self.fruits.append(bbox)
            else:
                labels = watershed(-dist_max, markers, mask=bw_bbox)
                for i in range(n_peak):
                    label = u8_from_bw(labels == i + 1)  # 0 is background
                    local_bbox = contour_bounding_rect(label)
                    local_bbox[:2] += bbox[:2]
                    self.fruits.append(local_bbox)
