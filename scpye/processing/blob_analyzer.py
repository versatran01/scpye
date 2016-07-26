from __future__ import (print_function, division, absolute_import)

import cv2
import numpy as np


class BlobAnalyzer(object):
    fruit_dtype = [('bbox', np.int, 4), ('num', np.int, 1)]

    def __init__(self, max_cntr_area=100, max_aspect=1.3, min_extent=0.62,
                 min_solidity=0.90, min_dist=4):
        self.max_cntr_area = max_cntr_area
        self.max_aspect = max_aspect
        self.min_extent = min_extent
        self.min_solidity = min_solidity
        self.min_dist = min_dist

    def is_single_blob(self, prop):
        """
        Check if this blob is a single blob
        :param prop:
        :return:
        """
        area, aspect, extent, solidity = prop

        return area < self.max_cntr_area \
               or (extent > self.min_extent and aspect < self.max_aspect) \
               or solidity > self.min_solidity

    def analyze(self, bgr, bw, props):
        """
        :param bgr: color image
        :param bw: mask
        :param props: region props
        :return: fruits
        """
        # Clean original bw
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        gray[bw == 0] = 0

        # areas = blobs['prop'][:, 0]
        # self.area_thresh = np.mean(areas)
        # fruits = [self.split_blob(blob, gray) for blob in blobs]
        # fruits = np.vstack(fruits)
        # return fruits, bw_clean
        return gray
