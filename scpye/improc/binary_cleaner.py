from __future__ import (print_function, division, absolute_import)

import logging

from scpye.improc.image_processing import (clean_bw, fill_bw, u8_from_bw)
from scpye.improc.contour_analysis import (analyze_contours_bw)


class BinaryCleaner(object):
    def __init__(self, ksize=3, iters=2, min_area=5):
        """
        :param ksize:
        :param iters:
        :param min_area:
        """
        self.ksize = ksize
        self.iters = iters
        self.min_area = min_area

        self.logger = logging.getLogger(__name__)

    def clean(self, bw):
        """
        Clean up binary image and generate blobs and contours
        :param bw: binary image, with 255 max
        :return:
        """
        bw = u8_from_bw(bw, val=255)
        bw_clean = clean_bw(bw, ksize=self.ksize, iters=self.iters)

        region_props = analyze_contours_bw(bw_clean, min_area=self.min_area)

        cntrs = [rp.cntr for rp in region_props]
        bw_fill = fill_bw(bw_clean, cntrs)

        return bw_fill, region_props
