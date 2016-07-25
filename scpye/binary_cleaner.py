from __future__ import (print_function, division, absolute_import)

from collections import namedtuple
from scpye.image_processing import (clean_bw, fill_bw, uint8_from_bw)
from scpye.contour_analysis import (analyze_contours_bw)

RegionProps = namedtuple('RegionProps', ('blobs', 'cntrs'))


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

    def clean(self, bw):
        """
        Clean up binary image and generate blobs and contours
        :param bw: binary image
        :return:
        """
        bw = uint8_from_bw(bw)
        bw_cleaned = clean_bw(bw, ksize=self.ksize, iters=self.iters)

        blobs, cntrs = analyze_contours_bw(bw_cleaned, min_area=self.min_area)
        bw_filled = fill_bw(bw_cleaned, cntrs)

        return bw_filled, RegionProps(blobs=blobs, cntrs=cntrs)
