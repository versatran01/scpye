from __future__ import (print_function, absolute_import, division)

import matplotlib.pyplot as plt
from scpye.improc.image_processing import u8_from_bw


class FruitVisualizer(object):
    def __init__(self, pause_time=0.01):
        self.fig = plt.figure()
        plt.ion()
        self.ax_bgr = self.fig.add_subplot(121)
        self.ax_bw = self.fig.add_subplot(122)
        self.h_bgr = None
        self.h_bw = None

        self.pause_time = pause_time

    def show(self, disp_bgr, disp_bw):

        if self.h_bgr is None:
            self.h_bgr = self.ax_bgr.imshow(disp_bgr)
            self.h_bw = self.ax_bw.imshow(disp_bw)
        else:
            self.h_bgr.set_data(disp_bgr)
            self.h_bw.set_data(disp_bw)

        plt.pause(self.pause_time)
