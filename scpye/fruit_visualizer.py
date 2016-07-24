from __future__ import (print_function, absolute_import, division)

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


class FruitVisualizer(object):
    def __init__(self, image_dir, bag_ind):
        self.fig = plt.figure()
        plt.ion()
        self.ax_bgr = self.fig.add_subplot(121)
        self.ax_bw = self.fig.add_subplot(122)
        self.h_bgr = None
        self.h_bw = None

        self.pause_time = 0.001
        self.i = 0

        self.image_dir = os.path.join(image_dir, 'frame' + str(bag_ind))
        self.bag_ind = bag_ind

        self.bgr_name = 'bgr{0:04d}.png'
        self.bw_name = 'bw{0:04d}.png'

    def show(self, disp_bgr, disp_bw):
        disp_bw = np.array(disp_bw, dtype='uint8') * 255
        if self.h_bgr is None:
            self.h_bgr = self.ax_bgr.imshow(disp_bgr)
            self.h_bw = self.ax_bw.imshow(disp_bw)
        else:
            self.h_bgr.set_data(disp_bgr)
            self.h_bw.set_data(disp_bw)

        self.i += 1

        if self.image_dir is not None:
            bgr_name = os.path.join(self.image_dir,
                                    self.bgr_name.format(self.i))
            bw_name = os.path.join(self.image_dir,
                                   self.bw_name.format(self.i))
            cv2.imwrite(bgr_name, disp_bgr)
            cv2.imwrite(bw_name, disp_bw)

        plt.pause(self.pause_time)
