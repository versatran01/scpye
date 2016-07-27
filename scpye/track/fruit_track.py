from __future__ import (print_function, division, absolute_import)

import numpy as np
from scpye.track.kalman_filter import KalmanFilter
from scpye.track.bounding_box import (bbox_center, bbox_shift)


def cov2ellipse(P, ns=3):
    U, D, _ = np.linalg.svd(P)
    w, h = np.sqrt(D) * ns
    a = np.arctan2(U[1, 0], U[0, 0])
    return w, h, np.rad2deg(a)


class FruitTrack(object):
    def __init__(self, fruit, flow):
        self.bbox = fruit
        self.age = 1
        self.hist = []

        # every track will initialize its own kalman filter
        pos = bbox_center(self.bbox)
        vel = np.array(flow)
        x0 = np.hstack((pos, vel))
        self.kf = KalmanFilter(x0=x0)

    @property
    def pos_cov(self):
        return self.kf.P[:2, :2]

    @property
    def pos(self):
        return self.kf.x[:2]

    @property
    def cov_ellipse(self):
        wha = cov2ellipse(self.pos_cov)
        return np.hstack((self.pos, wha))

    def predict(self):
        """
        Predict new location of the tracking
        """
        self.kf.predict()
        self.bbox = bbox_shift(self.bbox, self.kf.x[:2])

    def correct(self, fruit):
        """
        Correct location of the tracking
        :param fruit:
        """
        # bbox_new = fruit[:4]
        # self.num = fruit[-1]
        # # Update flow first
        # self.flow += (bbox_new[:2] - self.bbox[:2]) / 2
        # # Then update bbox
        # self.bbox = bbox_new
        # # Increment age
        # self.age += 1
        pass
