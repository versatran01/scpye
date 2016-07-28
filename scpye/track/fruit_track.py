from __future__ import (print_function, division, absolute_import)

import numpy as np
from scpye.track.kalman_filter import KalmanFilter
from scpye.track.bounding_box import (bbox_center, shift_bbox)


def cov2ellipse(P, ns=3):
    """
    Covariance to ellipse
    :param P: covariance
    :param ns: scale
    :return:
    """
    assert np.shape(P) == (2, 2)

    U, D, _ = np.linalg.svd(P)
    w, h = np.sqrt(D) * ns
    a = np.arctan2(U[1, 0], U[0, 0])

    return w, h, np.rad2deg(a)


class FruitTrack(object):
    def __init__(self, fruit, init_flow, proc_cov):
        self.bbox = fruit
        self.age = 1
        self.hist = []
        self.prev_pos = None

        # every track will initialize its own kalman filter
        init_pos = bbox_center(self.bbox)
        init_state = np.hstack((init_pos, init_flow))

        if np.ndim(proc_cov) == 1:
            proc_cov = np.diag(proc_cov)
        self.kf = KalmanFilter(x0=init_state, Q=proc_cov)

    @property
    def pos(self):
        return self.kf.x[:2].copy()

    @property
    def vel(self):
        return self.kf.x[2:].copy()

    @property
    def pos_cov(self):
        return self.kf.P[:2, :2].copy()

    @property
    def cov_ellipse(self):
        wha = cov2ellipse(self.pos_cov)
        return np.hstack((self.pos, wha))

    def predict(self):
        """
        Predict new location of the tracking
        """
        self.prev_pos = self.pos
        self.kf.predict()
        self.bbox = shift_bbox(self.bbox, self.pos)

    def correct_pos(self, pos, pos_cov):
        """
        Correct location of the track from pos input
        :param pos:
        :param pos_cov:
        """
        vel = pos - self.prev_pos
        z = np.hstack((pos, vel))
        R = np.hstack((pos_cov, np.ones(2)))
        self.kf.update(z, R)
        self.bbox = shift_bbox(self.bbox, self.pos)

    def correct_bbox(self, bbox, bbox_cov):
        """
        Correct location of the track from bbox input
        :param bbox:
        :param bbox_cov:
        :return:
        """
        pos = bbox_center(bbox)
        self.correct_pos(pos, bbox_cov)
