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
    def __init__(self, bbox, init_flow, proc_cov):
        self.bbox = bbox
        self.age = 1
        self.hist = []
        self.prev_pos = None

        # every track will initialize its own kalman filter
        init_pos = bbox_center(self.bbox)
        init_state = np.hstack((init_pos, init_flow))

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

    def correct_flow(self, pos, flow_cov):
        """
        Correct location of the track from pos input
        :param pos:
        :param flow_cov:
        """
        self.kf.update_pos(pos, flow_cov)
        self.bbox = shift_bbox(self.bbox, self.pos)

    def correct_assign(self, bbox, assign_cov):
        """
        Correct location of the track hungarian assignment
        :param bbox:
        :param assign_cov:
        :return:
        """
        pos = bbox_center(bbox)
        self.kf.update_pos(pos, assign_cov)
        self.bbox = shift_bbox(self.bbox, self.pos)
        # Don't forget to update bbox dimension
        self.bbox[2:] = bbox[2:]
