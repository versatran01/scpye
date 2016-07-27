from __future__ import (print_function, division, absolute_import)

import numpy as np
import numpy.linalg as la


class KalmanFilter(object):
    def __init__(self, dim_x=4):
        assert dim_x > 0, "dim_x must be > 0"
        self.dim_x = dim_x

        self.x = np.zeros(dim_x)  # state
        self.P = np.eye(dim_x)  # state cov
        self.Q = np.eye(dim_x)  # process cov

        # These are fixed
        self.I = np.eye(dim_x)
        self.F = np.array([[1.0, 1], [0, 0]])  # state transition matrix
        self.F_T = np.transpose(self.F)
        self.H = np.array([[1.0, 0], [0, 0]])  # measurement function
        self.H_T = np.transpose(self.H)

    def init(self, x0, P0):
        # TODO: initialize x, P, Q
        self.x = x0
        self.P = P0

    def predict(self):
        # x <- F * x + B * u
        self.x = np.dot(self.F, self.x)
        # P <- F * P * F^T + Q
        self.P = self.F.dot(self.P).dot(self.F) + self.Q

    def update(self, z_p, R):
        # z is [z_x, z_y, 0, 0]
        z = np.hstack((z_p, np.zeros(2)))

        # y = z - H * z
        Hx = np.dot(self.H, self.x)
        y = z - Hx
        # S = H * P * H^T + R
        S = self.H.dot(self.P).dot(self.H_T) + R
        # K = P * H^T * S^-1
        K = self.P.dot(self.H_T).dot(la.inv(S))
        # x <- x + K * y
        self.x += np.dot(K, y)
        # P <- (I - K * H) * P
        I_KH = self.I - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(K, R).dot(K.T)
