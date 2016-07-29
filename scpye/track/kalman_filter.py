from __future__ import (print_function, division, absolute_import)

import numpy as np
import numpy.linalg as la


class KalmanFilter(object):
    def __init__(self, x0=None, P0=None, Q=None):
        self.dim_x = 4

        # state
        if x0 is None:
            self.x = np.zeros(self.dim_x)
        else:
            self.x = x0

        # state cov
        if P0 is None:
            self.P = np.eye(self.dim_x)
        else:
            if np.ndim(P0) == 1:
                P0 = np.diag(P0)
            self.P = P0

        # process cov
        if Q is None:
            self.Q = np.eye(self.dim_x)
        else:
            if np.ndim(Q) == 1:
                Q = np.diag(Q)
            self.Q = Q

        # These are fixed
        self.I = np.eye(self.dim_x)
        I2 = np.eye(2)
        self.F = np.zeros((self.dim_x, self.dim_x))  # state transition matrix
        self.F[:2, :2] = I2
        self.F[:2, 2:] = I2
        self.F[2:, 2:] = I2
        self.F_T = np.transpose(self.F)
        self.H = np.eye(self.dim_x)  # measurement function
        self.H_T = np.transpose(self.H)

        self.Hp = np.zeros((2, self.dim_x))
        self.Hp[:2, :2] = I2
        self.Hp_T = np.transpose(self.Hp)

    def predict(self):
        """
        Prediction step of a Kalman filter
        :return:
        """
        # x = F * x + B * u
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = self.F.dot(self.P).dot(self.F) + self.Q

    def update(self, z, R=None):
        """
        Update step of a Kalman filter
        :param z:
        :param R:
        :return:
        """
        # Fix R dimension
        if np.ndim(R) == 1:
            R = np.diag(R)

        Hx = np.dot(self.H, self.x)
        # y = z - H * z
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

    def update_pos(self, z_p, R_p=None):
        if np.ndim(R_p) == 1:
            R_p = np.diag(R_p)

        Hx_p = np.dot(self.Hp, self.x)
        y = z_p - Hx_p
        S = self.H.dot(self.P).dot(self.Hp_T) + R_p
        K = self.P.dot(self.Hp_T).dot(la.inv(S))
        self.x += np.dot(K, y)

        I_KH = self.I - np.dot(K, self.Hp)
        self.P = np.dot(I_KH, self.P).dot(I_KH.T) + np.dot(K, R_p).dot(K.T)
