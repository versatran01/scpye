from __future__ import (print_function, division, absolute_import)

import numpy as np
import numpy.linalg as la


def diagonalize(M):
    if np.ndim(M) == 1:
        M = np.diag(M)
    return M


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
            self.P = diagonalize(P0)

        # process cov
        if Q is None:
            self.Q = np.eye(self.dim_x)
        else:
            self.Q = diagonalize(Q)

        # These are fixed
        self.I = np.eye(self.dim_x)
        I2 = np.eye(2)
        self.F = np.zeros((self.dim_x, self.dim_x))  # state transition matrix
        self.F[:2, :2] = I2
        self.F[:2, 2:] = I2
        self.F[2:, 2:] = I2

        self.H_p = np.zeros((2, self.dim_x))
        self.H_p[:, :2] = np.eye(2)

        self.H_v = np.zeros((2, self.dim_x))
        self.H_v[:, 2:] = np.eye(2)

    def predict(self):
        """
        Prediction step of a Kalman filter
        :return:
        """
        # x = F * x + B * u
        self.x = self.F.dot(self.x)
        # P = F * P * F^T + Q
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update_pos(self, z_p, R_p):
        """
        Kalman update with position measurement
        :param z_p:
        :param R_p:
        :return:
        """
        R_p = diagonalize(R_p)
        self.update(z_p, R_p, self.H_p)

    def update_vel(self, z_v, R_v):
        """
        Kalman update with velocity measurement
        :param z_v:
        :param R_v:
        :return:
        """
        R_v = diagonalize(R_v)
        self.update(z_v, R_v, self.H_v)

    def update(self, z, R, H):
        """
        General Kalman update step
        :param z:
        :param R:
        :param H:
        :return:
        """
        # y = z - H * x
        y = z - H.dot(self.x)
        # S = H * P * H.T + R
        S = H.dot(self.P).dot(H.T) + R
        # K  = P * H.T * S^-1
        K = self.P.dot(H.T).dot(la.inv(S))
        # x = x + K * y
        self.x += K.dot(y)
        I_KH = self.I - K.dot(H)
        # P = (I - K * H) * P * (I - K * H).T + K * R * K.T
        self.P = I_KH.dot(self.P).dot(I_KH.T) + K.dot(R).dot(K.T)
