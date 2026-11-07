"""Constant-velocity Kalman filter on (cx, cy, w, h) with noise scaled by body height and elapsed seconds.

Noise model (documented, uncalibrated):
  * measurement std of every coordinate = pos_std_h * h;
  * process variance per second: position (pos_std_h * h)^2, velocity (vel_std_h_per_s * h)^2.
Time is in seconds, so behaviour is independent of the video frame rate.
"""
from __future__ import annotations

import numpy as np

from ..config import KalmanConfig
from ..geometry import affine_scale

_DIM = 4


class BoxKalman:
    def __init__(self, cxcywh: np.ndarray, cfg: KalmanConfig):
        self.cfg = cfg
        self.x = np.zeros(2 * _DIM)
        self.x[:_DIM] = cxcywh
        h = max(float(cxcywh[3]), 1.0)
        p = (cfg.pos_std_h * h) ** 2
        v = (cfg.init_vel_std_h_per_s * h) ** 2
        self.P = np.diag([p] * _DIM + [v] * _DIM)

    @property
    def height(self) -> float:
        return max(float(self.x[3]), 1.0)

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = np.eye(2 * _DIM)
        F[:_DIM, _DIM:] = np.eye(_DIM) * dt
        h = self.height
        q_pos = (self.cfg.pos_std_h * h) ** 2 * dt
        q_vel = (self.cfg.vel_std_h_per_s * h) ** 2 * dt
        Q = np.diag([q_pos] * _DIM + [q_vel] * _DIM)
        self.x = F @ self.x
        self.x[2] = max(self.x[2], 1.0)
        self.x[3] = max(self.x[3], 1.0)
        self.P = F @ self.P @ F.T + Q

    def update(self, cxcywh: np.ndarray) -> None:
        H = np.zeros((_DIM, 2 * _DIM))
        H[:, :_DIM] = np.eye(_DIM)
        R = np.eye(_DIM) * (self.cfg.pos_std_h * max(float(cxcywh[3]), 1.0)) ** 2
        y = np.asarray(cxcywh, dtype=float) - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2 * _DIM) - K @ H) @ self.P

    def position_mahalanobis2(self, xy: np.ndarray) -> float:
        """Squared Mahalanobis distance of a measured centre from the predicted centre, using the predicted
        position covariance plus measurement noise. Grows tolerant with elapsed time and velocity uncertainty."""
        S = self.P[:2, :2] + np.eye(2) * (self.cfg.pos_std_h * self.height) ** 2
        y = np.asarray(xy, dtype=float) - self.x[:2]
        return float(y @ np.linalg.solve(S, y))

    def freeze_velocity(self) -> None:
        self.x[_DIM:] = 0.0

    def apply_camera(self, affine: np.ndarray) -> None:
        """Map the state through the inter-frame similarity transform (prev -> current image)."""
        A = np.asarray(affine, dtype=float)
        L, t = A[:, :2], A[:, 2]
        s = affine_scale(A)
        T = np.zeros((2 * _DIM, 2 * _DIM))
        T[0:2, 0:2] = L
        T[2, 2] = T[3, 3] = s
        T[4:6, 4:6] = L
        T[6, 6] = T[7, 7] = s
        x = T @ self.x
        x[0:2] += t
        self.x = x
        self.P = T @ self.P @ T.T
