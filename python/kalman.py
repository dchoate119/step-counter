"""Kalman filter for live step count and cadence.

State is [count (steps), cadence (steps/s), CNN bias (steps/s)]. Only cadence is
measured: the CNN reads cadence + bias every window, the stride detector reads
cadence (2 / stride interval) with no bias. The count is cadence integrated
over time. CNN only, the bias can't be seen, so count σ grows with time; with
the detector, the bias is estimated and cancelled. Shared by the laptop KF
notebook and the board.
"""

import numpy as np

Q_CADENCE = 0.1         # (steps/s)^2 per s; how fast cadence may drift between measurements
DET_SIGMA = 0.1         # steps/s; σ of a detector cadence measurement
CNN_BIAS_SIGMA = 0.015  # steps/s; prior σ of the CNN's slow offset, held for a whole walk
CADENCE0_SIGMA = 1.0    # steps/s; initial cadence uncertainty

H_CNN = np.array([0.0, 1.0, 1.0])
H_DET = np.array([0.0, 1.0, 0.0])


class StepKF:
    def __init__(self):
        self.x = np.zeros(3)
        self.P = np.diag([0.0, CADENCE0_SIGMA ** 2, CNN_BIAS_SIGMA ** 2])

    def predict(self, dt):
        F = np.eye(3)
        F[0, 1] = dt
        Q = np.zeros((3, 3))
        Q[:2, :2] = Q_CADENCE * np.array([[dt ** 3 / 3, dt ** 2 / 2], [dt ** 2 / 2, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update_cnn(self, cadence, sigma):
        self._update(H_CNN, cadence, sigma)

    def update_det(self, cadence):
        self._update(H_DET, cadence, DET_SIGMA)

    def _update(self, H, z, sigma):
        PH = self.P @ H
        K = PH / (H @ PH + sigma ** 2)
        self.x = self.x + K * (z - H @ self.x)
        self.P = self.P - np.outer(K, PH)

    def count(self):
        """(steps, σ)"""
        return self.x[0], np.sqrt(self.P[0, 0])

    def cadence(self):
        """(steps/s, σ)"""
        return self.x[1], np.sqrt(self.P[1, 1])
