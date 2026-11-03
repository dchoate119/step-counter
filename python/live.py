"""Live step count: stride detector + still gate + CNN, fused by the Kalman filter.

Fed raw read_batch() lines. A CNN window's cadence describes its centre, so
the KF runs CNN_DELAY_S behind the newest sample: readings queue up and are
applied in time order once the KF clock reaches them. state() predicts a copy
of the KF forward to the newest sample for display. Settings match
laptop/kf/kf.ipynb.
"""

import copy
from collections import deque

import numpy as np

import cnn
from detector import StrideDetector
from kalman import StepKF

HOP = 25                            # samples between CNN windows (0.5 s)
CNN_DELAY_S = cnn.WIN / cnn.FS / 2  # 2 s, half the window
CNN_SIGMA_SCALE = 3.0               # overlapping windows: ~1 independent reading per 4 s, not 8
PAUSE_S = 2.5                       # stride intervals longer than this are pauses, not cadence


class LiveCounter:
    def __init__(self):
        self.params = cnn.load()
        self.use_det = True
        self.reset()

    def reset(self):
        self.kf = StepKF()
        self.det = StrideDetector()
        self.buf = deque(maxlen=cnn.WIN)
        self.since_cnn = 0
        self.pending = []           # (t_s, kind, cadence, σ) not yet applied to the KF
        self.kf_t = None            # KF clock (s)
        self.t_now = None           # newest sample (s)
        self.last_stride = None
        self.strides = 0

    def set_mode(self, use_det):
        self.use_det = use_det
        self.reset()

    def add(self, lines):
        for line in lines:
            v = [float(x) for x in line.split(",")]
            t = v[0] / 1000.0
            if self.kf_t is None:
                self.kf_t = t
            self.t_now = t
            self.buf.append(v[1:])

            stride = self.det.add(v[0], v[6])
            if stride is not None:
                ts = stride / 1000.0
                self.strides += 1
                if self.use_det and self.last_stride is not None and ts - self.last_stride <= PAUSE_S:
                    self.pending.append((ts, "det", 2 / (ts - self.last_stride), None))
                self.last_stride = ts

            self.since_cnn += 1
            if len(self.buf) == cnn.WIN and self.since_cnn >= HOP:
                self.since_cnn = 0
                x = np.array(self.buf)
                if cnn.is_still(x):
                    mu, sd = 0.0, cnn.STILL_SIGMA
                else:
                    mu, logvar = cnn.forward_np(self.params, x)
                    sd = np.exp(0.5 * logvar)
                self.pending.append((t - CNN_DELAY_S, "cnn", mu, sd * CNN_SIGMA_SCALE))
        self._run_kf()

    def _run_kf(self):
        if self.t_now is None:
            return
        horizon = self.t_now - CNN_DELAY_S
        self.pending.sort(key=lambda e: e[0])
        while self.pending and self.pending[0][0] <= horizon:
            t, kind, c, sig = self.pending.pop(0)
            if t > self.kf_t:
                self.kf.predict(t - self.kf_t)
                self.kf_t = t
            if kind == "cnn":
                self.kf.update_cnn(c, sig)
            else:
                self.kf.update_det(c)

    def state(self):
        kf = copy.deepcopy(self.kf)
        if self.t_now is not None and self.t_now > self.kf_t:
            kf.predict(self.t_now - self.kf_t)
        n, n_sd = kf.count()
        c, c_sd = kf.cadence()
        return {
            "count": round(float(n), 1),
            "count_sigma": round(float(n_sd), 1),
            "cadence": round(float(c), 2),
            "cadence_sigma": round(float(c_sd), 2),
            "detector": 2 * self.strides,
            "use_det": self.use_det,
        }
