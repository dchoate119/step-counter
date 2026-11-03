"""Cadence CNN forward pass in numpy, from the JSON exported by laptop/model.py.

Shared by the board and the laptop (model.py imports from here), so both run
the same still gate and the same network.
"""

import json
from pathlib import Path

import numpy as np

FS = 50
WIN = 200                               # 4 s at 50 Hz
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
KERNEL = 7
STRIDE = 2
STILL_GYRO = 100.0                      # deg/s; still windows peak under ~50, walking over ~200
STILL_SIGMA = 0.05                      # steps/s, cadence σ reported for still windows
WEIGHTS = Path(__file__).resolve().parent / "cnn.json"


def is_still(x):
    """x: (WIN, 6) raw samples. True if gyro magnitude never reaches STILL_GYRO (cadence 0, skip the CNN)."""
    return np.linalg.norm(x[:, 3:], axis=1).max() < STILL_GYRO


def load(path=WEIGHTS):
    with open(path) as f:
        p = json.load(f)
    return {
        "mean": np.asarray(p["mean"]),
        "std": np.asarray(p["std"]),
        "conv": [(np.asarray(c["w"]), np.asarray(c["b"])) for c in p["conv"]],
        "head": (np.asarray(p["head"]["w"]), np.asarray(p["head"]["b"])),
    }


def forward_np(params, x):
    """x: (WIN, 6) raw samples, oldest first. Returns (mean, logvar)."""
    h = ((x - params["mean"]) / params["std"]).T          # (6, WIN)
    pad = KERNEL // 2
    for w, b in params["conv"]:                            # w: (out, in, k)
        hp = np.pad(h, ((0, 0), (pad, pad)))
        win = np.lib.stride_tricks.sliding_window_view(hp, KERNEL, axis=1)[:, ::STRIDE]
        h = np.maximum(np.einsum("oik,ilk->ol", w, win) + b[:, None], 0)
    w, b = params["head"]
    mu, logvar = w @ h.mean(axis=1) + b
    return float(mu), float(logvar)
