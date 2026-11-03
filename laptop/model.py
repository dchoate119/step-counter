"""Windowed cadence CNN: 4 s of raw 6-axis IMU in, cadence mean and log-variance out.

The torch model is trained in laptop/train/train.ipynb. export() writes its
weights and the input normalization to JSON, and forward_np() runs the same
network in numpy from that JSON, which is what the board will use.
"""

import json

import numpy as np
import torch
from torch import nn

FS = 50
WIN = 200                               # 4 s at 50 Hz
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
CONV = [(6, 16), (16, 32), (32, 32)]    # (in, out) channels per layer
KERNEL = 7
STRIDE = 2
STILL_GYRO = 100.0                      # deg/s; still windows peak under ~50, walking over ~200
STILL_SIGMA = 0.05                      # steps/s, cadence σ reported for still windows


class CadenceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for cin, cout in CONV:
            layers += [nn.Conv1d(cin, cout, KERNEL, STRIDE, padding=KERNEL // 2), nn.ReLU()]
        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(CONV[-1][1], 2)

    def forward(self, x):
        """x: (batch, 6, WIN), normalized. Returns (mean, logvar), each (batch,)."""
        out = self.head(self.conv(x).mean(dim=2))
        return out[:, 0], out[:, 1]


def is_still(x):
    """x: (WIN, 6) raw samples. True if gyro magnitude never reaches STILL_GYRO (cadence 0, skip the CNN)."""
    return np.linalg.norm(x[:, 3:], axis=1).max() < STILL_GYRO


def gaussian_nll(mu, logvar, y):
    return 0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar)).mean()


def export(model, mean, std, path):
    convs = [m for m in model.conv if isinstance(m, nn.Conv1d)]
    params = {
        "mean": np.asarray(mean).tolist(),
        "std": np.asarray(std).tolist(),
        "conv": [{"w": c.weight.tolist(), "b": c.bias.tolist()} for c in convs],
        "head": {"w": model.head.weight.tolist(), "b": model.head.bias.tolist()},
    }
    with open(path, "w") as f:
        json.dump(params, f)


def load(path):
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
