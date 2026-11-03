"""Windowed cadence CNN: 4 s of raw 6-axis IMU in, cadence mean and log-variance out.

The torch model is trained in laptop/train/train.ipynb. export() writes its
weights and the input normalization to JSON. The numpy forward pass and the
still gate live in python/cnn.py, shared with the board, and are re-exported
here.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from cnn import (CHANNELS, FS, KERNEL, STILL_GYRO, STILL_SIGMA, STRIDE, WIN,  # noqa: E402,F401
                 forward_np, is_still, load)

CONV = [(6, 16), (16, 32), (32, 32)]    # (in, out) channels per layer


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
