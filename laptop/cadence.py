"""Cadence labels from labeled stride times.

Stride phase rises by 1 per stride: linearly between strides, and flat during
pauses (gaps over PAUSE_S), before the first stride and after the last one. At
each edge of a pause the phase still ramps by 0.5 over half a stride, so the
total rise equals the number of strides. Cadence over any window is
2 x phase change / window length, in steps/s.
"""

import numpy as np

PAUSE_S = 2.5


def phase_knots(strides_s):
    """(times, phases) for np.interp, from sorted stride times in seconds."""
    s = np.asarray(strides_s, dtype=float)
    if len(s) == 0:
        return np.array([0.0]), np.array([0.0])
    iv = np.diff(s)
    walking = iv[iv <= PAUSE_S]
    typical = np.median(walking) if len(walking) else 1.2

    def half(i):   # half the stride interval next to gap i, for pause edges
        return (iv[i] if 0 <= i < len(iv) and iv[i] <= PAUSE_S else typical) / 2

    t, p = [s[0] - half(0)], [-0.5]
    for i, si in enumerate(s):
        t.append(si)
        p.append(float(i))
        if i + 1 < len(s) and iv[i] > PAUSE_S:   # pause: finish this stride, wait, start the next
            t += [si + half(i - 1), s[i + 1] - half(i + 1)]
            p += [i + 0.5, i + 0.5]
    t.append(s[-1] + half(len(iv) - 1))
    p.append(len(s) - 0.5)
    return np.array(t), np.array(p) + 0.5


def phase(t_s, knots):
    return np.interp(t_s, *knots)


def cadence(t0_s, t1_s, knots):
    """Mean cadence (steps/s) between t0 and t1."""
    return 2 * (phase(t1_s, knots) - phase(t0_s, knots)) / (t1_s - t0_s)
