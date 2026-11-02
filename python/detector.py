"""Stride detector: one peak per stride of the ankle foot in gz (deg/s).

Shared by the laptop labeling notebook and the board so both find exactly the
same strides. Works sample by sample: a stride starts when gz rises above HIGH,
and is emitted at its highest point once gz falls back below LOW. Strides
closer than MIN_GAP_S to the previous one are ignored. Steps = 2 x strides.
"""

FS = 50
HIGH = 80.0        # deg/s; slow strides peak around 100, fidgeting stays under ~30
LOW = 20.0         # deg/s; hysteresis so one swing can't trigger twice
MIN_GAP_S = 0.5    # fastest stride recorded is ~0.8 s


class StrideDetector:
    def __init__(self):
        self.armed = False
        self.peak_t = None
        self.peak_v = 0.0
        self.last_t = None

    def add(self, t_ms, gz):
        """Feed one sample. Returns the stride's peak time (ms) when one ends, else None."""
        if not self.armed:
            if gz > HIGH:
                self.armed, self.peak_t, self.peak_v = True, t_ms, gz
            return None
        if gz > self.peak_v:
            self.peak_t, self.peak_v = t_ms, gz
        if gz >= LOW:
            return None
        self.armed = False
        if self.last_t is not None and self.peak_t - self.last_t < MIN_GAP_S * 1000:
            return None
        self.last_t = self.peak_t
        return self.peak_t


def find_strides(t_ms, gz):
    """Stride peak times (ms) for a whole recording."""
    det = StrideDetector()
    return [s for s in (det.add(t, g) for t, g in zip(t_ms, gz)) if s is not None]
