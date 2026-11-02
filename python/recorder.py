"""CSV recording: one file per Start->Stop, named <label>_<NN>.csv in data/.

Rows are the sketch's read_batch() lines, written as-is and flushed every batch.
The step count for each recording goes in data/steps.csv (file,steps): typed in
on the page for walks, 0 automatically for still.
"""

import csv
import time
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # /app/data on the board
HEADER = "t_ms,ax,ay,az,gx,gy,gz\n"
STEPS_FILE = DATA_DIR / "steps.csv"
LABELS = ("still", "walk")


class Recorder:
    def __init__(self):
        self.file = None
        self.path = None
        self.label = ""
        self.rows = 0
        self.started = 0.0

    @property
    def recording(self):
        return self.file is not None

    def start(self, label):
        if label not in LABELS:
            raise ValueError(f"unknown label {label!r}")
        self.stop()
        DATA_DIR.mkdir(exist_ok=True)
        self.label = label
        self.path = next_free(label)
        self.file = open(self.path, "w")
        self.file.write(HEADER)
        self.rows = 0
        self.started = time.monotonic()
        print(f"recording {self.path.name}", flush=True)

    def write(self, lines):
        if not self.recording or not lines:
            return
        self.file.write("\n".join(lines) + "\n")
        self.file.flush()
        self.rows += len(lines)

    def stop(self):
        if not self.recording:
            return
        self.file.close()
        self.file = None
        print(f"saved {self.path.name}: {self.rows} rows, {self.elapsed():.1f} s", flush=True)
        if self.label == "still":
            set_steps(self.path.name, 0)

    def elapsed(self):
        return time.monotonic() - self.started if self.recording else 0.0

    def status(self):
        return {
            "recording": self.recording,
            "file": self.path.name if self.path else "",
            "label": self.label,
            "rows": self.rows,
            "elapsed": round(self.elapsed(), 1),
        }


def next_free(label):
    """walk_01.csv, walk_02.csv, ... first number not already taken."""
    n = 1
    while (DATA_DIR / f"{label}_{n:02d}.csv").exists():
        n += 1
    return DATA_DIR / f"{label}_{n:02d}.csv"


def load_steps():
    """{file: steps} from steps.csv."""
    if not STEPS_FILE.exists():
        return {}
    with open(STEPS_FILE) as f:
        return {row["file"]: int(row["steps"]) for row in csv.DictReader(f)}


def set_steps(name, steps):
    """Add or replace one recording's step count."""
    path = DATA_DIR / name
    if steps < 0 or path.parent != DATA_DIR or path == STEPS_FILE or not path.exists():
        raise ValueError(f"bad step count {steps} for {name!r}")
    counts = load_steps()
    counts[name] = steps
    with open(STEPS_FILE, "w") as f:
        f.write("file,steps\n")
        f.writelines(f"{k},{v}\n" for k, v in sorted(counts.items()))
    print(f"steps {name}: {steps}", flush=True)


def recordings():
    """Saved files, newest first, for the page's list. steps is None until entered."""
    counts = load_steps()
    files = [p for p in DATA_DIR.glob("*.csv") if p != STEPS_FILE]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [{"file": p.name, "kb": round(p.stat().st_size / 1024), "steps": counts.get(p.name)}
            for p in files]
