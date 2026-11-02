"""Step counter data logger.

Pulls 50 Hz IMU samples from the sketch every 200 ms, records them to CSV
between Start and Stop, saves the step count typed in for each walk, and
streams samples and status to the web page (http://<board-ip>:7000).
"""

import threading
import time

from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge

import recorder
from recorder import Recorder

POLL_S = 0.2
RATE_WINDOW_S = 5.0
MAX_BATCH = 50   # must match the sketch

ui = WebUI()
rec = Recorder()
lock = threading.Lock()   # page messages arrive on the web server's thread

# Rate stats, refreshed every RATE_WINDOW_S
window_start = time.monotonic()
window_count = 0
window_gap = 0
last_t = None
stats = {"hz": 0.0, "max_gap": 0, "dropped": 0}


def read_all():
    """Call read_batch until the sketch's buffer is empty."""
    lines = []
    while True:
        batch = Bridge.call("read_batch").splitlines()
        lines += batch
        if len(batch) < MAX_BATCH:
            return lines


def status():
    return {**rec.status(), **stats}


def send_recordings(sid=None):
    ui.send_message("recordings", {"files": recorder.recordings()}, room=sid)


def on_start(sid, data):
    with lock:
        rec.start(data.get("label", ""))
    ui.send_message("status", status())
    send_recordings()


def on_stop(sid, data):
    with lock:
        rec.stop()
    ui.send_message("status", status())
    send_recordings()


def on_steps(sid, data):
    try:
        recorder.set_steps(data.get("file", ""), int(data.get("steps")))
    except (TypeError, ValueError) as e:
        print(f"steps not saved: {e}", flush=True)
    send_recordings()


def on_connect(sid):
    ui.send_message("status", status(), room=sid)
    send_recordings(sid)


def update_stats(lines):
    global window_start, window_count, window_gap, last_t
    for line in lines:
        t = int(line.split(",", 1)[0])
        if last_t is not None:
            window_gap = max(window_gap, t - last_t)
        last_t = t
    window_count += len(lines)

    now = time.monotonic()
    if now - window_start >= RATE_WINDOW_S:
        stats["hz"] = round(window_count / (now - window_start), 1)
        stats["max_gap"] = window_gap
        stats["dropped"] = Bridge.call("imu_dropped")
        window_start, window_count, window_gap = now, 0, 0


def loop():
    try:
        lines = read_all()
    except Exception as e:  # sketch may not be ready yet
        print(f"waiting for sketch: {e}", flush=True)
        time.sleep(1)
        return

    with lock:
        rec.write(lines)
    update_stats(lines)

    samples = [[float(x) for x in line.split(",")] for line in lines]
    ui.send_message("samples", {"samples": samples})
    ui.send_message("status", status())
    time.sleep(POLL_S)


ui.on_message("start", on_start)
ui.on_message("stop", on_stop)
ui.on_message("steps", on_steps)
ui.on_connect(on_connect)

try:
    App.run(user_loop=loop)
finally:
    with lock:
        rec.stop()   # close an open recording on app stop
