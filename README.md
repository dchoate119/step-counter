# 😀 step-counter

### Description

Step counting with uncertainty from an ankle-worn IMU on an Arduino UNO Q. Three parts run on the board:

- **Live detector**: gyro peak detection, one peak per stride (steps = 2 × strides). Fast, but noisy per step.
- **Windowed model**: small 1D CNN (trained in PyTorch on the laptop, exported as JSON weights) predicts cadence ± σ from a 4 s window, every 1 s.
- **Kalman filter**: fuses both into live cadence ± σ and total steps ± σ, shown in the WebUI.

Only walking (any pace up to jogging) and standing still are covered.

### System layout

- **UNO Q MCU** (`sketch/sketch.ino`): IMU at 50 Hz → Bridge
- **UNO Q MPU** (`python/`): recorder, live detector, windowed model, Kalman filter, WebUI
- **Laptop** (`laptop/`): labeling, training, export

### Execution plan

1. [x] **Recording app**: port sketch, recorder and WebUI from gait-identifier. Labels `walk` / `still`; enter step count N at stop.
2. [x] **Collect**: mixed-pace walks of 50–100 steps, plus still recordings.
3. [x] **Label**: notebook detects gyro peaks, hand-fix, check 2 × peaks = N → stride timestamps.
4. [ ]**Live detector**: tune threshold and min gap against labels.
5. [ ]**Windowed CNN**: train with Gaussian NLL, export JSON, check numpy forward pass matches torch.
6. [ ]**Kalman filter**: run offline on recordings, tune noise, check σ calibration on fixed-count trials.
7. [ ]**Board app**: detector + CNN + KF + WebUI live.
