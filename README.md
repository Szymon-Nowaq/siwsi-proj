# Finger counter

Counts the fingers shown to the camera (0–5, or 0–10 with `--hands 2`) using
MediaPipe Hands plus a few geometry rules on its 21 landmarks.

## How it works

1. MediaPipe runs in `VIDEO` mode. A palm detector finds the hand, then a landmark
   model tracks it. While tracking holds, the palm detector is skipped, which saves CPU.
2. For each hand (`counter.py: extended_fingers`), using only distance ratios so it works in any rotation:
   - index–pinky: the finger is up if its tip is farther from the wrist than its PIP joint;
   - thumb: up if its tip is farther from the pinky base than the thumb IP joint,
     and farther from the index base than `0.6 × palm length` (`THUMB_SPREAD`).
3. The shown number is the most common count over the last 5 frames.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
mkdir -p models && curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

## Run

```bash
.venv/bin/python main.py
.venv/bin/python main.py --hands 2  
```

## Costs

CPU time per frame is measured with `.venv/bin/python benchmark.py` (320×240 input, one hand in view,
mean over 100 frames, pinned to one CPU core, camera read excluded). Inference runs on CPU only.

|                          | Laptop 1                                 | Laptop 2 |
|--------------------------|------------------------------------------|----------|
| Model                    | Lenovo Legion Pro 7 16IRX9H              |          |
| CPU                      | Intel Core i9-14900HX (24 cores)         |          |
| RAM                      | 32 GB                                    |          |
| OS                       | Ubuntu 22.04.5 LTS (kernel 6.8)          |          |
| CPU time per frame       | 16.2 ms                                  |          |
| Peak memory (RSS)        | ~220 MB                                  |          |