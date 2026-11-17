"""Mean CPU time per frame (all threads) for detection + counting, with exactly one hand in view."""
import argparse
import os
import resource
import statistics
import time

import cv2

from counter import HandDetector, extended_fingers
from main import DEFAULT_MODEL, OpenCVCamera, PiCamera

WARMUP_FRAMES = 10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--camera", default="0", help="camera index, or 'picam'")
    p.add_argument("--frames", type=int, default=100, help="frames with one hand to measure")
    p.add_argument("--timeout", type=float, default=60, help="give up after this many seconds")
    p.add_argument("--core", type=int, default=0, help="CPU core to pin to")
    args = p.parse_args()

    # MediaPipe sizes its thread pool to the available cores and the sync overhead
    # grows with it, so pin to one core to get a number comparable across machines.
    os.sched_setaffinity(0, {args.core})

    camera = PiCamera(320, 240) if args.camera == "picam" else OpenCVCamera(int(args.camera), 320, 240)
    detector = HandDetector(DEFAULT_MODEL)
    samples = []
    seen = 0
    start = time.monotonic()
    last_ts = -1
    print(f"Show one hand to the camera, collecting {args.frames} frames...")

    try:
        while len(samples) < args.frames and time.monotonic() - start < args.timeout:
            frame = camera.read()
            if frame is None:
                break
            c0 = time.process_time()
            frame = cv2.flip(frame, 1)
            ts = max(int((time.monotonic() - start) * 1000), last_ts + 1)
            last_ts = ts
            hands = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)
            for pts in hands:
                extended_fingers(pts)
            cpu_ms = (time.process_time() - c0) * 1000

            seen += 1
            if seen > WARMUP_FRAMES and len(hands) == 1:
                samples.append(cpu_ms)
    finally:
        detector.close()
        camera.release()

    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # ru_maxrss is in KB on Linux
    print(f"Peak memory (RSS): {peak_mb:.0f} MB")
    if not samples:
        print("No frames with exactly one hand were captured.")
        return
    print(f"CPU time per frame (1 hand in view): {statistics.mean(samples):.1f} ms "
          f"(mean of {len(samples)} frames)")


if __name__ == "__main__":
    main()
