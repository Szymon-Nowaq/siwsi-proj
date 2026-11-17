import argparse
import os
import time

import cv2

from counter import HandDetector, Smoother, extended_fingers

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hand_landmarker.task")
TIPS = [4, 8, 12, 16, 20]
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
         (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17),
         (17, 18), (18, 19), (19, 20)]


class OpenCVCamera:
    def __init__(self, index, width, height):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


class PiCamera:
    def __init__(self, width, height):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        config = self.cam.create_video_configuration(
            main={"size": (width, height), "format": "RGB888"}
        )
        self.cam.configure(config)
        self.cam.start()

    def read(self):
        return self.cam.capture_array()

    def release(self):
        self.cam.stop()


def parse_args():
    p = argparse.ArgumentParser(description="Finger counter (MediaPipe Hands)")
    p.add_argument("--camera", default="0", help="camera index, or 'picam' for Raspberry Pi CSI camera")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    p.add_argument("--fps", type=float, default=15, help="processing rate cap (0 = unlimited)")
    p.add_argument("--hands", type=int, default=1, choices=(1, 2), help="2 = count up to 10")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--headless", action="store_true", help="no window, print count to stdout")
    p.add_argument("--no-mirror", action="store_true")
    return p.parse_args()


def draw(frame, hands, states, count):
    for pts, fingers in zip(hands, states):
        ipts = [(int(x), int(y)) for x, y in pts]
        for a, b in BONES:
            cv2.line(frame, ipts[a], ipts[b], (200, 200, 200), 1)
        for tip, up in zip(TIPS, fingers):
            cv2.circle(frame, ipts[tip], 5, (0, 255, 0) if up else (0, 0, 255), -1)
    cv2.putText(frame, str(count), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, "q: quit", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


def main():
    args = parse_args()
    camera = PiCamera(args.width, args.height) if args.camera == "picam" \
        else OpenCVCamera(int(args.camera), args.width, args.height)
    detector = HandDetector(args.model, max_hands=args.hands)
    smoother = Smoother()
    frame_period = 1.0 / args.fps if args.fps > 0 else 0
    start = time.monotonic()
    last_ts = -1
    last_printed = None

    try:
        while True:
            t0 = time.monotonic()
            frame = camera.read()
            if frame is None:
                break
            if not args.no_mirror:
                frame = cv2.flip(frame, 1)

            ts = max(int((t0 - start) * 1000), last_ts + 1)
            last_ts = ts
            hands = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), ts)
            states = [extended_fingers(pts) for pts in hands]
            count = smoother.push(sum(sum(s) for s in states))

            if args.headless:
                if count != last_printed:
                    print(count, flush=True)
                    last_printed = count
            else:
                draw(frame, hands, states, count)
                cv2.imshow("finger counter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_period:
                spare = frame_period - (time.monotonic() - t0)
                if spare > 0:
                    time.sleep(spare)
    except KeyboardInterrupt:
        pass
    finally:
        detector.close()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
