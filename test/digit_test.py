#!/usr/bin/env python3
import cv2
import numpy as np
import time
from pathlib import Path

# --- USER SETTINGS ---
# Monitoring usb port: sudo dmesg -w
# Find yours via:  ls -l /dev/v4l/by-id/
DIGIT_R = "/dev/v4l/by-id/usb-Facebook_DIGIT_D21119-video-index0"
DIGIT_L = "/dev/v4l/by-id/usb-Facebook_DIGIT_D21273-video-index0"
W, H, FPS = 320, 240, 15         # capture resolution for each cam
WINDOW_NAME = "DIGIT R | L (1x2)"


def open_one_cam(cam):
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        cv2.imshow("VideoFrame", frame)
        time.sleep(0.033)

    capture.release()
    cv2.destroyAllWindows()

def open_cam(src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)  # CAP_V4L2 is helpful on Linux
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)            # fallback
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {src}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) # Force MJPG to lower USB bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS,          FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1) # Reduce buffering/lag (supported on many OpenCV builds)
    return cap

def main():
    capR = open_cam(DIGIT_R)
    capL = open_cam(DIGIT_L)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    while True:
        okR, fR = capR.read()
        okL, fL = capL.read()
        if not okR or not okL:
            print("Frame grab failed (check cables/paths).")
            break

        # Make heights equal for clean side-by-side (1x2)
        hR, wR = fR.shape[:2]
        hL, wL = fL.shape[:2]
        common_h = min(hR, hL)

        # fRr = cv2.resize(fR, (int(wR * (common_h / hR)), common_h))
        # fLr = cv2.resize(fL, (int(wL * (common_h / hL)), common_h))

        side = np.hstack([fR, fL])   # 1x2 concat
        cv2.imshow(WINDOW_NAME, side)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    capR.release(); capL.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    x = input("Press 1 for DIGIT_R, Press 2 for DIGIT_L, Press 3 for Both: \n")
    if x == '1':
        open_one_cam(DIGIT_R)
    elif x == '2':
        open_one_cam(DIGIT_L)
    else:
        main()