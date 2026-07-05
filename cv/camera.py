# cv/camera.py

import cv2 as cv
from config import CAMERA_INDEX


def open_camera():
    """
    Opens the camera and returns the camera object.
    """

    cap = cv.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Check CAMERA_INDEX or camera connection.")

    return cap


def read_frame(cap):
    """
    Reads one frame from the camera and returns it
    """

    ret, frame = cap.read()

    if not ret:
        raise RuntimeError("Could not read frame from camera.")

    return frame


def release_camera(cap):
    """
    Releases/closes the camera and OpenCV windows properly.
    """

    cap.release()
    cv.destroyAllWindows()