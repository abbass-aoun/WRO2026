# cv/test_vision.py

import cv2 as cv
from camera import open_camera, read_frame, release_camera
from config import WINDOW_NAME


def main():
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        cv.imshow(WINDOW_NAME, frame)

        # Press q to close the camera window
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(cap)


if __name__ == "__main__":
    main()