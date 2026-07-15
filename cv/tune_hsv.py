import cv2 as cv
import numpy as np

from camera import open_camera, read_frame, release_camera


CONTROL_WINDOW = "HSV Controls"
ORIGINAL_WINDOW = "Original Frame"
MASK_WINDOW = "HSV Mask"


def nothing(value):
    """
    Empty callback function required by OpenCV trackbars.

    Args:
        value: Current trackbar value.
    """

    pass


def create_hsv_trackbars():
    """
    Creates trackbars for HSV lower and upper limits.
    """

    cv.namedWindow(CONTROL_WINDOW)

    cv.createTrackbar("Lower H", CONTROL_WINDOW, 0, 179, nothing)
    cv.createTrackbar("Lower S", CONTROL_WINDOW, 0, 255, nothing)
    cv.createTrackbar("Lower V", CONTROL_WINDOW, 0, 255, nothing)

    cv.createTrackbar("Upper H", CONTROL_WINDOW, 179, 179, nothing)
    cv.createTrackbar("Upper S", CONTROL_WINDOW, 255, 255, nothing)
    cv.createTrackbar("Upper V", CONTROL_WINDOW, 255, 255, nothing)


def get_hsv_trackbar_values():
    """
    Reads the current HSV values from the trackbars.

    Returns:
        lower_hsv: NumPy array containing lower HSV limit.
        upper_hsv: NumPy array containing upper HSV limit.
    """

    lower_h = cv.getTrackbarPos("Lower H", CONTROL_WINDOW)
    lower_s = cv.getTrackbarPos("Lower S", CONTROL_WINDOW)
    lower_v = cv.getTrackbarPos("Lower V", CONTROL_WINDOW)

    upper_h = cv.getTrackbarPos("Upper H", CONTROL_WINDOW)
    upper_s = cv.getTrackbarPos("Upper S", CONTROL_WINDOW)
    upper_v = cv.getTrackbarPos("Upper V", CONTROL_WINDOW)

    lower_hsv = np.array([lower_h, lower_s, lower_v], dtype=np.uint8)
    upper_hsv = np.array([upper_h, upper_s, upper_v], dtype=np.uint8)

    return lower_hsv, upper_hsv


def main():
    print("Opening camera for HSV tuning...")
    print("Press q to quit.")
    print("Press p to print the current HSV values.")

    cap = open_camera()

    create_hsv_trackbars()

    while True:
        frame = read_frame(cap)

        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower_hsv, upper_hsv = get_hsv_trackbar_values()

        mask = cv.inRange(hsv_frame, lower_hsv, upper_hsv)

        cv.imshow(ORIGINAL_WINDOW, frame)
        cv.imshow(MASK_WINDOW, mask)

        key = cv.waitKey(1) & 0xFF

        if key == ord("p"):
            print("Current HSV values:")
            print(f"LOWER = ({lower_hsv[0]}, {lower_hsv[1]}, {lower_hsv[2]})")
            print(f"UPPER = ({upper_hsv[0]}, {upper_hsv[1]}, {upper_hsv[2]})")

        if key == ord("q"):
            break

    release_camera(cap)
    print("HSV tuning closed.")


if __name__ == "__main__":
    main()