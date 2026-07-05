import cv2 as cv
import numpy as np

from config import(
    LOWER_RED_1,
    UPPER_RED_1,
    LOWER_RED_2,
    UPPER_RED_2,
    LOWER_GREEN,
    UPPER_GREEN
)

def convert_to_hsv(frame):
    """
    Converts a BGR camera frame to HSV color space.

    Args:
        frame: image from the camera in BGR format
    
    Returns:
        hsv_frame: The image converted to HSV format
    """

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return hsv_frame

def create_red_mask(hsv_frame):
    """
    Masks red regions in the hsv frame

    Args:
        hsv_frame: frame after being transformed to HSV format
    Returns:
        red_mask: an image where red pixels are white and other pixels are black.
    """
    # converting HSV range boundaries from tuples to numpy arrays
    lower_red_1 = np.array(LOWER_RED_1, dtype = np.uint8)
    upper_red_1 = np.array(UPPER_RED_1, dtype = np.uint8)

    lower_red_2 = np.array(LOWER_RED_2, dtype = np.uint8)
    upper_red_2 = np.array(UPPER_RED_2, dtype = np.uint8)

    red_mask_1 = cv.inRange(hsv_frame, lower_red_1, upper_red_1)
    red_mask_2 = cv.inRange(hsv_frame, lower_red_2, upper_red_2)

    # combining the two masks
    red_mask = cv.bitwise_or(red_mask_1, red_mask_2)

    return red_mask

def create_green_mask(hsv_frame):
    """
    Masks green regions in the hsv frame

    Args:
        hsv_frame: frame after being transformed to HSV format
    Returns:
        green_mask: an image where green pixels are white and other pixels are black.
    """
    # converting HSV range boundaries from tuples to numpy arrays
    lower_green = np.array(LOWER_GREEN, dtype = np.uint8)
    upper_green = np.array(UPPER_GREEN, dtype = np.uint8)

    green_mask = cv.inRange(hsv_frame, lower_green, upper_green)

    return green_mask