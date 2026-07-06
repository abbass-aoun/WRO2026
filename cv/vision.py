import cv2 as cv
import numpy as np

from config import(
    LOWER_RED_1,
    UPPER_RED_1,
    LOWER_RED_2,
    UPPER_RED_2,
    LOWER_GREEN,
    UPPER_GREEN,
    MIN_PILLAR_AREA,
    BOUNDING_BOX_THICKNESS,
    CENTER_DOT_RADIUS
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
    Masks red regions in the hsv frame.

    Args:
        hsv_frame: frame after being transformed to HSV format.

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
    Masks green regions in the hsv frame.

    Args:
        hsv_frame: frame after being transformed to HSV format.

    Returns:
        green_mask: an image where green pixels are white and other pixels are black.
    """
    # converting HSV range boundaries from tuples to numpy arrays
    lower_green = np.array(LOWER_GREEN, dtype = np.uint8)
    upper_green = np.array(UPPER_GREEN, dtype = np.uint8)

    green_mask = cv.inRange(hsv_frame, lower_green, upper_green)

    return green_mask


def detect_pillars(mask, color_name):
    """
    Detects pillar-like colored regions from a binary mask.

    Args:
        mask: Binary image where the target color pixels are white.
        color_name: Name of the color being detected, such as "red" or "green".

    Returns:
        detections: A list of dictionaries containing information about each pillar detected.
    """

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []

    for contour in contours:
        area = cv.contourArea(contour)

        if area < MIN_PILLAR_AREA:
            continue

        x, y, width, height = cv.boundingRect(contour)

        center_x = x + width // 2
        center_y = y + height // 2

        detection = {
            "color": color_name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
        }

        detections.append(detection)

    return detections


def draw_detections(frame, detections):
    """
    Draws bounding boxes around, center point, and labels for detected pillars.

    Args:
        frame: image from the camera in BGR format
        detections: List of detected pillars
    
    Returns:
        output_frame: Frame with detection drawings
    """

    output_frame = frame.copy()

    for detection in detections:
        x = detection["x"]
        y = detection["y"]
        width = detection["width"]
        height = detection["height"]
        center_x = detection["center_x"]
        center_y = detection["center_y"]
        color_name = detection["color"]
        area = detection["area"]

        if color_name == "red":
            box_color = (0, 0, 255)
        else:
            box_color = (0, 255, 0)
        
        cv.rectangle(
            output_frame,
            (x, y),
            (x + width, y + height),
            box_color,
            BOUNDING_BOX_THICKNESS
        )

        cv.circle(
            output_frame,
            (center_x, center_y),
            CENTER_DOT_RADIUS,
            box_color,
            -1
        )

        label = f"{color_name} area = {int(area)}"

        cv.putText(
            output_frame,
            label,
            (x, y - 10),
            cv.FONT_HERSHEY_COMPLEX,
            0.6,
            box_color,
            2
        )

    return output_frame
