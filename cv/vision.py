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
    MIN_PILLAR_WIDTH,
    MIN_PILLAR_HEIGHT,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_EXTENT,
    MIN_CONFIDENCE,
    LEFT_REGION_RATIO,
    RIGHT_REGION_RATIO,
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


def calculate_detection_confidence(area, width, height, aspect_ratio, extent):
    """
    Calculates a simple confidence score for a detected pillar.

    Args:
        area: Contour area.
        width: Bounding box width.
        height: Bounding box height.
        aspect_ratio: height / width.
        extent: contour area / bounding box area.

    Returns:
        confidence: A value between 0 and 1.
    """

    area_score = min(area / 3000, 1.0)
    height_score = min(height / 120, 1.0)
    extent_score = min(extent / 0.8, 1.0)

    if MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
        aspect_score = 1.0
    
    else:
        aspect_score = 0.0
    
    # The confidence weights are manually chosen heuristic (not optimal) values.
    # They are not trained probabilities.
    # Area is weighted most because small blobs are often noise.
    # Height and extent help confirm pillar-like shape.
    # Aspect ratio is useful but given lower weight because perspective can distort it.
    confidence = (
        0.35 * area_score 
        + 0.25 * height_score 
        + 0.25 * extent_score 
        + 0.15 * aspect_score   
    )

    return confidence


def is_valid_pillar(area, width, height, aspect_ratio, extent, confidence):
    """
    Checks whether a detected contour should be accepted as a pillar.

    Args:
        area: Contour area.
        width: Bounding box width.
        height: Bounding box height.
        aspect_ratio: height / width.
        extent: contour area / bounding box area.
        confidence: Detection confidence score.

    Returns:
        True if the contour is accepted as a pillar, and false otherwise.
    """

    if area < MIN_PILLAR_AREA:
       return False
    
    if width < MIN_PILLAR_WIDTH:
        return False
    
    if height < MIN_PILLAR_HEIGHT:
        return False
    
    if aspect_ratio < MIN_ASPECT_RATIO or aspect_ratio > MAX_ASPECT_RATIO:
        return False
    
    if extent < MIN_EXTENT:
        return False
    
    if confidence < MIN_CONFIDENCE:
        return False
    
    return True


def classify_horizontal_position(center_x, frame_width):
    """
    Classifies an object's horizontal position in the camera frame.

    Args:
        center_x: The abscissa of the object's center.
        frame_width: The width of the camera frame

    Returns:
        position: "left", "center", or "right"
    """

    left_boundary = int(frame_width * LEFT_REGION_RATIO)
    right_boundary = int(frame_width * RIGHT_REGION_RATIO)

    if(center_x < left_boundary):
        return "left"
    
    if(center_x > right_boundary):
        return "right"
    
    return "center"


def detect_pillars(mask, color_name, frame_width):
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

        bounding_box_area = width * height

        if bounding_box_area == 0:
            continue
        
        aspect_ratio = height / width
        extent = area / bounding_box_area

        confidence = calculate_detection_confidence(
            area,
            width,
            height,
            aspect_ratio,
            extent,
        )

        if not is_valid_pillar(
            area,
            width,
            height,
            aspect_ratio,
            extent,
            confidence,
        ):
            continue

        center_x = x + width // 2
        center_y = y + height // 2

        horizontal_position = classify_horizontal_position(center_x, frame_width)

        detection = {
            "color": color_name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "horizontal_position": horizontal_position,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "extent": extent,
            "confidence": confidence
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
        horizontal_position = detection["horizontal_position"]
        color_name = detection["color"]
        # area = detection["area"]
        color_name = detection["color"]
        confidence = detection["confidence"]
        
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

        label = f"{color_name} {horizontal_position} conf = {confidence:.2f}"

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
