import cv2 as cv
import numpy as np
import math

from cv.config import(
    LOWER_RED_1,
    MIN_WALL_SLICE_DENSITY,
    UPPER_RED_1,
    LOWER_RED_2,
    UPPER_RED_2,
    LOWER_GREEN,
    UPPER_GREEN,
    LOWER_PINK,
    UPPER_PINK,
    PARKING_MARKER_HEIGHT_MM,
    PARKING_LOT_LENGTH_MM,
    PARKING_LOT_WIDTH_MM,
    MIN_PARKING_MARKER_AREA,
    MIN_PARKING_MARKER_WIDTH,
    MIN_PARKING_MARKER_HEIGHT,
    MIN_PARKING_CONFIDENCE,
    MORPH_KERNEL_SIZE,
    MORPH_ITERATIONS,
    MIN_PILLAR_AREA,
    MIN_PILLAR_WIDTH,
    MIN_PILLAR_HEIGHT,
    MIN_ASPECT_RATIO,
    MAX_ASPECT_RATIO,
    MIN_EXTENT,
    MIN_CONFIDENCE,
    NAVIGATION_MIN_CONFIDENCE,
    REAL_PILLAR_HEIGHT_MM,
    FOCAL_LENGTH_PIXELS,
    BOUNDING_BOX_THICKNESS,
    CENTER_DOT_RADIUS,
    LOWER_BLACK,
    UPPER_BLACK,
    REAL_WALL_HEIGHT_MM,
    WALL_ROI_START_RATIO,
    WALL_SLICE_WIDTH_PX,
    MIN_WALL_SLICE_PIXELS,
    MIN_WALL_SLICE_HEIGHT_PX,
    LEFT_WALL_ZONE_RATIO,
    RIGHT_WALL_ZONE_RATIO,
    MIN_WALL_CONFIDENCE,
    LOWER_ORANGE,
    UPPER_ORANGE,
    LOWER_BLUE,
    UPPER_BLUE,
    MIN_TRACK_LINE_AREA,
    MIN_TRACK_LINE_WIDTH,
    TRACK_LINE_NEAR_Y_RATIO,
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

    red_mask = clean_mask(red_mask)
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

    green_mask = clean_mask(green_mask)
    return green_mask


def create_pink_mask(hsv_frame):
    """
    Creates a cleaned binary mask for pink/magenta parking markers.

    Args:
        hsv_frame: The camera frame converted to HSV.

    Returns:
        pink_mask: A cleaned binary image where pink/magenta pixels are white.
    """
    lower_pink = np.array(LOWER_PINK, dtype=np.uint8)
    upper_pink = np.array(UPPER_PINK, dtype=np.uint8)

    pink_mask = cv.inRange(hsv_frame, lower_pink, upper_pink)
    pink_mask = clean_mask(pink_mask)

    return pink_mask


def create_black_wall_mask(hsv_frame):
    """
    Creates a cleaned binary mask for black walls.
    """

    lower_black = np.array(LOWER_BLACK, dtype=np.uint8)
    upper_black = np.array(UPPER_BLACK, dtype=np.uint8)

    black_mask = cv.inRange(hsv_frame, lower_black, upper_black)
    black_mask = clean_mask(black_mask)

    return black_mask


def create_orange_line_mask(hsv_frame):
    lower = np.array(LOWER_ORANGE, dtype=np.uint8)
    upper = np.array(UPPER_ORANGE, dtype=np.uint8)

    mask = cv.inRange(hsv_frame, lower, upper)

    return clean_mask(mask)


def create_blue_line_mask(hsv_frame):
    lower = np.array(LOWER_BLUE, dtype=np.uint8)
    upper = np.array(UPPER_BLUE, dtype=np.uint8)

    mask = cv.inRange(hsv_frame, lower, upper)

    return clean_mask(mask)


def clean_mask(mask):
    """
    Cleans a binary mask using morphological operations.

    Args:
        mask: Binary image where target pixels are white and background pixels are black.

    Returns:
        cleaned_mask: Mask after noise removal and gap filling.
    """

    kernel = np.ones(
        (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), 
        dtype=np.uint8
    )

    opened_mask = cv.morphologyEx(
        mask,
        cv.MORPH_OPEN,
        kernel,
        iterations=MORPH_ITERATIONS,
    )

    cleaned_mask = cv.morphologyEx(
        opened_mask,
        cv.MORPH_CLOSE,
        kernel,
        iterations=MORPH_ITERATIONS,
    )

    return cleaned_mask


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

    area_score = min(area / 3000.0, 1.0)
    height_score = min(height / 120.0, 1.0)
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


def estimate_object_distance_mm(pixel_height, real_height_mm):
    """
    Estimates distance from the camera to an object using its real height
    and detected pixel height (using pinhole camera model).

    Args:
        pixel_height: Detected object height in pixels.
        real_height_mm: Real object height in millimeters.

    Returns:
        estimated_distance_mm: Approximate distance from camera to object in millimeters.
    """

    if pixel_height <= 0:
        return None

    estimated_distance_mm = (
        real_height_mm * FOCAL_LENGTH_PIXELS
    ) / pixel_height

    return estimated_distance_mm


def estimate_horizontal_angle(center_x, frame_width):
    """
    Estimates the horizontal angle of the object from the camera central axis.

    Args:
        center_x: The abscissa of the detected pillar.
        frame_width: Width of the camera frame.

    Returns:
        angle_rad: Horizontal angle in radians.
        angle_deg: Horizontal angle in degrees.
    """

    image_center_x = frame_width / 2
    offset_x_pixels = center_x - image_center_x

    angle_rad = math.atan(offset_x_pixels / FOCAL_LENGTH_PIXELS)
    angle_deg = math.degrees(angle_rad)

    return angle_rad, angle_deg


def estimate_camera_relative_position(estimated_distance_mm, angle_rad):
    """
    Estimates object position relative to the camera.

    Camera coordinate system:
        x = left/right offset from camera axis
        y = forward distance from camera

    Args:
        estimated_distance_mm: Approximate distance from camera to pillar.
        angle_rad: Horizontal angle from camera center axis.

    Returns:
        relative_x_mm: Sideways offset from camera center axis.
        relative_y_mm: Forward distance from camera.
    """

    if estimated_distance_mm is None:
        return None, None

    relative_x_mm = estimated_distance_mm * math.sin(angle_rad)
    relative_y_mm = estimated_distance_mm * math.cos(angle_rad)

    return relative_x_mm, relative_y_mm


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


        estimated_distance = estimate_object_distance_mm(height, REAL_PILLAR_HEIGHT_MM)
        angle_rad, angle_deg = estimate_horizontal_angle(center_x, frame_width)
        relative_x, relative_y = estimate_camera_relative_position(estimated_distance, angle_rad)

        detection = {
            "color": color_name,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "estimated_distance_mm": estimated_distance,
            "angle_deg": angle_deg,
            "relative_x_mm": relative_x,
            "relative_y_mm": relative_y,
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
        estimated_distance = detection["estimated_distance_mm"]
        relative_x = detection["relative_x_mm"]
        relative_y = detection["relative_y_mm"]
        angle_deg = detection["angle_deg"]
        color_name = detection["color"]
        area = detection["area"]
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
        if estimated_distance is not None:
            label = (
                f"h = {height} "
                #f"X = {relative_x} "
                #f"Y = {relative_y}"
                f"d = {estimated_distance / 10:.0f}cm "     
            )

        else:
            label = f"{color_name} - conf = {confidence:.2f}"

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


def select_primary_detection(detections):
    """
    Selects the most important detection for navigation.

    Args:
        detections: A list of accepted pillars.

    Returns:
        primary_detection: The detection with the highest priority, or None.
    """

    if not detections:
        return None
    
    usable_detections = []

    for detection in detections:
        if detection["confidence"] < NAVIGATION_MIN_CONFIDENCE:
            continue
        
        if detection["estimated_distance_mm"] is None:
            continue

        usable_detections.append(detection)

    if not usable_detections:
        return None
    
    primary_detection = min(
        usable_detections,
        key=lambda detection: (
            detection["estimated_distance_mm"],
            -detection["confidence"],
            -detection["area"],
        )
    )

    return primary_detection


def detect_parking_markers(mask, frame_width):
    """
    Detects magenta parking limitation markers from a binary mask.

    Args:
        mask: Binary image where parking marker pixels are white.
        frame_width: Width of the original camera frame.

    Returns:
        parking_markers: List of detected parking marker dictionaries.
    """

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    parking_markers = []

    for contour in contours:
        area = cv.contourArea(contour)

        if area < MIN_PARKING_MARKER_AREA:
            continue

        x, y, width, height = cv.boundingRect(contour)

        if width < MIN_PARKING_MARKER_WIDTH:
            continue

        if height < MIN_PARKING_MARKER_HEIGHT:
            continue

        center_x = x + width // 2
        center_y = y + height // 2

        estimated_distance = estimate_object_distance_mm(
            height,
            PARKING_MARKER_HEIGHT_MM,
        )

        angle_rad, angle_deg = estimate_horizontal_angle(center_x, frame_width)

        relative_x, relative_y = estimate_camera_relative_position(
            estimated_distance,
            angle_rad,
        )

        marker_confidence = min(area / 6000, 1.0)

        if(marker_confidence < MIN_PARKING_CONFIDENCE):
            continue

        marker = {
            "type": "parking_marker",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y,
            "area": area,
            "estimated_distance_mm": estimated_distance,
            "angle_deg": angle_deg,
            "relative_x_mm": relative_x,
            "relative_y_mm": relative_y,
            "confidence": marker_confidence
        }

        parking_markers.append(marker)

    parking_markers.sort(key=lambda marker: marker["area"], reverse=True)

    return parking_markers


def create_parking_output(parking_markers):
    """
    Creates parking output from detected parking markers.

    Cases:
        0 markers:
            Parking is not detected.

        1 marker:
            Parking is partially detected.
            The visible marker is stored as marker_1.

        2 or more markers:
            Full parking slot is detected.
            The two best markers are stored as marker_1 and marker_2.

    Args:
        parking_markers: List of detected parking marker dictionaries.

    Returns:
        parking_output: Dictionary describing the detected parking markers and slot.
    """

    if len(parking_markers) == 0:
        return {
            "parking_detected": False,
            "parking_status": "not_detected",
            "reason": "no_markers_detected",
            "marker_count": 0,

            "marker_1": None,
            "marker_2": None,

            "marker_1_relative_x_mm": None,
            "marker_1_relative_y_mm": None,
            "marker_2_relative_x_mm": None,
            "marker_2_relative_y_mm": None,

            "slot_center_relative_x_mm": None,
            "slot_center_relative_y_mm": None,
            "slot_distance_mm": None,
            "slot_angle_deg": None,

            "parking_lot_length_mm": PARKING_LOT_LENGTH_MM,
            "parking_lot_width_mm": PARKING_LOT_WIDTH_MM,
        }

    if len(parking_markers) == 1:
        marker_1 = parking_markers[0]
        marker_2 = None

        marker_1_x = marker_1["relative_x_mm"]
        marker_1_y = marker_1["relative_y_mm"]

        if marker_1_x is not None and marker_1_y is not None:
            slot_center_x = marker_1_x
            slot_center_y = marker_1_y + (PARKING_LOT_LENGTH_MM / 2)

            slot_distance_mm = math.sqrt(
                slot_center_x ** 2
                + slot_center_y ** 2
            )

            slot_angle_rad = math.atan2(slot_center_x, slot_center_y)
            slot_angle_deg = math.degrees(slot_angle_rad)
        else:
            slot_center_x = None
            slot_center_y = None
            slot_distance_mm = None
            slot_angle_deg = None

        return {
            "parking_detected": True,
            "parking_status": "partial_slot_detected",
            "reason": "one_marker_detected",
            "marker_count": 1,

            "marker_1": marker_1,
            "marker_2": marker_2,

            "marker_1_relative_x_mm": marker_1_x,
            "marker_1_relative_y_mm": marker_1_y,
            "marker_2_relative_x_mm": None,
            "marker_2_relative_y_mm": None,

            "slot_center_relative_x_mm": slot_center_x,
            "slot_center_relative_y_mm": slot_center_y,
            "slot_distance_mm": slot_distance_mm,
            "slot_angle_deg": slot_angle_deg,

            "parking_lot_length_mm": PARKING_LOT_LENGTH_MM,
            "parking_lot_width_mm": PARKING_LOT_WIDTH_MM,
        }

    marker_1 = parking_markers[0]
    marker_2 = parking_markers[1]

    marker_1_x = marker_1["relative_x_mm"]
    marker_1_y = marker_1["relative_y_mm"]
    marker_2_x = marker_2["relative_x_mm"]
    marker_2_y = marker_2["relative_y_mm"]

    if (
        marker_1_x is not None
        and marker_1_y is not None
        and marker_2_x is not None
        and marker_2_y is not None
    ):
        slot_center_x = (marker_1_x + marker_2_x) / 2
        slot_center_y = (marker_1_y + marker_2_y) / 2

        slot_distance_mm = math.sqrt(
            slot_center_x ** 2
            + slot_center_y ** 2
        )

        slot_angle_rad = math.atan2(slot_center_x, slot_center_y)
        slot_angle_deg = math.degrees(slot_angle_rad)
    else:
        slot_center_x = None
        slot_center_y = None
        slot_distance_mm = None
        slot_angle_deg = None

    return {
        "parking_detected": True,
        "parking_status": "full_slot_detected",
        "reason": "two_markers_detected",
        "marker_count": len(parking_markers),

        "marker_1": marker_1,
        "marker_2": marker_2,

        "marker_1_relative_x_mm": marker_1_x,
        "marker_1_relative_y_mm": marker_1_y,
        "marker_2_relative_x_mm": marker_2_x,
        "marker_2_relative_y_mm": marker_2_y,

        "slot_center_relative_x_mm": slot_center_x,
        "slot_center_relative_y_mm": slot_center_y,
        "slot_distance_mm": slot_distance_mm,
        "slot_angle_deg": slot_angle_deg,

        "parking_lot_length_mm": PARKING_LOT_LENGTH_MM,
        "parking_lot_width_mm": PARKING_LOT_WIDTH_MM,
    }


def draw_parking_markers(frame, parking_markers):
    """
    Draws every detected parking marker.
    If a slot center is available, it can also draw slot information.
    """

    output_frame = frame.copy()

    for marker in parking_markers:
        x = marker["x"]
        y = marker["y"]
        width = marker["width"]
        height = marker["height"]
        estimated_distance = marker["estimated_distance_mm"]
        relative_x = marker["relative_x_mm"]
        relative_y = marker["relative_y_mm"]
        marker_confidence = marker["confidence"]

        cv.rectangle(
            output_frame,
            (x, y),
            (x + width, y + height),
            (255, 0, 255),
            BOUNDING_BOX_THICKNESS,
        )

        if relative_x is not None and relative_y is not None:
            label = (
                f"d={estimated_distance / 10:.0f}cm "
                f"conf={marker_confidence:.2f} "
                f"X = {relative_x:.0f} "
                f"Y = {relative_y:.0f}"
            )
        else:
            label = f"conf={marker_confidence:.2f}"

        cv.putText(
            output_frame,
            label,
            (x, max(y - 10, 20)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            1,
        )

    return output_frame


def classify_wall_side(slice_center_x, frame_width):
    """
    Classifies where the wall slice appears in the image.
    """

    left_boundary = int(frame_width * LEFT_WALL_ZONE_RATIO)
    right_boundary = int(frame_width * RIGHT_WALL_ZONE_RATIO)

    if slice_center_x < left_boundary:
        return "left"

    if slice_center_x > right_boundary:
        return "right"

    return "front"


def detect_wall_slices(black_mask, frame_width, frame_height):
    """
    Detects black wall slices and estimates their distance.

    The wall is not treated as one object because it can extend across
    the whole frame or appear on the side and front at the same time.

    Args:
        black_mask: Binary mask for black wall pixels.
        frame_width: Width of the camera frame.
        frame_height: Height of the camera frame.

    Returns:
        wall_slices: List of detected wall slice dictionaries.
    """

    roi_start_y = int(frame_height * WALL_ROI_START_RATIO)

    roi_mask = black_mask.copy()
    roi_mask[:roi_start_y, :] = 0

    wall_slices = []

    for x_start in range(0, frame_width, WALL_SLICE_WIDTH_PX):
        x_end = min(x_start + WALL_SLICE_WIDTH_PX, frame_width)

        slice_mask = roi_mask[:, x_start:x_end]

        black_pixels = cv.countNonZero(slice_mask)

        slice_area = slice_mask.shape[0] * slice_mask.shape[1]
        black_density = black_pixels / slice_area

        if black_density < MIN_WALL_SLICE_DENSITY:
            continue

        if black_pixels < MIN_WALL_SLICE_PIXELS:
            continue

        ys, xs = np.where(slice_mask > 0)

        if len(ys) == 0:
            continue

        y_min = int(np.min(ys))
        y_max = int(np.max(ys))

        wall_pixel_height = y_max - y_min + 1

        if wall_pixel_height < MIN_WALL_SLICE_HEIGHT_PX:
            continue

        slice_center_x = x_start + (x_end - x_start) // 2
        slice_center_y = (y_min + y_max) // 2

        estimated_distance = estimate_object_distance_mm(
            wall_pixel_height,
            REAL_WALL_HEIGHT_MM,
        )

        angle_rad, angle_deg = estimate_horizontal_angle(
            slice_center_x,
            frame_width,
        )

        relative_x, relative_y = estimate_camera_relative_position(
            estimated_distance,
            angle_rad,
        )

        wall_side = classify_wall_side(slice_center_x, frame_width)

        confidence = min(wall_pixel_height / 100, 1.0)

        if confidence < MIN_WALL_CONFIDENCE:
            continue

        wall_slice = {
            "type": "wall_slice",

            "wall_side": wall_side,

            "x_start": x_start,
            "x_end": x_end,
            "slice_center_x": slice_center_x,
            "slice_center_y": slice_center_y,

            "y_min": y_min,
            "y_max": y_max,

            "black_pixels": black_pixels,
            "wall_pixel_height": wall_pixel_height,

            "estimated_distance_mm": estimated_distance,
            "angle_deg": angle_deg,
            "relative_x_mm": relative_x,
            "relative_y_mm": relative_y,

            "confidence": confidence,
        }

        wall_slices.append(wall_slice)

    return wall_slices


def create_wall_output(wall_slices):
    """
    Creates a summarized wall output from detected wall slices.

    Returns the nearest left, front, and right wall if they appear.
    """

    left_wall_slices = []
    front_wall_slices = []
    right_wall_slices = []

    for wall_slice in wall_slices:
        if wall_slice["wall_side"] == "left":
            left_wall_slices.append(wall_slice)
        elif wall_slice["wall_side"] == "front":
            front_wall_slices.append(wall_slice)
        elif wall_slice["wall_side"] == "right":
            right_wall_slices.append(wall_slice)

    if left_wall_slices:
        nearest_left_wall = min(
            left_wall_slices,
            key=lambda wall_slice: wall_slice["estimated_distance_mm"],
        )
    else:
        nearest_left_wall = None

    if front_wall_slices:
        nearest_front_wall = min(
            front_wall_slices,
            key=lambda wall_slice: wall_slice["estimated_distance_mm"],
        )
    else:
        nearest_front_wall = None

    if right_wall_slices:
        nearest_right_wall = min(
            right_wall_slices,
            key=lambda wall_slice: wall_slice["estimated_distance_mm"],
        )
    else:
        nearest_right_wall = None

    return {
        "left_wall_detected": nearest_left_wall is not None,
        "front_wall_detected": nearest_front_wall is not None,
        "right_wall_detected": nearest_right_wall is not None,

        "nearest_left_wall": nearest_left_wall,
        "nearest_front_wall": nearest_front_wall,
        "nearest_right_wall": nearest_right_wall,

        "left_wall_distance_mm": (
            nearest_left_wall["estimated_distance_mm"]
            if nearest_left_wall is not None
            else None
        ),
        "front_wall_distance_mm": (
            nearest_front_wall["estimated_distance_mm"]
            if nearest_front_wall is not None
            else None
        ),
        "right_wall_distance_mm": (
            nearest_right_wall["estimated_distance_mm"]
            if nearest_right_wall is not None
            else None
        ),

        "inner_outer_classification_available": False,

        "total_wall_slice_count": len(wall_slices),
    }


def draw_single_nearest_wall_label(frame, wall_slice, color):
    """
    Draws label for one nearest wall slice.
    """

    x = wall_slice["slice_center_x"]
    y = wall_slice["y_min"]

    estimated_distance = wall_slice["estimated_distance_mm"]
    relative_x = wall_slice["relative_x_mm"]
    relative_y = wall_slice["relative_y_mm"]

    if (
        estimated_distance is not None
        and relative_x is not None
        and relative_y is not None
    ):
        label = (
            f"d={estimated_distance / 10:.0f}cm "
            f"X = {relative_x:.0f}mm "
            f"Y = s{relative_y:.0f}mm"
        )
    else:
        label = f"wall"

    cv.putText(
        frame,
        label,
        (x, max(y - 10, 20)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
    )


def draw_wall_slices(frame, wall_slices, wall_output):
    """
    Draws detected wall slices and labels nearest side-based walls.
    """

    output_frame = frame.copy()

    for wall_slice in wall_slices:
        x_start = wall_slice["x_start"]
        x_end = wall_slice["x_end"]
        y_min = wall_slice["y_min"]
        y_max = wall_slice["y_max"]

        wall_side = wall_slice["wall_side"]

        if wall_side == "left":
            box_color = (255, 255, 0)
        elif wall_side == "front":
            box_color = (180, 180, 180)
        else:
            box_color = (0, 255, 255)

        cv.rectangle(
            output_frame,
            (x_start, y_min),
            (x_end, y_max),
            box_color,
            1,
        )

    nearest_left_wall = wall_output["nearest_left_wall"]
    nearest_front_wall = wall_output["nearest_front_wall"]
    nearest_right_wall = wall_output["nearest_right_wall"]

    if nearest_left_wall is not None:
        draw_single_nearest_wall_label(
            output_frame,
            nearest_left_wall,
            (255, 255, 0),
        )

    if nearest_front_wall is not None:
        draw_single_nearest_wall_label(
            output_frame,
            nearest_front_wall,
            (180, 180, 180),
        )

    if nearest_right_wall is not None:
        draw_single_nearest_wall_label(
            output_frame,
            nearest_right_wall,
            (0, 255, 255),
        )

    return output_frame


def detect_track_line(mask, color_name, frame_height):
    """
    Detect the nearest valid orange/blue track line.

    Returns information about whether the line exists and
    whether it has entered the camera's near/arming zone.
    """

    contours, _ = cv.findContours(
        mask,
        cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE,
    )

    valid_lines = []

    for contour in contours:

        area = cv.contourArea(contour)

        if area < MIN_TRACK_LINE_AREA:
            continue

        x, y, width, height = cv.boundingRect(contour)

        if width < MIN_TRACK_LINE_WIDTH:
            continue

        bottom_y = y + height

        valid_lines.append(
            {
                "color": color_name,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "bottom_y": bottom_y,
            }
        )

    if not valid_lines:
        return {
            "detected": False,
            "close": False,
            "bottom_y": None,
            "area": None,
        }

    # Closest visible line = one extending lowest in image.
    nearest = max(
        valid_lines,
        key=lambda line: line["bottom_y"],
    )

    near_zone_start = int(
        frame_height * TRACK_LINE_NEAR_Y_RATIO
    )

    close = nearest["bottom_y"] >= near_zone_start

    return {
        "detected": True,
        "close": close,
        "bottom_y": nearest["bottom_y"],
        "area": nearest["area"],
    }


def detect_track_lines(
    orange_mask,
    blue_mask,
    frame_height,
):
    orange = detect_track_line(
        orange_mask,
        "orange",
        frame_height,
    )

    blue = detect_track_line(
        blue_mask,
        "blue",
        frame_height,
    )

    return {
        "orange": orange,
        "blue": blue,
    }


def process_image(frame):
    """
    Processes one camera frame and returns all useful visual information.

    Args:
        frame:
            BGR image captured from the camera.

    Returns:
        vision_result:
            Dictionary containing:
            - all detected pillars
            - summarized parking information
            - summarized wall information
    """

    if frame is None:
        return {
            "pillars": [],
            "parking": create_parking_output([]),
            "walls": create_wall_output([]),
        }

    # Get frame dimensions once.
    frame_height, frame_width = frame.shape[:2]

    # Convert BGR -> HSV once.
    # All color detection uses this same HSV frame.
    hsv_frame = convert_to_hsv(frame)

    # --------------------------------------------------
    # 1. PILLAR DETECTION
    # --------------------------------------------------

    red_mask = create_red_mask(hsv_frame)
    green_mask = create_green_mask(hsv_frame)

    red_pillars = detect_pillars(
        red_mask,
        "red",
        frame_width,
    )

    green_pillars = detect_pillars(
        green_mask,
        "green",
        frame_width,
    )

    # We want ALL visible pillars.
    pillars = red_pillars + green_pillars

    # --------------------------------------------------
    # 2. PARKING DETECTION
    # --------------------------------------------------

    pink_mask = create_pink_mask(hsv_frame)

    parking_markers = detect_parking_markers(
        pink_mask,
        frame_width,
    )

    # Convert raw marker detections into useful
    # parking-slot information.
    parking = create_parking_output(parking_markers)

    # --------------------------------------------------
    # 3. WALL DETECTION
    # --------------------------------------------------

    black_mask = create_black_wall_mask(hsv_frame)

    wall_slices = detect_wall_slices(
        black_mask,
        frame_width,
        frame_height,
    )

    # Convert many raw wall slices into useful information
    # about the nearest left/front/right walls.
    walls = create_wall_output(wall_slices)


    orange_line_mask = create_orange_line_mask(hsv_frame)
    blue_line_mask = create_blue_line_mask(hsv_frame)

    track_lines = detect_track_lines(
        orange_line_mask,
        blue_line_mask,
        frame_height,
    )


    # --------------------------------------------------
    # 4. RETURN COMPLETE VISION SNAPSHOT
    # --------------------------------------------------

    return {
        "pillars": pillars,
        "parking": parking,
        "walls": walls,
        "track_lines": track_lines
    }
