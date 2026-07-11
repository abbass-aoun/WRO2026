CAMERA_INDEX = 0

WINDOW_NAME = "Camera Feed"
RED_MASK_WINDOW = "Red Mask"
GREEN_MASK_WINDOW = "Green Mask"


# Hue: 0 to 179
# Saturation: 0 to 255
# Value: 0 to 255

# HSV color ranges for red and green. 
# These values are starting points and will need tuning using the real pillars

# Red wraps around the HSV hue scale, so we need two red ranges:

# one from 0 to 10.
LOWER_RED_1 = (0, 140, 80)
UPPER_RED_1 = (10, 255, 255)

# one from 170 to 179.
LOWER_RED_2 = (170, 140, 80)
UPPER_RED_2 = (179, 255, 255)

# Green range
LOWER_GREEN = (35, 60, 30)
UPPER_GREEN = (90, 255, 200)


# Minimum contour area
# Smaller detected regions are ignored as noise
MIN_PILLAR_AREA = 1000

# Minimum bounding box size.
MIN_PILLAR_WIDTH = 15
MIN_PILLAR_HEIGHT = 30

# Aspect ratio = height / width.
# Pillars are expected to appear taller than they are wide.
# The official pillar dimensions are 50 mm × 50 mm × 100 mm.
# Since the visible front face is approximately 50 mm wide and 100 mm tall, the expected height-to-width ratio is about 2.0.
# The detector therefore uses an aspect ratio range around this value, while keeping tolerance for perspective distortion, camera angle, and partial detection.
MIN_ASPECT_RATIO = 1.4
MAX_ASPECT_RATIO = 3.2

# Extent = contour area / bounding box area.
# Low extent means the contour does not fill its rectangle well.
MIN_EXTENT = 0.35

# Minimum confidence required to accept a detection.
MIN_CONFIDENCE = 0.45


# Drawing settings 
BOUNDING_BOX_THICKNESS = 2
CENTER_DOT_RADIUS = 5


# Camera-relative position thresholds.
# If center_x is less than 33% of the frame width, the object is on the left.
# If center_x is greater than 66% of the frame width, the object is on the right.
# Otherwise, it is in the center.
LEFT_REGION_RATIO = 0.33
RIGHT_REGION_RATIO = 0.66

# Distance estimation thresholds based on bounding box height in pixels.
# In the camera image, a closer pillar appears taller.
# Initial values - to be tuned with the rasberry pi camera.
FAR_PILLAR_HEIGHT = 60
CLOSE_PILLAR_HEIGHT = 150


# Navigation decision thresholds

# Minimum confidence needed for a detection to be used for navigation
NAVIGATION_MIN_CONFIDENCE = 0.55

# Distance levels that are important for navigation.
# For now, we will ignore far pillars and react only to medium or close pillars.
ACTION_DISTANCE_LEVELS = ["medium", "close"]


# Camera-relative distance and angle estimation

# Official pillar height.
REAL_PILLAR_HEIGHT_MM = 100

# Temporary focal length for current camera setup.
# This must be calibrated for the laptop webcam first,
# then recalibrated for the Raspberry Pi camera later.
FOCAL_LENGTH_PIXELS = 700


# Mask cleaning settings

# Kernel size used for morphological operations.
# Larger values clean more aggressively but may remove small valid detections.
MORPH_KERNEL_SIZE = 6

# Number of times the operation is repeated.
MORPH_ITERATIONS = 1


# Parking detection window
PINK_MASK_WINDOW = "Pink Parking Mask"

# HSV range for pink/magenta parking markers.
# Starting values; tune later using tune_hsv.py.
LOWER_MAGENTA = (160, 20, 110)
UPPER_MAGENTA = (179, 150, 255)

# Official parking limitation dimensions in mm:
# 200 mm x 20 mm x 100 mm
PARKING_MARKER_LENGTH_MM = 200
PARKING_MARKER_DEPTH_MM = 20
PARKING_MARKER_HEIGHT_MM = 100

# Replace this with the actual robot length later.
ROBOT_LENGTH_MM = 300

# Parking lot length = 1.5 × robot length.
PARKING_LOT_LENGTH_MM = int(1.5 * ROBOT_LENGTH_MM)

# Parking lot width is always 200 mm.
PARKING_LOT_WIDTH_MM = 200

# Parking marker filtering
MIN_PARKING_MARKER_AREA = 1200
MIN_PARKING_MARKER_WIDTH = 20
MIN_PARKING_MARKER_HEIGHT = 40

# Parking alignment tolerance
PARKING_ALIGNMENT_TOLERANCE_PX = 40