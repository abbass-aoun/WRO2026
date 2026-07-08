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
# one from 0 to 10 and one from 170 to 179.
LOWER_RED_1 = (0, 140, 80)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (170, 140, 80)
UPPER_RED_2 = (179, 255, 255)

# Green range
LOWER_GREEN = (40, 70, 70)
UPPER_GREEN = (85, 255, 255)


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