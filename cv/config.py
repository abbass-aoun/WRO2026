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
LOWER_RED_1 = (0, 100, 100)
UPPER_RED_1 = (10, 255, 255)

LOWER_RED_2 = (170, 100, 100)
UPPER_RED_2 = (179, 255, 255)

# Green range
LOWER_GREEN = (40, 70, 70)
UPPER_GREEN = (85, 255, 255)


# Minimum contour area
# Smaller detected regions are ignored as noise
MIN_PILLAR_AREA = 3500

# Drawing settings 
BOUNDING_BOX_THICKNESS = 2
CENTER_DOT_RADIUS = 5