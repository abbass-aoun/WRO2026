import math


def camera_to_world(
    robot_x_cm: float,
    robot_y_cm: float,
    robot_theta_rad: float,
    relative_x_mm: float,
    relative_y_mm: float,
) -> tuple[float, float]:
    """
    Convert camera-relative CV coordinates to global coordinates.

    CV convention:
        relative_x_mm:
            positive = right
            negative = left

        relative_y_mm:
            positive = forward

    Global convention:
        x, y in cm
        theta = 0 faces global +X
        positive theta = CCW
    """

    # Convert mm -> cm
    right_cm = relative_x_mm / 10.0
    forward_cm = relative_y_mm / 10.0

    cos_t = math.cos(robot_theta_rad)
    sin_t = math.sin(robot_theta_rad)

    # Forward direction in world coordinates:
    # (cos(theta), sin(theta))
    #
    # Right direction in world coordinates:
    # (sin(theta), -cos(theta))

    global_x = (
        robot_x_cm
        + forward_cm * cos_t
        + right_cm * sin_t
    )

    global_y = (
        robot_y_cm
        + forward_cm * sin_t
        - right_cm * cos_t
    )

    return global_x, global_y