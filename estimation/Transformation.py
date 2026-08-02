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

    # Convert mm -> cm and change coordinate system to fit the usual rotation matrix
    robot_local_x_cm = relative_y_mm / 10.0
    robot_local_y_cm = -relative_x_mm / 10.0

    cos_t = math.cos(robot_theta_rad)
    sin_t = math.sin(robot_theta_rad)
    
    
    global_x = (
        robot_x_cm
        + robot_local_x_cm * cos_t
        - robot_local_y_cm * sin_t
    )

    global_y = (
        robot_y_cm
        + robot_local_x_cm * sin_t
        + robot_local_y_cm * cos_t
    )

    return global_x, global_y