import math
from estimation.ekf import EKF, EKF_R_GYRO_R2

def read_sensors_and_update_ekf(encoders, color, ekf, robot, dt):
    """
    Reads all sensors, runs EKF predict + gyro update, updates robot state.

    Returns:
        speed       : average forward speed (cm/s)
        v_l, v_r    : individual wheel speeds (cm/s)
        omega       : yaw rate from gyro (rad/s)
        x, y, theta : EKF estimated pose
        orange_seen : bool
        blue_seen   : bool
    """
    # 1. Encoders
    v_l, v_r = encoders.get_linear_speeds()
    speed    = 0.5 * (v_l + v_r)

    # 2. Gyro
    omega = encoders.get_yaw_rate()   # rad/s, 0.0 if IMU absent

    # 3. EKF predict (motion model)
    steer_rad = math.radians(robot.steer_angle)
    ekf.predict(speed, steer_rad, dt)

    # 4. EKF update (gyro correction)
    # explicit flag, set once at startup
    if encoders.has_imu:
        ekf.update_gyro_rate(omega, dt, R_gyro=EKF_R_GYRO_R2)

    # 5. Push EKF result into robot state
    x, y, theta = ekf.state
    robot.update_pose(x, y, theta)
    robot.update_speed(speed)

    # 6. Color sensor (background thread — just read the flags)
    orange_seen = color.orange_seen
    blue_seen   = color.blue_seen

    return speed, v_l, v_r, omega, x, y, theta, orange_seen, blue_seen
