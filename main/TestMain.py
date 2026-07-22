"""
main/TestMain.py — Sensor + EKF loop test (no motor movement).

Initialises all hardware, waits for start button, then runs the
read_sensors_and_update_ekf loop at 50 Hz and prints the state.

HOW TO RUN:
    python3 -m main.TestMain
"""

import math
import time
import numpy as np
from gpiozero import Button

from config import (
    WHEELBASE_CM, DT_S,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2, EKF_R_GYRO_R2,
)
from control.allEncodersClass import RobotEncoders
from control.color_sensor     import ColorSensor
from control.robot            import Robot
from estimation.ekf           import EKF

# ── Hardware pins ─────────────────────────────────────────────────────────────
PIN_ENC_LEFT     = 7
PIN_ENC_RIGHT    = 5
PIN_COLOR_S0     = 17
PIN_COLOR_S1     = 27
PIN_COLOR_S2     = 22
PIN_COLOR_S3     = 23
PIN_COLOR_OUT    = 24
PIN_COLOR_LED    = 25

# ── EKF noise matrices ────────────────────────────────────────────────────────
_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])

def calibrate_gyro(encoders, samples=200):
    """Robot MUST be perfectly still during this."""
    print("Calibrating gyro — do NOT move the robot...")
    total = 0.0
    for _ in range(samples):
        total += encoders.get_yaw_rate()
        time.sleep(0.01)
    bias = total / samples
    print(f"Gyro bias = {math.degrees(bias):+.2f} deg/s\n")
    return bias

# ── Sensor read + EKF update (one tick) ──────────────────────────────────────
def read_sensors_and_update_ekf(encoders, color, ekf, robot, dt,gyro_bias):
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
    omega = encoders.get_yaw_rate() -gyro_bias   # rad/s, 0.0 if IMU absent

    # 3. EKF predict (motion model)
    steer_rad = math.radians(robot.steer_angle)
    ekf.predict(speed, steer_rad, dt)

    # 4. EKF update (gyro correction)
    # explicit flag, set once at startup
    ekf.update_gyro_rate(omega, dt, R_gyro=EKF_R_GYRO_R2)

    # 5. Push EKF result into robot state
    x, y, theta = ekf.state
    robot.update_pose(x, y, theta)
    robot.update_speed(speed)

    # 6. Color sensor (background thread — just read the flags)
    orange_seen = color.orange_seen
    blue_seen   = color.blue_seen

    return speed, v_l, v_r, omega, x, y, theta, orange_seen, blue_seen



# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Initialise hardware
    encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
    color    = ColorSensor(PIN_COLOR_S0, PIN_COLOR_S1,
                           PIN_COLOR_S2, PIN_COLOR_S3,
                           PIN_COLOR_OUT, PIN_COLOR_LED)
    ekf      = EKF(wheelbase=WHEELBASE_CM, Q=_Q, R_imu=EKF_R_GYRO_R2)
    robot    = Robot()  

    gyro_bias = calibrate_gyro(encoders) 
     
    # Initialise EKF at starting position
    ekf.initialize(x0=0.0, y0=0.0, theta0=0.0)
    encoders.reset()
    tick = 0
    last_time = time.monotonic()

    try:
        while True:
            t_start = time.monotonic()

            now = time.monotonic()
            dt = now - last_time if tick > 0 else DT_S
            last_time = now
            
            speed, v_l, v_r, omega, x, y, theta, orange, blue = \
                read_sensors_and_update_ekf(encoders, color, ekf, robot, dt, gyro_bias)

            if tick % 25 == 0:
                print(
                    f"[{tick * DT_S:6.1f}s]  "
                    f"x={x:6.1f}  y={y:6.1f}  theta={math.degrees(theta):+6.2f}deg  "
                    f"spd={speed:5.1f}  vL={v_l:5.1f}  vR={v_r:5.1f}  "
                    f"gyro={math.degrees(omega):+6.2f}deg/s  "
                    f"orange={'YES' if orange else 'no ':3}  "
                    f"blue={'YES' if blue else 'no '}"
                )

            tick += 1
            leftover = DT_S - (time.monotonic() - t_start)
            if leftover > 0:
                time.sleep(leftover)

    except KeyboardInterrupt:
        print("\nStopped.")

    finally:
        color.stop()


if __name__ == "__main__":
    main()
