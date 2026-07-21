"""
main/test_ekf_straight.py — Straight-line EKF verification.

Drives straight forward. Uses gyro + encoders in EKF.
Heading correction: if robot is at an angle, servo corrects it back to straight.
Prints sensor readings, EKF state, and control action every 0.5 s.

HOW TO RUN:
    python3 -m main.test_ekf_straight

PASS criteria:
    - x grows by roughly the real distance driven
    - y stays near 50 (small drift OK)
    - theta converges to ~0 even if robot starts at an angle
"""

import math
import time
import numpy as np
from gpiozero import Button

from config import (
    WHEELBASE_CM, DT_S,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2, EKF_R_GYRO_R2,
    SERVO_MAX_DEG,
)
from control.car_controller   import CarController
from control.allEncodersClass import RobotEncoders
from estimation.ekf           import EKF

PIN_MOTOR_IN1    = 18
PIN_MOTOR_IN2    = 13
PIN_MOTOR_ENA    = 19
PIN_SERVO        = 12
PIN_ENC_LEFT     = 7
PIN_ENC_RIGHT    = 5
PIN_START_BUTTON = 8

DRIVE_DUTY   = 0.70   # forward duty
RUN_TIME_S   = 5.0    # run duration
HEADING_KP   = 30.0   # proportional gain for heading correction (deg/rad)

_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])


def main():
    car      = CarController(PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO)
    encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
    ekf      = EKF(wheelbase=WHEELBASE_CM, Q=_Q, R_imu=EKF_R_GYRO_R2)

    car.stop()

    btn = Button(PIN_START_BUTTON, pull_up=True, bounce_time=0.05)
    print("Straight-line EKF test.  Press start button to begin.")
    btn.wait_for_press()
    print("GO — driving straight for %.1f s" % RUN_TIME_S)

    ekf.initialize(x0=150.0, y0=50.0, theta0=0.0)
    encoders.reset()

    t_end = time.monotonic() + RUN_TIME_S
    tick  = 0

    try:
        while time.monotonic() < t_end:
            t_start = time.monotonic()

            # ── Sensors ───────────────────────────────────────────────
            v_l, v_r = encoders.get_linear_speeds()
            speed    = 0.5 * (v_l + v_r)
            omega    = encoders.get_yaw_rate()   # rad/s (0.0 if IMU absent)

            # ── EKF ───────────────────────────────────────────────────
            ekf.predict(speed, 0.0, DT_S)
            if omega != 0.0:
                ekf.update_gyro_rate(omega, DT_S, R_gyro=EKF_R_GYRO_R2)

            x, y, theta = ekf.state

            # ── Heading correction (auto-straighten) ──────────────────
            # theta > 0 = robot heading left, so steer right (negative)
            steer_deg = -HEADING_KP * theta
            steer_deg = max(-SERVO_MAX_DEG, min(SERVO_MAX_DEG, steer_deg))

            # ── Control ───────────────────────────────────────────────
            car.set_steering(steer_deg)
            car.set_motor('f', DRIVE_DUTY)

            # ── Log every 25 ticks (0.5 s) ───────────────────────────
            if tick % 25 == 0:
                print(
                    f"[{tick*DT_S:5.1f}s]  "
                    f"x={x:6.1f}  y={y:6.1f}  theta={math.degrees(theta):+6.2f}deg  "
                    f"spd={speed:5.1f}  vL={v_l:5.1f}  vR={v_r:5.1f}  "
                    f"gyro={math.degrees(omega):+6.2f}deg/s  "
                    f"steer={steer_deg:+5.1f}deg  duty={DRIVE_DUTY:.2f}"
                )

            tick += 1
            leftover = DT_S - (time.monotonic() - t_start)
            if leftover > 0:
                time.sleep(leftover)

    except KeyboardInterrupt:
        print("\nStopped by user.")

    finally:
        car.stop()
        x, y, theta = ekf.state
        print(f"\nFinal EKF:  x={x:.1f}  y={y:.1f}  theta={math.degrees(theta):+.2f}deg")
        print(f"EKF distance: {x - 150.0:.1f} cm — compare to tape measure.")


if __name__ == "__main__":
    main()
