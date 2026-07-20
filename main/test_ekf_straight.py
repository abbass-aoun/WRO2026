"""
main/test_ekf_straight.py — Straight-line EKF verification.

Drives straight forward at a fixed duty with the servo centred.
Prints EKF x, y, theta every 0.5 s so you can compare to a tape measure.

HOW TO RUN:
    python3 -m main.test_ekf_straight

PASS criteria:
    - x grows by roughly the real distance driven
    - y stays near 50 (no drift sideways)
    - theta stays near 0 (no heading drift)
"""

import math
import time
import numpy as np
from gpiozero import Button

from config import (
    WHEELBASE_CM, DT_S,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2, EKF_R_GYRO_R2,
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

DRIVE_DUTY  = 0.60   # fixed forward duty — TUNE if too fast/slow
RUN_TIME_S  = 5.0    # how long to drive before auto-stop

_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])


def main():
    car      = CarController(PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO)
    encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
    ekf      = EKF(wheelbase=WHEELBASE_CM, Q=_Q, R_imu=EKF_R_GYRO_R2)

    car.stop()
    car.set_steering(0.0)   # servo centred

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

            v_l, v_r  = encoders.get_linear_speeds()
            speed     = 0.5 * (v_l + v_r)
            omega     = encoders.get_yaw_rate()   # 0.0 if IMU absent

            ekf.predict(speed, 0.0, DT_S)
            # Gyro disabled here — bias would curve the straight-line estimate.
            # Enable once bias calibration is done.

            x, y, theta = ekf.state

            car.set_steering(0.0)   # force servo centred every tick
            car.set_motor('f', DRIVE_DUTY)

            if tick % 25 == 0:
                print(
                    f"[{tick*DT_S:5.1f}s]  "
                    f"x={x:6.1f}  y={y:6.1f}  θ={math.degrees(theta):+6.1f}°  "
                    f"spd={speed:5.1f} cm/s  vL={v_l:5.1f}  vR={v_r:5.1f}"
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
        print(f"\nFinal EKF:  x={x:.1f}  y={y:.1f}  θ={math.degrees(theta):+.1f}°")
        print("Measure the real distance driven and compare to (x - 150.0).")
        car.stop()


if __name__ == "__main__":
    main()
