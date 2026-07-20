"""
control/allEncodersClass.py — Wheel encoders + MPU-6050 IMU reader.
====================================================================

Ported from last year's VSCode/sensors/allEncodersClass.py.

WHAT THIS DOES:
    Counts pulses from two IR slot sensors on the rear wheels.
    Each pulse = one tooth of the encoder wheel passing the sensor.
    From pulse timing we derive:
        - Forward speed (cm/s)
        - Total distance driven (cm)
    Also reads angular velocity (yaw rate) from the MPU-6050 gyroscope.

HARDWARE:
    Left  encoder  → GPIO pin 27 (default, TUNE ON REAL ROBOT)
    Right encoder  → GPIO pin 19 (default, TUNE ON REAL ROBOT)
    MPU-6050       → I2C bus (address 0x68)

INSTALL:
    pip install mpu6050-raspberrypi

TUNING:
    pulses_per_rev : count the encoder teeth on your wheel disc. TUNE ON REAL ROBOT.
    wheel_circ_cm  : measure the wheel circumference (π × diameter). TUNE ON REAL ROBOT.

NOTE ON IMU AXIS:
    get_yaw_rate() reads gyro['x'] from the MPU-6050.
    IMU is mounted with X axis pointing UP (accel x ≈ +9.8 m/s²), so X = yaw.
    The returned value is in rad/s (already converted from deg/s).

NOTE ON ABSOLUTE HEADING:
    The MPU-6050 only has a gyroscope — no magnetometer.
    It gives angular RATE (rad/s), not absolute heading (degrees).
    Our EKF uses update_gyro_rate() for this sensor.
    If you upgrade to a BNO055 (which gives absolute heading), you can also
    call ekf.update_imu(heading_rad) for a drift-free heading fix.
"""

import math
from gpiozero import Button
from time import time

try:
    from mpu6050 import mpu6050 as MPU6050Lib
    _IMU_AVAILABLE = True
except ImportError:
    _IMU_AVAILABLE = False


class RobotEncoders:
    """
    Reads wheel speed and IMU yaw rate from the real robot hardware.

    Usage:
        enc = RobotEncoders()
        v_l, v_r = enc.get_linear_speeds()   # cm/s
        omega     = enc.get_yaw_rate()        # rad/s
        speed     = enc.get_speed_cm_s()      # cm/s (average of L and R)
    """

    def __init__(self,
                 wheel_left_pin:  int   = 27,
                 wheel_right_pin: int   = 19,
                 pulses_per_rev:  int   = 50,
                 wheel_circ_cm:   float = 20.48):
        """
        Args:
            wheel_left_pin  : BCM GPIO for left  encoder IR sensor.  TUNE ON REAL ROBOT.
            wheel_right_pin : BCM GPIO for right encoder IR sensor.  TUNE ON REAL ROBOT.
            pulses_per_rev  : teeth on the encoder disc per full revolution. TUNE ON REAL ROBOT.
            wheel_circ_cm   : wheel circumference in cm (π × diameter).      TUNE ON REAL ROBOT.
        """
        self.PULSES_PER_REV = pulses_per_rev
        self.WHEEL_CIRC     = wheel_circ_cm
        self._cm_per_pulse  = wheel_circ_cm / pulses_per_rev

        self._left_count          = 0
        self._right_count         = 0
        self._left_last_pulse_t   = time()
        self._right_last_pulse_t  = time()
        self._left_total_dist_cm  = 0.0
        self._right_total_dist_cm = 0.0

        self._left_sensor  = Button(wheel_left_pin,  pull_up=True)
        self._right_sensor = Button(wheel_right_pin, pull_up=True)
        self._left_sensor.when_pressed  = self._on_left_pulse
        self._right_sensor.when_pressed = self._on_right_pulse

        if _IMU_AVAILABLE:
            try:
                self._imu = MPU6050Lib(0x68)
            except OSError:
                print("WARNING: MPU-6050 not responding — gyro disabled, encoders only")
                self._imu = None
        else:
            self._imu = None

    # ------------------------------------------------------------------
    # Interrupt callbacks — fired automatically by gpiozero
    # ------------------------------------------------------------------

    def _on_left_pulse(self) -> None:
        self._left_count         += 1
        self._left_total_dist_cm += self._cm_per_pulse
        self._left_last_pulse_t   = time()

    def _on_right_pulse(self) -> None:
        self._right_count         += 1
        self._right_total_dist_cm += self._cm_per_pulse
        self._right_last_pulse_t  = time()

    # ------------------------------------------------------------------
    # Speed
    # ------------------------------------------------------------------

    def get_linear_speeds(self) -> tuple:
        """
        Return (v_left, v_right) in cm/s.
        Speed = cm_per_pulse / time_since_last_pulse.
        If no pulse for > 1 s, speed is considered 0.
        """
        now  = time()
        dt_l = now - self._left_last_pulse_t
        dt_r = now - self._right_last_pulse_t
        v_l  = self._cm_per_pulse / dt_l if dt_l < 1.0 else 0.0
        v_r  = self._cm_per_pulse / dt_r if dt_r < 1.0 else 0.0
        return v_l, v_r

    def get_speed_cm_s(self) -> float:
        """Return average forward speed (cm/s) from both wheels."""
        v_l, v_r = self.get_linear_speeds()
        return 0.5 * (v_l + v_r)

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def get_distances(self) -> tuple:
        """Return (left_dist_cm, right_dist_cm) total distances driven."""
        return self._left_total_dist_cm, self._right_total_dist_cm

    def reset(self) -> None:
        """Reset pulse counts and distance totals (call at start of each run)."""
        self._left_count          = 0
        self._right_count         = 0
        self._left_total_dist_cm  = 0.0
        self._right_total_dist_cm = 0.0
        self._left_last_pulse_t   = time()
        self._right_last_pulse_t  = time()

    # ------------------------------------------------------------------
    # IMU — yaw rate from MPU-6050 gyroscope
    # ------------------------------------------------------------------

    def get_yaw_rate(self) -> float:
        """
        Return yaw angular velocity in rad/s from the MPU-6050 gyroscope.

        TUNE ON REAL ROBOT: change 'z' to 'x' or 'y' if the IMU is mounted
        in a different orientation on your robot.

        Returns 0.0 if the IMU library is not installed.
        """
        if self._imu is None:
            return 0.0
        gyro   = self._imu.get_gyro_data()   # deg/s
        return math.radians(gyro['x'])        # convert to rad/s  — X axis points UP (accel x≈+9.8)
