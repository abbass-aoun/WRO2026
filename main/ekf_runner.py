"""
EKF Runner — main/ekf_runner.py
================================

Shows the full per-tick control loop with EKF state estimation.
Replace every  # MOCK  line with the real hardware call when wiring sensors.

Sensor wiring map  (WRO 2026 robot):
    Encoder  -- wheel encoder on rear axle, gives speed in cm/s
    IMU      -- BNO055 (I2C), gives absolute heading + gyro angular rate
    Servo    -- front steering servo, commanded via servo_cal.steer_deg_to_pwm()

Usage (real robot):
    from main.ekf_runner import EKFRunner
    runner = EKFRunner()
    runner.initialize(start_x=150.0, start_y=50.0, start_theta=0.0)

    while racing:
        steer_cmd = ...                   # from steering PID
        speed_cmd = ...                   # from driving PID
        x, y, theta = runner.tick(speed_cmd, steer_cmd, dt=0.02)
        robot.update_pose(x, y, theta)    # write back for controllers to read

Usage (simulation / unit test):
    runner = EKFRunner()
    runner.initialize(150.0, 50.0, 0.0)
    x, y, th = runner.tick(40.0, 0.0, 0.02)   # straight at 40 cm/s
"""

import math
import numpy as np

from estimation.ekf import EKF

# ─────────────────────────────────────────────────────────────────────────────
# All parameters come from config.py — edit there, not here.
# ─────────────────────────────────────────────────────────────────────────────
from config import (
    WHEELBASE_CM,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2,
    EKF_R_IMU_R2, EKF_R_GYRO_R2,
    EKF_IMU_PERIOD,
)

Q                = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])
R_IMU_HEADING    = EKF_R_IMU_R2
R_GYRO           = EKF_R_GYRO_R2
IMU_UPDATE_PERIOD = EKF_IMU_PERIOD


class EKFRunner:
    """
    Thin wrapper around EKF that manages the update schedule and sensor calls.

    Replace every # MOCK block with real hardware calls.
    The rest of the code (EKF math, scheduling) stays the same.
    """

    def __init__(self):
        self._ekf        = EKF(wheelbase=WHEELBASE_CM, Q=Q, R_imu=R_IMU_HEADING)
        self._tick_count = 0

    def initialize(self, start_x: float, start_y: float, start_theta: float):
        """
        Set the known starting pose before the race begins.
        Call once, when the car is stationary at the start line.

        Args:
            start_x, start_y : position in cm (from track layout)
            start_theta      : heading in radians (0 = East / facing +x)
        """
        self._ekf.initialize(
            start_x, start_y, start_theta,
            pos_std    = 1.5,    # initial position uncertainty (cm, 1-sigma)   # TUNE ON REAL ROBOT
            heading_std= 0.035,  # initial heading uncertainty (~2 deg, 1-sigma)# TUNE ON REAL ROBOT
        )
        self._tick_count = 0

    def tick(
        self,
        speed_cmd:     float,
        steer_deg_cmd: float,
        dt:            float,
    ) -> tuple[float, float, float]:
        """
        Run one complete EKF cycle at ~50 Hz.

        Call this once per control loop iteration, AFTER computing speed and
        steering commands but BEFORE using the pose to make the next decision.

        Args:
            speed_cmd:     commanded forward speed in cm/s
            steer_deg_cmd: commanded steering angle in degrees
            dt:            elapsed seconds since the last tick (target 0.02 s)

        Returns:
            (x, y, theta) — best pose estimate after fusing all sensors this tick
        """
        # ── 1. Read sensors  ─────────────────────────────────────────────────
        speed_meas  = _read_encoder_speed()       # cm/s   # MOCK
        omega_gyro  = _read_gyro_rate()           # rad/s  # MOCK
        theta_abs   = _read_imu_heading()         # rad    # MOCK

        # ── 2. Predict — omega_gyro replaces Ackermann dtheta (no double-count)
        steer_rad = math.radians(steer_deg_cmd)
        self._ekf.predict(speed_meas, steer_rad, dt, omega_gyro=omega_gyro)

        # ── 4. Update: IMU absolute heading every N ticks (drift correction)  ─
        self._tick_count += 1
        if self._tick_count % IMU_UPDATE_PERIOD == 0:
            self._ekf.update_imu(theta_abs)

        return self._ekf.state   # (x, y, theta)

    @property
    def uncertainty(self) -> tuple[float, float]:
        """Returns (position_unc_cm, heading_unc_deg) — useful for logging."""
        return (self._ekf.position_uncertainty,
                self._ekf.heading_uncertainty_deg)


# ─────────────────────────────────────────────────────────────────────────────
# MOCK sensor reads  —  replace with real hardware calls
# ─────────────────────────────────────────────────────────────────────────────

def _read_encoder_speed() -> float:
    """Return wheel speed in cm/s from the rear encoder.  # MOCK
    Real call example (Arduino Serial bridge):
        return encoder.get_speed_cm_s()
    """
    return 0.0   # MOCK

def _read_gyro_rate() -> float:
    """Return yaw angular velocity in rad/s from BNO055 gyro.  # MOCK
    Real call example:
        return math.radians(imu.gyro[2])  # BNO055 gyro z-axis in deg/s -> rad/s
    """
    return 0.0   # MOCK

def _read_imu_heading() -> float:
    """Return absolute yaw heading in radians, wrapped to [-pi, pi].  # MOCK
    Real call example:
        raw_deg = imu.euler[0]          # BNO055 Euler heading, 0-360 deg
        return math.atan2(math.sin(math.radians(raw_deg)),
                          math.cos(math.radians(raw_deg)))
    """
    return 0.0   # MOCK


# =============================================================================
# Self-test  —  run from project root:  python -m main.ekf_runner
# =============================================================================
if __name__ == "__main__":
    print("EKF Runner self-test (mock sensors — all readings are 0)")
    runner = EKFRunner()
    runner.initialize(150.0, 50.0, 0.0)

    DT = 0.02
    N  = 50    # 1 second at 50 Hz

    for i in range(N):
        # Simulate driving straight East at 40 cm/s
        x, y, theta = runner.tick(speed_cmd=40.0, steer_deg_cmd=0.0, dt=DT)
        pos_unc, hdg_unc = runner.uncertainty

    expected_x = 150.0 + 40.0 * N * DT   # = 150 + 40 = 190 cm (from mock encoder = 0, stays 150)
    print(f"  After {N} ticks (mock encoder=0): x={x:.2f}, y={y:.2f}, theta={math.degrees(theta):.2f} deg")
    print(f"  Position uncertainty: {pos_unc:.3f} cm")
    print(f"  Heading uncertainty : {hdg_unc:.3f} deg")
    print("  (x stays at 150 because mock encoder returns 0 -- replace # MOCK to see motion)")
