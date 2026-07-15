"""
EXTENDED KALMAN FILTER (EKF) -- estimation/ekf.py
==================================================

PURPOSE:
    Estimate the car's true position (x, y, theta) by fusing two noisy sources:
      - Encoder speed  --> feeds the PREDICT step (bicycle model)
      - IMU heading    --> feeds the UPDATE step (heading correction)

CONCEPT IN ONE PARAGRAPH:
    We keep two things: a state estimate [x, y, theta] and a covariance matrix P.
    P is a 3x3 grid of numbers that tracks our UNCERTAINTY about each state variable
    and how they are correlated. Every loop tick we PREDICT (move the estimate forward
    using the motion model, which grows P because motion adds error). Whenever the IMU
    sends a heading reading, we UPDATE (pull the estimate toward the measurement, which
    shrinks P because new information reduces uncertainty). The Kalman Gain K decides
    HOW MUCH to pull: if P is large (model uncertain) and R is small (sensor accurate),
    K is large and we trust the sensor. If R is large, K is small and we stay closer
    to the model.

WHY "EXTENDED"?
    A regular Kalman Filter only handles LINEAR equations.
    Our bicycle model has sin(theta) and cos(theta) -- nonlinear.
    The "Extended" fix: at each predict step, compute the JACOBIAN F (partial
    derivatives of the motion model w.r.t. the state). F is the best linear
    approximation of the motion model at the current operating point.

STATE VECTOR  x = [x, y, theta]
    x     -- cm, East is positive
    y     -- cm, North is positive
    theta -- radians, 0 = East, pi/2 = North, wrapped to [-pi, pi]

COVARIANCE MATRIX P (3x3)
    P[0,0] -- variance of x estimate (cm^2)
    P[1,1] -- variance of y estimate (cm^2)
    P[2,2] -- variance of theta estimate (rad^2)
    P[i,j] -- covariance between variable i and j (captures correlations)
    sqrt(P[i,i]) is the standard deviation -- the "radius of uncertainty"

NOISE PARAMETERS (all TUNE ON REAL ROBOT)
    Q      -- 3x3 process noise: uncertainty the motion model adds per step
    R_imu  -- scalar: IMU heading measurement noise variance (rad^2)
              BNO055 accuracy ~1 deg -> R_imu ~ (0.017)^2 = 0.0003
"""

import math
import numpy as np
from control.robot import Robot


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _wrap(angle: float) -> float:
    """Wrap angle to [-pi, pi] to avoid the 179 -> -179 degree jump problem."""
    return math.atan2(math.sin(angle), math.cos(angle))


# ---------------------------------------------------------------------------
# EKF class
# ---------------------------------------------------------------------------

class EKF:
    """
    Extended Kalman Filter for WRO car position and heading estimation.

    State:   [x (cm), y (cm), theta (rad)]
    Predict: bicycle kinematic model driven by encoder speed
    Update:  IMU absolute heading correction
    """

    def __init__(self, wheelbase: float, Q: np.ndarray, R_imu: float):
        """
        Args:
            wheelbase: front-to-rear axle distance in cm    # TUNE ON REAL ROBOT
            Q:         3x3 process noise covariance matrix  # TUNE ON REAL ROBOT
            R_imu:     IMU heading measurement noise (rad^2)# TUNE ON REAL ROBOT
        """
        self._L     = float(wheelbase)
        self._Q     = np.asarray(Q, dtype=float).reshape(3, 3)
        self._R_imu = float(R_imu)

        # State estimate -- will be set properly by initialize()
        self._x = np.zeros(3, dtype=float)

        # Covariance -- large until initialize() is called
        self._P = np.eye(3) * 1e6

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self, x0: float, y0: float, theta0: float,
                   pos_std: float = 2.0, heading_std: float = 0.05):
        """
        Set the known starting position before the race begins.

        Args:
            x0, y0:      starting position in cm (from track layout)
            theta0:      starting heading in radians (0 = East)
            pos_std:     how well we know the start position (cm, 1-sigma)
            heading_std: how well we know the start heading (rad, 1-sigma)
        """
        self._x = np.array([x0, y0, theta0], dtype=float)
        self._P = np.diag([pos_std ** 2, pos_std ** 2, heading_std ** 2])

    # ------------------------------------------------------------------
    # Predict step
    # ------------------------------------------------------------------

    def predict(self, speed: float, steer_angle: float, dt: float):
        """
        Advance the state estimate by one time step using the bicycle model.

        Called every loop tick (~50 Hz) with encoder-measured speed.

        WHAT HAPPENS INSIDE:
            1. Apply bicycle kinematics to get new x, y, theta.
            2. Compute Jacobian F (linearise the motion around current state).
            3. Propagate covariance: P = F*P*F^T + Q
               (P grows because motion adds uncertainty)

        JACOBIAN F  (why we need it):
            f([x, y, th]) = [x + v*cos(th)*dt,
                             y + v*sin(th)*dt,
                             th + (v/L)*tan(d)*dt]
            F = df/d[x,y,th] evaluated at current state:
            F = [[1, 0, -v*sin(th)*dt],
                 [0, 1,  v*cos(th)*dt],
                 [0, 0,  1           ]]
            The only nonlinear terms are in the third column
            (how heading change affects x and y).

        Args:
            speed:       current speed in cm/s  (from encoder)     # MOCK until encoder wired
            steer_angle: current steer angle in radians (from servo)# MOCK until servo wired
            dt:          time step in seconds (typically 0.02 s)
        """
        x, y, theta = self._x
        v = speed
        d = steer_angle
        L = self._L

        # -- Bicycle kinematics (nonlinear motion model) --
        dx     = v * math.cos(theta) * dt
        dy     = v * math.sin(theta) * dt
        dtheta = (v / L) * math.tan(d) * dt

        self._x[0] = x + dx
        self._x[1] = y + dy
        self._x[2] = _wrap(theta + dtheta)

        # -- Jacobian F = df/d[x, y, theta] at current (x, y, theta) --
        F = np.array([
            [1.0, 0.0, -v * math.sin(theta) * dt],
            [0.0, 1.0,  v * math.cos(theta) * dt],
            [0.0, 0.0,  1.0],
        ], dtype=float)

        # -- Covariance propagation --
        # Moving adds uncertainty: P grows by Q each step
        self._P = F @ self._P @ F.T + self._Q

    # ------------------------------------------------------------------
    # Update step -- IMU heading
    # ------------------------------------------------------------------

    def update_imu(self, theta_measured: float):
        """
        Correct the heading estimate using an IMU absolute heading reading.

        WHAT HAPPENS INSIDE:
            1. Compute innovation: y = theta_measured - theta_predicted
               (angle-wrapped to handle the 179 -> -179 jump)
            2. Compute Kalman Gain: K = P * H^T * (H*P*H^T + R)^{-1}
               H = [0, 0, 1]  because we directly measure theta
            3. Update state: x = x + K * y
               (K tells us how far to move toward the measurement)
            4. Update covariance: P = (I - K*H) * P
               (P shrinks because the measurement gave us new information)

        UNDERSTANDING K FOR THIS UPDATE:
            K is a 3-vector: [K_x, K_y, K_theta]
            Right after initialize() with diagonal P, off-diagonal terms are
            zero, so K_x = K_y = 0 and only theta is corrected.
            After curved driving, off-diagonals grow (position and heading
            become correlated), so an IMU update also slightly corrects x, y.

        Args:
            theta_measured: heading from IMU in radians, in [-pi, pi]
                            # MOCK until IMU wired (e.g. BNO055 euler[0] in rad)
        """
        # Measurement model: we observe theta directly
        # h(x) = theta  -->  H = [0, 0, 1]
        H = np.array([[0.0, 0.0, 1.0]])        # shape (1, 3)
        R = np.array([[self._R_imu]])           # shape (1, 1)

        # Innovation: what the sensor says minus what we predicted
        # Must be angle-wrapped to handle the -pi/pi boundary
        y = _wrap(theta_measured - self._x[2]) # scalar

        # Innovation covariance: S = H*P*H^T + R  --> scalar = P[2,2] + R_imu
        S = float((H @ self._P @ H.T + R)[0, 0])

        # Kalman gain: K = P * H^T / S  --> shape (3,)
        # This is the 3rd column of P divided by S
        K = (self._P @ H.T).flatten() / S      # shape (3,)

        # State correction: move state toward measurement by K * innovation
        self._x     = self._x + K * y
        self._x[2]  = _wrap(self._x[2])

        # Covariance update: P = (I - K*H) * P
        # Reformulate K to shape (3,1) for outer product
        self._P = (np.eye(3) - K.reshape(3, 1) @ H) @ self._P

    # ------------------------------------------------------------------
    # Update step -- gyro angular rate
    # ------------------------------------------------------------------

    def update_gyro_rate(self, omega_measured: float, dt: float,
                         R_gyro: float = 1e-5):
        """
        Correct the heading estimate using a gyro angular-rate reading.

        The gyro measures dtheta/dt = omega (rad/s).  We integrate one step
        to get the expected heading change and treat that as a soft heading
        measurement.  Because gyro drift is slow (~0.01 deg/s for MEMS), the
        noise R_gyro can be very small, making this update highly trusted.

        WHEN TO CALL:
            Every tick (50 Hz) immediately after predict().
            If the IMU also gives an absolute heading, call update_imu() as
            well (e.g. every 5th tick) to correct long-term gyro drift.

        Args:
            omega_measured: gyro angular velocity in rad/s        # MOCK until wired
                            (BNO055: imu.gyro[2] for yaw axis)
            dt:             time step in seconds (typically 0.02 s)
            R_gyro:         gyro noise variance (rad^2 per step)  # TUNE ON REAL ROBOT
                            Default 1e-5 ≈ (0.003 rad)^2 ≈ 0.18 deg per step — typical MEMS.
        """
        # Heading implied by integrating the gyro one step from current estimate
        theta_from_gyro = _wrap(self._x[2] + omega_measured * dt)

        # EKF update — same structure as update_imu() but with R_gyro instead of R_imu
        H = np.array([[0.0, 0.0, 1.0]])
        R = np.array([[R_gyro]])
        y = _wrap(theta_from_gyro - self._x[2])          # innovation (angle-wrapped)
        S = float((H @ self._P @ H.T + R)[0, 0])
        K = (self._P @ H.T).flatten() / S                # Kalman gain, shape (3,)
        self._x    = self._x + K * y
        self._x[2] = _wrap(self._x[2])
        self._P    = (np.eye(3) - K.reshape(3, 1) @ H) @ self._P

    # ------------------------------------------------------------------
    # Write to Robot
    # ------------------------------------------------------------------

    def update_robot(self, robot: Robot):
        """
        Write the current EKF estimate into the shared Robot object.

        Call this at the END of each control loop tick, after predict()
        and any update() calls. The main loop and controllers read robot.pose.
        """
        robot.update_pose(
            float(self._x[0]),
            float(self._x[1]),
            float(self._x[2])
        )

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> tuple:
        """Current estimate as (x_cm, y_cm, theta_rad)."""
        return (float(self._x[0]), float(self._x[1]), float(self._x[2]))

    @property
    def covariance(self) -> np.ndarray:
        """Current 3x3 covariance matrix P (a copy -- safe to modify)."""
        return self._P.copy()

    @property
    def position_uncertainty(self) -> float:
        """RMS position uncertainty in cm = sqrt(var_x + var_y)."""
        return float(math.sqrt(self._P[0, 0] + self._P[1, 1]))

    @property
    def heading_uncertainty_deg(self) -> float:
        """1-sigma heading uncertainty in degrees."""
        return float(math.degrees(math.sqrt(abs(self._P[2, 2]))))

    def __repr__(self):
        x, y, th = self._x
        return (
            f"EKF(x={x:.2f}cm, y={y:.2f}cm, "
            f"theta={math.degrees(th):.2f}deg | "
            f"pos_unc={self.position_uncertainty:.2f}cm, "
            f"hdg_unc={self.heading_uncertainty_deg:.2f}deg)"
        )


# ===========================================================================
# TEST -- run from project root:  python -m estimation.ekf
# ===========================================================================
if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("EKF TEST SUITE")
    print("=" * 60)

    Q = np.diag([0.01, 0.01, 0.0001])   # process noise (TUNE ON REAL ROBOT)
    ekf = EKF(wheelbase=16.5, Q=Q, R_imu=0.001)

    # ------------------------------------------------------------------
    # Test 1: initialize
    # ------------------------------------------------------------------
    print("\nTEST 1: Initialize at (100, 50, 0 deg)")
    ekf.initialize(100.0, 50.0, 0.0, pos_std=2.0, heading_std=0.05)
    x, y, th = ekf.state
    print(f"  x={x:.2f}  y={y:.2f}  theta={math.degrees(th):.2f} deg")
    print(f"  position uncertainty : {ekf.position_uncertainty:.3f} cm  (expected ~2.83)")
    print(f"  heading uncertainty  : {ekf.heading_uncertainty_deg:.3f} deg (expected ~2.86)")
    assert abs(x - 100.0) < 1e-9 and abs(y - 50.0) < 1e-9 and abs(th) < 1e-9
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 2: predict straight East
    # ------------------------------------------------------------------
    print("\nTEST 2: Predict -- straight East at 40 cm/s for 1 second")
    ekf.initialize(0.0, 0.0, 0.0)
    for _ in range(50):          # 50 * 0.02 s = 1 s
        ekf.predict(speed=40.0, steer_angle=0.0, dt=0.02)
    x, y, th = ekf.state
    print(f"  x={x:.2f} cm  (expected ~40.00)")
    print(f"  y={y:.4f} cm  (expected ~0.0000)")
    print(f"  theta={math.degrees(th):.4f} deg  (expected ~0.0)")
    print(f"  position uncertainty : {ekf.position_uncertainty:.3f} cm")
    assert abs(x - 40.0) < 0.1, f"x should be ~40, got {x}"
    assert abs(y) < 0.01
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 3: IMU corrects injected heading error
    # ------------------------------------------------------------------
    print("\nTEST 3: Inject 5 deg heading error -- IMU update corrects it")
    ekf.initialize(0.0, 0.0, 0.0)
    for _ in range(10):
        ekf.predict(speed=40.0, steer_angle=0.0, dt=0.02)
    ekf._x[2] = math.radians(5.0)   # inject error
    th_before = math.degrees(ekf.state[2])
    print(f"  theta before IMU update: {th_before:.2f} deg  (injected 5 deg)")
    ekf.update_imu(math.radians(1.0))
    th_after = math.degrees(ekf.state[2])
    print(f"  theta after IMU update : {th_after:.4f} deg  (expected between 1 and 5)")
    assert 1.0 < th_after < 5.0, f"Expected 1 < theta < 5, got {th_after}"
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 4: covariance shrinks after update
    # ------------------------------------------------------------------
    print("\nTEST 4: Covariance P[theta,theta] shrinks after IMU update")
    ekf.initialize(0.0, 0.0, 0.0, pos_std=1.0, heading_std=0.5)
    p_before = ekf.covariance[2, 2]
    ekf.update_imu(0.0)
    p_after = ekf.covariance[2, 2]
    print(f"  P[2,2] before: {p_before:.6f}")
    print(f"  P[2,2] after : {p_after:.6f}  (must be smaller)")
    assert p_after < p_before
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 5: update_robot writes into Robot
    # ------------------------------------------------------------------
    print("\nTEST 5: update_robot() writes EKF state into Robot")
    from control.robot import Robot
    robot = Robot()
    ekf.initialize(150.0, 75.0, math.radians(45.0))
    ekf.update_robot(robot)
    rx, ry, rth = robot.pose
    print(f"  robot.pose: x={rx:.2f}, y={ry:.2f}, theta={math.degrees(rth):.2f} deg")
    assert abs(rx - 150.0) < 1e-9
    assert abs(ry - 75.0) < 1e-9
    assert abs(math.degrees(rth) - 45.0) < 1e-6
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 6: heading wraps at +-180
    # ------------------------------------------------------------------
    print("\nTEST 6: Heading wraps correctly across +-180 degrees")
    ekf.initialize(0.0, 0.0, math.radians(170.0))
    for _ in range(10):
        ekf.predict(speed=40.0, steer_angle=math.radians(20.0), dt=0.02)
    _, _, th = ekf.state
    print(f"  theta after turning past 180: {math.degrees(th):.2f} deg  (must be in [-180, 180])")
    assert -math.pi <= th <= math.pi
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 7: repr
    # ------------------------------------------------------------------
    print("\nTEST 7: __repr__")
    ekf.initialize(100.0, 200.0, math.radians(90.0))
    print(f"  {ekf}")
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL 7 TESTS PASSED")
    print("=" * 60)
