import math


class Robot:
    """
    Stores the car's current physical state.

    WHAT IS THIS?
    -------------
    Think of this as the car's "dashboard" in memory.
    It holds the latest known values for position, heading, and speed.

    WHO WRITES TO IT?
        - The EKF (estimation module) updates x, y, theta after each sensor fusion step.
        - The encoder reading updates speed.
        - The car controller updates steer_angle after each servo command.

    WHO READS FROM IT?
        - The steering controller reads x, y, theta.
        - The driving controller reads speed.
        - The main loop reads everything to make decisions.

    COORDINATE SYSTEM:
        - x, y are in centimetres, in the world/global frame.
        - theta is in radians. 0 = car is facing the +x direction.
          Positive theta = turned counter-clockwise (left).
        - speed is in cm/s. Positive = moving forward.
        - steer_angle is in degrees. 0 = wheels straight.
          Positive = steering left. TUNE SIGN ON REAL ROBOT.
    """

    def __init__(self):
        self._x: float = 0.0
        self._y: float = 0.0
        self._theta: float = 0.0
        self._speed: float = 0.0
        self._steer_angle: float = 0.0

    # ------------------------------------------------------------------
    # Properties (read-only access from outside; only updated via methods)
    # ------------------------------------------------------------------

    @property
    def x(self) -> float:
        return self._x

    @property
    def y(self) -> float:
        return self._y

    @property
    def theta(self) -> float:
        return self._theta

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def steer_angle(self) -> float:
        return self._steer_angle

    @property
    def pose(self) -> tuple:
        """Return (x, y, theta) as a single tuple."""
        return self._x, self._y, self._theta

    # ------------------------------------------------------------------
    # Update methods (called by EKF, encoder, and car controller)
    # ------------------------------------------------------------------

    def update_pose(self, x: float, y: float, theta: float) -> None:
        """
        Called by the EKF after each position estimation step.
        theta is normalised to [-pi, pi].
        """
        self._x = x
        self._y = y
        self._theta = math.atan2(math.sin(theta), math.cos(theta))

    def update_speed(self, speed: float) -> None:
        """Called after reading the wheel encoders."""
        self._speed = speed

    def update_steering(self, angle: float) -> None:
        """Called after sending a steering command to the servo."""
        self._steer_angle = angle

    def reset(self) -> None:
        """Reset all state to zero (e.g. at the start of a new run)."""
        self._x = 0.0
        self._y = 0.0
        self._theta = 0.0
        self._speed = 0.0
        self._steer_angle = 0.0

    def __repr__(self) -> str:
        return (
            f"Robot(x={self._x:.2f} cm, y={self._y:.2f} cm, "
            f"theta={math.degrees(self._theta):.1f}°, "
            f"speed={self._speed:.2f} cm/s, "
            f"steer={self._steer_angle:.1f}°)"
        )


# ----------------------------------------------------------------------
# TEST — run this file directly:  python control/robot.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    robot = Robot()
    print("Initial state:", robot)

    robot.update_pose(x=10.0, y=20.0, theta=1.5708)  # 1.5708 rad ≈ 90°
    robot.update_speed(35.0)
    robot.update_steering(-5.0)
    print("After update: ", robot)
    # Expected: x=10.00 cm, y=20.00 cm, theta≈90.0°, speed=35.00 cm/s, steer=-5.0°

    x, y, theta = robot.pose
    print(f"Pose tuple:    x={x}, y={y}, theta={math.degrees(theta):.1f}°")
    # Expected: x=10.0, y=20.0, theta=90.0°

    robot.reset()
    print("After reset:  ", robot)
    # Expected: all zeros
