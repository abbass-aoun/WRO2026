import math
from control.pid_controller import PIDController
from trajectory.base import TrajectoryBase


class SteeringPIDController(PIDController):
    """
    Decides how much to turn the steering wheel at each moment.

    WHAT IS THIS?
    -------------
    The car is somewhere on the track. There is a path it should follow.
    This controller measures two things:
        1. Cross-track error (CTE):
               How far LEFT or RIGHT is the car from the path?
               Example: car is 5 cm to the left → CTE = +5
        2. Heading error (HE):
               How much is the car's nose pointing away from the path direction?
               Example: car is aimed 15° more to the left → HE = +0.26 rad

    It combines them:  error = CTE + heading_weight × HE
    Then feeds that error into a PID controller, which outputs a steering angle.

    WHY BOTH?
    ---------
    CTE alone reacts too late (you've already drifted).
    Heading error alone lets you drift without correcting position.
    Together, the car corrects early (heading) and doesn't accumulate drift (CTE).

    SIGN CONVENTION:
    ----------------
    Positive error → car is to the LEFT of / pointing left of the path.
    Whether positive output means "steer left" or "steer right" depends on
    how the servo is mounted.  TUNE THE SIGN OF Kp ON THE REAL ROBOT.

    TRACKING PATH PROGRESS:
    -----------------------
    After every compute() call, read self.current_s.
    This is the arc-length of the closest point found on the path.
    Pass it as par_s on the NEXT call so the search starts nearby
    (much faster than scanning the whole path from scratch).
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        output_limits: tuple,
        windup_limit: float,
        heading_weight: float = 1.0,
    ):
        """
        Args:
            Kp, Ki, Kd     : PID gains.  TUNE ON REAL ROBOT.
            output_limits  : (min, max) steering angle in degrees, e.g. (-27, 27).
            windup_limit   : cap on the integral term to prevent windup.
            heading_weight : how strongly heading error pulls vs. CTE.
                             0.0 = ignore heading, pure cross-track only.
                             TUNE ON REAL ROBOT.
        """
        super().__init__(Kp, Ki, Kd, output_limits, windup_limit)
        self.heading_weight: float = heading_weight
        self.current_s: float = 0.0  # updated every compute(); read by main loop
        self.last_cte: float = 0.0   # raw cross-track error from last compute(); read by logger

    def compute_error(
        self,
        curr_x: float,
        curr_y: float,
        curr_theta: float,
        trajectory: TrajectoryBase,
        par_s: float,
    ) -> float:
        """
        Compute the combined steering error.

        Args:
            curr_x, curr_y : car position in cm
            curr_theta     : car heading in radians
            trajectory     : any object that inherits TrajectoryBase
            par_s          : arc-length guess for the closest-point search (cm)

        Returns:
            Scalar error (positive = car is left of / pointing left of path).
        """
        # Step 1 — find the point on the path closest to the car.
        s = trajectory.find_closest(curr_x, curr_y, near_s=par_s)
        self.current_s = s

        px, py = trajectory.get_point(s)
        tx, ty = trajectory.get_tangent(s)  # unit vector along path direction

        # Step 2 — cross-track error (signed perpendicular distance).
        # Vector from closest path point → car:
        ex = curr_x - px
        ey = curr_y - py
        # Left-facing normal of tangent (tx, ty) is (-ty, tx).
        # CTE = dot((ex, ey), (-ty, tx)).
        # Positive = car is to the left of the path.
        cte = (-ty) * ex + tx * ey
        self.last_cte = cte

        # Step 3 — heading error (difference between car and path direction).
        path_angle = math.atan2(ty, tx)
        he = math.atan2(
            math.sin(curr_theta - path_angle),
            math.cos(curr_theta - path_angle),
        )  # normalised to [-π, π]; positive = car points more left than path

        return cte + self.heading_weight * he

    def compute(
        self,
        curr_x: float,
        curr_y: float,
        curr_theta: float,
        trajectory: TrajectoryBase,
        par_s: float,
    ) -> float:
        """
        Compute and return the steering angle correction.

        Returns:
            Steering angle correction in degrees (clamped to output_limits).
        """
        error = self.compute_error(curr_x, curr_y, curr_theta, trajectory, par_s)
        return self._compute(error)


# ----------------------------------------------------------------------
# TEST — run this file directly:  python control/steering_controller.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from trajectory.base import TrajectoryBase

    class _StraightPath(TrajectoryBase):
        """Mock: a straight horizontal path along y=0, from x=0 to x=200 cm."""
        def find_closest(self, x, y, near_s=None):
            return float(x)          # arc-length = x for a horizontal path
        def get_point(self, s):
            return (s, 0.0)
        def get_tangent(self, s):
            return (1.0, 0.0)        # always pointing in +x direction
        def get_curvature(self, s):
            return 0.0
        @property
        def total_length(self):
            return 200.0

    path = _StraightPath()
    ctrl = SteeringPIDController(
        Kp=1.0, Ki=0.0, Kd=0.0,
        output_limits=(-27, 27),
        windup_limit=10.0,
        heading_weight=1.0,
    )

    print("=== Steering controller tests ===\n")

    # Test 1: car is 5 cm LEFT of the path, heading aligned → expect positive error
    error = ctrl.compute_error(50.0, 5.0, 0.0, path, 50.0)
    print(f"Test 1  car 5 cm left, heading aligned")
    print(f"  error = {error:.4f}  (expected: +5.0)\n")

    # Test 2: car is 3 cm RIGHT of the path, heading aligned → expect negative error
    ctrl.reset()
    error = ctrl.compute_error(50.0, -3.0, 0.0, path, 50.0)
    print(f"Test 2  car 3 cm right, heading aligned")
    print(f"  error = {error:.4f}  (expected: -3.0)\n")

    # Test 3: car is ON the path but heading 10° left → pure heading error
    ctrl.reset()
    he_rad = math.radians(10.0)
    error = ctrl.compute_error(50.0, 0.0, he_rad, path, 50.0)
    print(f"Test 3  car on path, heading 10° left")
    print(f"  error = {error:.4f}  (expected: +{he_rad:.4f} ~= +0.1745)\n")

    # Test 4: full compute() — PID output with Kp=1
    ctrl.reset()
    output = ctrl.compute(50.0, 5.0, 0.0, path, 50.0)
    print(f"Test 4  full compute, car 5 cm left, Kp=1")
    print(f"  steering output = {output:.4f} deg  (expected ~= +5.0 deg, clamped to +-27)")
