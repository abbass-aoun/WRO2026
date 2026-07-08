from control.pid_controller import PIDController
from trajectory.base import TrajectoryBase


class DrivingPIDController(PIDController):
    """
    Decides how fast the car should go at each moment.

    WHAT IS THIS?
    -------------
    The car is following a path. Some parts of the path are straight,
    some are tight curves. This controller:
        1. Looks at how sharply the path is curving where the car is right now.
        2. Computes a SAFE target speed for that curvature.
        3. Measures the gap between target speed and actual speed.
        4. Feeds that gap into PID, which outputs a throttle correction.

    HOW IS TARGET SPEED COMPUTED?
    ------------------------------
    target_speed = base_speed / (1 + k_curve × |curvature|)

    Examples (base_speed=50 cm/s, k_curve=1.0):
        curvature = 0.00 (straight)  → target = 50 / 1.00 = 50.0 cm/s  ✓ full speed
        curvature = 0.05 (gentle)    → target = 50 / 1.05 = 47.6 cm/s
        curvature = 0.20 (moderate)  → target = 50 / 1.20 = 41.7 cm/s
        curvature = 1.00 (sharp)     → target = 50 / 2.00 = 25.0 cm/s  ✓ half speed

    SIGN CONVENTION FOR OUTPUT:
    ----------------------------
    Positive output → speed up (increase throttle).
    Negative output → slow down (reduce throttle or brake).

    TRACKING PATH PROGRESS:
    -----------------------
    After every compute() call, read self.current_s.
    Pass it as par_s on the NEXT call for an efficient closest-point search.
    """

    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        output_limits: tuple,
        windup_limit: float,
        base_speed: float,
        k_curve: float = 1.0,
    ):
        """
        Args:
            Kp, Ki, Kd    : PID gains.  TUNE ON REAL ROBOT.
            output_limits : (min, max) throttle output, e.g. (-1.0, 1.0).
            windup_limit  : cap on the integral term.
            base_speed    : maximum target speed on a straight (cm/s).
                            TUNE ON REAL ROBOT.
            k_curve       : how aggressively to slow down for curves.
                            Higher = slower in curves.
                            TUNE ON REAL ROBOT.
        """
        super().__init__(Kp, Ki, Kd, output_limits, windup_limit)
        self.base_speed: float = base_speed
        self.k_curve: float = k_curve
        self.current_s: float = 0.0  # updated every compute(); read by main loop

    def _target_speed(self, curvature: float) -> float:
        """Compute target speed from path curvature. Pure math — no hardware."""
        return self.base_speed / (1.0 + self.k_curve * abs(curvature))

    def compute_error(
        self,
        curr_x: float,
        curr_y: float,
        curr_theta: float,
        trajectory: TrajectoryBase,
        par_s: float,
        curr_speed: float = 0.0,
    ) -> float:
        """
        Compute the speed error (target_speed - current_speed).

        Args:
            curr_x, curr_y : car position in cm
            curr_theta     : car heading in radians
            trajectory     : any object that inherits TrajectoryBase
            par_s          : arc-length guess for the closest-point search (cm)
            curr_speed     : current forward speed from encoders (cm/s)

        Returns:
            Scalar error in cm/s.
            Positive → car is slower than target → PID says speed up.
            Negative → car is faster than target → PID says slow down.
        """
        # Step 1 — find the point on the path closest to the car.
        s = trajectory.find_closest(curr_x, curr_y, near_s=par_s)
        self.current_s = s

        # Step 2 — get curvature at that point.
        curvature = trajectory.get_curvature(s)

        # Step 3 — compute target speed and return the error.
        target = self._target_speed(curvature)
        return target - curr_speed

    def compute(
        self,
        curr_x: float,
        curr_y: float,
        curr_theta: float,
        trajectory: TrajectoryBase,
        par_s: float,
        curr_speed: float = 0.0,
    ) -> float:
        """
        Compute and return the throttle correction.

        Returns:
            Throttle correction (clamped to output_limits).
        """
        error = self.compute_error(curr_x, curr_y, curr_theta, trajectory, par_s, curr_speed)
        return self._compute(error)


# ----------------------------------------------------------------------
# TEST — run this file directly:  python control/driving_controller.py
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from trajectory.base import TrajectoryBase

    class _StraightPath(TrajectoryBase):
        """Mock: straight path, zero curvature."""
        def find_closest(self, x, y, near_s=None): return float(x)
        def get_point(self, s): return (s, 0.0)
        def get_tangent(self, s): return (1.0, 0.0)
        def get_curvature(self, s): return 0.0
        @property
        def total_length(self): return 200.0

    class _CurvedPath(TrajectoryBase):
        """Mock: curved path, constant curvature of 0.2."""
        def find_closest(self, x, y, near_s=None): return float(x)
        def get_point(self, s): return (s, 0.0)
        def get_tangent(self, s): return (1.0, 0.0)
        def get_curvature(self, s): return 0.2
        @property
        def total_length(self): return 200.0

    ctrl = DrivingPIDController(
        Kp=1.0, Ki=0.0, Kd=0.0,
        output_limits=(-1.0, 1.0),
        windup_limit=50.0,
        base_speed=50.0,
        k_curve=1.0,
    )

    print("=== Driving controller tests ===\n")

    # Test 1: straight path, car stopped → should accelerate to full base_speed
    error = ctrl.compute_error(50.0, 0.0, 0.0, _StraightPath(), 50.0, curr_speed=0.0)
    print(f"Test 1  straight path, curr_speed=0")
    print(f"  error = {error:.2f} cm/s  (expected: +50.0)\n")

    # Test 2: straight path, car already at base_speed → no correction needed
    ctrl.reset()
    error = ctrl.compute_error(50.0, 0.0, 0.0, _StraightPath(), 50.0, curr_speed=50.0)
    print(f"Test 2  straight path, curr_speed=50 (at target)")
    print(f"  error = {error:.2f} cm/s  (expected:  0.0)\n")

    # Test 3: curved path (curvature=0.2), car stopped
    # target = 50 / (1 + 1.0 * 0.2) = 50 / 1.2 = 41.67
    ctrl.reset()
    error = ctrl.compute_error(50.0, 0.0, 0.0, _CurvedPath(), 50.0, curr_speed=0.0)
    print(f"Test 3  curved path (k=0.2), curr_speed=0")
    print(f"  error = {error:.2f} cm/s  (expected: +41.67)\n")

    # Test 4: curved path, car going too fast → should slow down
    ctrl.reset()
    error = ctrl.compute_error(50.0, 0.0, 0.0, _CurvedPath(), 50.0, curr_speed=50.0)
    print(f"Test 4  curved path (k=0.2), curr_speed=50 (too fast)")
    print(f"  error = {error:.2f} cm/s  (expected: -8.33)\n")

    # Test 5: full compute() — PID output with Kp=1
    ctrl.reset()
    output = ctrl.compute(50.0, 0.0, 0.0, _StraightPath(), 50.0, curr_speed=30.0)
    print(f"Test 5  full compute, straight path, curr_speed=30")
    print(f"  throttle output = {output:.4f}  (expected ~= +1.0, clamped from +20)")
