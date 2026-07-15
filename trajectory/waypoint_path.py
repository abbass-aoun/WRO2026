"""
WHAT IS A WAYPOINT PATH?
------------------------
A series of (x, y) points connected by straight lines.

    (x0,y0) ──────── (x1,y1) ──────── (x2,y2) ── ...

WHEN IS IT USED?
    - Straight sections of the track where there are no pillars.
      The car just drives from one end of the section to the other.
    - As the reference line for Pure Pursuit (explained below).

WHAT IS PURE PURSUIT?
---------------------
Pure Pursuit is a steering technique. Instead of a full PID on cross-track
error, it does one simple thing:

    1. Find a point on the path a fixed distance AHEAD of the car (the
       "look-ahead point", typically 20-40 cm in front).
    2. Compute the steering angle needed to arc directly to that point.

For our WRO car, the steering controller already uses cross-track + heading
PID, so we don't implement pure pursuit as a separate steering law. But the
WaypointPath class provides the `look_ahead_point()` helper that the main
loop can optionally use to pick smarter target points.

CURVATURE NOTE:
    Straight lines have zero curvature, so the driving controller will
    always target base_speed on a WaypointPath. For cornering, use a
    BezierSegment instead (which has non-zero curvature).
"""

import math
import numpy as np
from trajectory.base import TrajectoryBase


class WaypointPath(TrajectoryBase):
    """
    A path made of straight-line segments connecting a list of waypoints.

    WHAT IT IS:
        The simplest possible path — just a series of (x, y) points.
        Used for straight track sections and Pure Pursuit reference lines.

    HOW THE CONTROLLERS USE IT:
        The steering controller will steer toward the path (CTE + heading).
        The driving controller will run at full base_speed (curvature = 0).
    """

    def __init__(self, waypoints: list):
        """
        Args:
            waypoints: list of (x, y) tuples in cm. Must have at least 2 points.
        """
        if len(waypoints) < 2:
            raise ValueError("WaypointPath needs at least 2 waypoints.")
        self._pts = [np.array(p, dtype=float) for p in waypoints]
        self._build_table()

    # ------------------------------------------------------------------
    # Internal: build arc-length table
    # ------------------------------------------------------------------

    def _build_table(self):
        """Compute cumulative straight-line distances between waypoints."""
        self._s = [0.0]
        for i in range(1, len(self._pts)):
            d = float(np.linalg.norm(self._pts[i] - self._pts[i - 1]))
            self._s.append(self._s[-1] + d)

    def _locate(self, s: float):
        """
        Return (segment_index, alpha) for arc-length s.
        alpha ∈ [0, 1] is how far along that segment.
        """
        s = max(0.0, min(s, self._s[-1]))
        for i in range(len(self._pts) - 1):
            if self._s[i] <= s <= self._s[i + 1]:
                seg_len = self._s[i + 1] - self._s[i]
                alpha = (s - self._s[i]) / seg_len if seg_len > 1e-12 else 0.0
                return i, alpha
        return len(self._pts) - 2, 1.0

    # ------------------------------------------------------------------
    # TrajectoryBase interface
    # ------------------------------------------------------------------

    def find_closest(self, x: float, y: float, near_s: float = None) -> float:
        """
        Project (x, y) onto the nearest line segment.

        We project the car position perpendicularly onto each segment
        (not just find the nearest waypoint), which is more accurate.
        """
        xy = np.array([x, y])
        best_s = 0.0
        best_dist = float('inf')

        for i in range(len(self._pts) - 1):
            A = self._pts[i]
            B = self._pts[i + 1]
            AB = B - A
            len_sq = float(np.dot(AB, AB))

            if len_sq < 1e-12:
                # Degenerate segment (zero length) — just check endpoint
                t = 0.0
            else:
                t = float(np.dot(xy - A, AB) / len_sq)
                t = max(0.0, min(1.0, t))

            closest = A + t * AB
            dist = float(np.linalg.norm(xy - closest))

            if dist < best_dist:
                best_dist = dist
                best_s = self._s[i] + t * (self._s[i + 1] - self._s[i])

        return best_s

    def get_point(self, s: float) -> tuple:
        """Return (x, y) on the path at arc-length s."""
        i, alpha = self._locate(s)
        p = self._pts[i] + alpha * (self._pts[i + 1] - self._pts[i])
        return (float(p[0]), float(p[1]))

    def get_tangent(self, s: float) -> tuple:
        """Return unit tangent direction at arc-length s."""
        i, _ = self._locate(s)
        d = self._pts[i + 1] - self._pts[i]
        norm = float(np.linalg.norm(d))
        if norm < 1e-9:
            return (1.0, 0.0)
        return (float(d[0] / norm), float(d[1] / norm))

    def get_curvature(self, s: float) -> float:
        """Always 0 — straight line segments have no curvature."""
        return 0.0

    @property
    def total_length(self) -> float:
        """Total path length in cm."""
        return self._s[-1]

    # ------------------------------------------------------------------
    # Pure Pursuit helper
    # ------------------------------------------------------------------

    def look_ahead_point(self, x: float, y: float, distance: float) -> tuple:
        """
        Return the (x, y) point on the path that is `distance` cm ahead
        of the closest point to (x, y).

        This is the look-ahead point used in Pure Pursuit steering.
        If the look-ahead distance goes past the end of the path,
        the endpoint is returned.

        Args:
            x, y     : current car position in cm
            distance : look-ahead distance in cm (TUNE ON REAL ROBOT)

        Returns:
            (x, y) look-ahead point in cm
        """
        s_closest = self.find_closest(x, y)
        s_ahead = min(s_closest + distance, self.total_length)
        return self.get_point(s_ahead)

    def __repr__(self):
        return (f"WaypointPath({len(self._pts)} points, "
                f"length={self.total_length:.2f} cm)")


# ===========================================================================
# TEST — run from project root:  python -m trajectory.waypoint_path
# ===========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("TEST 1: Simple 2-point straight path (0,0) -> (100,0)")
    print("=" * 55)

    path = WaypointPath([(0, 0), (100, 0)])
    print(path)
    print(f"  total_length      : {path.total_length:.2f} cm   (expected 100)")

    p0 = path.get_point(0)
    pL = path.get_point(100)
    pm = path.get_point(50)
    print(f"  point at s=0      : {p0}   (expected (0,0))")
    print(f"  point at s=100    : {pL} (expected (100,0))")
    print(f"  point at s=50     : {pm}  (expected (50,0))")

    t = path.get_tangent(50)
    print(f"  tangent anywhere  : ({t[0]:.3f}, {t[1]:.3f})  (expected (1,0))")

    k = path.get_curvature(50)
    print(f"  curvature         : {k:.3f}            (expected 0.0)")

    # Car is above the path at (50, 8)
    s = path.find_closest(50, 8)
    print(f"  closest to (50,8) : s={s:.2f} cm   (expected ~50)")

    lap = path.look_ahead_point(30, 5, 25)
    print(f"  look-ahead 25cm from (30,5): {lap}  (expected ~(55, 0))")

    print()
    print("=" * 55)
    print("TEST 2: L-shaped path (0,0)->(100,0)->(100,100)")
    print("=" * 55)

    path2 = WaypointPath([(0, 0), (100, 0), (100, 100)])
    print(path2)
    print(f"  total_length      : {path2.total_length:.2f} cm  (expected 200)")

    # At s=100 we should be at the corner (100,0)
    corner = path2.get_point(100)
    print(f"  point at s=100    : ({corner[0]:.2f}, {corner[1]:.2f})  (expected 100,0)")

    # At s=150 we should be halfway up the second segment → (100, 50)
    mid2 = path2.get_point(150)
    print(f"  point at s=150    : ({mid2[0]:.2f}, {mid2[1]:.2f}) (expected 100,50)")

    # Tangent on second segment should point in +y
    t2 = path2.get_tangent(150)
    print(f"  tangent at s=150  : ({t2[0]:.3f}, {t2[1]:.3f}) (expected (0,1))")

    # Closest to a point beside the vertical segment
    s2 = path2.find_closest(95, 60)
    print(f"  closest to (95,60): s={s2:.2f} cm  (expected ~160)")
