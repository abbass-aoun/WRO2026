"""
WHAT IS THE TRAJECTORY BUILDER?
---------------------------------
This is the "traffic controller" of the trajectory module.

The main loop never creates BezierSegment or WaypointPath directly.
Instead it calls TrajectoryBuilder methods and gets back a ready-to-use
TrajectoryBase object.  The main loop doesn't need to know whether the
result is a Bezier curve or a straight line — the controllers work the
same either way.

PILLAR COLOR CONVENTION (matches vision team's output):
    RED   = 0  →  pass on the RIGHT side of the pillar
    GREEN = 1  →  pass on the LEFT  side of the pillar

COORDINATE SYSTEM:
    x, y in centimetres, same world frame as robot.py.
    theta in radians (0 = facing +x, counter-clockwise positive).
"""

import math
import numpy as np
from trajectory.bezier import BezierSegment, BezierPath
from trajectory.waypoint_path import WaypointPath

# Pillar color constants — must match whatever the vision team sends
RED   = 0
GREEN = 1

# How far to the side the bypass waypoint is placed from the pillar centre (cm).
# Must be > pillar radius (2.5 cm) + half car width + safety margin.
# TUNE ON REAL ROBOT once you know the car's exact width.
PILLAR_CLEARANCE_CM = 20.0


class TrajectoryBuilder:
    """
    Factory class that builds the correct TrajectoryBase object
    for each situation the car encounters on the track.

    ALL METHODS ARE STATIC — no instance needed.
    Usage:  path = TrajectoryBuilder.straight(...)
    """

    # ------------------------------------------------------------------
    # 1. Straight section (no pillars in the way)
    # ------------------------------------------------------------------

    @staticmethod
    def straight(start_x: float, start_y: float,
                 end_x: float, end_y: float) -> WaypointPath:
        """
        Build a straight-line path from start to end.

        WHEN TO USE:
            When the car is in a straight section and no pillar is visible.
            The car will drive at full base_speed (curvature = 0).

        Args:
            start_x, start_y : entry point of the section (cm)
            end_x,   end_y   : exit point of the section  (cm)

        Returns:
            WaypointPath with 2 points.
        """
        return WaypointPath([(start_x, start_y), (end_x, end_y)])

    # ------------------------------------------------------------------
    # 2. Corner (90-degree turn)
    # ------------------------------------------------------------------

    @staticmethod
    def corner(start_x: float, start_y: float, start_theta: float,
               turn_direction: int, radius: float = 50.0) -> BezierSegment:
        """
        Build a smooth 90-degree corner as a Bezier curve.

        WHEN TO USE:
            When the car is entering a corner section.
            The Bezier curve will have non-zero curvature, so the driving
            controller will automatically slow the car down.

        HOW IT WORKS:
            The end point and end tangent are computed from the geometry:
              - Start tangent  = car's current heading
              - End tangent    = start tangent rotated 90° in turn_direction
              - End point      = start + radius * (turn normal) + radius * end_tangent

        Args:
            start_x, start_y : entry point of the corner (cm)
            start_theta      : car heading at entry (radians)
            turn_direction   : +1 = left turn (CCW), -1 = right turn (CW)
            radius           : approximate turn radius in cm. TUNE ON REAL ROBOT.

        Returns:
            BezierSegment for the corner arc.
        """
        v0 = np.array([math.cos(start_theta), math.sin(start_theta)])

        # Rotate start tangent by 90° × turn_direction to get end tangent
        angle = turn_direction * (math.pi / 2.0)
        ca, sa = math.cos(angle), math.sin(angle)
        v3 = np.array([ca * v0[0] - sa * v0[1],
                       sa * v0[0] + ca * v0[1]])

        # Perpendicular direction (toward the inside of the turn)
        perp = np.array([-turn_direction * v0[1],
                          turn_direction * v0[0]])

        P0 = np.array([start_x, start_y])
        # Circular-arc endpoint: P3 = P0 + R*v0 + R*perp.
        # This puts P3 diagonally (45° between heading and turn direction),
        # keeping Bezier control point P1 inside the track rather than
        # pushing it toward the outer wall.
        P3 = P0 + radius * v0 + radius * perp

        return BezierSegment.from_endpoints_and_tangents(P0, P3, v0, v3)

    # ------------------------------------------------------------------
    # 3. Pillar swerve (obstacle avoidance)
    # ------------------------------------------------------------------

    @staticmethod
    def pillar_swerve(
        start_x: float, start_y: float, start_theta: float,
        pillar_x: float, pillar_y: float, pillar_color: int,
        end_x: float, end_y: float, end_theta: float,
        clearance: float = PILLAR_CLEARANCE_CM,
    ) -> BezierPath:
        """
        Build a 2-segment Bezier path that bypasses a single pillar.

        WHEN TO USE:
            When the vision team detects a pillar and sends its (x, y)
            and colour.  Call this immediately and switch to the returned
            path.

        HOW IT WORKS:
            It places a "bypass waypoint" 'clearance' cm to the correct
            side of the pillar, then builds:
                Segment 1:  start  → bypass waypoint
                Segment 2:  bypass → end (reconnect to track centreline)

            The tangents at all three points are aligned with the driving
            direction (start_theta), giving a smooth S-curve.

        PILLAR SIDE RULE (WRO rules):
            RED   pillar (color=0) → pass on the RIGHT
            GREEN pillar (color=1) → pass on the LEFT

        Args:
            start_x/y, start_theta : car's current position + heading
            pillar_x/y, pillar_color: from vision team
            end_x/y, end_theta     : target reconnect point on centreline
            clearance              : side offset from pillar centre (cm)
                                     TUNE ON REAL ROBOT

        Returns:
            BezierPath with 2 segments.
        """
        # Path tangent (direction of travel) at the pillar's location
        # We approximate it as the car's current heading.
        tx = math.cos(start_theta)
        ty = math.sin(start_theta)

        # Left-facing normal of the path direction: (-ty, tx)
        # side = +1 → bypass to the LEFT  (GREEN)
        # side = -1 → bypass to the RIGHT (RED)
        side = 1.0 if pillar_color == GREEN else -1.0

        # Bypass waypoint: shift the pillar position sideways
        bypass_x = pillar_x + side * (-ty) * clearance
        bypass_y = pillar_y + side * ( tx) * clearance

        # Start and end tangents: known from heading
        v_start = np.array([tx, ty])
        v_end   = np.array([math.cos(end_theta), math.sin(end_theta)])

        # Bypass tangent: WRO 2025 team formula
        #   t = (p_next - p_curr) + kc * |p_next - p_curr| * (p_curr - p_prev)
        # Blends the forward direction with a backward contribution to reduce
        # curvature at the waypoint.  from_endpoints_and_tangents normalises it.
        v_bypass = TrajectoryBuilder._blend_tangent(
            (start_x, start_y), (bypass_x, bypass_y), (end_x, end_y)
        )

        seg1 = BezierSegment.from_endpoints_and_tangents(
            (start_x, start_y), (bypass_x, bypass_y), v_start, v_bypass
        )
        seg2 = BezierSegment.from_endpoints_and_tangents(
            (bypass_x, bypass_y), (end_x, end_y), v_bypass, v_end
        )

        return BezierPath([seg1, seg2])

    # ------------------------------------------------------------------
    # Internal: bypass-point tangent formula (WRO 2025 team)
    # ------------------------------------------------------------------

    @staticmethod
    def _blend_tangent(p_prev, p_curr, p_next, kc: float = 0.0085) -> np.ndarray:
        """
        Compute the tangent direction at p_curr using the formula:

            t = (p_next - p_curr)  +  kc * |p_next - p_curr| * (p_curr - p_prev)

        This blends the forward direction (toward p_next) with a small
        contribution from the backward direction (away from p_prev), which
        reduces curvature at the waypoint.

        kc = 0.0085 is the constant from the WRO 2025 team code.
        The result is passed to from_endpoints_and_tangents which normalises it,
        so only the direction matters.
        """
        p_prev = np.asarray(p_prev, dtype=float)
        p_curr = np.asarray(p_curr, dtype=float)
        p_next = np.asarray(p_next, dtype=float)
        tdv = p_next - p_curr
        return tdv + kc * np.linalg.norm(tdv) * (p_curr - p_prev)

    # ------------------------------------------------------------------
    # 4. Parking approach path
    # ------------------------------------------------------------------

    @staticmethod
    def parking_approach(start_x: float, start_y: float, start_theta: float,
                         lot_x: float, lot_y: float,
                         lot_theta: float) -> BezierSegment:
        """
        Build a smooth path from the car's current position to the
        entry of the parking lot.

        WHEN TO USE:
            After completing lap 3, when the vision team has confirmed
            the parking lot position.

        Args:
            start_x/y, start_theta : car's position + heading after lap 3
            lot_x/y, lot_theta     : parking lot entry point + entry heading
                                     (provided by vision team)

        Returns:
            BezierSegment from car to parking lot entry.
        """
        v0 = np.array([math.cos(start_theta), math.sin(start_theta)])
        v3 = np.array([math.cos(lot_theta), math.sin(lot_theta)])
        return BezierSegment.from_endpoints_and_tangents(
            (start_x, start_y), (lot_x, lot_y), v0, v3
        )


# ===========================================================================
# TEST — run from project root:  python -m trajectory.builder
# ===========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("TEST 1: straight()")
    print("=" * 55)
    path = TrajectoryBuilder.straight(0, 0, 200, 0)
    print(path)
    print(f"  length={path.total_length:.1f} cm  (expected 200)")
    print(f"  tangent: {path.get_tangent(100)}  (expected (1,0))")

    print()
    print("=" * 55)
    print("TEST 2: corner() — left turn from (0,0) heading right")
    print("=" * 55)
    c = TrajectoryBuilder.corner(0, 0, start_theta=0.0,
                                  turn_direction=+1, radius=50)
    print(c)
    t_start = c.get_tangent(0)
    t_end   = c.get_tangent(c.total_length)
    print(f"  tangent at start: ({t_start[0]:.2f}, {t_start[1]:.2f})  (expected ~1,0)")
    print(f"  tangent at end  : ({t_end[0]:.2f}, {t_end[1]:.2f})  (expected ~0,1 for left turn)")
    k_mid = c.get_curvature(c.total_length / 2)
    print(f"  curvature at mid: {k_mid:.5f}  (expected > 0)")

    print()
    print("=" * 55)
    print("TEST 3: pillar_swerve() — RED pillar (pass right)")
    print("=" * 55)
    # Car going in +x, pillar at (100, 0), RED = pass right = go to -y side
    red_path = TrajectoryBuilder.pillar_swerve(
        start_x=0, start_y=0, start_theta=0.0,
        pillar_x=100, pillar_y=0, pillar_color=RED,
        end_x=200, end_y=0, end_theta=0.0,
    )
    print(red_path)
    # The bypass waypoint should be at (100, -20)
    # The path's minimum y should be negative (car went right = went to -y side)
    s_vals = [red_path.total_length * i / 50 for i in range(51)]
    ys = [red_path.get_point(s)[1] for s in s_vals]
    print(f"  bypass y  = {min(ys):.1f} cm  (expected ~-20, car passed RIGHT)")

    print()
    print("=" * 55)
    print("TEST 4: pillar_swerve() — GREEN pillar (pass left)")
    print("=" * 55)
    green_path = TrajectoryBuilder.pillar_swerve(
        start_x=0, start_y=0, start_theta=0.0,
        pillar_x=100, pillar_y=0, pillar_color=GREEN,
        end_x=200, end_y=0, end_theta=0.0,
    )
    print(green_path)
    s_vals2 = [green_path.total_length * i / 50 for i in range(51)]
    ys2 = [green_path.get_point(s)[1] for s in s_vals2]
    print(f"  bypass y  = {max(ys2):.1f} cm  (expected ~+20, car passed LEFT)")

    print()
    print("=" * 55)
    print("TEST 5: Both swerves end near (200, 0)")
    print("=" * 55)
    red_end = red_path.get_point(red_path.total_length)
    grn_end = green_path.get_point(green_path.total_length)
    print(f"  RED   end: ({red_end[0]:.1f}, {red_end[1]:.1f})   (expected ~200,0)")
    print(f"  GREEN end: ({grn_end[0]:.1f}, {grn_end[1]:.1f})  (expected ~200,0)")
