"""
WHAT ARE BEZIER CURVES?
-----------------------
A Bezier curve is a smooth mathematical curve defined by "control points."
For a CUBIC Bezier (the type we use) there are exactly 4 control points:

    P0 ──► P1        P2 ◄── P3
    (start)  *      *  (end)
              *    *
               curve

P0 = start point  (the car is here)
P3 = end point    (where the car needs to go)
P1 = pulls the curve toward the start tangent direction
P2 = pulls the curve toward the end tangent direction

The car NEVER passes through P1 or P2 — they only shape the curve.

The formula is:
    B(t) = (1-t)³·P0  +  3(1-t)²t·P1  +  3(1-t)t²·P2  +  t³·P3
    where t goes from 0 (start) to 1 (end)

WHY DO WE USE THEM?
-------------------
When a pillar is detected, we need a smooth path that:
  - starts where the car currently is (P0)
  - passes on the correct SIDE of the pillar
  - ends back on the track centerline (P3)
  - has no sharp kinks (steering wheel doesn't jerk)

Bezier curves give us all of that with simple math.

ARC-LENGTH PARAMETERIZATION
----------------------------
The controllers use arc-length 's' (distance in cm along the curve), not
parameter 't'. The mapping from s → t is NOT linear (the curve might move
faster or slower in t at different points). We pre-compute a lookup table
at __init__ time so every call to get_point(s) / get_tangent(s) is fast.
"""

import math
import numpy as np
from trajectory.base import TrajectoryBase


class BezierSegment(TrajectoryBase):
    """
    A single cubic Bezier curve segment.

    WHAT IT IS:
        One smooth curve from P0 to P3, shaped by two internal control
        points P1 and P2.  Think of it as one "swerve" manoeuvre.

    HOW TO CREATE ONE:
        Use the class method `from_endpoints_and_tangents()` — you give it
        a start point, end point, and the direction the curve should be
        pointing at each end.  It figures out P1 and P2 automatically.

    HOW THE CONTROLLERS USE IT:
        They call find_closest(), get_point(), get_tangent(), get_curvature()
        — all defined in TrajectoryBase and implemented below.
    """

    N_SAMPLES = 300  # resolution of the arc-length lookup table

    def __init__(self, P0, P1, P2, P3):
        """
        Args:
            P0, P1, P2, P3: the four control points, each an (x, y) pair in cm.
        """
        self._P = [np.array(p, dtype=float) for p in (P0, P1, P2, P3)]
        self._build_table()

    # ------------------------------------------------------------------
    # Factory method: the clean way to construct a segment
    # ------------------------------------------------------------------

    @classmethod
    def from_endpoints_and_tangents(cls, P0, P3, v0, v3):
        """
        Build a segment from two endpoints and the desired tangent directions.

        How P1 and P2 are computed:
            d  = distance(P0, P3) / 3
            P1 = P0 + d * v0      (one third of the way, along start tangent)
            P2 = P3 - d * v3      (one third back from end, along end tangent)

        This guarantees the curve leaves P0 in direction v0 and
        arrives at P3 in direction v3 — no kinks, perfectly smooth.

        Args:
            P0, P3 : (x, y) start and end points in cm
            v0, v3 : (tx, ty) tangent direction vectors (need not be unit)
        """
        P0 = np.array(P0, dtype=float)
        P3 = np.array(P3, dtype=float)
        v0 = np.array(v0, dtype=float)
        v3 = np.array(v3, dtype=float)
        v0 = v0 / np.linalg.norm(v0)
        v3 = v3 / np.linalg.norm(v3)

        d = np.linalg.norm(P3 - P0) / 3.0
        P1 = P0 + d * v0
        P2 = P3 - d * v3

        return cls(P0, P1, P2, P3)

    # ------------------------------------------------------------------
    # Internal math — evaluating the curve at parameter t ∈ [0, 1]
    # ------------------------------------------------------------------

    def _point_at_t(self, t):
        """B(t) — position on curve."""
        P = self._P
        mt = 1.0 - t
        return mt**3 * P[0] + 3*mt**2*t * P[1] + 3*mt*t**2 * P[2] + t**3 * P[3]

    def _deriv1_at_t(self, t):
        """B'(t) — first derivative (velocity vector, NOT unit)."""
        P = self._P
        mt = 1.0 - t
        return 3*mt**2*(P[1]-P[0]) + 6*mt*t*(P[2]-P[1]) + 3*t**2*(P[3]-P[2])

    def _deriv2_at_t(self, t):
        """B''(t) — second derivative (acceleration vector)."""
        P = self._P
        mt = 1.0 - t
        return 6*mt*(P[2] - 2*P[1] + P[0]) + 6*t*(P[3] - 2*P[2] + P[1])

    # ------------------------------------------------------------------
    # Arc-length lookup table (built once at construction)
    # ------------------------------------------------------------------

    def _build_table(self):
        """
        Sample the curve at N_SAMPLES points, compute cumulative arc lengths.

        Stores:
            _t_samples  : array of t values [0 .. 1]
            _s_samples  : cumulative arc length at each t (in cm)
            _pts_cache  : pre-computed (x,y) at each sample (for find_closest)
        """
        N = self.N_SAMPLES
        ts = np.linspace(0.0, 1.0, N)
        P = self._P
        mt = 1.0 - ts  # shape (N,)

        # Vectorised Bezier evaluation — no loop needed
        pts = (mt**3)[:, None]*P[0] + \
              (3*mt**2*ts)[:, None]*P[1] + \
              (3*mt*ts**2)[:, None]*P[2] + \
              (ts**3)[:, None]*P[3]

        diffs = np.diff(pts, axis=0)
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])

        self._t_samples  = ts
        self._s_samples  = np.concatenate([[0.0], np.cumsum(seg_lens)])
        self._pts_cache  = pts

    def _s_to_t(self, s: float) -> float:
        """Convert arc-length s → curve parameter t via linear interpolation."""
        s = float(np.clip(s, 0.0, self._s_samples[-1]))
        idx = int(np.searchsorted(self._s_samples, s, side='right')) - 1
        idx = max(0, min(idx, len(self._s_samples) - 2))
        s0, s1 = self._s_samples[idx], self._s_samples[idx + 1]
        t0, t1 = self._t_samples[idx], self._t_samples[idx + 1]
        if abs(s1 - s0) < 1e-12:
            return float(t0)
        alpha = (s - s0) / (s1 - s0)
        return float(t0 + alpha * (t1 - t0))

    # ------------------------------------------------------------------
    # TrajectoryBase interface — used by the controllers
    # ------------------------------------------------------------------

    def find_closest(self, x: float, y: float, near_s: float = None) -> float:
        """
        Return arc-length s of the point on the curve closest to (x, y).

        If near_s is given, only searches a window around that position
        (much faster in the main loop when the car hasn't moved far).
        """
        if near_s is not None:
            near_idx = int(np.searchsorted(self._s_samples, near_s))
            window = 40
            lo = max(0, near_idx - window)
            hi = min(len(self._s_samples), near_idx + window + 1)
            pts = self._pts_cache[lo:hi]
            dists = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
            idx = lo + int(np.argmin(dists))
        else:
            dists = np.hypot(self._pts_cache[:, 0] - x, self._pts_cache[:, 1] - y)
            idx = int(np.argmin(dists))
        return float(self._s_samples[idx])

    def get_point(self, s: float) -> tuple:
        """Return (x, y) on the curve at arc-length s."""
        t = self._s_to_t(s)
        p = self._point_at_t(t)
        return (float(p[0]), float(p[1]))

    def get_tangent(self, s: float) -> tuple:
        """Return unit tangent (tx, ty) at arc-length s."""
        t = self._s_to_t(s)
        d = self._deriv1_at_t(t)
        norm = math.hypot(float(d[0]), float(d[1]))
        if norm < 1e-9:
            return (1.0, 0.0)
        return (float(d[0] / norm), float(d[1] / norm))

    def get_curvature(self, s: float) -> float:
        """
        Return curvature κ = |B'×B''| / |B'|³ at arc-length s.

        κ = 0   → straight
        κ = 0.1 → radius = 10 cm (very tight for this robot)
        κ large → very sharp turn → driving controller slows down
        """
        t = self._s_to_t(s)
        d1 = self._deriv1_at_t(t)
        d2 = self._deriv2_at_t(t)
        # 2D cross product: d1.x·d2.y − d1.y·d2.x
        # Sign convention: positive = CCW (left) turn, negative = CW (right) turn.
        cross = float(d1[0]*d2[1] - d1[1]*d2[0])
        speed_sq = float(d1[0]**2 + d1[1]**2)
        speed_cubed = speed_sq ** 1.5
        if speed_cubed < 1e-9:
            return 0.0
        return cross / speed_cubed

    @property
    def total_length(self) -> float:
        """Total arc length of this segment in cm."""
        return float(self._s_samples[-1])

    def __repr__(self):
        return (f"BezierSegment(P0={self._P[0].tolist()}, "
                f"P3={self._P[3].tolist()}, length={self.total_length:.2f} cm)")


# ===========================================================================

class BezierPath(TrajectoryBase):
    """
    A chain of BezierSegment objects connected end-to-end.

    WHAT IT IS:
        Two (or more) Bezier segments joined together into one continuous path.
        The arc-length parameter 's' runs continuously from 0 across all segments.

    WHEN IS IT USED:
        For a pillar swerve: the path goes
            car position → bypass point beside the pillar → back to centerline
        That needs 2 segments (one to swerve out, one to swerve back).

    HOW IT WORKS INTERNALLY:
        Each segment has its own arc-length [0 .. seg.total_length].
        BezierPath stores cumulative offsets so it can convert a global s
        to (which segment, local s within that segment).
    """

    def __init__(self, segments: list):
        """
        Args:
            segments: list of BezierSegment objects, in order.
        """
        if not segments:
            raise ValueError("BezierPath needs at least one segment.")
        self._segs = segments
        # Cumulative offsets: _offsets[i] = arc-length at start of segment i
        self._offsets = [0.0]
        for seg in segments:
            self._offsets.append(self._offsets[-1] + seg.total_length)

    def _locate(self, s: float):
        """
        Return (segment_index, local_s) for a global arc-length s.
        local_s is the arc-length within that specific segment.
        """
        s = float(np.clip(s, 0.0, self.total_length))
        for i in range(len(self._segs) - 1, -1, -1):
            if s >= self._offsets[i]:
                return i, s - self._offsets[i]
        return 0, 0.0

    def find_closest(self, x: float, y: float, near_s: float = None) -> float:
        """Search all segments, return the global s of the closest point."""
        best_global_s = 0.0
        best_dist = float('inf')
        for i, seg in enumerate(self._segs):
            # Pass a local near_s hint if available
            local_near = None
            if near_s is not None:
                local_near = max(0.0, near_s - self._offsets[i])
            local_s = seg.find_closest(x, y, near_s=local_near)
            pt = seg.get_point(local_s)
            dist = math.hypot(pt[0] - x, pt[1] - y)
            if dist < best_dist:
                best_dist = dist
                best_global_s = self._offsets[i] + local_s
        return best_global_s

    def get_point(self, s: float) -> tuple:
        i, local_s = self._locate(s)
        return self._segs[i].get_point(local_s)

    def get_tangent(self, s: float) -> tuple:
        i, local_s = self._locate(s)
        return self._segs[i].get_tangent(local_s)

    def get_curvature(self, s: float) -> float:
        i, local_s = self._locate(s)
        return self._segs[i].get_curvature(local_s)

    @property
    def total_length(self) -> float:
        return self._offsets[-1]

    def __repr__(self):
        return (f"BezierPath({len(self._segs)} segments, "
                f"total_length={self.total_length:.2f} cm)")


# ===========================================================================
# TEST — run from project root:  python -m trajectory.bezier
# ===========================================================================
if __name__ == "__main__":
    print("=" * 55)
    print("TEST 1: Straight segment (P0->P3 both pointing right)")
    print("=" * 55)

    seg = BezierSegment.from_endpoints_and_tangents(
        P0=(0, 0), P3=(100, 0), v0=(1, 0), v3=(1, 0)
    )
    print(seg)
    print(f"  total_length     : {seg.total_length:.2f} cm   (expected ~100)")
    p0 = seg.get_point(0)
    pL = seg.get_point(seg.total_length)
    pm = seg.get_point(seg.total_length / 2)
    print(f"  point at s=0     : ({p0[0]:.2f}, {p0[1]:.2f})   (expected 0,0)")
    print(f"  point at s=L     : ({pL[0]:.2f}, {pL[1]:.2f}) (expected ~100,0)")
    print(f"  point at s=L/2   : ({pm[0]:.2f}, {pm[1]:.2f})   (expected ~50,0)")
    t0 = seg.get_tangent(0)
    tL = seg.get_tangent(seg.total_length)
    print(f"  tangent at s=0   : ({t0[0]:.3f}, {t0[1]:.3f})  (expected 1,0)")
    print(f"  tangent at s=L   : ({tL[0]:.3f}, {tL[1]:.3f})  (expected 1,0)")
    k = seg.get_curvature(seg.total_length / 2)
    print(f"  curvature mid    : {k:.6f}           (expected ~0.0)")
    s_near = seg.find_closest(50, 5)
    print(f"  closest to (50,5): s={s_near:.2f} cm    (expected ~50)")

    print()
    print("=" * 55)
    print("TEST 2: Curved segment (ends pointing down-right)")
    print("=" * 55)
    seg2 = BezierSegment.from_endpoints_and_tangents(
        P0=(0, 0), P3=(100, -30), v0=(1, 0), v3=(1, 0)
    )
    print(seg2)
    # NOTE: s=L/2 is the inflection point of this symmetric S-curve, so curvature
    # is mathematically 0 there. Check at L/4 instead (first half of the swerve).
    k2 = seg2.get_curvature(seg2.total_length / 4)
    print(f"  curvature at L/4 : {k2:.6f}  (expected > 0, curve bends here)")

    print()
    print("=" * 55)
    print("TEST 3: BezierPath (2 segments joined)")
    print("=" * 55)
    seg_a = BezierSegment.from_endpoints_and_tangents(
        P0=(0, 0), P3=(50, -20), v0=(1, 0), v3=(1, 0)
    )
    seg_b = BezierSegment.from_endpoints_and_tangents(
        P0=(50, -20), P3=(100, 0), v0=(1, 0), v3=(1, 0)
    )
    path = BezierPath([seg_a, seg_b])
    print(path)
    print(f"  total_length     : {path.total_length:.2f} cm")
    pt_mid = path.get_point(path.total_length / 2)
    print(f"  point at s=L/2   : ({pt_mid[0]:.2f}, {pt_mid[1]:.2f})")
    pt_end = path.get_point(path.total_length)
    print(f"  point at s=L     : ({pt_end[0]:.2f}, {pt_end[1]:.2f})  (expected ~100,0)")
    s_c = path.find_closest(50, -20)
    print(f"  closest to (50,-20): s={s_c:.2f} cm  (expected ~{seg_a.total_length:.2f})")
