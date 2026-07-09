"""
WRO 2026 Full Race Simulation — CCW + CW, Random Pillars, Parking
==================================================================
Run from project root:   python -m simulation.race_sim

Produces: simulation_result.png

WHAT IS TESTED:
    - Both CCW (left turns) and CW (right turns) lap directions
    - 2 random pillars per direction (random x/y position, random RED/GREEN)
    - All 4 Bezier corners per direction
    - PID steering with realistic noise each step
    - Parking approach at end of lap (smooth Bezier curve to lot entry)

TRACK MATCHES REAL WRO 2026 IMAGES:
    - 300x300 cm outer boundary with rounded corners
    - Inner square obstacle (100-200 cm) with rounded corners
    - Diagonal orange lines at bottom-left and top-right corners
    - Diagonal blue lines at bottom-right and top-left corners
    - Pink parking lot rectangle near the starting area
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from trajectory.builder import TrajectoryBuilder, RED, GREEN
from control.steering_controller import SteeringPIDController

# ── Track geometry (cm) ──────────────────────────────────────────────────────
OUTER    = 300        # mat is 300 x 300 cm
INNER_LO = 100        # inner obstacle starts at x/y = 100
INNER_HI = 200        # inner obstacle ends  at x/y = 200
CL       = 50         # centerline near wall  (y=50 bottom, x=50 left)
CH       = 250        # centerline far wall   (y=250 top,   x=250 right)

# ── Car / simulation parameters ──────────────────────────────────────────────
BASE_SPEED   = 40.0   # cm/s on straights              (TUNE ON REAL ROBOT)
MIN_SPEED    = 8.0    # cm/s minimum (servo authority) (TUNE ON REAL ROBOT)
A_LAT_MAX    = 12.5   # cm/s² lateral acceleration limit for speed-from-curvature
                      # v_max = sqrt(A_LAT_MAX / |κ|)  — from WRO 2025 team formula
                      # 12.5 cm/s² ≈ 0.013 g (very conservative; TUNE ON REAL ROBOT)
DT           = 0.02   # 50 Hz control loop
WHEELBASE    = 16.5   # cm front-to-rear axle       (TUNE ON REAL ROBOT)
ROBOT_LENGTH = 18.0   # cm total car length         (TUNE ON REAL ROBOT)

# ── Parking lots ──────────────────────────────────────────────────────────────
# WRO 2026 rules:
#   Width = always 20 cm (gap between the two magenta marker blocks)
#   Depth = 1.5 × robot length  (calculated per-team once robot is built)
#   Lot is in the starting straight; car drives South (y decreasing) to enter.
CCW_LOT_X  = 150.0               # default centre-x; randomised in main()
CCW_LOT_Y  = float(CL)           # entry edge on the centreline (y=50)
CW_LOT_X   = 150.0               # default centre-x; randomised in main()
CW_LOT_Y   = float(CL)
LOT_THETA  = -math.pi / 2        # car heading to drive INTO the lot (South)
LOT_WIDTH  = 20.0                 # cm — fixed by WRO 2026 rules
LOT_DEPTH  = 1.5 * ROBOT_LENGTH  # = 27.0 cm
MARKER_W   = 2.0                  # cm — magenta block thickness (2 cm wide)
MARKER_L   = 20.0                 # cm — magenta block length   (20 cm long)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ctrl():
    # heading_weight reduced from 18→8: prevents servo saturation at corner entry.
    # heading_weight=18 caused the car to slam to ±27° immediately on every corner,
    # overshooting the planned Bezier and accumulating drift each lap.
    # Kd=0: derivative of noisy sensor readings saturates output. (TUNE ON REAL ROBOT)
    return SteeringPIDController(
        Kp=1.1, Ki=0.0, Kd=0.0,
        output_limits=(-27, 27),
        windup_limit=20.0,
        heading_weight=8.0,
    )


def _color_name(c):
    return "RED" if c == RED else "GRN"


# ── Pillar generation ─────────────────────────────────────────────────────────

def generate_pillars():
    """
    Random pillar positions — one pillar per straight section (4 per direction).
    Color is random RED or GREEN each run.

    CCW straights (each runs 100 cm: 100→200 or 200→100):
        sec 0  East   (y=CL):   x in [120,180]
        sec 2  North  (x=CH):   y in [120,180]
        sec 4  West   (y=CH):   x in [120,180]
        sec 6  South  (x=CL):   y in [120,180]

    CW straights (mirror layout):
        sec 0  West   (y=CL):   x in [120,180]
        sec 2  North  (x=CL):   y in [120,180]
        sec 4  East   (y=CH):   x in [120,180]
        sec 6  South  (x=CH):   y in [120,180]
    """
    rng = np.random.default_rng()

    def rc():
        return int(rng.integers(0, 2))

    ccw = {
        'sec0': (float(rng.uniform(120, 180)), float(CL),  rc()),
        'sec2': (float(CH),  float(rng.uniform(120, 180)), rc()),
        'sec4': (float(rng.uniform(120, 180)), float(CH),  rc()),
        'sec6': (float(CL),  float(rng.uniform(120, 180)), rc()),
    }
    cw = {
        'sec0': (float(rng.uniform(120, 180)), float(CL),  rc()),
        'sec2': (float(CL),  float(rng.uniform(120, 180)), rc()),
        'sec4': (float(rng.uniform(120, 180)), float(CH),  rc()),
        'sec6': (float(CH),  float(rng.uniform(120, 180)), rc()),
    }
    return ccw, cw


# ── Section builders ─────────────────────────────────────────────────────────

def build_ccw_sections_open():
    """
    8 sections for one CCW lap — OPEN CHALLENGE (no traffic signs).

    Corner geometry: each corner starts 50 cm before the outer wall and
    exits on the adjacent centreline — circular-arc endpoints so the Bezier
    stays inside the track instead of bulging toward the outer wall.
    """
    PI = math.pi
    return [
        (TrajectoryBuilder.straight(100, CL, 200, CL),      'Sec0 East  straight',    []),
        (TrajectoryBuilder.corner(200, CL,  0,    +1, 50),  'Sec1 Corner BR (E->N)',  []),
        (TrajectoryBuilder.straight(250, 100, 250, 200),    'Sec2 North straight',    []),
        (TrajectoryBuilder.corner(250, 200, PI/2, +1, 50),  'Sec3 Corner TR (N->W)',  []),
        (TrajectoryBuilder.straight(200, 250, 100, 250),    'Sec4 West  straight',    []),
        (TrajectoryBuilder.corner(100, 250, PI,   +1, 50),  'Sec5 Corner TL (W->S)',  []),
        (TrajectoryBuilder.straight(50, 200, 50, 100),      'Sec6 South straight',    []),
        (TrajectoryBuilder.corner(50,  100, -PI/2,+1, 50),  'Sec7 Corner BL (S->E)', []),
    ]


def build_cw_sections_open():
    """
    8 sections for one CW lap — OPEN CHALLENGE (no traffic signs).

    Same circular-arc corner geometry as the CCW version.
    """
    PI = math.pi
    return [
        (TrajectoryBuilder.straight(200, CL, 100, CL),      'Sec0 West  straight',    []),
        (TrajectoryBuilder.corner(100, CL,  PI,   -1, 50),  'Sec1 Corner BL (W->N)',  []),
        (TrajectoryBuilder.straight(50, 100, 50, 200),      'Sec2 North straight',    []),
        (TrajectoryBuilder.corner(50,  200, PI/2, -1, 50),  'Sec3 Corner TL (N->E)',  []),
        (TrajectoryBuilder.straight(100, 250, 200, 250),    'Sec4 East  straight',    []),
        (TrajectoryBuilder.corner(200, 250, 0,    -1, 50),  'Sec5 Corner TR (E->S)',  []),
        (TrajectoryBuilder.straight(250, 200, 250, 100),    'Sec6 South straight',    []),
        (TrajectoryBuilder.corner(250, 100, -PI/2,-1, 50),  'Sec7 Corner BR (S->W)', []),
    ]


def build_ccw_sections(p):
    """
    8 sections for one CCW lap — pillar swerve on every straight (4 pillars).
    Swerve start/end aligned with the new circular-arc corner geometry.
    """
    PI = math.pi
    p0 = p['sec0']   # East   straight  (y=CL,  x: 100→200)
    p2 = p['sec2']   # North  straight  (x=CH,  y: 100→200)
    p4 = p['sec4']   # West   straight  (y=CH,  x: 200→100)
    p6 = p['sec6']   # South  straight  (x=CL,  y: 200→100)

    return [
        (TrajectoryBuilder.pillar_swerve(
             100, CL, 0, p0[0], p0[1], p0[2], 200, CL, 0),
         f'Sec0 East  +{_color_name(p0[2])}@({p0[0]:.0f},{p0[1]:.0f})', [p0]),

        (TrajectoryBuilder.corner(200, CL, 0, +1, 50),
         'Sec1 Corner BR (E->N)', []),

        (TrajectoryBuilder.pillar_swerve(
             250, 100, PI/2, p2[0], p2[1], p2[2], 250, 200, PI/2),
         f'Sec2 North +{_color_name(p2[2])}@({p2[0]:.0f},{p2[1]:.0f})', [p2]),

        (TrajectoryBuilder.corner(250, 200, PI/2, +1, 50),
         'Sec3 Corner TR (N->W)', []),

        (TrajectoryBuilder.pillar_swerve(
             200, 250, PI, p4[0], p4[1], p4[2], 100, 250, PI),
         f'Sec4 West  +{_color_name(p4[2])}@({p4[0]:.0f},{p4[1]:.0f})', [p4]),

        (TrajectoryBuilder.corner(100, 250, PI, +1, 50),
         'Sec5 Corner TL (W->S)', []),

        (TrajectoryBuilder.pillar_swerve(
             50, 200, -PI/2, p6[0], p6[1], p6[2], 50, 100, -PI/2),
         f'Sec6 South +{_color_name(p6[2])}@({p6[0]:.0f},{p6[1]:.0f})', [p6]),

        (TrajectoryBuilder.corner(50, 100, -PI/2, +1, 50),
         'Sec7 Corner BL (S->E)', []),
    ]


def build_cw_sections(p):
    """
    8 sections for one CW lap — pillar swerve on every straight (4 pillars).
    Swerve start/end aligned with the new circular-arc corner geometry.
    """
    PI = math.pi
    p0 = p['sec0']   # West   straight  (y=CL,  x: 200→100)
    p2 = p['sec2']   # North  straight  (x=CL,  y: 100→200)
    p4 = p['sec4']   # East   straight  (y=CH,  x: 100→200)
    p6 = p['sec6']   # South  straight  (x=CH,  y: 200→100)

    return [
        (TrajectoryBuilder.pillar_swerve(
             200, CL, PI, p0[0], p0[1], p0[2], 100, CL, PI),
         f'Sec0 West  +{_color_name(p0[2])}@({p0[0]:.0f},{p0[1]:.0f})', [p0]),

        (TrajectoryBuilder.corner(100, CL, PI, -1, 50),
         'Sec1 Corner BL (W->N)', []),

        (TrajectoryBuilder.pillar_swerve(
             50, 100, PI/2, p2[0], p2[1], p2[2], 50, 200, PI/2),
         f'Sec2 North +{_color_name(p2[2])}@({p2[0]:.0f},{p2[1]:.0f})', [p2]),

        (TrajectoryBuilder.corner(50, 200, PI/2, -1, 50),
         'Sec3 Corner TL (N->E)', []),

        (TrajectoryBuilder.pillar_swerve(
             100, 250, 0, p4[0], p4[1], p4[2], 200, 250, 0),
         f'Sec4 East  +{_color_name(p4[2])}@({p4[0]:.0f},{p4[1]:.0f})', [p4]),

        (TrajectoryBuilder.corner(200, 250, 0, -1, 50),
         'Sec5 Corner TR (E->S)', []),

        (TrajectoryBuilder.pillar_swerve(
             250, 200, -PI/2, p6[0], p6[1], p6[2], 250, 100, -PI/2),
         f'Sec6 South +{_color_name(p6[2])}@({p6[0]:.0f},{p6[1]:.0f})', [p6]),

        (TrajectoryBuilder.corner(250, 100, -PI/2, -1, 50),
         'Sec7 Corner BR (S->W)', []),
    ]


# ── Section simulator ─────────────────────────────────────────────────────────

def simulate_section(path, start_x, start_y, start_theta, label=''):
    """
    Simulate the car following one path section using BICYCLE KINEMATICS.

    Why bicycle model (not ideal-tracking + noise):
        Ideal tracking sets the car's heading to the path tangent every step,
        so the PID only corrects tiny drift — corners never appear in the
        steering chart.  The bicycle model makes the car's heading a REAL
        state variable that must be steered by the PID, so:
            - straights  → near-zero steering
            - corners    → PID saturates servo at ±27° to turn fast enough
            - pillar swerves → moderate left/right pulses
        This is what the real robot will produce.

    How it works:
        Each tick:
          1. PID reads (x, y, theta) and outputs a steering angle in degrees.
          2. Bicycle kinematics advance x, y, theta by DT.
          3. The arc-length parameter s is advanced by the current speed * DT
             from the PID's last found closest point (so s stays ahead of the
             car and the look-ahead search window stays valid).
          4. Small Gaussian noise on x, y, theta models floor imperfections.

    Returns: xs, ys, thetas, steers, speeds  (lists, one entry per tick)
    """
    ctrl   = _make_ctrl()
    end_pt = path.get_point(path.total_length)
    s      = path.find_closest(start_x, start_y)

    x, y, theta = start_x, start_y, start_theta
    xs     = [x];     ys    = [y];     thetas = [theta]
    steers = [0.0];   speeds = [BASE_SPEED]

    max_steps = int(path.total_length / (BASE_SPEED * DT) * 5 + 50)

    for _ in range(max_steps):
        # Terminate on arc-length: once the path-lookahead pointer reaches the
        # end, the section is done regardless of where the car physically is.
        # (Distance-to-endpoint fails when the car overshoots the endpoint.)
        if s >= path.total_length:
            break

        # Physics-based speed from WRO 2025 team approach:
        #   lateral acceleration  a_y = v² · |κ|  ≤  A_LAT_MAX
        #   → v_max = sqrt(A_LAT_MAX / |κ|)
        curvature = path.get_curvature(s)
        if abs(curvature) > 1e-4:
            speed = min(BASE_SPEED, math.sqrt(A_LAT_MAX / abs(curvature)))
        else:
            speed = BASE_SPEED
        speed = max(MIN_SPEED, speed)

        # Ackermann feedforward: only on CORNER sections.
        # Swerve bypass paths spike to very high curvature near the bypass waypoint —
        # applying feedforward there saturates the servo and sends the car off track.
        # Straights have κ≈0 so feedforward = 0 anyway.
        # Sign: positive steer_deg = theta decreases; CCW κ>0 → steer_ff<0 (steer left).
        #
        # CRITICAL: clamp |κ| to 1/CORNER_RADIUS before computing feedforward.
        # The Bezier endpoints have κ≈0.032 (60% above design 0.02) due to the
        # cubic approximation.  Without clamping, the servo saturates to 27° and
        # the car overshoots the 90° turn, causing a spiral in CW direction.
        # Clamped at exactly 1/50 = 0.02, feedforward turns the car precisely 90°
        # in the expected number of steps with no overshoot.
        if 'Corner' in label:
            kappa_ff = max(-0.02, min(0.02, curvature))  # clamp to corner design radius
            steer_ff = -math.degrees(math.atan(WHEELBASE * kappa_ff))
        else:
            steer_ff = 0.0

        # PID: reads actual car pose, returns CORRECTION in degrees
        steer_corr = ctrl.compute(x, y, theta, path, s)
        # Advance arc-length pointer one step ahead of the closest-found point
        s = min(ctrl.current_s + speed * DT, path.total_length)

        # Total steering = feedforward (handles the turn geometry) +
        #                  PID correction (handles residual CTE/heading error)
        steer_deg = max(-27.0, min(27.0, steer_ff + steer_corr))

        # ── Bicycle kinematics (identical to WRO 2025 EKF motion model) ──
        # x' = v·cos(θ)·dt   y' = v·sin(θ)·dt   θ' = –(v/L)·tan(δ)·dt
        # Minus sign: positive δ = right turn → θ decreases.
        steer_rad  = math.radians(steer_deg)
        x     += speed * math.cos(theta) * DT + np.random.normal(0, 0.04)
        y     += speed * math.sin(theta) * DT + np.random.normal(0, 0.04)
        theta -= (speed / WHEELBASE) * math.tan(steer_rad) * DT
        theta += np.random.normal(0, 0.0004)
        theta  = math.atan2(math.sin(theta), math.cos(theta))  # wrap ±π
        # ────────────────────────────────────────────────────────────────

        xs.append(x);  ys.append(y);  thetas.append(theta)
        steers.append(steer_deg);     speeds.append(speed)

    return xs, ys, thetas, steers, speeds


def simulate_all(sections, sx, sy, stheta, tag=''):
    """Simulate every section in sequence, thread positions together."""
    xs_all, ys_all, th_all, st_all, sp_all = [], [], [], [], []
    cx, cy, cth = sx, sy, stheta

    for path, label, _ in sections:
        xs, ys, ths, sts, sps = simulate_section(path, cx, cy, cth, label)
        xs_all.extend(xs);  ys_all.extend(ys);  th_all.extend(ths)
        st_all.extend(sts); sp_all.extend(sps)
        cx, cy, cth = xs[-1], ys[-1], ths[-1]
        print(f"  [{tag}] {label:<52}  end=({cx:.0f},{cy:.0f})  steps={len(xs)}")

    return xs_all, ys_all, th_all, st_all, sp_all, (cx, cy, cth)


# ── Drawing helpers ───────────────────────────────────────────────────────────

def draw_track(ax, title, lot_x, lot_y, start_x=150.0):
    """
    Render the WRO 2026 track per official game rules (Figure 11):
        - White 300x300 cm mat, black walls
        - Rounded inner obstacle (100-200 cm)
        - Corner lines at 30 deg: orange at 30 deg from horizontal,
          blue at 60 deg from horizontal (= 30 deg from vertical inner wall)
        - Dashed gray centreline loop
        - 20x50 cm starting zone (dashed, CMYK 0,0,0,30)
        - Parking lot: light fill + two 20x2 cm magenta marker blocks
    Colors per official CMYK spec:
        Orange  CMYK(0,60,100,0)  -> RGB(255,102,0)
        Blue    CMYK(100,80,0,0)  -> RGB(0,51,255)
        Gray    CMYK(0,0,0,30)    -> RGB(179,179,179)
        Magenta                   -> RGB(255,0,255)
    """
    # Official colors
    ORG  = '#FF6600'   # CMYK(0,60,100,0)
    BLU  = '#0033FF'   # CMYK(100,80,0,0)
    GRAY = '#B3B3B3'   # CMYK(0,0,0,30) — dashed lines
    MAG  = '#FF00FF'   # RGB(255,0,255) — parking markers

    ax.set_facecolor('#f0f0f0')

    # Outer mat (white)
    ax.add_patch(mpatches.FancyBboxPatch(
        (2, 2), OUTER - 4, OUTER - 4,
        boxstyle="round,pad=4",
        lw=4, edgecolor='#111111', facecolor='white', zorder=0
    ))

    # Inner obstacle (rounded square, dark gray)
    ax.add_patch(mpatches.FancyBboxPatch(
        (INNER_LO + 2, INNER_LO + 2),
        INNER_HI - INNER_LO - 4, INNER_HI - INNER_LO - 4,
        boxstyle="round,pad=10",
        lw=2.5, edgecolor='#222222', facecolor='#aaaaaa', zorder=1
    ))

    # ── Corner lines — 30° geometry (WRO 2026 Figure 11) ─────────────────────
    # T30 = distance along the inner wall face where the line terminates.
    # Orange = steep line (60° from horiz, toward side face of inner obstacle)
    # Blue   = shallow line (30° from horiz, toward bottom/top face)
    T30 = INNER_LO * math.tan(math.radians(30))   # ≈ 57.7 cm
    LW  = 2.5

    # BL: outer (0,0) → inner obstacle corner (100,100)
    ax.plot([0,     INNER_LO], [0,     T30      ], color=BLU, lw=LW, zorder=3)
    ax.plot([0,     T30      ], [0,     INNER_LO ], color=ORG, lw=LW, zorder=3)
    # BR: outer (300,0) → inner (200,100)
    ax.plot([OUTER, INNER_HI], [0,     T30      ], color=BLU, lw=LW, zorder=3)
    ax.plot([OUTER, OUTER-T30], [0,    INNER_LO ], color=ORG, lw=LW, zorder=3)
    # TR: outer (300,300) → inner (200,200)
    ax.plot([OUTER, INNER_HI], [OUTER, OUTER-T30], color=BLU, lw=LW, zorder=3)
    ax.plot([OUTER, OUTER-T30], [OUTER, INNER_HI], color=ORG, lw=LW, zorder=3)
    # TL: outer (0,300) → inner (100,200)
    ax.plot([0,     INNER_LO], [OUTER, OUTER-T30], color=BLU, lw=LW, zorder=3)
    ax.plot([0,     T30      ], [OUTER, INNER_HI ], color=ORG, lw=LW, zorder=3)

    # ── Dashed centreline loop  CMYK(0,0,0,30) ───────────────────────────────
    ax.plot([CL, CH, CH, CL, CL], [CL, CL, CH, CH, CL],
            '--', color=GRAY, lw=0.9, alpha=0.60, zorder=2)

    # ── Starting zone  200x500 mm = 20x50 cm  CMYK(0,0,0,30) ─────────────────
    # 20 cm along driving direction centred at start_x, 50 cm across (y=0→CL)
    ax.add_patch(mpatches.Rectangle(
        (start_x - 10, 0), 20, CL,
        fill=False, linestyle='--', linewidth=0.9,
        edgecolor=GRAY, alpha=0.80, zorder=2
    ))

    # ── Parking lot ───────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (lot_x - LOT_WIDTH / 2, lot_y - LOT_DEPTH),
        LOT_WIDTH, LOT_DEPTH,
        lw=1, edgecolor=MAG, facecolor='#FFE0FF', alpha=0.70, zorder=4
    ))
    ax.text(lot_x, lot_y - LOT_DEPTH / 2,
            'PARK', fontsize=6, ha='center', va='center',
            color='#880088', fontweight='bold', zorder=5)
    # Left marker block  200x20 mm = 20x2 cm, RGB(255,0,255)
    ax.add_patch(mpatches.Rectangle(
        (lot_x - LOT_WIDTH / 2 - MARKER_W, lot_y - MARKER_L),
        MARKER_W, MARKER_L,
        lw=0, facecolor=MAG, zorder=5
    ))
    # Right marker block
    ax.add_patch(mpatches.Rectangle(
        (lot_x + LOT_WIDTH / 2, lot_y - MARKER_L),
        MARKER_W, MARKER_L,
        lw=0, facecolor=MAG, zorder=5
    ))

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xlim(-30, 330)
    ax.set_ylim(-45, 325)
    ax.set_aspect('equal')
    ax.set_xlabel('x (cm)', fontsize=9)
    ax.set_ylabel('y (cm)', fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.grid(True, alpha=0.08, zorder=0)


def _plot_path(ax, path, color, lw=2.3, alpha=0.72):
    ss  = np.linspace(0, path.total_length, 250)
    pts = [path.get_point(float(s)) for s in ss]
    ax.plot([p[0] for p in pts], [p[1] for p in pts],
            '-', color=color, lw=lw, alpha=alpha, zorder=6)


def draw_pillars(ax, pillar_list):
    """
    Draw each traffic sign with its official WRO elements:
        - Evaluation circle: diameter 85 mm = 8.5 cm, CMYK(20,0,100,0) = RGB(204,255,0)
        - Seat square:       50x50 mm = 5x5 cm, CMYK(0,0,0,30) = RGB(179,179,179)
        - Pillar body:       50x50x100 mm = 5x5 cm footprint
          Red   = RGB(238,39,55)   Green = RGB(68,214,44)
    """
    RED_COL  = '#EE2737'   # RGB(238, 39, 55)
    GRN_COL  = '#44D62C'   # RGB(68, 214, 44)
    SEAT_COL = '#B3B3B3'   # CMYK(0,0,0,30)
    CIRC_COL = '#CCFF00'   # CMYK(20,0,100,0)

    for p in pillar_list:
        if p is None:
            continue
        px, py, c = p
        col = RED_COL if c == RED else GRN_COL
        lbl = 'R' if c == RED else 'G'

        # Evaluation circle (dia=8.5cm, r=4.25cm)
        ax.add_patch(plt.Circle((px, py), 4.25,
                                edgecolor=CIRC_COL, facecolor='none',
                                lw=0.6, alpha=0.75, zorder=5))
        # Traffic sign seat (50x50mm = 5x5 cm square, centered)
        ax.add_patch(mpatches.Rectangle(
            (px - 2.5, py - 2.5), 5.0, 5.0,
            fill=False, edgecolor=SEAT_COL, lw=0.8, alpha=0.85, zorder=6
        ))
        # Pillar body (5x5 cm, filled square)
        ax.add_patch(mpatches.Rectangle(
            (px - 2.5, py - 2.5), 5.0, 5.0,
            facecolor=col, edgecolor='none', alpha=0.90, zorder=7
        ))
        ax.text(px, py + 5.5, lbl, fontsize=7.5, ha='center',
                color=col, fontweight='bold', zorder=8)


def draw_arrows(ax, xs, ys, thetas, every=28):
    for i in range(0, len(xs), every):
        dx = 7.0 * math.cos(thetas[i])
        dy = 7.0 * math.sin(thetas[i])
        ax.annotate('', xy=(xs[i]+dx, ys[i]+dy), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle='->', color='#003300', lw=0.9),
                    zorder=9)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    np.random.seed(None)
    PI = math.pi

    # ── Randomise starting direction (WRO draws this before each round) ───────
    start_ccw = bool(np.random.randint(0, 2))
    dir_tag   = 'CCW' if start_ccw else 'CW'
    start_th  = 0.0 if start_ccw else PI
    print(f"Starting direction: {dir_tag}")

    # Pillar positions — fixed for the whole round, random each run
    print("Generating pillar positions...")
    ccw_p, cw_p = generate_pillars()
    pillars_by_dir = ccw_p if start_ccw else cw_p
    for k, v in pillars_by_dir.items():
        print(f"  {dir_tag} {k}: ({v[0]:.0f},{v[1]:.0f}) {_color_name(v[2])}")

    # Section templates
    open_secs = build_ccw_sections_open() if start_ccw else build_cw_sections_open()
    obs_secs  = build_ccw_sections(ccw_p) if start_ccw else build_cw_sections(cw_p)
    rev_secs  = build_cw_sections(cw_p)  if start_ccw else build_ccw_sections(ccw_p)
    rev_th    = PI if start_ccw else 0.0

    # Random lot position on the bottom straight (y=CL) — WRO places it randomly.
    # Valid range: clear of corner entries at x=100 and x=200, so [115, 185].
    lot_x     = float(np.random.uniform(115, 185))
    lot_y     = float(CL)
    rev_lot_x = float(np.random.uniform(115, 185))   # fresh random if direction reverses

    # =========================================================================
    # OPEN CHALLENGE  --  3 clean laps, no traffic signs
    # =========================================================================
    print(f"\n=== OPEN CHALLENGE ({dir_tag}, 3 laps, no signs) ===")
    open_laps_xs, open_laps_ys = [], []
    open_st, open_sp = [], []
    cx, cy, cth = 150.0, float(CL), start_th

    for lap in range(3):
        xs, ys, ths, sts, sps, (cx, cy, cth) = \
            simulate_all(open_secs, cx, cy, cth, f'OPEN {dir_tag} L{lap+1}')
        open_laps_xs.append(xs)
        open_laps_ys.append(ys)
        open_st.extend(sts)
        open_sp.extend(sps)

    t_open = sum(len(lx) for lx in open_laps_xs) * DT

    # =========================================================================
    # OBSTACLE CHALLENGE  --  3 laps with traffic signs + parking
    # WRO 2026 special rule: last sign after lap 2 RED=reverse, GREEN=same
    # =========================================================================
    print(f"\n=== OBSTACLE CHALLENGE ({dir_tag}, 3 laps + parking) ===")
    obs_laps_xs, obs_laps_ys = [], []
    obs_st, obs_sp = [], []
    cx, cy, cth = 150.0, float(CL), start_th
    last_sign = None

    for lap in range(3):
        if lap < 2:
            secs = obs_secs
        else:
            if last_sign == RED:
                print(f"  Last sign RED -> lap 3 REVERSES direction")
                secs = rev_secs
                cth  = rev_th
            else:
                print(f"  Last sign {'GREEN' if last_sign==GREEN else 'none'} -> lap 3 SAME direction")
                secs = obs_secs

        xs, ys, ths, sts, sps, (cx, cy, cth) = \
            simulate_all(secs, cx, cy, cth, f'OBS {dir_tag} L{lap+1}')
        obs_laps_xs.append(xs)
        obs_laps_ys.append(ys)
        obs_st.extend(sts)
        obs_sp.extend(sps)

        if lap < 2:
            for _, _, pillars in secs:
                if pillars:
                    last_sign = pillars[0][2]

    # Parking after lap 3
    final_lot_x = rev_lot_x if (last_sign == RED) else lot_x
    print(f"  Parking at x={final_lot_x:.0f}...")
    park_path = TrajectoryBuilder.parking_approach(
        cx, cy, cth, final_lot_x, lot_y, LOT_THETA)
    pxp, pyp, _, _, _ = simulate_section(park_path, cx, cy, cth, 'parking')

    # Drive-in: straight south from lot entry into the space between the pink walls
    cx_a, cy_a, cth_a = pxp[-1], pyp[-1], LOT_THETA
    stop_dist = LOT_DEPTH - ROBOT_LENGTH / 2   # = 18.0 cm
    drive_in_path = TrajectoryBuilder.straight(
        final_lot_x, lot_y,
        final_lot_x, lot_y - stop_dist)
    pxd, pyd, _, _, _ = simulate_section(drive_in_path, cx_a, cy_a, cth_a, 'park_drive_in')

    t_obs = sum(len(lx) for lx in obs_laps_xs) * DT
    print(f"\nOpen Challenge:     {t_open:.1f} s  ({int(t_open/DT)} steps)")
    print(f"Obstacle Challenge: {t_obs:.1f} s  ({int(t_obs/DT)} steps) + parking")

    # =========================================================================
    # Figure  --  2 challenge maps (top) + steering + speed (bottom)
    # =========================================================================
    fig = plt.figure(figsize=(22, 16))
    gs  = GridSpec(3, 2, figure=fig,
                   height_ratios=[4.0, 0.9, 0.9],
                   hspace=0.48, wspace=0.26)

    ax_open = fig.add_subplot(gs[0, 0])
    ax_obs  = fig.add_subplot(gs[0, 1])
    ax_st   = fig.add_subplot(gs[1, :])
    ax_sp   = fig.add_subplot(gs[2, :])

    LAP_COLS  = ['#009933', '#0044cc', '#cc5500']
    LAP_NAMES = ['Lap 1', 'Lap 2', 'Lap 3']
    C_CORNER   = '#7722bb'
    C_STRAIGHT = '#1155bb'
    C_SWERVE   = '#cc4400'
    C_PARK_PLN = '#886600'
    C_PARK_CAR = '#ff8800'

    # ── OPEN CHALLENGE MAP ───────────────────────────────────────────────────
    draw_track(ax_open,
               f'OPEN CHALLENGE  ({dir_tag})  --  3 laps, no traffic signs',
               lot_x, lot_y)

    for path, lbl, _ in open_secs:
        col = C_CORNER if 'Corner' in lbl else C_STRAIGHT
        _plot_path(ax_open, path, col, lw=1.8, alpha=0.35)

    for i, (lxs, lys) in enumerate(zip(open_laps_xs, open_laps_ys)):
        ax_open.plot(lxs, lys, '-', color=LAP_COLS[i], lw=1.1,
                     alpha=0.85, label=LAP_NAMES[i], zorder=8)

    ax_open.plot(150, CL, 's', color='gold', ms=11, mec='#444', zorder=10)
    ax_open.text(158, CL+10, 'START', fontsize=7, color='#553300', zorder=11)
    ax_open.text(150, CH+10,
                 f'{dir_tag}  |  3 laps  |  {t_open:.1f} s  (no parking)',
                 fontsize=8, ha='center', color='#333')
    ax_open.legend(loc='upper right', fontsize=8, framealpha=0.88)

    # ── OBSTACLE CHALLENGE MAP ───────────────────────────────────────────────
    draw_track(ax_obs,
               f'OBSTACLE CHALLENGE  ({dir_tag})  --  3 laps + parallel parking',
               final_lot_x, lot_y)

    all_pillars = []
    for path, lbl, pil in obs_secs:
        col = C_SWERVE if pil else C_CORNER if 'Corner' in lbl else C_STRAIGHT
        _plot_path(ax_obs, path, col, lw=1.8, alpha=0.30)
        all_pillars.extend(pil)

    if last_sign == RED:
        for path, lbl, _ in rev_secs:
            col = C_CORNER if 'Corner' in lbl else C_STRAIGHT
            _plot_path(ax_obs, path, col, lw=1.2, alpha=0.18)

    for i, (lxs, lys) in enumerate(zip(obs_laps_xs, obs_laps_ys)):
        lap_label = LAP_NAMES[i]
        if i == 2 and last_sign == RED:
            lap_label += ' (REVERSED)'
        ax_obs.plot(lxs, lys, '-', color=LAP_COLS[i], lw=1.1,
                    alpha=0.85, label=lap_label, zorder=8)

    _plot_path(ax_obs, park_path,    C_PARK_PLN, lw=2.0, alpha=0.85)
    _plot_path(ax_obs, drive_in_path, C_PARK_PLN, lw=2.0, alpha=0.85)
    ax_obs.plot(pxp, pyp, '-', color=C_PARK_CAR, lw=1.4, alpha=0.9, zorder=8)
    ax_obs.plot(pxd, pyd, '-', color=C_PARK_CAR, lw=1.4, alpha=0.9, zorder=8)

    ax_obs.plot(150, CL, 's', color='gold', ms=11, mec='#444', zorder=10)
    ax_obs.text(158, CL+10, 'START', fontsize=7, color='#553300', zorder=11)
    ax_obs.plot(pxd[-1], pyd[-1], '*', color='gold', ms=14, mec='#444', zorder=10)
    ax_obs.text(pxd[-1]+3, pyd[-1]+2, 'PARKED', fontsize=7, color='#880088')

    rule_note = ('Lap 3: REVERSED (last sign=RED)'
                 if last_sign == RED else
                 'Lap 3: same dir (last sign=GREEN/none)')
    ax_obs.text(150, CH+10,
                f'{dir_tag}  |  3 laps  |  {t_obs:.1f} s  |  {rule_note}',
                fontsize=7.5, ha='center', color='#333')
    draw_pillars(ax_obs, all_pillars)

    leg_obs = [
        mpatches.Patch(color=C_SWERVE,   label='Pillar swerve (Bezier)'),
        mpatches.Patch(color=C_CORNER,   label='Corner (Bezier)'),
        mpatches.Patch(color=C_STRAIGHT, label='Straight'),
        mpatches.Patch(color=C_PARK_PLN, label='Parking (planned)'),
        plt.Line2D([0],[0], color=C_PARK_CAR, lw=1.5, label='Parking (car)'),
        plt.Line2D([0],[0], color='#EE2737', marker='s', ms=7, ls='', label='RED  -> pass RIGHT'),
        plt.Line2D([0],[0], color='#44D62C', marker='s', ms=7, ls='', label='GREEN -> pass LEFT'),
    ] + [plt.Line2D([0],[0], color=c, lw=1.5, label=n)
         for c, n in zip(LAP_COLS, LAP_NAMES)]
    ax_obs.legend(handles=leg_obs, loc='upper right', fontsize=7, framealpha=0.88)

    # ── Steering chart (obstacle challenge) ──────────────────────────────────
    t_arr    = np.arange(len(obs_st)) * DT
    lap_ends = np.cumsum([len(lx) for lx in obs_laps_xs]) * DT
    for i, (t0, t1) in enumerate(zip([0]+list(lap_ends[:-1]), lap_ends)):
        ax_st.axvspan(t0, t1, alpha=0.07, color=LAP_COLS[i], zorder=0)
        ax_st.text((t0+t1)/2, 29, LAP_NAMES[i], fontsize=7.5,
                   ha='center', color=LAP_COLS[i], fontweight='bold')
    ax_st.plot(t_arr, obs_st, color='#cc4400', lw=0.65, zorder=2)
    ax_st.axhline(0,   color='black', lw=0.5)
    ax_st.axhline( 27, color='gray',  lw=0.5, ls='--')
    ax_st.axhline(-27, color='gray',  lw=0.5, ls='--')
    ax_st.set_ylim(-35, 35)
    ax_st.set_title('Obstacle Challenge -- Steering angle  (+/-27 deg servo limit)', fontsize=9)
    ax_st.set_xlabel('Time (s)', fontsize=8)
    ax_st.set_ylabel('deg', fontsize=8)
    ax_st.grid(True, alpha=0.22)

    # ── Speed chart ───────────────────────────────────────────────────────────
    t_arr2 = np.arange(len(obs_sp)) * DT
    for i, (t0, t1) in enumerate(zip([0]+list(lap_ends[:-1]), lap_ends)):
        ax_sp.axvspan(t0, t1, alpha=0.07, color=LAP_COLS[i], zorder=0)
    ax_sp.plot(t_arr2, obs_sp, color='#005599', lw=0.65, zorder=2)
    ax_sp.axhline(BASE_SPEED, color='gray', lw=0.5, ls='--',
                  label=f'Max {BASE_SPEED} cm/s')
    ax_sp.set_ylim(0, BASE_SPEED + 8)
    ax_sp.set_title('Obstacle Challenge -- Speed  (auto-reduced in curves)', fontsize=9)
    ax_sp.set_xlabel('Time (s)', fontsize=8)
    ax_sp.set_ylabel('cm/s', fontsize=8)
    ax_sp.legend(fontsize=7)
    ax_sp.grid(True, alpha=0.22)

    plt.suptitle(
        'WRO 2026 Future Engineers  --  Both Challenges  |  3 Laps Each  |  '
        f'Starting direction: {dir_tag}  |  {int(1/DT)} Hz PID  |  Bicycle kinematics',
        fontsize=11, y=1.005
    )

    out = 'simulation_result.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSaved -> {out}')


if __name__ == '__main__':
    main()
