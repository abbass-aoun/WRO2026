"""
WRO 2026 Real-Time Race Simulation  --  pygame Visualiser
==========================================================

Animated top-down race simulation.  Shows the FULL obstacle challenge:
  - Lap 1 and Lap 2 in the starting direction (CCW or CW)
  - WRO special rule after lap 2: last sign RED → lap 3 reverses, GREEN → same
  - Parking approach Bezier + drive-in straight between the two pink walls

Run from project root:
    python -m simulation.pygame_sim

Controls:
    SPACE   pause / resume
    F       fast mode (5×)
    R       re-roll random pillars + lot position and restart
    ESC     quit
"""

import sys
import math
import collections

try:
    import pygame
except ImportError:
    sys.exit("pygame not found.  Run:   pip install pygame")

import numpy as np

from simulation.race_sim import (
    generate_pillars,
    build_ccw_sections, build_cw_sections,
    simulate_section,
    CCW_LOT_Y,
)
from trajectory.builder import TrajectoryBuilder, RED, GREEN
from config import (
    BASE_SPEED_CM_S as BASE_SPEED,
    DT_S            as DT,
    WHEELBASE_CM    as WHEELBASE,
    ROBOT_LENGTH_CM as ROBOT_LENGTH,
    LOT_THETA_RAD   as LOT_THETA,
    LOT_WIDTH_CM    as LOT_WIDTH,
    LOT_DEPTH_CM    as LOT_DEPTH,
    MARKER_W_CM     as MARKER_W,
    MARKER_L_CM     as MARKER_L,
    SIM_PILLAR_HIT_CM as PILLAR_HIT_CM,
)

# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────
SCALE   = 2.0
TW      = int(300 * SCALE)
TH      = int(300 * SCALE)
T_LEFT  = 18
T_TOP   = 44
PAN_X   = T_LEFT + TW + 14
PAN_W   = 262
WIN_W   = PAN_X + PAN_W + 6
WIN_H   = T_TOP + TH + 28
FPS     = 60
TRAIL_N = 320

LOT_Y   = float(CCW_LOT_Y)   # = 50.0  (both CCW and CW park on the south straight)

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
BG       = (11,  11,  22)
MAT      = (236, 236, 236)
WALL_C   = (22,  22,  22)
INNER_F  = (160, 160, 160)
INNER_E  = (34,  34,  34)
ORG_L    = (255, 102,   0)
BLU_L    = (  0,  51, 255)
CL_COL   = (179, 179, 179)
LOT_F    = (255, 224, 255)
LOT_E    = (255,   0, 255)
C_CCW    = (  0, 232,  82)
C_CW     = ( 52, 158, 255)
FRONT_C  = (255, 238,  95)
R_PIK    = (238,  39,  55)
G_PIK    = ( 68, 214,  44)
SEAT_COL = (179, 179, 179)
CIRC_COL = (204, 255,   0)
PAN_BG   = ( 17,  17,  30)
TXT_W    = (212, 212, 212)
TXT_D    = (100, 100, 122)
GOLD_C   = (255, 208,  48)
BAR_BG   = ( 42,  42,  60)
WHITE    = (255, 255, 255)
BLACK    = (  0,   0,   0)
RED_C    = (238,  39,  55)
GRN_C    = ( 68, 214,  44)

COL_SWV  = (200,  75,  10)
COL_COR  = (130,  45, 185)
COL_STR  = ( 28,  85, 178)
COL_PRK  = (120,  95,   8)
COL_REV  = ( 90,  50, 160)   # reversed-lap planned path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def w2s(x, y):
    return (int(T_LEFT + x * SCALE),
            int(T_TOP  + (300 - y) * SCALE))


def _draw_path(surf, path, color, width=2):
    n = max(60, int(path.total_length))
    pts = [w2s(*path.get_point(path.total_length * i / n)) for i in range(n + 1)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, width)


# ─────────────────────────────────────────────────────────────────────────────
# Static track surface
# ─────────────────────────────────────────────────────────────────────────────

def make_track_surface(run, fnt_sm):
    """Build a static pygame Surface from one run dict."""
    obs_secs      = run['obs_secs']
    lap3_secs     = run['lap3_secs']
    lap3_reversed = run['lap3_reversed']
    park_path     = run['park_path']
    drive_in_path = run['drive_in_path']
    lot_x         = run['final_lot_x']
    pills         = run['pills']

    surf = pygame.Surface((WIN_W, WIN_H))
    surf.fill(BG)

    # Mat
    pygame.draw.rect(surf, MAT,    (T_LEFT+3, T_TOP+3, TW-6, TH-6), border_radius=10)
    pygame.draw.rect(surf, WALL_C, (T_LEFT,   T_TOP,   TW,   TH),   5, border_radius=10)

    # Inner obstacle
    x1, y1 = w2s(100, 200)
    x2, y2 = w2s(200, 100)
    pygame.draw.rect(surf, INNER_F, pygame.Rect(x1, y1, x2-x1, y2-y1), border_radius=14)
    pygame.draw.rect(surf, INNER_E, pygame.Rect(x1, y1, x2-x1, y2-y1), 3, border_radius=14)

    # Dashed centrelines
    corners = [w2s(50,50), w2s(250,50), w2s(250,250), w2s(50,250), w2s(50,50)]
    for i in range(4):
        p0, p1 = corners[i], corners[i+1]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L = math.hypot(dx, dy)
        if L < 1: continue
        n_dash = max(1, int(L / 14))
        for j in range(0, n_dash, 2):
            t0, t1 = j/n_dash, min((j+1)/n_dash, 1.0)
            pygame.draw.line(surf, CL_COL,
                             (int(p0[0]+dx*t0), int(p0[1]+dy*t0)),
                             (int(p0[0]+dx*t1), int(p0[1]+dy*t1)), 1)

    # Corner marker lines
    # Orange = steep (60° from horiz)   Blue = shallow (30° from horiz)
    T30 = 100 * math.tan(math.radians(30))
    def cl(a, b, col):
        pygame.draw.line(surf, col, w2s(*a), w2s(*b), 3)
    cl((0,   0),   (100,   T30),     BLU_L)
    cl((0,   0),   (T30,   100),     ORG_L)
    cl((300, 0),   (200,   T30),     BLU_L)
    cl((300, 0),   (300-T30, 100),   ORG_L)
    cl((300,300),  (200, 300-T30),   BLU_L)
    cl((300,300),  (300-T30, 200),   ORG_L)
    cl((0,  300),  (100, 300-T30),   BLU_L)
    cl((0,  300),  (T30, 200),       ORG_L)

    # Parking lot
    lx1, ly1 = w2s(lot_x - LOT_WIDTH/2, LOT_Y)
    lx2, ly2 = w2s(lot_x + LOT_WIDTH/2, LOT_Y - LOT_DEPTH)
    lot_rect = pygame.Rect(lx1, ly1, lx2-lx1, ly2-ly1)
    pygame.draw.rect(surf, LOT_F, lot_rect)
    pygame.draw.rect(surf, LOT_E, lot_rect, 1)
    p_s = fnt_sm.render("PARK", True, LOT_E)
    surf.blit(p_s, ((lx1+lx2)//2 - p_s.get_width()//2,
                    (ly1+ly2)//2 - p_s.get_height()//2))
    mx1, my1 = w2s(lot_x - LOT_WIDTH/2 - MARKER_W, LOT_Y)
    mx2, my2 = w2s(lot_x - LOT_WIDTH/2,             LOT_Y - MARKER_L)
    pygame.draw.rect(surf, LOT_E, pygame.Rect(mx1, my1, mx2-mx1, my2-my1))
    rx1, ry1 = w2s(lot_x + LOT_WIDTH/2,             LOT_Y)
    rx2, ry2 = w2s(lot_x + LOT_WIDTH/2 + MARKER_W,  LOT_Y - MARKER_L)
    pygame.draw.rect(surf, LOT_E, pygame.Rect(rx1, ry1, rx2-rx1, ry2-ry1))

    # Planned paths — laps 1+2 (obs_secs) in normal colours
    for path, label, _ in obs_secs:
        has_pillar = ('+' in label)
        is_corner  = ('Corner' in label)
        col = COL_SWV if has_pillar else COL_COR if is_corner else COL_STR
        _draw_path(surf, path, col, 2)

    # Lap 3 path — different colour if reversed
    if lap3_reversed:
        for path, label, _ in lap3_secs:
            has_pillar = ('+' in label)
            is_corner  = ('Corner' in label)
            col = COL_REV if has_pillar else COL_COR if is_corner else COL_STR
            _draw_path(surf, path, col, 1)

    # Parking planned paths
    if park_path is not None:
        _draw_path(surf, park_path, COL_PRK, 2)
    if drive_in_path is not None:
        _draw_path(surf, drive_in_path, COL_PRK, 2)

    # Starting zone dashes
    sz_pts = [w2s(140,0), w2s(160,0), w2s(160,50), w2s(140,50), w2s(140,0)]
    for i in range(len(sz_pts)-1):
        p0, p1 = sz_pts[i], sz_pts[i+1]
        dx, dy = p1[0]-p0[0], p1[1]-p0[1]
        L = max(1, math.hypot(dx, dy))
        n_dash = max(1, int(L / 8))
        for j in range(0, n_dash, 2):
            t0, t1 = j/n_dash, min((j+1)/n_dash, 1.0)
            pygame.draw.line(surf, CL_COL,
                             (int(p0[0]+dx*t0), int(p0[1]+dy*t0)),
                             (int(p0[0]+dx*t1), int(p0[1]+dy*t1)), 1)

    # Evaluation circles + seats
    for px, py, _ in pills:
        scx, scy = w2s(px, py)
        pygame.draw.circle(surf, CIRC_COL, (scx, scy), int(4.25 * SCALE), 1)
        half = int(2.5 * SCALE)
        pygame.draw.rect(surf, SEAT_COL,
                         pygame.Rect(scx-half, scy-half, 2*half, 2*half), 1)

    # Pillars
    for px, py, c in pills:
        col = R_PIK if c == RED else G_PIK
        scx, scy = w2s(px, py)
        half = int(2.5 * SCALE)
        glow = tuple(min(255, v // 3 + 20) for v in col)
        pygame.draw.rect(surf, glow,
                         pygame.Rect(scx-half-4, scy-half-4, 2*half+8, 2*half+8),
                         border_radius=3)
        pygame.draw.rect(surf, col,   pygame.Rect(scx-half, scy-half, 2*half, 2*half))
        pygame.draw.rect(surf, WHITE, pygame.Rect(scx-half, scy-half, 2*half, 2*half), 1)
        lbl = fnt_sm.render("R" if c == RED else "G", True, WHITE)
        surf.blit(lbl, (scx - lbl.get_width()//2, scy - half - 14))

    return surf


# ─────────────────────────────────────────────────────────────────────────────
# Car sprite + trail
# ─────────────────────────────────────────────────────────────────────────────

def draw_car(surf, x, y, theta, color, length_cm=18.0, width_cm=12.0):
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hl, hw = length_cm / 2, width_cm / 2
    def corner(fx, fy):
        return w2s(x + fx*cos_t - fy*sin_t, y + fx*sin_t + fy*cos_t)
    pts = [corner(hl,hw), corner(hl,-hw), corner(-hl,-hw), corner(-hl,hw)]
    pygame.draw.polygon(surf, (0,0,0), [(px+3,py+3) for px,py in pts])
    pygame.draw.polygon(surf, color, pts)
    pygame.draw.polygon(surf, tuple(max(0,c-75) for c in color), pts, 2)
    pygame.draw.line(surf, FRONT_C, pts[0], pts[1], 3)
    pygame.draw.circle(surf, FRONT_C, w2s(x, y), 2)


def draw_full_path(surf, full_trail, color):
    if len(full_trail) < 2:
        return
    dim = tuple(max(0, int(c * 0.38)) for c in color)
    pygame.draw.lines(surf, dim, False, list(full_trail), 1)


def draw_trail(surf, trail, color):
    n = len(trail)
    if n < 2: return
    for i, pos in enumerate(trail):
        t = i / n
        r = int(BG[0] + (color[0]-BG[0]) * t)
        g = int(BG[1] + (color[1]-BG[1]) * t)
        b = int(BG[2] + (color[2]-BG[2]) * t)
        pygame.draw.circle(surf, (r,g,b), pos, max(1, int(t * 3.5)))


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(surf, fonts, run, sec_label, speed, steer,
               elapsed, frame_i, total_frames, parked):
    direction      = run['dir']
    car_col        = run['color']
    lap3_reversed  = run['lap3_reversed']
    last_sign      = run['last_sign']
    lap_end_frames = run.get('lap_end_frames', [])

    font_big, font_med, font_sm = fonts
    pygame.draw.rect(surf, PAN_BG, (PAN_X, T_TOP, PAN_W, TH))
    pygame.draw.rect(surf, (38,38,55), (PAN_X, T_TOP, PAN_W, TH), 1)

    pad = 11
    y   = T_TOP + 12
    bw  = PAN_W - 2*pad

    def text(msg, fnt, col=TXT_W, ox=0):
        nonlocal y
        s = fnt.render(msg, True, col)
        surf.blit(s, (PAN_X + pad + ox, y))
        y += s.get_height()

    def gap(n=7): nonlocal y; y += n

    def hbar(val, lo, hi, fg, height=15, center=False):
        nonlocal y
        rect = pygame.Rect(PAN_X+pad, y, bw, height)
        pygame.draw.rect(surf, BAR_BG, rect, border_radius=4)
        if center:
            mx = PAN_X + pad + bw//2
            pygame.draw.line(surf, TXT_D, (mx,y),(mx,y+height), 1)
            frac = max(-1, min(1, val / max(abs(lo),abs(hi),1)))
            fw = int(abs(frac) * bw//2)
            ox = mx if frac >= 0 else mx - fw
            if fw > 1:
                pygame.draw.rect(surf, fg, pygame.Rect(ox,y,fw,height), border_radius=3)
        else:
            frac = max(0, min(1, (val-lo) / max(hi-lo,1)))
            fw = int(frac * bw)
            if fw > 1:
                pygame.draw.rect(surf, fg, pygame.Rect(PAN_X+pad,y,fw,height), border_radius=4)
        y += height

    # Direction
    sym = "<<  CCW" if direction == "CCW" else "CW  >>"
    text(sym, font_big, car_col); gap(5)

    status_col  = GOLD_C if parked else TXT_W
    status_text = "** PARKED **" if parked else "  RACING  "
    text(status_text, font_med, status_col); gap(10)

    # Lap indicator (parsed from sec_label "L1 ...", "L2 ...", "L3 ...")
    lap_num = 0
    if sec_label.startswith('L') and len(sec_label) > 1 and sec_label[1].isdigit():
        lap_num = int(sec_label[1])
    if lap_num > 0:
        lap_col = GOLD_C if (lap_num == 3 and lap3_reversed) else TXT_W
        lap_tag = "  (REVERSED)" if (lap_num == 3 and lap3_reversed) else ""
        text(f"LAP  {lap_num} / 3{lap_tag}", font_med, lap_col); gap(8)
    else:
        text("PARKING", font_med, (200, 160, 50)); gap(8)

    # Speed
    text("SPEED", font_sm, TXT_D); gap(4)
    hbar(speed, 0, BASE_SPEED, car_col)
    gap(2)
    text(f"  {speed:.1f}  /  {BASE_SPEED:.0f} cm/s", font_sm); gap(10)

    # Steering
    steer_col = (220,75,75) if steer > 2 else (75,220,75) if steer < -2 else (140,140,140)
    text("STEERING", font_sm, TXT_D); gap(4)
    hbar(steer, -27, 27, steer_col, center=True)
    gap(2)
    text(f"  {steer:+.1f} deg  (limit +-27)", font_sm); gap(10)

    # Section + Ackermann feedforward badge
    text("SECTION", font_sm, TXT_D); gap(3)
    name = sec_label.split('+')[0].strip()
    if len(name) > 20: name = name[-20:]
    text(name, font_med); gap(4)
    if 'Corner' in sec_label:
        pygame.draw.rect(surf, (130, 45, 185),
                         pygame.Rect(PAN_X+pad, y, bw, 18), border_radius=4)
        ff_s = font_sm.render("  ACKERMANN FF  ON", True, WHITE)
        surf.blit(ff_s, (PAN_X+pad+2, y+2))
        y += 22
    gap(6)

    # Race progress
    text("RACE PROGRESS", font_sm, TXT_D); gap(4)
    hbar(frame_i, 0, max(total_frames,1), car_col, height=10)
    gap(3)
    pct = int(100 * frame_i // max(total_frames,1))
    text(f"  {pct}%   ({elapsed:.1f} s)", font_sm); gap(10)

    # Lap split timers — one compact line per lap
    splits_parts = []
    for lap_i, end_f in enumerate(lap_end_frames):
        lap_t   = end_f * DT
        done    = frame_i >= end_f
        active  = (not done and (lap_i == 0 or frame_i >= lap_end_frames[lap_i-1]))
        marker  = ">" if active else ("✓" if done else " ")
        splits_parts.append((f"L{lap_i+1}:{lap_t:.0f}s", GOLD_C if done else (car_col if active else TXT_D)))
    text("SPLITS", font_sm, TXT_D); gap(2)
    sx = PAN_X + pad
    for part_txt, part_col in splits_parts:
        ps = font_sm.render(part_txt, True, part_col)
        surf.blit(ps, (sx, y))
        sx += ps.get_width() + 6
    y += font_sm.get_height(); gap(8)

    # Special-rule result
    text("LAP 3 RULE", font_sm, TXT_D); gap(3)
    if last_sign == RED:
        pygame.draw.circle(surf, R_PIK, (PAN_X+pad+7, y+7), 6)
        text("   Last=RED  REVERSED", font_sm, RED_C); gap(6)
    elif last_sign == GREEN:
        pygame.draw.circle(surf, G_PIK, (PAN_X+pad+7, y+7), 6)
        text("   Last=GRN  same dir", font_sm, GRN_C); gap(6)
    else:
        text("   (after lap 2)", font_sm, TXT_D); gap(6)

    # Pillar legend
    text("PILLAR RULE", font_sm, TXT_D); gap(4)
    pygame.draw.circle(surf, R_PIK, (PAN_X+pad+7, y+6), 6)
    text("   RED  = pass RIGHT", font_sm, TXT_W); gap(2)
    pygame.draw.circle(surf, G_PIK, (PAN_X+pad+7, y+6), 6)
    text("   GRN  = pass LEFT",  font_sm, TXT_W); gap(12)

    # Path legend
    for col, lbl in [
        (COL_SWV, "Swerve lap1+2"),
        (COL_REV, "Swerve lap3 rev"),
        (COL_COR, "Corner (Bezier)"),
        (COL_STR, "Straight"),
        (COL_PRK, "Parking path"),
    ]:
        pygame.draw.rect(surf, col, (PAN_X+pad, y+4, 16, 6), border_radius=2)
        text(f"   {lbl}", font_sm, TXT_D, ox=16); gap(1)
    gap(8)

    # Controls
    y = T_TOP + TH - 68
    text("SPACE  pause / resume", font_sm, TXT_D)
    gap(1); text("F      fast (5x)",      font_sm, TXT_D)
    gap(1); text("R      re-roll + restart", font_sm, TXT_D)
    gap(1); text("ESC    quit",            font_sm, TXT_D)


# ─────────────────────────────────────────────────────────────────────────────
# 3-lap trajectory computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_run(ccw_secs, cw_secs, start_is_ccw, lot_x):
    """
    Simulate the full WRO obstacle challenge:
      Lap 1 & 2  — starting direction (CCW or CW)
      Lap 3      — WRO special rule: last sign RED → reverse, GREEN → same
      Parking    — Bezier approach + straight drive-in between pink walls

    Returns a run dict with frames and all metadata needed for rendering.
    """
    obs_secs = ccw_secs if start_is_ccw else cw_secs
    rev_secs = cw_secs  if start_is_ccw else ccw_secs
    start_th = 0.0 if start_is_ccw else math.pi
    rev_th   = math.pi if start_is_ccw else 0.0
    direction = 'CCW' if start_is_ccw else 'CW'
    car_col   = C_CCW  if start_is_ccw else C_CW

    cx, cy, cth    = 150.0, LOT_Y, start_th
    last_sign      = None
    lap3_reversed  = False
    lap3_secs      = obs_secs
    frames         = []
    frame_count    = 0
    lap_end_frames = []   # frame index at end of each lap  (for split timer)
    pillar_results = []   # (end_frame, px, py, min_d) per swerve (for badges)

    for lap in range(3):
        if lap < 2:
            secs = obs_secs
        else:
            # WRO special rule: last sign after lap 2 sets lap 3 direction
            if last_sign == RED:
                lap3_reversed = True
                lap3_secs     = rev_secs
                cth           = rev_th    # force heading for reversed start
            secs = lap3_secs

        for sec_i, (path, label, pillars) in enumerate(secs):
            # Rebuild swerve from actual car position to keep bypass clearance
            # regardless of drift accumulated through previous corners.
            if pillars:
                px, py, pc = pillars[0]
                end_xy  = path.get_point(path.total_length)
                end_tan = path.get_tangent(path.total_length)
                end_th  = math.atan2(end_tan[1], end_tan[0])
                path = TrajectoryBuilder.pillar_swerve(
                    cx, cy, cth, px, py, pc,
                    float(end_xy[0]), float(end_xy[1]), end_th
                )
            xs, ys, ths, sts, sps, _ = simulate_section(path, cx, cy, cth, label)
            if pillars:
                seg_d = np.hypot(np.array(xs) - px, np.array(ys) - py)
                pillar_results.append((frame_count + len(xs), px, py,
                                       float(seg_d.min()), pc))
            for row in zip(xs, ys, ths, sts, sps):
                frames.append((*row, lap * 10 + sec_i, f'L{lap+1} {label}'))
            frame_count += len(xs)
            cx, cy, cth = xs[-1], ys[-1], ths[-1]

        lap_end_frames.append(frame_count)

        # Track last sign only across lap 1 and lap 2
        if lap < 2:
            for _, _, pills in secs:
                if pills:
                    last_sign = pills[0][2]

    # Parking: Bezier approach to lot entry, then straight drive-in
    final_lot_x = lot_x
    park_path = TrajectoryBuilder.parking_approach(cx, cy, cth, final_lot_x, LOT_Y, LOT_THETA)
    xs, ys, ths, sts, sps, _ = simulate_section(park_path, cx, cy, cth)
    for row in zip(xs, ys, ths, sts, sps):
        frames.append((*row, 30, 'Parking approach'))
    cx, cy, cth = xs[-1], ys[-1], LOT_THETA

    stop_dist     = LOT_DEPTH - ROBOT_LENGTH / 2
    drive_in_path = TrajectoryBuilder.straight(
        final_lot_x, LOT_Y,
        final_lot_x, LOT_Y - stop_dist)
    xs, ys, ths, sts, sps, _ = simulate_section(drive_in_path, cx, cy, cth)
    for row in zip(xs, ys, ths, sts, sps):
        frames.append((*row, 31, 'Parking drive-in'))

    return {
        'dir':            direction,
        'color':          car_col,
        'frames':         frames,
        'obs_secs':       obs_secs,
        'lap3_secs':      lap3_secs,
        'lap3_reversed':  lap3_reversed,
        'last_sign':      last_sign,
        'park_path':      park_path,
        'drive_in_path':  drive_in_path,
        'final_lot_x':    final_lot_x,
        'pills':          [],           # filled by _regenerate
        'lap_end_frames': lap_end_frames,
        'pillar_results': pillar_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Generate random scenario
# ─────────────────────────────────────────────────────────────────────────────

def _regenerate():
    """Roll random pillars + lot position, simulate both CCW and CW starts."""
    ccw_p, cw_p = generate_pillars()
    lot_x = float(np.random.uniform(115, 185))

    ccw_secs = build_ccw_sections(ccw_p)
    cw_secs  = build_cw_sections(cw_p)

    runs = []
    for start_is_ccw in (True, False):
        r = compute_run(ccw_secs, cw_secs, start_is_ccw, lot_x)
        pills = list(ccw_p.values()) if start_is_ccw else list(cw_p.values())
        r['pills'] = pills
        runs.append(r)
        n = len(r['frames'])
        rev_tag = '  [LAP3 REVERSED]' if r['lap3_reversed'] else ''
        print(f"  {r['dir']}: {n} frames ({n*DT:.1f} s)  lot_x={lot_x:.0f}{rev_tag}")

    return runs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("WRO 2026 Pygame Simulation  —  3 laps + parking")
    print("  Generating random pillars + lot position...")

    pygame.init()
    pygame.display.set_caption("WRO 2026 Future Engineers  —  3-Lap Obstacle Challenge")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 19, bold=True)
    font_med = pygame.font.SysFont("consolas", 14)
    font_sm  = pygame.font.SysFont("consolas", 12)
    fonts    = (font_big, font_med, font_sm)

    runs = _regenerate()

    run_idx    = 0
    frame_idx  = 0
    step_acc   = 0.0
    trail      = collections.deque(maxlen=TRAIL_N)
    full_trail = []
    paused     = False
    fast       = False
    state      = 'run'
    trans_timer = 0
    track_surf  = None

    def load_run(idx):
        nonlocal run_idx, frame_idx, step_acc, trail, full_trail, state, track_surf
        run_idx    = idx
        frame_idx  = 0
        step_acc   = 0.0
        trail.clear()
        full_trail = []
        state      = 'run'
        track_surf = make_track_surface(runs[idx], font_sm)

    load_run(0)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    pygame.quit(); return
                if k == pygame.K_SPACE:
                    paused = not paused
                if k == pygame.K_f:
                    fast = not fast
                if k == pygame.K_r:
                    print("  Re-rolling pillars + lot position...")
                    runs[:] = _regenerate()
                    load_run(0); paused = False

        run   = runs[run_idx]
        frames = run['frames']
        total  = len(frames)
        car_col = run['color']

        # Advance — speed-proportional with 3× base multiplier, 0.5 floor
        if not paused:
            if state == 'run':
                fi_peek   = min(frame_idx, total - 1)
                cur_speed = frames[fi_peek][4]
                speed_ratio = max(0.5, cur_speed / BASE_SPEED)
                mult        = 15.0 if fast else 3.0
                step_acc   += mult * speed_ratio
                advance     = int(step_acc)
                step_acc   -= advance
                frame_idx   = min(frame_idx + advance, total)
                if frame_idx >= total:
                    state       = 'transition'
                    trans_timer = FPS * 2

            elif state == 'transition':
                trans_timer -= 1
                if trans_timer <= 0:
                    nxt = run_idx + 1
                    if nxt < len(runs):
                        load_run(nxt)
                        run     = runs[run_idx]
                        frames  = run['frames']
                        total   = len(frames)
                        car_col = run['color']
                    else:
                        state = 'done'

        # Draw
        screen.blit(track_surf, (0, 0))

        fi = max(0, min(frame_idx - 1, total - 1))
        if frame_idx > 0:
            x, y, theta, steer, speed, sec_i, sec_label = frames[fi]
            parked = (frame_idx >= total)

            pt = w2s(x, y)
            trail.append(pt)
            if not full_trail or full_trail[-1] != pt:
                full_trail.append(pt)

            draw_full_path(screen, full_trail, car_col)
            draw_trail(screen, trail, car_col)
            draw_car(screen, x, y, theta, car_col)

            # Pillar pass badges — shown once the car has passed each swerve
            for end_f, px, py, min_d, pc in run.get('pillar_results', []):
                if frame_idx >= end_f:
                    hit  = min_d < PILLAR_HIT_CM
                    bcol = RED_C if hit else GRN_C
                    bx, by = w2s(px, py)
                    pygame.draw.circle(screen, bcol, (bx, by - int(9*SCALE)), int(5*SCALE))
                    badge = font_sm.render(
                        f"!!{min_d:.0f}" if hit else f"{min_d:.0f}",
                        True, bcol)
                    screen.blit(badge, (bx - badge.get_width()//2,
                                        by - int(9*SCALE) - badge.get_height()))

            if parked:
                cx_s, cy_s = w2s(x, y)
                for angle in range(0, 360, 40):
                    a  = math.radians(angle)
                    ex = int(cx_s + 20 * math.cos(a))
                    ey = int(cy_s + 20 * math.sin(a))
                    pygame.draw.line(screen, GOLD_C, (cx_s,cy_s), (ex,ey), 2)
                pygame.draw.circle(screen, GOLD_C, (cx_s,cy_s), 6)

            draw_panel(screen, fonts, run, sec_label, speed, steer,
                       frame_idx * DT, frame_idx, total, parked)
        else:
            draw_panel(screen, fonts, run, '—', 0.0, 0.0,
                       0.0, 0, total, False)

        # Title
        ttl = font_big.render(
            f"WRO 2026 Future Engineers   |   {run['dir']} start   |   3 Laps + Parking",
            True, TXT_W)
        screen.blit(ttl, (T_LEFT, 10))

        if fast:
            f_s = font_sm.render("FAST 5x", True, (255,140,0))
            screen.blit(f_s, (T_LEFT + TW - f_s.get_width() - 6, T_TOP + 3))

        if paused:
            ov = font_big.render("--  PAUSED  --", True, GOLD_C)
            screen.blit(ov, (T_LEFT + TW//2 - ov.get_width()//2,
                              T_TOP  + TH//2 - ov.get_height()//2))

        if state == 'transition':
            nxt_dir = runs[run_idx+1]['dir'] if run_idx+1 < len(runs) else None
            if nxt_dir:
                msg = f"Done!  Next: {nxt_dir} start in {trans_timer//FPS + 1}s..."
            else:
                msg = "All runs complete.  Press R to re-roll."
            ov = font_big.render(msg, True, GOLD_C)
            alpha_s = pygame.Surface(ov.get_size(), pygame.SRCALPHA)
            alpha_s.fill((0,0,0,0))
            alpha_s.blit(ov, (0,0))
            alpha_s.set_alpha(min(255, trans_timer * 3))
            screen.blit(alpha_s, (T_LEFT + TW//2 - ov.get_width()//2,
                                   T_TOP  + TH//2 + 40))

        if state == 'done':
            ov = font_big.render(
                "SIMULATION COMPLETE     Press R to re-roll", True, GOLD_C)
            screen.blit(ov, (WIN_W//2 - ov.get_width()//2, T_TOP + TH//2))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
