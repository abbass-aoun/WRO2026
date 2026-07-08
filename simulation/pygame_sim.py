"""
WRO 2026 Real-Time Race Simulation  --  pygame Visualiser
==========================================================

Animated top-down race visualisation.  Pre-computes both CCW and CW
trajectories (bicycle model + PID), then plays them back at 60 FPS
with a dark "racing game" aesthetic.

Run from project root:
    python -m simulation.pygame_sim

Controls:
    SPACE   pause / resume
    F       fast mode (5x)
    R       restart from the beginning
    ESC     quit

What you see:
    Left panel   -- 300 x 300 cm WRO 2026 field (to scale at 2 px/cm)
                    * Orange diagonal lines  = CCW corner markers
                    * Blue   diagonal lines  = CW  corner markers
                    * Rounded gray square    = inner obstacle
                    * Pink rectangle         = parking lot
                    * Coloured dots          = pillars  (R=red, G=green)
                    * Dim coloured lines     = planned Bezier paths
                    * Bright coloured line   = actual car path (with trail)
                    * Car sprite             = rotated rectangle, yellow front

    Right panel  -- live telemetry:
                    * Speed gauge
                    * Steering angle bar  (+/- 27 deg servo limit)
                    * Current section name
                    * Lap progress bar
                    * Elapsed time

Simulation runs CCW first (2 seconds gap), then CW automatically.
"""

import sys
import math
import collections

try:
    import pygame
except ImportError:
    sys.exit("pygame not found.  Run:   pip install pygame")

import numpy as np

# ── Import back-end from race_sim ─────────────────────────────────────────────
from simulation.race_sim import (
    generate_pillars,
    build_ccw_sections, build_cw_sections,
    simulate_section,
    CCW_LOT_X, CCW_LOT_Y,
    CW_LOT_X,  CW_LOT_Y,
    LOT_THETA, LOT_WIDTH, LOT_DEPTH, MARKER_W, MARKER_L,
    BASE_SPEED, DT, WHEELBASE,
)
from trajectory.builder import TrajectoryBuilder, RED, GREEN

# ─────────────────────────────────────────────────────────────────────────────
# Layout constants
# ─────────────────────────────────────────────────────────────────────────────
SCALE   = 2.0             # pixels per cm
TW      = int(300 * SCALE)  # 600  track width  (px)
TH      = int(300 * SCALE)  # 600  track height (px)
T_LEFT  = 18              # track left margin
T_TOP   = 44              # track top  margin (below title)
PAN_X   = T_LEFT + TW + 14
PAN_W   = 262
WIN_W   = PAN_X + PAN_W + 6
WIN_H   = T_TOP + TH + 28
FPS     = 60
TRAIL_N = 280             # fading-tail length (frames) — full path stored separately

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────
BG       = (11,  11,  22)
MAT      = (236, 236, 236)
WALL_C   = (22,  22,  22)
INNER_F  = (160, 160, 160)
INNER_E  = (34,  34,  34)
ORG_L    = (255, 102,   0)   # CMYK(0,60,100,0)  — official WRO orange
BLU_L    = (  0,  51, 255)   # CMYK(100,80,0,0)  — official WRO blue
CL_COL   = (179, 179, 179)   # CMYK(0,0,0,30)    — dashed lines
LOT_F    = (255, 224, 255)   # light magenta fill
LOT_E    = (255,   0, 255)   # RGB(255,0,255)     — official parking marker
C_CCW    = (  0, 232,  82)   # neon green
C_CW     = ( 52, 158, 255)   # neon blue
FRONT_C  = (255, 238,  95)
R_PIK    = (238,  39,  55)   # RGB(238,39,55)     — official red sign
G_PIK    = ( 68, 214,  44)   # RGB(68,214,44)     — official green sign
SEAT_COL = (179, 179, 179)   # CMYK(0,0,0,30)
CIRC_COL = (204, 255,   0)   # CMYK(20,0,100,0)   — evaluation circle
PAN_BG   = ( 17,  17,  30)
TXT_W    = (212, 212, 212)
TXT_D    = (100, 100, 122)
GOLD_C   = (255, 208,  48)
BAR_BG   = ( 42,  42,  60)
WHITE    = (255, 255, 255)
BLACK    = (  0,   0,   0)

# Planned path colours
COL_SWV  = (200,  75,  10)   # pillar swerve
COL_COR  = (130,  45, 185)   # 90-degree corner
COL_STR  = ( 28,  85, 178)   # straight
COL_PRK  = (120,  95,   8)   # parking approach


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def w2s(x, y):
    """World cm → screen pixels.  Y axis flipped (world Y-up, screen Y-down)."""
    return (int(T_LEFT + x * SCALE),
            int(T_TOP  + (300 - y) * SCALE))


def _draw_path(surf, path, color, width=2):
    n = max(60, int(path.total_length))
    pts = [w2s(*path.get_point(path.total_length * i / n)) for i in range(n + 1)]
    if len(pts) >= 2:
        pygame.draw.lines(surf, color, False, pts, width)


# ─────────────────────────────────────────────────────────────────────────────
# Static track surface  (built once per run)
# ─────────────────────────────────────────────────────────────────────────────

def make_track_surface(sections, park_path, lot_x, lot_y, pillars, fnt_sm):
    surf = pygame.Surface((WIN_W, WIN_H))
    surf.fill(BG)

    # ── White mat
    pygame.draw.rect(surf, MAT,
                     (T_LEFT+3, T_TOP+3, TW-6, TH-6), border_radius=10)
    pygame.draw.rect(surf, WALL_C,
                     (T_LEFT, T_TOP, TW, TH), 5, border_radius=10)

    # ── Inner obstacle (rounded grey square)
    x1, y1 = w2s(100, 200)
    x2, y2 = w2s(200, 100)
    pygame.draw.rect(surf, INNER_F,
                     pygame.Rect(x1, y1, x2-x1, y2-y1), border_radius=14)
    pygame.draw.rect(surf, INNER_E,
                     pygame.Rect(x1, y1, x2-x1, y2-y1), 3, border_radius=14)

    # ── Dashed centreline loop (50 cm and 250 cm rails)
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

    # ── Diagonal corner markers — 30° geometry (WRO 2026 Figure 11)
    # T30 = 100 * tan(30°) ≈ 57.7 cm — where each line meets the inner wall face
    # Orange: 30° from horizontal   Blue: 60° from horizontal (30° from vertical)
    T30 = 100 * math.tan(math.radians(30))   # ≈ 57.7
    def cl(a, b, col):
        pygame.draw.line(surf, col, w2s(*a), w2s(*b), 3)

    # BL outer (0,0)
    cl((0, 0),   (100, T30),       ORG_L)
    cl((0, 0),   (T30, 100),       BLU_L)
    # BR outer (300,0)
    cl((300, 0), (200, T30),       ORG_L)
    cl((300, 0), (300-T30, 100),   BLU_L)
    # TR outer (300,300)
    cl((300,300),(200, 300-T30),   ORG_L)
    cl((300,300),(300-T30, 200),   BLU_L)
    # TL outer (0,300)
    cl((0, 300), (100, 300-T30),   ORG_L)
    cl((0, 300), (T30, 200),       BLU_L)

    # ── Parking lot (light fill + two magenta marker blocks)
    lx1, ly1 = w2s(lot_x - LOT_WIDTH/2, lot_y)
    lx2, ly2 = w2s(lot_x + LOT_WIDTH/2, lot_y - LOT_DEPTH)
    lot_rect = pygame.Rect(lx1, ly1, lx2-lx1, ly2-ly1)
    pygame.draw.rect(surf, LOT_F, lot_rect)
    pygame.draw.rect(surf, LOT_E, lot_rect, 1)
    p_s = fnt_sm.render("PARK", True, LOT_E)
    surf.blit(p_s, ((lx1+lx2)//2 - p_s.get_width()//2,
                    (ly1+ly2)//2 - p_s.get_height()//2))
    # Left marker block
    mx1, my1 = w2s(lot_x - LOT_WIDTH/2 - MARKER_W, lot_y)
    mx2, my2 = w2s(lot_x - LOT_WIDTH/2,             lot_y - MARKER_L)
    pygame.draw.rect(surf, LOT_E, pygame.Rect(mx1, my1, mx2-mx1, my2-my1))
    # Right marker block
    rx1, ry1 = w2s(lot_x + LOT_WIDTH/2,             lot_y)
    rx2, ry2 = w2s(lot_x + LOT_WIDTH/2 + MARKER_W,  lot_y - MARKER_L)
    pygame.draw.rect(surf, LOT_E, pygame.Rect(rx1, ry1, rx2-rx1, ry2-ry1))

    # ── Planned path overlay  (dim so the car stands out)
    for path, label, _ in sections:
        has_pillar = ('+' in label)
        is_corner  = ('Corner' in label)
        col = COL_SWV if has_pillar else COL_COR if is_corner else COL_STR
        _draw_path(surf, path, col, 2)
    if park_path is not None:
        _draw_path(surf, park_path, COL_PRK, 2)

    # ── Starting zone  200x500 mm = 20x50 cm, dashed CMYK(0,0,0,30)
    # 20 cm centred at x=150 along the bottom straight, y=0..50
    sz_pts = [w2s(140, 0), w2s(160, 0), w2s(160, 50), w2s(140, 50), w2s(140, 0)]
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

    # ── Traffic sign seats + evaluation circles at pillar positions
    for px, py, c in pillars:
        scx, scy = w2s(px, py)
        # Evaluation circle dia=85mm=8.5cm → r=4.25cm → 4.25*SCALE px
        r_circ = int(4.25 * SCALE)
        pygame.draw.circle(surf, CIRC_COL, (scx, scy), r_circ, 1)
        # Seat square 50x50mm=5x5cm
        half = int(2.5 * SCALE)
        pygame.draw.rect(surf, SEAT_COL,
                         pygame.Rect(scx-half, scy-half, 2*half, 2*half), 1)

    # ── Pillars  (50x50mm = 5x5cm square body, official RGB colors)
    for px, py, c in pillars:
        col = R_PIK if c == RED else G_PIK
        scx, scy = w2s(px, py)
        half = int(2.5 * SCALE)   # 2.5 cm * 2 px/cm = 5 px
        # Soft glow
        glow = tuple(min(255, v // 3 + 20) for v in col)
        pygame.draw.rect(surf, glow,
                         pygame.Rect(scx-half-4, scy-half-4, 2*half+8, 2*half+8),
                         border_radius=3)
        # Pillar body (5x5cm square)
        pygame.draw.rect(surf, col,
                         pygame.Rect(scx-half, scy-half, 2*half, 2*half))
        pygame.draw.rect(surf, WHITE,
                         pygame.Rect(scx-half, scy-half, 2*half, 2*half), 1)
        lbl = fnt_sm.render("R" if c == RED else "G", True, WHITE)
        surf.blit(lbl, (scx - lbl.get_width()//2, scy - half - 14))

    return surf


# ─────────────────────────────────────────────────────────────────────────────
# Car sprite
# ─────────────────────────────────────────────────────────────────────────────

def draw_car(surf, x, y, theta, color, length_cm=18.0, width_cm=12.0):
    """Rotated rectangle with front indicator stripe."""
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    hl, hw = length_cm / 2, width_cm / 2

    def corner(fx, fy):
        return w2s(x + fx*cos_t - fy*sin_t,
                   y + fx*sin_t + fy*cos_t)

    pts = [corner(hl,hw), corner(hl,-hw),
           corner(-hl,-hw), corner(-hl,hw)]

    # Drop shadow
    shadow = [(px+3, py+3) for px, py in pts]
    pygame.draw.polygon(surf, (0, 0, 0), shadow)
    # Body
    pygame.draw.polygon(surf, color, pts)
    # Outline
    dark = tuple(max(0, c - 75) for c in color)
    pygame.draw.polygon(surf, dark, pts, 2)
    # Front stripe (bright yellow)
    pygame.draw.line(surf, FRONT_C, pts[0], pts[1], 3)
    # Centre dot
    pygame.draw.circle(surf, FRONT_C, w2s(x, y), 2)


# ─────────────────────────────────────────────────────────────────────────────
# Trail
# ─────────────────────────────────────────────────────────────────────────────

def draw_full_path(surf, full_trail, color):
    """Thin dim line showing every position visited since lap start."""
    if len(full_trail) < 2:
        return
    dim = tuple(max(0, int(c * 0.38)) for c in color)
    pygame.draw.lines(surf, dim, False, list(full_trail), 1)


def draw_trail(surf, trail, color):
    """Bright fading tail near the car."""
    n = len(trail)
    if n < 2:
        return
    for i, pos in enumerate(trail):
        t = i / n                   # 0 = oldest (dim), 1 = newest (bright)
        r = int(BG[0] + (color[0]-BG[0]) * t)
        g = int(BG[1] + (color[1]-BG[1]) * t)
        b = int(BG[2] + (color[2]-BG[2]) * t)
        pygame.draw.circle(surf, (r, g, b), pos, max(1, int(t * 3.5)))


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry panel
# ─────────────────────────────────────────────────────────────────────────────

def draw_panel(surf, fonts, direction, sec_name, sec_i,
               speed, steer, elapsed, frame_i, total_frames,
               car_color, parked):
    font_big, font_med, font_sm = fonts
    pw = PAN_W
    pygame.draw.rect(surf, PAN_BG, (PAN_X, T_TOP, pw, TH))
    pygame.draw.rect(surf, (38, 38, 55), (PAN_X, T_TOP, pw, TH), 1)

    pad = 11
    y = T_TOP + 12
    bw = pw - 2*pad   # bar width

    def text(msg, fnt, col=TXT_W, ox=0):
        nonlocal y
        s = fnt.render(msg, True, col)
        surf.blit(s, (PAN_X + pad + ox, y))
        y += s.get_height()

    def gap(n=7):
        nonlocal y; y += n

    def hbar(val, lo, hi, fg, height=15, center=False):
        nonlocal y
        rect = pygame.Rect(PAN_X+pad, y, bw, height)
        pygame.draw.rect(surf, BAR_BG, rect, border_radius=4)
        if center:
            mx = PAN_X + pad + bw//2
            pygame.draw.line(surf, TXT_D, (mx, y), (mx, y+height), 1)
            frac = max(-1, min(1, val / max(abs(lo), abs(hi), 1)))
            fw = int(abs(frac) * bw//2)
            ox = mx if frac >= 0 else mx - fw
            if fw > 1:
                pygame.draw.rect(surf, fg,
                                 pygame.Rect(ox, y, fw, height), border_radius=3)
        else:
            frac = max(0, min(1, (val-lo) / max(hi-lo, 1)))
            fw = int(frac * bw)
            if fw > 1:
                pygame.draw.rect(surf, fg,
                                 pygame.Rect(PAN_X+pad, y, fw, height), border_radius=4)
        y += height

    # ── Direction header
    sym = "◄◄  CCW" if direction == "CCW" else "CW  ►► "
    text(sym, font_big, car_color); gap(5)

    status_col  = GOLD_C if parked else TXT_W
    status_text = "** PARKED **" if parked else "  RACING  "
    text(status_text, font_med, status_col); gap(14)

    # ── Speed
    text("SPEED", font_sm, TXT_D); gap(4)
    hbar(speed, 0, BASE_SPEED, car_color)
    gap(2)
    text(f"  {speed:.1f}  /  {BASE_SPEED:.0f} cm/s", font_sm); gap(10)

    # ── Steering
    steer_col = (220,75,75) if steer > 2 else (75,220,75) if steer < -2 else (140,140,140)
    text("STEERING", font_sm, TXT_D); gap(4)
    hbar(steer, -27, 27, steer_col, center=True)
    gap(2)
    text(f"  {steer:+.1f} deg  (limit ±27)", font_sm); gap(10)

    # ── Section
    text("SECTION", font_sm, TXT_D); gap(3)
    name = sec_name.split('+')[0].strip()[:20]
    text(f"{sec_i:02d}  {name}", font_med); gap(10)

    # ── Progress
    text("LAP PROGRESS", font_sm, TXT_D); gap(4)
    hbar(frame_i, 0, max(total_frames,1), car_color, height=10)
    gap(3)
    pct = int(100 * frame_i // max(total_frames,1))
    text(f"  {pct}%", font_sm); gap(10)

    # ── Time
    text("ELAPSED", font_sm, TXT_D); gap(4)
    t_s = font_big.render(f"  {elapsed:.1f} s", True, TXT_W)
    surf.blit(t_s, (PAN_X+pad, y)); y += t_s.get_height(); gap(14)

    # ── Pillar legend
    text("PILLAR RULE", font_sm, TXT_D); gap(4)
    pygame.draw.circle(surf, R_PIK, (PAN_X+pad+7, y+6), 6)
    text("   RED  = pass RIGHT", font_sm, TXT_W); gap(2)
    pygame.draw.circle(surf, G_PIK, (PAN_X+pad+7, y+6), 6)
    text("   GRN  = pass LEFT",  font_sm, TXT_W); gap(14)

    # ── Path legend
    for col, lbl in [
        (COL_SWV, "Pillar swerve"),
        (COL_COR, "Corner (Bezier)"),
        (COL_STR, "Straight"),
        (COL_PRK, "Parking curve"),
    ]:
        pygame.draw.rect(surf, col, (PAN_X+pad, y+4, 16, 6), border_radius=2)
        text(f"   {lbl}", font_sm, TXT_D, ox=16); gap(1)
    gap(10)

    # ── Controls  (pinned to bottom)
    y = T_TOP + TH - 78
    text("SPACE  pause / resume", font_sm, TXT_D)
    gap(1)
    text("F      fast (5×)",      font_sm, TXT_D)
    gap(1)
    text("R      restart",        font_sm, TXT_D)
    gap(1)
    text("ESC    quit",           font_sm, TXT_D)


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory pre-computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_run(sections, start_theta, lot_x, lot_y):
    """
    Run bicycle-model simulation through all sections + parking approach.
    Returns (frames_list, park_path) where each frame is a tuple:
        (x, y, theta, steer_deg, speed_cm_s, sec_index, sec_label)
    """
    frames = []
    cx, cy, cth = 150.0, 50.0, float(start_theta)

    for sec_i, (path, label, _) in enumerate(sections):
        xs, ys, ths, sts, sps = simulate_section(path, cx, cy, cth)
        for row in zip(xs, ys, ths, sts, sps):
            frames.append((*row, sec_i, label))
        cx, cy, cth = xs[-1], ys[-1], ths[-1]

    park_path = TrajectoryBuilder.parking_approach(
        cx, cy, cth, lot_x, lot_y, LOT_THETA)
    xs, ys, ths, sts, sps = simulate_section(park_path, cx, cy, cth)
    for row in zip(xs, ys, ths, sts, sps):
        frames.append((*row, 8, "Parking approach"))

    return frames, park_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Pre-compute (this happens before the window opens — gives clean start)
    print("WRO 2026 Pygame Simulation")
    print("  Generating random pillars...")
    ccw_p, cw_p = generate_pillars()

    print("  Building CCW sections...")
    ccw_secs = build_ccw_sections(ccw_p)
    print("  Building CW  sections...")
    cw_secs  = build_cw_sections(cw_p)

    print("  Simulating CCW trajectory (bicycle model)...")
    ccw_frames, ccw_park = compute_run(ccw_secs, 0.0, CCW_LOT_X, CCW_LOT_Y)
    print("  Simulating CW  trajectory (bicycle model)...")
    cw_frames,  cw_park  = compute_run(cw_secs, math.pi, CW_LOT_X, CW_LOT_Y)

    ccw_pills = list(ccw_p.values())   # 4 pillars: sec0,sec2,sec4,sec6
    cw_pills  = list(cw_p.values())

    print(f"  CCW: {len(ccw_frames)} frames ({len(ccw_frames)*DT:.1f} s sim)")
    print(f"  CW:  {len(cw_frames)}  frames ({len(cw_frames)*DT:.1f} s sim)")
    print("  Opening window...")

    # ── Pygame init
    pygame.init()
    pygame.display.set_caption("WRO 2026 Future Engineers  —  Real-Time Simulation")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    font_big = pygame.font.SysFont("consolas", 19, bold=True)
    font_med = pygame.font.SysFont("consolas", 14)
    font_sm  = pygame.font.SysFont("consolas", 12)
    fonts = (font_big, font_med, font_sm)

    # ── Run registry
    runs = [
        ('CCW', C_CCW, ccw_frames, ccw_secs, ccw_park, CCW_LOT_X, CCW_LOT_Y, ccw_pills),
        ('CW',  C_CW,  cw_frames,  cw_secs,  cw_park,  CW_LOT_X,  CW_LOT_Y,  cw_pills),
    ]

    # ── Mutable state
    run_idx     = 0
    frame_idx   = 0
    trail       = collections.deque(maxlen=TRAIL_N)
    full_trail  = []          # every position since lap start (unbounded)
    paused      = False
    fast        = False
    state       = 'run'     # 'run' | 'transition' | 'done'
    trans_timer = 0
    track_surf  = None

    def load_run(idx):
        nonlocal run_idx, frame_idx, trail, full_trail, state, track_surf
        run_idx   = idx
        frame_idx = 0
        trail.clear()
        full_trail = []
        state = 'run'
        _, _, _, secs, park, lx, ly, pills = runs[idx]
        track_surf = make_track_surface(secs, park, lx, ly, pills, font_sm)

    load_run(0)

    # ── Main loop
    while True:
        # Events
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
                    load_run(0); paused = False

        # Unpack current run
        direction, car_col, frames, _, _, lot_x, lot_y, _ = runs[run_idx]
        total = len(frames)

        # Advance
        if not paused:
            if state == 'run':
                frame_idx = min(frame_idx + (5 if fast else 1), total)
                if frame_idx >= total:
                    state       = 'transition'
                    trans_timer = FPS * 2      # 2-second pause between runs

            elif state == 'transition':
                trans_timer -= 1
                if trans_timer <= 0:
                    nxt = run_idx + 1
                    if nxt < len(runs):
                        load_run(nxt)
                        direction, car_col, frames, _, _, lot_x, lot_y, _ = runs[run_idx]
                        total = len(frames)
                    else:
                        state = 'done'

        # ── Draw static track
        screen.blit(track_surf, (0, 0))

        # ── Dynamic: car + trail
        fi = max(0, min(frame_idx - 1, total - 1))
        if frame_idx > 0:
            x, y, theta, steer, speed, sec_i, sec_name = frames[fi]
            parked = (frame_idx >= total)

            pt = w2s(x, y)
            trail.append(pt)
            if not full_trail or full_trail[-1] != pt:
                full_trail.append(pt)

            draw_full_path(screen, full_trail, car_col)
            draw_trail(screen, trail, car_col)
            draw_car(screen, x, y, theta, car_col)

            # Gold starburst when parked
            if parked:
                cx_s, cy_s = w2s(x, y)
                for angle in range(0, 360, 40):
                    a = math.radians(angle)
                    ex = int(cx_s + 20 * math.cos(a))
                    ey = int(cy_s + 20 * math.sin(a))
                    pygame.draw.line(screen, GOLD_C, (cx_s, cy_s), (ex, ey), 2)
                pygame.draw.circle(screen, GOLD_C, (cx_s, cy_s), 6)

            draw_panel(screen, fonts, direction, sec_name, sec_i,
                       speed, steer, frame_idx * DT,
                       frame_idx, total, car_col, parked)
        else:
            draw_panel(screen, fonts, direction, "—", 0,
                       0.0, 0.0, 0.0, 0, total, car_col, False)

        # ── Title bar
        ttl = font_big.render(
            f"WRO 2026 Future Engineers   |   {direction} Direction", True, TXT_W)
        screen.blit(ttl, (T_LEFT, 10))

        if fast:
            f_s = font_sm.render("FAST 5×", True, (255, 140, 0))
            screen.blit(f_s, (T_LEFT + TW - f_s.get_width() - 6, T_TOP + 3))

        # ── Overlays
        if paused:
            ov = font_big.render("——  PAUSED  ——", True, GOLD_C)
            screen.blit(ov, (T_LEFT + TW//2 - ov.get_width()//2,
                             T_TOP  + TH//2 - ov.get_height()//2))

        if state == 'transition':
            nxt_dir = runs[run_idx+1][0] if run_idx+1 < len(runs) else None
            if nxt_dir:
                msg = f"CCW DONE!   Starting CW in {trans_timer//FPS + 1}s..."
            else:
                msg = "All runs complete.  Press R to replay."
            ov = font_big.render(msg, True, GOLD_C)
            alpha_surf = pygame.Surface(ov.get_size(), pygame.SRCALPHA)
            alpha_surf.fill((0, 0, 0, 0))
            alpha_surf.blit(ov, (0, 0))
            alpha_surf.set_alpha(min(255, trans_timer * 3))
            screen.blit(alpha_surf,
                        (T_LEFT + TW//2 - ov.get_width()//2, T_TOP + TH//2 + 40))

        if state == 'done':
            ov = font_big.render(
                "SIMULATION COMPLETE     Press R to replay", True, GOLD_C)
            screen.blit(ov, (WIN_W//2 - ov.get_width()//2, T_TOP + TH//2))

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    main()
