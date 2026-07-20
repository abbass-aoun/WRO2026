"""
config.py  —  Single source of truth for ALL robot and simulation parameters.
==============================================================================

Change values here for testing and competition tuning.
Every module imports from this file — you never need to hunt through
multiple source files to adjust a constant.

HOW TO USE ON COMPETITION DAY:
    1. Open this file only.
    2. Adjust the TUNE ON REAL ROBOT values for the measured robot.
    3. Save and re-run any simulation or the real control loop.

SECTIONS:
    Robot Geometry     — physical dimensions measured on the real robot
    Motion             — speed and control loop rate
    PID Steering       — cross-track + heading PID gains
    Ackermann FF       — feedforward corner steer pre-angle
    Track Geometry     — WRO 2026 field spec (do NOT change)
    Pillar Swerve      — bypass clearance from pillar centre
    Parking Lot        — auto-computed from robot length
    EKF / Estimation   — Kalman filter noise parameters
    Simulation         — noise model, thresholds, stat runs (sim only)
"""

import math

# ─────────────────────────────────────────────────────────────────────────────
# Robot Geometry                                         TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
WHEELBASE_CM    = 16.5   # front-to-rear axle distance (cm)
ROBOT_LENGTH_CM = 30.0   # overall body length (cm)
ROBOT_WIDTH_CM  = 12.0   # overall body width  (cm)
SERVO_MAX_DEG   = 27.0   # maximum steering angle each way (deg)

# ─────────────────────────────────────────────────────────────────────────────
# Motion                                                 TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
BASE_SPEED_CM_S = 40.0   # nominal driving speed on straights (cm/s)
MIN_SPEED_CM_S  =  8.0   # speed floor at tight corners (cm/s)
A_LAT_MAX       = 12.5   # lateral acceleration limit (cm/s²)
                          # corner speed = sqrt(A_LAT_MAX / |curvature|)
                          # at R=50 cm: v_corner = sqrt(12.5/0.02) = 25 cm/s
CONTROL_HZ      = 50     # control loop frequency (Hz)
DT_S            = 1.0 / CONTROL_HZ   # = 0.02 s per tick

# ─────────────────────────────────────────────────────────────────────────────
# PID Steering                                           TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
# Combined error:  err = CTE_cm  +  HEADING_W * heading_error_rad
# Ackermann feedforward handles corner geometry; PID corrects residual drift.
#
# Tuning guide:
#   Start: Kp=1.0, Ki=0, Kd=0, HEADING_W=6
#   Increase Kp until car tracks straights, backs off if oscillating
#   Raise HEADING_W to reduce heading lag at corner entry
#   Add small Kd only if oscillations persist
PID_KP         =  1.1   # proportional gain
PID_KI         =  0.0   # integral gain  (add only if steady offset observed)
PID_KD         =  0.0   # derivative gain
PID_HEADING_W  =  8.0   # heading error weight (cm / rad)
PID_WINDUP_LIM = 20.0   # integral anti-windup clamp (deg)

# ─────────────────────────────────────────────────────────────────────────────
# Ackermann Feedforward                                  TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
# Pre-steers the servo at corners so the PID only needs to correct residual error.
# Applied ONLY on Corner sections; never on pillar swerves.
# Formula: steer_ff = -atan(WHEELBASE * clamp(kappa, ±1/CORNER_RADIUS))
#
# CORNER_RADIUS_CM is clamped to keep the cubic Bezier approximation from
# over-saturating the servo (the Bezier endpoints have kappa ~60% above design).
CORNER_RADIUS_CM = 50.0   # 90-degree arc design radius (cm)

# ─────────────────────────────────────────────────────────────────────────────
# Track Geometry  (WRO 2026 specification — do NOT change)
# ─────────────────────────────────────────────────────────────────────────────
TRACK_CM    = 300.0   # outer mat side length (square)
INNER_LO_CM = 100.0   # inner obstacle x/y start
INNER_HI_CM = 200.0   # inner obstacle x/y end
CL_CM       =  50.0   # centreline near wall (y=50 bottom, x=50 left)
CH_CM       = 250.0   # centreline far  wall (y=250 top,   x=250 right)

# ─────────────────────────────────────────────────────────────────────────────
# Pillar Swerve                                          TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
# Bypass waypoint lateral offset from pillar centre (cm).
# Must be > half-pillar (2.5 cm) + half-car-width + safety margin.
PILLAR_CLEARANCE_CM = 20.0   # lateral offset from pillar centre
PILLAR_BODY_CM      =  5.0   # pillar side length (WRO 2026 spec: 50×50 mm)

# ─────────────────────────────────────────────────────────────────────────────
# Parking Lot                             (TUNE LOT_DEPTH if robot changes)
# ─────────────────────────────────────────────────────────────────────────────
# WRO 2026 rules: lot width fixed at 20 cm; depth = 1.5 × robot length.
# Update LOT_DEPTH_CM if ROBOT_LENGTH_CM changes.
LOT_WIDTH_CM  = 20.0                    # gap between marker blocks (WRO fixed)
LOT_DEPTH_CM  = 1.5 * ROBOT_LENGTH_CM  # = 27.0 cm at 18.0 cm robot
LOT_THETA_RAD = -math.pi / 2           # heading to enter lot (South = -π/2)
MARKER_W_CM   =  2.0                   # magenta marker block thickness (cm)
MARKER_L_CM   = 20.0                   # magenta marker block length    (cm)

# ─────────────────────────────────────────────────────────────────────────────
# EKF / Estimation                                       TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
EKF_Q_XY_CM2   = 0.10    # position  process noise per step (cm²)
EKF_Q_THETA_R2 = 0.0002  # heading   process noise per step (rad²)
EKF_R_IMU_R2   = (math.pi / 180.0) ** 2   # BNO055 absolute heading noise (~1 deg)
EKF_R_GYRO_R2  = (0.003) ** 2             # gyro angular-rate noise (rad²/step)
EKF_IMU_PERIOD = 5        # absolute IMU update every N ticks (50 Hz ÷ 5 = 10 Hz)

# ─────────────────────────────────────────────────────────────────────────────
# Simulation Only  (no effect on the real robot)
# ─────────────────────────────────────────────────────────────────────────────
SIM_NOISE_XY_CM     = 0.04    # Gaussian xy  noise per step (cm, 1-sigma)
SIM_NOISE_THETA_RAD = 0.0004  # Gaussian yaw noise per step (rad, 1-sigma)
SIM_PILLAR_MISS_CM  = 40.0    # flag: car never got within this distance of a pillar
SIM_PILLAR_HIT_CM   =  5.0    # flag: car came this close (sign body overlap)
SIM_STAT_RUNS       =  3      # random scenarios for end-of-run statistics (0 = skip)

# ─────────────────────────────────────────────────────────────────────────────
# Safety / Debug  (bring-up flags — adjust per test session, then lock in)
# ─────────────────────────────────────────────────────────────────────────────
# MOTOR_INVERTED : True  → 'f' command wired to backward; swaps direction in
#                  CarController so the rest of the code never needs to change.
# MAX_DUTY_SAFE  : hard PWM ceiling (0–1) applied in main.py before every
#                  motor command.  0.35 limits bench-test speed; raise to
#                  0.60–0.70 once straight-line tests pass.
# SHOW_VISION_DEBUG : open a live camera overlay window on the main thread.
#                     Needs a connected display.  Leave False at competition.
MOTOR_INVERTED    = True   # flip to True if robot drives backward on 'f'
MAX_DUTY_SAFE     = 0.35    # TUNE: raise gradually after bench tests pass
SHOW_VISION_DEBUG = False   # True = live camera window (display must be attached)