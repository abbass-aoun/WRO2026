# WRO 2026 — Future Engineers Self-Driving Car

Raspberry Pi–based autonomous vehicle for the WRO 2026 Future Engineers challenge.  
Runs a 50 Hz closed-loop controller that fuses wheel encoders and a gyroscope, follows pre-planned Bézier trajectories, avoids traffic-sign pillars, completes 3 laps, and parks automatically.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Hardware](#2-hardware)
3. [Software Architecture](#3-software-architecture)
4. [Module Reference](#4-module-reference)
   - [config.py](#41-configpy)
   - [control/](#42-control)
   - [estimation/](#43-estimation)
   - [trajectory/](#44-trajectory)
   - [main/](#45-main)
   - [simulation/](#46-simulation)
5. [The 50 Hz Control Loop](#5-the-50-hz-control-loop)
6. [Track Geometry](#6-track-geometry)
7. [Partner Integration](#7-partner-integration)
8. [Running the Robot](#8-running-the-robot)
9. [Running the Simulation](#9-running-the-simulation)
10. [Tuning Guide](#10-tuning-guide)

---

## 1. Project Overview

### What the robot does

- Detects race direction (CCW or CW) from the first colored floor line it crosses
- Completes 3 laps of the WRO track (8 sections per lap = 24 sections total)
- Avoids red and green traffic-sign pillars with smooth Bézier swerve paths
- After lap 3: if the last pillar seen in laps 1–2 was **RED**, reverses direction for lap 3 (WRO 2026 rule)
- Parks inside the marked lot at the end of the race

### Team split

| Partner | Responsibility |
|---------|---------------|
| This repo | Motion planning, state estimation (EKF), PID control, hardware I/O |
| Partner (cv-development branch) | Camera vision: pillar detection, parking lot detection |

The two halves connect at a single `VisionFrame` object constructed in `main/main.py` each tick.

---

## 2. Hardware

### Raspberry Pi pin assignments

All pin numbers are BCM (GPIO) numbers.

| Component | Signal | GPIO | Physical Pin |
|-----------|--------|------|--------------|
| L298N motor driver | IN1 | 18 | 12 |
| L298N motor driver | IN2 | 13 | 33 |
| L298N motor driver | ENA (PWM) | 19 | 35 |
| Steering servo | Signal | 12 | 32 |
| Left wheel encoder | IR pulse | 7 | 26 |
| Right wheel encoder | IR pulse | 5 | 29 |
| Start button | Signal | 8 | 24 |
| TCS3200 color sensor | S0 | 17 | 11 |
| TCS3200 color sensor | S1 | 27 | 13 |
| TCS3200 color sensor | S2 | 22 | 15 |
| TCS3200 color sensor | S3 | 23 | 16 |
| TCS3200 color sensor | OUT | 24 | 18 |
| TCS3200 color sensor | LED | 25 | 22 |
| VL53L0X ToF #1 XSHUT | — | 4 | 7 |
| VL53L0X ToF #2 XSHUT | — | 10 | 19 |
| VL53L0X ToF #3 XSHUT | — | 9 | 21 |
| VL53L0X ToF #4 XSHUT | — | 6 | 31 |
| MPU-6050 IMU | SDA | 2 | 3 |
| MPU-6050 IMU | SCL | 3 | 5 |

### Sensors

| Sensor | Purpose | Note |
|--------|---------|------|
| IR slot encoders ×2 | Wheel speed (cm/s) and distance | 50 pulses/rev, ~20.48 cm circumference |
| MPU-6050 | Yaw angular rate (rad/s) for EKF | No absolute heading — gyro only |
| TCS3200 color sensor | Detect orange/blue floor lines | Background thread, ~12 Hz update |
| VL53L0X ToF ×4 | Wall/obstacle distances in mm | ~55 ms per read; not in main loop |

### Software dependencies

```bash
pip install gpiozero mpu6050-raspberrypi smbus2 numpy pygame opencv-python
```

---

## 3. Software Architecture

```
WRO2026/
├── config.py                  # All tunable constants — edit here only
│
├── control/                   # Hardware I/O and control algorithms
│   ├── robot.py               # Car state (x, y, theta, speed, steer_angle)
│   ├── car_controller.py      # Motor + servo hardware wrapper
│   ├── allEncodersClass.py    # Wheel encoders + MPU-6050 gyro reader
│   ├── color_sensor.py        # TCS3200 floor-line detector (background thread)
│   ├── tof_sensor.py          # VL53L0X ToF sensors (4×, I2C address reassignment)
│   ├── steering_controller.py # Steering PID (cross-track + heading error)
│   ├── driving_controller.py  # Speed PID (curvature-adaptive target speed)
│   ├── pid_controller.py      # Base PID class (used by both controllers above)
│   ├── brake_controller.py    # Active braking PID (used at race end)
│   ├── servoClass.py          # Low-level servo PWM wrapper
│   └── servo_cal.py           # Servo angle ↔ PWM calibration
│
├── estimation/
│   └── ekf.py                 # Extended Kalman Filter — fuses encoders + gyro
│
├── trajectory/
│   ├── base.py                # Abstract interface all trajectories must implement
│   ├── waypoint_path.py       # Straight-line path (used for straight sections)
│   ├── bezier.py              # Cubic Bézier segment and multi-segment path
│   └── builder.py             # Factory: builds straight / corner / swerve / parking paths
│
├── main/
│   ├── main.py                # Entry point — the 50 Hz control loop
│   ├── race_manager.py        # Race state machine + track geometry + VisionFrame
│   ├── parking.py             # End-of-race parking state machine
│   └── ekf_runner.py          # EKF scheduling wrapper (reference / standalone use)
│
└── simulation/
    ├── pygame_sim.py          # Real-time Pygame visualiser
    └── race_sim.py            # Headless simulation runner (no display needed)
```

### Data flow (one tick)

```
Wheel encoders ──► speed_meas (cm/s)
MPU-6050 gyro  ──► omega_gyro (rad/s)
                                        ┌────────────────────────┐
Both sensors ──────────────────────────►│  EKF  (estimation)     │──► (x, y, θ)
                                        └────────────────────────┘
                                                   │
                                                   ▼
                                            Robot state object
                                                   │
TCS3200 color sensor  ──► orange_seen / blue_seen  │
Partner camera code   ──► pillars / parking_lot    │
                                                   ▼
                                        ┌────────────────────────┐
                                        │  RaceManager.update()  │──► trajectory
                                        └────────────────────────┘
                                                   │
                          ┌────────────────────────┴──────────────────────────┐
                          ▼                                                    ▼
               SteeringPIDController                              DrivingPIDController
               (CTE + heading error)                             (curvature-adaptive speed)
                          │                                                    │
                          ▼                                                    ▼
               + Ackermann feedforward                               duty cycle
               (corners only)                                                  │
                          │                                                    │
                          ▼                                                    ▼
                   car.set_steering(deg)                          car.set_motor('f', duty)
```

---

## 4. Module Reference

### 4.1 `config.py`

**Single source of truth for every tunable constant.**  
On competition day, only this file needs to be edited.

Key sections:

| Section | Constants | Notes |
|---------|-----------|-------|
| Robot Geometry | `WHEELBASE_CM`, `ROBOT_LENGTH_CM`, `SERVO_MAX_DEG` | Measure on physical robot |
| Motion | `BASE_SPEED_CM_S`, `CONTROL_HZ`, `DT_S` | 50 Hz loop → `DT_S = 0.02 s` |
| PID Steering | `PID_KP`, `PID_KI`, `PID_KD`, `PID_HEADING_W` | See tuning guide |
| Ackermann FF | `CORNER_RADIUS_CM` | 90° arc radius = 50 cm |
| Track Geometry | `TRACK_CM`, `CL_CM`, `CH_CM` | WRO spec — do not change |
| Pillar Swerve | `PILLAR_CLEARANCE_CM` | Lateral bypass offset from pillar centre |
| Parking | `LOT_WIDTH_CM`, `LOT_DEPTH_CM` | WRO spec + robot length |
| EKF | `EKF_Q_*`, `EKF_R_*`, `EKF_IMU_PERIOD` | Kalman filter noise parameters |
| Simulation | `SIM_NOISE_*`, `SIM_STAT_RUNS` | No effect on real robot |

---

### 4.2 `control/`

#### `robot.py` — Car state container

Holds the latest known values for `(x, y, theta, speed, steer_angle)`.  
Think of it as the car's in-memory dashboard.

- Written by: EKF (`update_pose`), encoders (`update_speed`), servo output (`update_steering`)
- Read by: controllers, race manager, main loop
- Coordinate system: x/y in cm, theta in radians (0 = East, positive = CCW)

```python
robot.x, robot.y, robot.theta   # position in world frame (cm, rad)
robot.speed                      # forward speed (cm/s)
robot.steer_angle                # current servo command (deg)
```

---

#### `car_controller.py` — Motor and servo hardware

Wraps gpiozero devices for the L298N motor driver and steering servo.

```python
car = CarController(IN1, IN2, ENA, SERVO_PIN)
car.set_motor('f', duty)     # forward at duty cycle 0.0–1.0
car.set_motor('b', duty)     # reverse
car.set_steering(angle_deg)  # servo angle in degrees
car.brake(encoders)          # PID active brake to zero speed
car.stop()                   # cut motor power
```

The servo is configured in `servoClass.py` with `center_angle=78°` and `max_deviation=27°` — tune on the real robot.

---

#### `allEncodersClass.py` — Wheel encoders + gyro

Reads IR slot-sensor pulses from both rear wheels via gpiozero interrupt callbacks.  
Also reads yaw rate from the MPU-6050 gyroscope.

```python
enc = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
v_l, v_r = enc.get_linear_speeds()   # cm/s per wheel
omega     = enc.get_yaw_rate()        # rad/s (from MPU-6050 gyro Z axis)
enc.reset()                           # zero counters at race start
```

Speed is estimated from time between pulses: `v = cm_per_pulse / dt_since_last_pulse`.  
If no pulse arrives for >1 s the wheel is considered stopped.

---

#### `color_sensor.py` — TCS3200 floor-line detector

Detects the orange start/finish line and blue section-divider lines painted on the WRO mat.

Runs in a **daemon background thread** (~12 Hz) so it never blocks the 50 Hz main loop.  
The main loop just reads two boolean flags:

```python
color = ColorSensor(S0, S1, S2, S3, OUT, LED)
color.orange_seen   # True when orange line is under the sensor
color.blue_seen     # True when blue line is under the sensor
color.stop()        # call at program exit
```

How it works:  
- S0/S1 pins set frequency scaling to 20% (mid-range sensitivity)
- S2/S3 pins select which color filter (Red/Green/Blue) the TCS3200 measures
- The OUT pin pulses at a rate proportional to color intensity
- Pulses are counted over a 10 ms window per channel → R, G, B "brightness" values
- Orange detected when: `r > 150 and g > 60 and b < 80 and r > b*2`
- Blue detected when: `b > 150 and b > r*2 and b > g`
- **TUNE thresholds on the real robot** by running `python -m control.color_sensor`

The onboard LED (GPIO 25) is turned ON at startup for consistent illumination regardless of ambient light.

---

#### `tof_sensor.py` — VL53L0X time-of-flight sensors (×4)

Manages four VL53L0X laser distance sensors that share the same I2C bus.

**Startup sequence** (critical — all four boot at address 0x29):
1. Pull all XSHUT pins LOW → all sensors off
2. Power up sensor 1, assign it address 0x30
3. Power up sensor 2, assign it address 0x31
4. Power up sensor 3, assign it address 0x32
5. Power up sensor 4, assign it address 0x33

```python
tof = ToFSensors()
d1, d2, d3, d4 = tof.read_all_mm()   # distances in mm (None on failure)
```

> **Warning**: each read takes ~55 ms; reading all four = ~220 ms.  
> Do NOT call `read_all_mm()` inside the 50 Hz main loop.  
> Use in a background thread or for diagnostic purposes only.

---

#### `steering_controller.py` — Steering PID

Computes how much to turn the steering wheel to keep the car on the path.

Combines two error signals:
- **Cross-track error (CTE)**: perpendicular distance from car to path (cm)
- **Heading error (HE)**: angular difference between car heading and path tangent (rad)

```
combined_error = CTE + HEADING_WEIGHT × HE
```

Heading error is weighted because it reacts early (you can see you're about to drift before you've drifted).

```python
steer_ctrl = SteeringPIDController(Kp=1.1, Ki=0.0, Kd=0.0,
                                    output_limits=(-27, 27),
                                    windup_limit=20.0,
                                    heading_weight=8.0)
steer_deg = steer_ctrl.compute(x, y, theta, trajectory, par_s)
par_s = steer_ctrl.current_s   # feed back for efficient next search
```

---

#### `driving_controller.py` — Speed PID

Computes throttle duty cycle. Target speed adapts to path curvature so the car slows automatically in corners:

```
target_speed = base_speed / (1 + k_curve × |curvature|)
```

Examples at `base_speed=40 cm/s`, `k_curve=1.0`:

| Section | Curvature | Target speed |
|---------|-----------|-------------|
| Straight | 0.00 | 40.0 cm/s |
| Corner (R=50 cm) | 0.02 | 39.2 cm/s |
| Sharp corner | 0.20 | 33.3 cm/s |

```python
pid_out = drive_ctrl.compute(x, y, theta, trajectory, par_s, speed_meas)
duty = BASE_DUTY + DUTY_GAIN * pid_out   # clamp 0.0–1.0
```

---

### 4.3 `estimation/`

#### `ekf.py` — Extended Kalman Filter

Estimates the car's pose `(x, y, θ)` by fusing wheel encoder speed with MPU-6050 gyro rate.

**State vector**: `[x_cm, y_cm, theta_rad]`  
**Motion model**: bicycle kinematic model  
```
x_new     = x + speed × cos(θ) × dt
y_new     = y + speed × sin(θ) × dt
theta_new = θ + speed × tan(steer_rad) / wheelbase × dt
```

**Two update steps per tick**:
1. `ekf.predict(speed, steer_rad, dt)` — propagate state using bicycle model
2. `ekf.update_gyro_rate(omega, dt, R_gyro)` — correct heading drift using gyro

Optionally, `ekf.update_imu(theta_abs)` can be called with a BNO055 absolute heading (not used with MPU-6050 which is gyro-only).

```python
ekf = EKF(wheelbase=WHEELBASE_CM, Q=Q_matrix, R_imu=R_IMU)
ekf.initialize(start_x=150.0, start_y=50.0, start_theta=0.0)

ekf.predict(speed_meas, math.radians(steer_deg), DT_S)
ekf.update_gyro_rate(omega_gyro, DT_S, R_gyro=EKF_R_GYRO_R2)

x, y, theta = ekf.state
```

The Jacobian `F` (how state uncertainty propagates through the bicycle model) is computed analytically for correct uncertainty growth around corners.

---

### 4.4 `trajectory/`

All trajectory types share the same interface defined in `base.py`.

#### `base.py` — Abstract trajectory interface

Every trajectory must implement four methods, all working in **arc-length** `s` (cm from start):

```python
find_closest(x, y, near_s=None) -> float   # arc-length of nearest point to (x,y)
get_point(s)                    -> (x, y)   # world position at arc-length s
get_tangent(s)                  -> (tx, ty) # unit tangent vector at s
get_curvature(s)                -> float    # signed curvature κ at s (+ = left/CCW)
total_length                    -> float    # property: total arc-length of path
```

#### `waypoint_path.py` — Straight-line path

Used for straight sections of the track.  
`get_curvature()` always returns 0.0.  
`find_closest()` uses perpendicular projection (not just nearest waypoint) for accurate CTE.

```python
path = WaypointPath([(x0,y0), (x1,y1), ...])
```

#### `bezier.py` — Cubic Bézier paths

`BezierSegment`: a single cubic Bézier with 4 control points.  
- Builds a 300-sample arc-length lookup table at construction time for accurate arc-length queries
- `get_curvature(s)` returns signed curvature: `κ = (d1 × d2) / |d1|³` (cross product for sign)
- Positive κ = curve turns left (CCW), negative = turns right (CW)
- `from_endpoints_and_tangents(P0, P3, v0, v3)` factory auto-computes control points P1, P2

`BezierPath`: chains multiple `BezierSegment` objects end-to-end with continuous arc-length.

#### `builder.py` — Trajectory factory

Builds the right trajectory type for each situation:

| Method | Returns | Used for |
|--------|---------|----------|
| `TrajectoryBuilder.straight(x0,y0, x1,y1)` | `WaypointPath` | Track straight sections |
| `TrajectoryBuilder.corner(entry, theta, direction, radius)` | `BezierSegment` | 90° track corners |
| `TrajectoryBuilder.pillar_swerve(car, pillar, color, end)` | `BezierPath` (2 segments) | Pillar bypass |
| `TrajectoryBuilder.parking_approach(car, lot)` | `BezierSegment` | Parking entry curve |

**Pillar swerve logic**:
- RED pillar → pass to the right (bypass point is offset in the `-y` direction relative to path)
- GREEN pillar → pass to the left (bypass point is offset in the `+y` direction)
- Lateral offset = `PILLAR_CLEARANCE_CM` (default 20 cm from pillar centre)
- Path: car position → bypass waypoint → section exit, all joined as a smooth 2-segment Bézier

---

### 4.5 `main/`

#### `race_manager.py` — Race state machine

The "brain" of the race. Called every tick with the current robot state and vision data; returns the trajectory to follow.

**`VisionFrame`** (dataclass — the bridge between vision and race logic):
```python
VisionFrame(
    pillars          = [(x_cm, y_cm, color), ...],  # partner fills (closest first)
    orange_line_seen = True/False,                   # color sensor fills
    blue_line_seen   = True/False,                   # color sensor fills
    parking_lot      = (x, y, theta) or None,        # partner fills
)
```

**`WROTrack`** — hard-coded track geometry (300×300 cm field):

CCW (all left turns):
```
Start (150,50) → East → Corner → North → Corner → West → Corner → South → Corner → back to Start
```

CW (all right turns):
```
Start (150,50) → West → Corner → North → Corner → East → Corner → South → Corner → back to Start
```

**`RaceManager`** states:

```
WAITING  ──(orange line)──► RACING(CCW)
         ──(blue line)────► RACING(CW)

RACING   ──(24 sections done)──► PARKING

PARKING  ──(car inside lot)──► DONE
```

**Each tick in RACING**:
1. If a pillar is visible on a straight section → replace current trajectory with `pillar_swerve()`
2. Track the last pillar color seen (for the WRO 2026 lap-3 direction rule)
3. If section is complete (line seen OR within 15 cm of exit): advance to next section
4. On lap 2→3 transition: if `last_sign_color == RED` → flip CCW↔CW

**WRO 2026 lap-3 direction rule**:
> After completing lap 2, if the last traffic sign seen was RED, lap 3 runs in the **opposite** direction. If GREEN (or no sign seen), direction is unchanged.

---

#### `parking.py` — Parking state machine

Manages the two-phase parking sequence after 3 laps.

```
IDLE ──(set_lot called)──► APPROACH ──(within 15 cm of entry)──► DRIVE_IN ──(inside lot)──► DONE
```

- **APPROACH**: follows a `parking_approach()` Bézier from current car position to lot entry
- **DRIVE_IN**: follows a short straight `stop_dist = lot_depth − robot_length/2` cm into the lot
  - At this stop point: rear clears the entry, front just touches the back wall
- **DONE**: `_is_inside_lot()` checks the car in lot-local coordinates (along-axis and perpendicular-axis)

Fallback lot positions if partner vision returns `None`:
- Both directions: `lot_y = LOT_DEPTH_CM` (≈ 27 cm from outer wall), `lot_theta = -π/2`
- CCW fallback: `lot_x = 200` (car finished heading East — lot estimated right of centre)
- CW fallback: `lot_x = 100` (car finished heading West — lot estimated left of centre)

---

#### `main.py` — 50 Hz entry point

The program that runs on the Raspberry Pi during the race. See [Section 5](#5-the-50-hz-control-loop) for the full loop description.

#### `ekf_runner.py` — EKF scheduling helper

A standalone wrapper around `ekf.py` that handles the update scheduling.  
Not used directly by `main.py` (which calls EKF directly), but useful as a clean reference or for testing the EKF in isolation.

---

### 4.6 `simulation/`

#### `race_sim.py` — Headless simulator

Runs the complete race logic without a display or real hardware. Useful for verifying trajectory planning and race state transitions.

#### `pygame_sim.py` — Visual simulator

Renders the track, car, trajectories, pillars, and parking lot in real time using Pygame. Run from the project root:

```bash
python -m simulation.pygame_sim
```

---

## 5. The 50 Hz Control Loop

Every 20 ms (`main/main.py`):

```
1. READ SENSORS
   ├── encoders.get_linear_speeds()     → v_left, v_right (cm/s)
   └── encoders.get_yaw_rate()          → omega_gyro (rad/s)

2. EKF
   ├── ekf.predict(speed, steer_rad, dt)
   ├── ekf.update_gyro_rate(omega, dt)
   └── ekf.state → (x, y, theta) → robot.update_pose(x, y, theta)

3. VISION FRAME
   ├── orange_line_seen = color.orange_seen   ← TCS3200 background thread
   ├── blue_line_seen   = color.blue_seen     ← TCS3200 background thread
   ├── pillars          = []                  ← MOCK: partner fills this
   └── parking_lot      = None               ← MOCK: partner fills this

4. RACE LOGIC
   └── trajectory = race.update(robot, vision_frame)

5. STEERING
   ├── steer_deg = steer_ctrl.compute(x, y, theta, trajectory, par_s)
   ├── [on corner sections] steer_deg += Ackermann_feedforward(kappa)
   └── car.set_steering(steer_deg)

6. SPEED
   ├── pid_out = drive_ctrl.compute(x, y, theta, trajectory, par_s, speed)
   ├── duty = BASE_DUTY + DUTY_GAIN × pid_out
   └── car.set_motor('f', duty)

7. SLEEP remaining time to hit 20 ms exactly

ON EXIT (KeyboardInterrupt or race.is_done):
   car.brake(encoders)   ← active PID brake
   car.stop()
   color.stop()          ← joins background thread, turns off LED
```

### Ackermann feedforward

Applied **only** on corner sections (never on pillar swerves or straights).

```python
kappa    = trajectory.get_curvature(s)                        # signed 1/radius
kappa    = clamp(kappa, ±1/CORNER_RADIUS_CM)                  # limit to ±1/50
steer_ff = -degrees(atan(WHEELBASE_CM × kappa))               # Ackermann formula
steer_deg += steer_ff
```

This pre-steers the servo to the geometric angle the corner requires, so the PID only needs to correct residual drift — not fight the entire corner geometry.

---

## 6. Track Geometry

WRO 2026 field: 300 × 300 cm. World frame: x = East, y = North.

```
(0,300)─────────────────────────(300,300)
   │                                │
   │   (50,250)────────(250,250)    │
   │      │    INNER BOX   │       │
   │   (50,50)─────────(250,50)    │
   │                                │
   │           [P]                  │   ← parking lot (bottom straight, outer wall)
(0,0)───────────────────────────(300,0)   (y=0 = outer wall)
```

Track centrelines: `y=50` (bottom), `y=250` (top), `x=50` (left), `x=250` (right)  
Outer wall: `y=0` (south), `y=300` (north), `x=0` (west), `x=300` (east)  
Corner radius: 50 cm (default)  
Start position: `(150, 50)` — centre of the bottom straight

**Parking lot** (WRO 2026 Figure 4):  
- Always on the bottom starting straight, against the south outer wall (`y=0`)  
- Two magenta blocks bound the lot; right block is at the outer wall (`y=0`), left block at `y=LOT_DEPTH_CM`  
- `lot_x` = x-centre of the gap between the blocks — **variable**, set by judges, detected by vision  
- `lot_y` = `LOT_DEPTH_CM` ≈ 27 cm (inner/entry edge of the lot)  
- `lot_theta` = `−π/2` (car enters heading South, toward `y=0`)  
- Width: 20 cm (fixed by WRO rules)  
- Depth: `1.5 × robot_length` from the outer wall into the track

**CCW section sequence** (one lap):

| # | Kind | Entry → Exit | Entry heading |
|---|------|-------------|---------------|
| 0 | straight | (150,50) → (250,50) | East (0°) |
| 1 | corner | (250,50) → (250,150) | East → North |
| 2 | straight | (250,150) → (250,250) | North (90°) |
| 3 | corner | (250,250) → (150,250) | North → West |
| 4 | straight | (150,250) → (50,250) | West (180°) |
| 5 | corner | (50,250) → (50,150) | West → South |
| 6 | straight | (50,150) → (50,50) | South (−90°) |
| 7 | corner | (50,50) → (150,50) | South → East |

CW is the mirror image (all turns are right turns).

---

## 7. Partner Integration

The vision partner's code provides:
- **Pillar positions**: list of `(x_cm, y_cm, color)` tuples, closest pillar first.  
  `color = 0` (RED) → pass to the right. `color = 1` (GREEN) → pass to the left.
- **Parking lot position**: `(x_cm, y_cm, theta_rad)` of the lot entry edge centre, or `None`.

These two values are the only connection points. In `main/main.py`, find these two lines and replace the `# MOCK` stubs:

```python
vision_frame = VisionFrame(
    orange_line_seen = color.orange_seen,       # already wired — TCS3200
    blue_line_seen   = color.blue_seen,         # already wired — TCS3200
    pillars          = [],                       # MOCK — partner fills this
    parking_lot      = None,                    # MOCK — partner fills this
)
```

The orange/blue floor line detection is **already handled** by the TCS3200 hardware color sensor — the partner does not need to provide these.

---

## 8. Running the Robot

### Prerequisites

```bash
# On the Raspberry Pi
pip install gpiozero mpu6050-raspberrypi smbus2 numpy opencv-python
```

### Start the race

```bash
cd /path/to/WRO2026
python -m main.main
```

The program waits for the start button (GPIO 8) to be pressed, then begins the control loop.

### Module self-tests

Each module has a built-in self-test. Run from the project root:

```bash
python -m control.color_sensor     # live R/G/B readings — use to tune thresholds
python -m control.tof_sensor       # live distance readings from all 4 ToF sensors
python -m estimation.ekf           # 7 EKF unit tests
python -m trajectory.bezier        # Bézier arc-length and curvature tests
python -m trajectory.builder       # path building tests
python -m main.race_manager        # 14 race state-machine tests
python -m main.parking             # 8 parking sequence tests
python -m main.ekf_runner          # EKF scheduling test (mock sensors)
```

---

## 9. Running the Simulation

```bash
pip install pygame numpy
python -m simulation.pygame_sim
```

The simulator uses all the same trajectory, race manager, and parking code as the real robot. Only the hardware layer (encoders, gyro, servo, motor) is replaced with a physics model.

---

## 10. Tuning Guide

### Competition day — step-by-step

**Step 1: Measure the robot**  
Open `config.py` and update:
- `WHEELBASE_CM` — front-to-rear axle distance
- `ROBOT_LENGTH_CM` — total body length
- `ROBOT_WIDTH_CM` — total body width
- `SERVO_MAX_DEG` — maximum servo throw each way

**Step 2: Tune encoder constants** (`allEncodersClass.py`)
- Count the teeth on each encoder disc → `pulses_per_rev`
- Measure wheel circumference (π × diameter) → `wheel_circ_cm`

**Step 3: Tune the color sensor thresholds**  
```bash
python -m control.color_sensor
```
Hold the sensor over orange tape, blue tape, and plain floor. Note the R/G/B counts, then edit the thresholds in `color_sensor.py` `_run()` method.

**Step 4: Tune motor duty cycle** (`main/main.py`)  
Set `BASE_DUTY` so the car drives at approximately `BASE_SPEED_CM_S`.  
Start at 0.50 and adjust until measured speed matches target.

**Step 5: Tune steering PID** (`config.py`)  
Start: `Kp=1.0, Ki=0, Kd=0, PID_HEADING_W=6`  
1. Increase `Kp` until the car tracks a straight well
2. Raise `PID_HEADING_W` to reduce heading lag entering corners
3. Add small `Kd` only if persistent oscillations remain
4. Add small `Ki` only if a steady sideways offset is observed on straights

**Step 6: Tune EKF noise** (`config.py`)  
These are the initial values — they work well in simulation:
- `EKF_Q_XY_CM2 = 0.10` — increase if the filter is too slow to follow real motion
- `EKF_Q_THETA_R2 = 0.0002` — increase if heading drifts noticeably
- `EKF_R_GYRO_R2 = (0.003)²` — decrease if your gyro is cleaner than average

**Step 7: Tune pillar clearance** (`config.py`)  
`PILLAR_CLEARANCE_CM = 20.0` — lateral distance from the car centre to the pillar centre when passing. Increase if the car clips pillars; decrease if the swerve path is too wide.

### Key `# TUNE ON REAL ROBOT` locations

| File | What to tune |
|------|-------------|
| `config.py` | Everything listed above |
| `main/main.py` | `BASE_DUTY`, `DUTY_GAIN`, servo `center_angle`, `max_deviation` |
| `main/ekf_runner.py` | `pos_std`, `heading_std` initial EKF uncertainty |
| `control/color_sensor.py` | Orange/blue detection thresholds in `_run()` |
| `control/allEncodersClass.py` | `pulses_per_rev`, `wheel_circ_cm`, gyro axis |
| `control/servoClass.py` | `center_angle`, `max_deviation` for the physical servo |
| `main/parking.py` | `ROBOT_LENGTH_CM`, `DEFAULT_LOT_DEPTH`, `APPROACH_DONE_CM` |
