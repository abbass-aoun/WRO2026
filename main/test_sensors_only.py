"""
main/test_sensors_only.py — Level 2 integration test.
======================================================

Runs the full 50 Hz control loop on the real robot using ONLY hardware sensors.
Camera / vision pipeline is intentionally excluded.

WHAT IS ACTIVE:
    Encoders    → speed measurement → driving PID + EKF predict
    Gyro        → yaw rate          → EKF update
    Color sensor→ orange/blue lines → direction detection + section advance
    EKF         → fused (x, y, θ) estimate
    Race manager→ trajectory per section, lap counting
    Steering PID→ cross-track + heading error → servo angle
    Driving PID → speed error → motor duty
    Motor/servo → hardware output

WHAT IS NOT ACTIVE:
    Camera / VisionThread — no pillar detection, no parking lot detection
    Pillar swerve trajectories will never be generated
    Parking uses fallback position (x=200 or 100, y=LOT_DEPTH_CM)

WHAT TO LOOK FOR:
    1. Direction detected from first orange or blue line crossing
    2. Section advances after each line crossing (console prints section number)
    3. Lap counter increments every 8 sections
    4. Robot stays near the track centreline on straights and corners
    5. After 3 laps: transitions to PARKING state and stops

HOW TO RUN (from project root on the Raspberry Pi):
    python -m main.test_sensors_only

TUNE BEFORE RUNNING:
    config.py  → BASE_SPEED_CM_S, MAX_DUTY_SAFE, MOTOR_INVERTED
    This file  → BASE_DUTY, DUTY_GAIN (lines marked TUNE ON REAL ROBOT)
"""

import math
import time

import numpy as np
from gpiozero import Button

from config import (
    WHEELBASE_CM,
    SERVO_MAX_DEG,
    BASE_SPEED_CM_S,
    DT_S,
    CORNER_RADIUS_CM,
    PID_KP, PID_KI, PID_KD,
    PID_HEADING_W, PID_WINDUP_LIM,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2,
    EKF_R_GYRO_R2,
    MAX_DUTY_SAFE,
)
from control.robot               import Robot
from control.car_controller      import CarController
from control.allEncodersClass    import RobotEncoders
from control.color_sensor        import ColorSensor
from control.steering_controller import SteeringPIDController
from control.driving_controller  import DrivingPIDController
from estimation.ekf              import EKF
from main.race_manager           import RaceManager, Direction, VisionFrame

# ─────────────────────────────────────────────────────────────────────────────
# Hardware pin numbers  (must match main/main.py)
# ─────────────────────────────────────────────────────────────────────────────
PIN_MOTOR_IN1    = 18
PIN_MOTOR_IN2    = 13
PIN_MOTOR_ENA    = 19
PIN_SERVO        = 12
PIN_ENC_LEFT     = 7
PIN_ENC_RIGHT    = 5
PIN_START_BUTTON = 8

PIN_COLOR_S0  = 17
PIN_COLOR_S1  = 27
PIN_COLOR_S2  = 22
PIN_COLOR_S3  = 23
PIN_COLOR_OUT = 24
PIN_COLOR_LED = 25

# ─────────────────────────────────────────────────────────────────────────────
# Motor duty calibration                               TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
BASE_DUTY = 0.55
DUTY_GAIN = 0.008

# ─────────────────────────────────────────────────────────────────────────────
# EKF process noise
# ─────────────────────────────────────────────────────────────────────────────
_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])


def main() -> None:
    car      = None
    encoders = None
    color    = None

    try:
        # ── 1. Hardware objects ───────────────────────────────────────────────
        car      = CarController(PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO)
        car.stop()
        encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
        color    = ColorSensor(PIN_COLOR_S0, PIN_COLOR_S1,
                               PIN_COLOR_S2, PIN_COLOR_S3,
                               PIN_COLOR_OUT, PIN_COLOR_LED)

        # ── 2. State estimation ───────────────────────────────────────────────
        ekf   = EKF(wheelbase=WHEELBASE_CM, Q=_Q, R_imu=EKF_R_GYRO_R2)
        robot = Robot()

        # ── 3. Race logic ─────────────────────────────────────────────────────
        race = RaceManager(corner_radius=CORNER_RADIUS_CM)

        # ── 4. Controllers ────────────────────────────────────────────────────
        steer_ctrl = SteeringPIDController(
            Kp             = PID_KP,
            Ki             = PID_KI,
            Kd             = PID_KD,
            output_limits  = (-SERVO_MAX_DEG, SERVO_MAX_DEG),
            windup_limit   = PID_WINDUP_LIM,
            heading_weight = PID_HEADING_W,
        )
        drive_ctrl = DrivingPIDController(
            Kp            = 0.010,      # TUNE ON REAL ROBOT
            Ki            = 0.0,
            Kd            = 0.0,
            output_limits = (-0.30, 0.30),
            windup_limit  = 10.0,
            base_speed    = BASE_SPEED_CM_S,
            k_curve       = 1.0,        # TUNE ON REAL ROBOT
        )

        # ── 5. Start button ───────────────────────────────────────────────────
        start_btn = Button(PIN_START_BUTTON, pull_up=True, bounce_time=0.05)
        time.sleep(0.1)
        print("Level 2 — sensors only.  Press start button to begin.")
        start_btn.wait_for_press()
        print("GO!")

        # ── 6. Initialise EKF and race state ─────────────────────────────────
        ekf.initialize(x0=150.0, y0=50.0, theta0=0.0)
        robot.update_pose(150.0, 50.0, 0.0)
        encoders.reset()
        race.start(robot, direction=Direction.UNKNOWN)

        steer_par_s = 0.0
        drive_par_s = 0.0
        tick        = 0
        prev_section = 0
        prev_lap     = 1

        # ── 7. Control loop ───────────────────────────────────────────────────
        while not race.is_done:
            t_start = time.monotonic()

            # ── Sensors ───────────────────────────────────────────────────────
            v_l, v_r   = encoders.get_linear_speeds()
            speed_meas = 0.5 * (v_l + v_r)
            omega_gyro = encoders.get_yaw_rate()

            # ── EKF ───────────────────────────────────────────────────────────
            steer_rad = math.radians(robot.steer_angle)
            ekf.predict(speed_meas, steer_rad, DT_S)
            ekf.update_gyro_rate(omega_gyro, DT_S, R_gyro=EKF_R_GYRO_R2)

            x, y, theta = ekf.state
            robot.update_pose(x, y, theta)
            robot.update_speed(speed_meas)

            # ── Vision frame — sensors only, no camera ────────────────────────
            vision_frame = VisionFrame(
                pillars          = [],
                orange_line_seen = color.orange_seen,
                blue_line_seen   = color.blue_seen,
                parking_lot      = None,
            )

            # ── Race logic ────────────────────────────────────────────────────
            trajectory = race.update(robot, vision_frame)

            # Print whenever direction is detected, section or lap changes
            if race.direction.value != "unknown":
                if race.section != prev_section or race.lap != prev_lap:
                    print(
                        f"[lap={race.lap} sec={race.section} "
                        f"dir={race.direction.value}]  "
                        f"x={x:.1f} y={y:.1f} θ={math.degrees(theta):+.1f}°"
                    )
                    prev_section = race.section
                    prev_lap     = race.lap

            if trajectory is None:
                # Waiting for first line — coast slowly until direction known
                car.set_motor('f', BASE_DUTY * 0.5)
                _sleep_rest(t_start, DT_S)
                tick += 1
                continue

            # ── Steering ──────────────────────────────────────────────────────
            steer_deg = steer_ctrl.compute(x, y, theta, trajectory, steer_par_s)
            steer_par_s = steer_ctrl.current_s

            sec = race._current_section
            if sec is not None and sec.kind == "corner":
                kappa    = trajectory.get_curvature(steer_par_s)
                kappa    = max(-1.0 / CORNER_RADIUS_CM,
                               min( 1.0 / CORNER_RADIUS_CM, kappa))
                steer_ff = -math.degrees(math.atan(WHEELBASE_CM * kappa))
                steer_deg += steer_ff

            steer_deg = max(-SERVO_MAX_DEG, min(SERVO_MAX_DEG, steer_deg))
            robot.update_steering(steer_deg)

            # ── Speed ─────────────────────────────────────────────────────────
            pid_out = drive_ctrl.compute(
                x, y, theta, trajectory, drive_par_s, speed_meas)
            drive_par_s = drive_ctrl.current_s
            duty = BASE_DUTY + DUTY_GAIN * pid_out
            duty = max(0.0, min(MAX_DUTY_SAFE, duty))

            # ── Hardware output ───────────────────────────────────────────────
            car.set_steering(steer_deg)
            car.set_motor('f', duty)

            # ── Log every 25 ticks (0.5 s) ───────────────────────────────────
            if tick % 25 == 0:
                print(
                    f"[t={tick:05d}] "
                    f"spd={speed_meas:5.1f} cm/s  "
                    f"vL={v_l:5.1f}  vR={v_r:5.1f}  "
                    f"duty={duty:.3f}  steer={steer_deg:+.1f}°  "
                    f"cte={steer_ctrl.last_cte:+.1f} cm  "
                    f"orange={'Y' if color.orange_seen else 'n'}  "
                    f"blue={'Y' if color.blue_seen else 'n'}"
                )

            tick += 1
            _sleep_rest(t_start, DT_S)

        print(f"Race complete — {race.lap - 1} laps done.  Stopping.")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — stopping.")

    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise

    finally:
        print("Shutting down...")
        if car is not None and encoders is not None:
            try:
                car.brake(encoders)
                car.stop()
            except Exception:
                pass
        if color is not None:
            try:
                color.stop()
            except Exception:
                pass
        print("Shutdown complete.")


def _sleep_rest(t_start: float, dt: float) -> None:
    leftover = dt - (time.monotonic() - t_start)
    if leftover > 0:
        time.sleep(leftover)


if __name__ == "__main__":
    main()
