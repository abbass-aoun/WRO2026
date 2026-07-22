"""
main/main.py — 50 Hz real-robot control loop.
==============================================

This is the entry point that runs on the physical robot.
It wires together every module in the project into one loop.

HOW TO RUN (on the Raspberry Pi, from the project root):
    python -m main.main

WHAT HAPPENS EACH TICK (every 20 ms at 50 Hz):
    1. Read wheel encoder speeds + IMU gyro rate
    2. EKF predict + update  →  best (x, y, θ) estimate
    3. Get vision frame from partner's camera code
    4. Race manager decides which path to follow
    5. Steering PID + Ackermann feedforward → servo angle
    6. Driving PID → motor duty cycle
    7. Send commands to hardware
    8. Sleep until the next tick

PARTNER INTEGRATION:
    Orange/blue floor-line detection is handled by the TCS3200 colour sensor
    (control/color_sensor.py) — no partner code needed for this.
    Partner provides pillar positions and parking lot via VisionFrame in the
    main loop (look for the two MOCK comments inside the control loop).

TUNING GUIDE (competition day):
    All robot-specific numbers are in config.py.
    Hardware pin numbers are at the top of this file (marked PIN_*).
    The two duty-cycle constants (BASE_DUTY, DUTY_PID_GAIN) are here because
    they depend on the physical battery voltage and motor, not robot geometry.
"""

import math
import time

from control.color_sensor import ColorSensor
from main.vision_adapter import VisionThread, build_vision_frame
from config import (
    WHEELBASE_CM,
    SERVO_MAX_DEG,
    BASE_SPEED_CM_S,
    DT_S,
    CORNER_RADIUS_CM,
    PID_KP, PID_KI, PID_KD,
    PID_HEADING_W, PID_WINDUP_LIM,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2,
    EKF_R_GYRO_R2, EKF_IMU_PERIOD,
    MAX_DUTY_SAFE,
    SHOW_VISION_DEBUG,
)

import numpy as np

from control.robot               import Robot
from control.car_controller      import CarController
from control.allEncodersClass    import RobotEncoders
from control.steering_controller import SteeringPIDController
from control.driving_controller  import DrivingPIDController
from estimation.ekf              import EKF
from main.race_manager           import RaceManager, Direction

# ─────────────────────────────────────────────────────────────────────────────
# Hardware pin numbers                              TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
PIN_MOTOR_IN1    = 18   # L298N IN1 (motor direction)       — Pin 12 / GPIO 18
PIN_MOTOR_IN2    = 13   # L298N IN2 (motor direction)       — Pin 33 / GPIO 13
PIN_MOTOR_ENA    = 19   # L298N ENA (PWM speed)             — Pin 35 / GPIO 19
PIN_SERVO        = 12   # Steering servo signal             — Pin 32 / GPIO 12
PIN_ENC_LEFT     = 7    # Left  wheel IR encoder            — Pin 26 / GPIO  7
PIN_ENC_RIGHT    = 5    # Right wheel encoder Channel A     — Pin 29 / GPIO  5
PIN_START_BUTTON = 8    # Push button to start the race     — Pin 24 / GPIO  8

# ─────────────────────────────────────────────────────────────────────────────
# TCS3200 colour sensor pin numbers  (Pins for Sensors.txt)
# ─────────────────────────────────────────────────────────────────────────────
PIN_COLOR_S0  = 17   # Frequency scaling — Pin 11 / GPIO 17
PIN_COLOR_S1  = 27   # Frequency scaling — Pin 13 / GPIO 27
PIN_COLOR_S2  = 22   # Filter select     — Pin 15 / GPIO 22
PIN_COLOR_S3  = 23   # Filter select     — Pin 16 / GPIO 23
PIN_COLOR_OUT = 24   # Pulse output      — Pin 18 / GPIO 24
PIN_COLOR_LED = 25   # Illumination LEDs — Pin 22 / GPIO 25

# ─────────────────────────────────────────────────────────────────────────────
# Motor duty-cycle calibration                      TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
# BASE_DUTY  : duty cycle (0–1) that makes the car travel at ~BASE_SPEED_CM_S
#              on a flat straight.  Start at 0.50 and adjust until speed matches.
# DUTY_GAIN  : how strongly the speed PID adjusts duty around BASE_DUTY.
#              Small value = gentle corrections.  Too large = oscillation.
BASE_DUTY  = 0.55   # TUNE ON REAL ROBOT
DUTY_GAIN  = 0.008  # TUNE ON REAL ROBOT

# ─────────────────────────────────────────────────────────────────────────────
# EKF noise matrices                                TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # Initialise to None so the finally block can safely skip un-created objects
    car      = None
    encoders = None
    color    = None
    vision   = None

    try:
        # ── 1. Create hardware objects ────────────────────────────────────────
        car      = CarController(PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO)
        car.stop()   # guarantee motor is off regardless of pin power-on state
        encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
        color    = ColorSensor(PIN_COLOR_S0, PIN_COLOR_S1,
                               PIN_COLOR_S2, PIN_COLOR_S3, PIN_COLOR_OUT,
                               PIN_COLOR_LED)
        vision   = VisionThread()          # starts camera pipeline in background

        # ── 2. Create state estimation ────────────────────────────────────────
        ekf   = EKF(wheelbase=WHEELBASE_CM, Q=_Q, R_imu=EKF_R_GYRO_R2)
        robot = Robot()

        # ── 3. Create race logic ──────────────────────────────────────────────
        race = RaceManager(corner_radius=CORNER_RADIUS_CM)

        # ── 4. Create controllers ─────────────────────────────────────────────
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

        # ── 5. Wait for start button ──────────────────────────────────────────
        from gpiozero import Button
        # pull_up=True  → pin rests HIGH; pressing (connects to GND) goes LOW
        # bounce_time   → ignore pulses shorter than 50 ms (debounce)
        start_btn = Button(PIN_START_BUTTON, pull_up=True, bounce_time=0.05)
        time.sleep(0.1)   # let the pin settle before we start watching it
        print("Ready — press the start button to begin.")
        start_btn.wait_for_press()
        print("GO!")

        # ── 6. Initialise EKF and race manager ───────────────────────────────
        # The car starts at position (150, 50) facing East (or West — unknown yet).
        # The race manager will detect direction from the first line the camera sees.
        ekf.initialize(x0=150.0, y0=50.0, theta0=0.0)   # theta0 is in radians
        robot.update_pose(150.0, 50.0, 0.0)
        encoders.reset()

        race.start(robot, direction=Direction.UNKNOWN)  # direction detected from first line

        steer_par_s = 0.0   # arc-length hint for steering closest-point search
        drive_par_s = 0.0   # arc-length hint for driving closest-point search
        tick        = 0     # tick counter (used for IMU update scheduling)

        # ── 7. Control loop ───────────────────────────────────────────────────
        while not race.is_done:
            t_start = time.monotonic()

            # ── SENSORS ───────────────────────────────────────────────────────
            # Encoder: average of left and right wheel speed
            v_l, v_r   = encoders.get_linear_speeds()      # cm/s each wheel
            speed_meas = 0.5 * (v_l + v_r)                 # forward speed cm/s

            # IMU: yaw rate in rad/s from MPU-6050 gyroscope
            omega_gyro = encoders.get_yaw_rate()            # rad/s

            # ── EKF ───────────────────────────────────────────────────────────
            steer_rad = math.radians(robot.steer_angle)
            ekf.predict(speed_meas, steer_rad, DT_S, omega_gyro=omega_gyro)
            # NOTE: ekf.update_imu(heading_abs) would go here if you have a
            #       BNO055 (absolute heading).  MPU-6050 only has gyro rate.

            x, y, theta = ekf.state
            robot.update_pose(x, y, theta)
            robot.update_speed(speed_meas)

            # ── VISION ───────────────────────────────────────────────────────
            # Orange/blue lines: TCS3200 color sensor (background thread).
            # Pillars + parking lot: partner's camera pipeline (VisionThread),
            # converted from camera-relative mm to world-frame cm via robot pose.
            vision_frame = build_vision_frame(
                vision.navigation_output,
                vision.parking_output,
                robot,
                color,
            )

            # ── RACE LOGIC ────────────────────────────────────────────────────
            trajectory = race.update(robot, vision_frame)
            if trajectory is None:
                # Waiting for direction detection — coast slowly
                car.set_motor('f', BASE_DUTY * 0.5)
                _sleep_rest(t_start, DT_S)
                tick += 1
                continue

            # ── STEERING ─────────────────────────────────────────────────────
            steer_deg = steer_ctrl.compute(x, y, theta, trajectory, steer_par_s)
            steer_par_s = steer_ctrl.current_s

            # Ackermann feedforward on corners: pre-steer the servo so the
            # PID only needs to correct residual drift, not the whole corner.
            sec = race._current_section
            if sec is not None and sec.kind == "corner":
                kappa    = trajectory.get_curvature(steer_par_s)
                kappa    = max(-1.0 / CORNER_RADIUS_CM,
                               min( 1.0 / CORNER_RADIUS_CM, kappa))
                steer_ff = -math.degrees(math.atan(WHEELBASE_CM * kappa))
                steer_deg += steer_ff

            steer_deg = max(-SERVO_MAX_DEG, min(SERVO_MAX_DEG, steer_deg))
            robot.update_steering(steer_deg)

            # ── SPEED ─────────────────────────────────────────────────────────
            pid_out = drive_ctrl.compute(
                x, y, theta, trajectory, drive_par_s, speed_meas)
            drive_par_s = drive_ctrl.current_s
            duty = BASE_DUTY + DUTY_GAIN * pid_out
            duty = max(0.0, min(MAX_DUTY_SAFE, duty))   # hard safety ceiling

            # ── HARDWARE OUTPUT ───────────────────────────────────────────────
            car.set_steering(steer_deg)
            car.set_motor('f', duty)

            # ── LOGGING (every 25 ticks = 0.5 s) ────────────────────────────
            if tick % 25 == 0:
                v_l_log, v_r_log = encoders.get_linear_speeds()
                print(
                    f"[t={tick:05d}] "
                    f"spd={speed_meas:5.1f} cm/s  "
                    f"vL={v_l_log:5.1f}  vR={v_r_log:5.1f}  "
                    f"duty={duty:.3f}  steer={steer_deg:+.1f}°"
                )

            # ── VISION DEBUG WINDOW ──────────────────────────────────────────
            if SHOW_VISION_DEBUG:
                import cv2 as _cv2
                dbg = vision.debug_frame
                if dbg is not None:
                    nav_action = vision.navigation_output.get("action", "")
                    _cv2.putText(dbg, nav_action, (10, 28),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    _cv2.imshow("WRO Vision", dbg)
                key = _cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Vision window closed — stopping.")
                    break

            # ── TIMING ────────────────────────────────────────────────────────
            tick += 1
            _sleep_rest(t_start, DT_S)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt — stopping.")

    except Exception as exc:
        print(f"\nFatal error: {exc}")
        raise

    finally:
        # Always stop background threads and motors, even if an exception
        # occurred before the control loop started.  Each step is guarded so
        # a failure in one cleanup action does not block the others.
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
        if vision is not None:
            vision.stop()   # sets stop_event and joins the thread
        if SHOW_VISION_DEBUG:
            try:
                import cv2 as _cv2
                _cv2.destroyAllWindows()
            except Exception:
                pass
        print("Shutdown complete.")


def _sleep_rest(t_start: float, dt: float) -> None:
    """Sleep for whatever time remains in the current tick."""
    leftover = dt - (time.monotonic() - t_start)
    if leftover > 0:
        time.sleep(leftover)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
