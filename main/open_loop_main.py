WHAT """
main/open_loop_main.py  —  WRO 2026 Open-Loop Challenge FSM
=============================================================
Sensors used:
  - ToF sensors  : wall distance → lane-centring steering
  - Vision thread: pillar detection + lane framing
  - Color sensor : orange/blue line → corner transitions

No EKF. No Bezier planner. Pure FSM + sensor feedback.

States:
  WAITING → STRAIGHT ↔ CORNER_TURNING
             STRAIGHT ↔ PILLAR_AVOID
             → FINISHED  (after TOTAL_LAPS)
"""

import math
import threading
import time
from enum import Enum, auto

from gpiozero import Button, LED, Motor
from rpi_hardware_pwm import HardwarePWM

from control.color_sensor   import ColorSensor
from control.tof_sensor     import ToFSensors
from main.vision_adapter    import VisionThread


# ─────────────────────────────────────────────────────────────────────────────
# GPIO / HARDWARE PINS
# ─────────────────────────────────────────────────────────────────────────────

PIN_MOTOR_FWD  = 13          # gpiozero Motor forward  (confirmed in loop_mat.py)
PIN_MOTOR_BWD  = 18          # gpiozero Motor backward
PIN_MOTOR_ENA  = 19          # Motor enable

PWM_CHANNEL    = 0           # HardwarePWM channel 0 = GPIO 12
PWM_CHIP       = 0

PIN_START_BTN  = 8
PIN_LEDS       = [16, 20, 21, 26]

PIN_COLOR_S0   = 17
PIN_COLOR_S1   = 27
PIN_COLOR_S2   = 22
PIN_COLOR_S3   = 23
PIN_COLOR_OUT  = 24
PIN_COLOR_LED  = 25

TOF_XSHUT_PINS = (4, 10, 9, 6)   # order: TUNE ON REAL ROBOT

# Which index of read_all_mm() maps to which sensor direction:
TOF_IDX_FRONT  = 0    # TUNE ON REAL ROBOT
TOF_IDX_LEFT   = 1    # TUNE ON REAL ROBOT
TOF_IDX_RIGHT  = 2    # TUNE ON REAL ROBOT


# ─────────────────────────────────────────────────────────────────────────────
# TUNABLE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Servo (absolute angles 0-180 degrees)
SERVO_CENTER     = 90     # TUNE ON REAL ROBOT: angle that drives straight
SERVO_MAX_LEFT   = 65     # TUNE ON REAL ROBOT
SERVO_MAX_RIGHT  = 115    # TUNE ON REAL ROBOT

# Speeds (0.0 – 1.0 duty cycle)
DRIVE_SPEED      = 0.35   # TUNE ON REAL ROBOT: straight speed
CORNER_SPEED     = 0.30   # TUNE ON REAL ROBOT
PILLAR_SPEED     = 0.28   # TUNE ON REAL ROBOT

# ToF wall following
TOF_TARGET_MM       = 500    # TUNE ON REAL ROBOT: target distance from left wall (mm)
TOF_WALL_KP         = 0.025  # TUNE ON REAL ROBOT: proportional gain for wall error
TOF_MAX_STEER_DEG   = 20.0   # TUNE ON REAL ROBOT: maximum ToF steering correction

# Vision framing (uses walls key from vision result)
VISION_FRAME_KP   = 0.05    # TUNE ON REAL ROBOT
VISION_FRAME_MAX  = 15.0    # TUNE ON REAL ROBOT: max degrees from vision correction

# Corner
CORNER_LOCK_DEG  = 25       # TUNE ON REAL ROBOT: servo offset during turn
CORNER_TURN_S    = 1.3      # TUNE ON REAL ROBOT: seconds at full lock
CORNER_SETTLE_S  = 0.4      # TUNE ON REAL ROBOT: straight pause after turn

# Pillar avoidance
PILLAR_TRIGGER_CM  = 80.0   # TUNE ON REAL ROBOT: start avoidance within this distance
PILLAR_STEER_DEG   = 22     # TUNE ON REAL ROBOT: extra steer toward avoidance side
PILLAR_AVOID_S     = 1.2    # TUNE ON REAL ROBOT: time to steer away
PILLAR_RECOVER_S   = 0.6    # TUNE ON REAL ROBOT: time to counter-steer back

# Line debounce
LINE_DEBOUNCE_S  = 1.0

# Race
TOTAL_LAPS       = 3
CORNERS_PER_LAP  = 4

# Control loop rate
DT_S = 0.02   # 50 Hz


# ─────────────────────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────────────────────

class State(Enum):
    WAITING        = auto()
    STRAIGHT       = auto()
    CORNER_TURNING = auto()
    PILLAR_AVOID   = auto()
    FINISHED       = auto()

class Direction(Enum):
    UNKNOWN = auto()
    CW      = auto()
    CCW     = auto()


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR + SERVO WRAPPERS
# ─────────────────────────────────────────────────────────────────────────────

class MotorCtrl:
    def __init__(self):
        self._m = Motor(
            forward  = PIN_MOTOR_FWD,
            backward = PIN_MOTOR_BWD,
            enable   = PIN_MOTOR_ENA,
        )

    def forward(self, speed=DRIVE_SPEED):
        self._m.forward(max(0.0, min(1.0, speed)))

    def stop(self):
        self._m.stop()


class ServoCtrl:
    def __init__(self):
        self._pwm = HardwarePWM(pwm_channel=PWM_CHANNEL, hz=50, chip=PWM_CHIP)
        self._pwm.start(0)
        self.center()

    def set(self, angle: float):
        angle = max(float(SERVO_MAX_LEFT), min(float(SERVO_MAX_RIGHT), float(angle)))
        self._pwm.change_duty_cycle(2.5 + (angle / 180.0) * 10.0)

    def center(self):
        self.set(SERVO_CENTER)

    def shutdown(self):
        self._pwm.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TOF BACKGROUND READER  (non-blocking — ToF takes ~55 ms per sensor)
# ─────────────────────────────────────────────────────────────────────────────

class ToFReader:
    def __init__(self):
        self._tof      = ToFSensors(xshut_pins=TOF_XSHUT_PINS)
        self._readings = [None, None, None, None]
        self._lock     = threading.Lock()
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                data = list(self._tof.read_all_mm())
                with self._lock:
                    self._readings = data
            except Exception:
                pass

    def get(self):
        with self._lock:
            return list(self._readings)

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


# ─────────────────────────────────────────────────────────────────────────────
# RACE STATE  (module-level globals)
# ─────────────────────────────────────────────────────────────────────────────

state             = State.WAITING
direction         = Direction.UNKNOWN
corners_completed = 0
laps              = 0

_phase_start_t    = 0.0   # monotonic time when current phase started
_last_line_t      = 0.0   # last confirmed line crossing (for debounce)
_pillar_phase     = 0     # 0 = steer away, 1 = recover
_pillar_dir       = 0     # +1 = steer right, -1 = steer left


# ─────────────────────────────────────────────────────────────────────────────
# STEERING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _wall_follow_steer(tof_readings) -> float:
    """
    Return a signed degree correction (added to SERVO_CENTER) to
    keep the robot centred in the lane using ToF wall distances.
    Positive = steer right; negative = steer left.
    """
    left_mm  = tof_readings[TOF_IDX_LEFT]
    right_mm = tof_readings[TOF_IDX_RIGHT]

    if left_mm is None and right_mm is None:
        return 0.0

    if left_mm is not None and right_mm is not None:
        error = (right_mm - left_mm) * 0.5          # centre-of-lane error
    elif left_mm is not None:
        error = left_mm - TOF_TARGET_MM
    else:
        error = -(right_mm - TOF_TARGET_MM)

    correction = TOF_WALL_KP * error
    return max(-TOF_MAX_STEER_DEG, min(TOF_MAX_STEER_DEG, correction))


def _vision_frame_steer(vision_result) -> float:
    """
    Return a signed correction from the vision lane-centre offset.
    Key 'lane_center_offset_px' must be provided by vision.py.
    TUNE: confirm key name with vision partner.
    """
    offset = vision_result.get("walls", {}).get("lane_center_offset_px")
    if offset is None:
        return 0.0
    correction = VISION_FRAME_KP * float(offset)
    return max(-VISION_FRAME_MAX, min(VISION_FRAME_MAX, correction))


def _nearest_pillar(vision_result):
    """
    Return {'color': str, 'dist_cm': float} for the nearest
    forward-facing pillar, or None if nothing relevant is visible.
    """
    best_dist = float("inf")
    best      = None

    for p in vision_result.get("pillars", []):
        rx    = p.get("relative_x_mm")
        ry    = p.get("relative_y_mm")
        color = p.get("color")

        if rx is None or ry is None or color not in ("red", "green"):
            continue

        dist_cm = math.hypot(rx, ry) / 10.0

        if dist_cm < best_dist:
            best_dist = dist_cm
            best = {"color": color, "dist_cm": dist_cm}

    return best


# ─────────────────────────────────────────────────────────────────────────────
# LINE CROSSING HANDLER
# ─────────────────────────────────────────────────────────────────────────────

def _check_line(orange_seen, blue_seen):
    """
    Act on a confirmed line crossing.  Locks driving direction on the
    first crossing, then switches to CORNER_TURNING on entry lines.
    Must only be called when state == State.STRAIGHT.
    """
    global direction, state, _last_line_t, _phase_start_t

    now = time.monotonic()
    if now - _last_line_t < LINE_DEBOUNCE_S:
        return
    if not (orange_seen or blue_seen):
        return

    # Lock direction from the very first line seen
    if direction == Direction.UNKNOWN:
        if blue_seen and not orange_seen:
            direction = Direction.CCW
            print("Direction locked: CCW")
        elif orange_seen and not blue_seen:
            direction = Direction.CW
            print("Direction locked: CW")
        else:
            return   # both seen simultaneously — ambiguous, skip

    cw  = direction == Direction.CW
    ccw = direction == Direction.CCW

    # Corner entry line
    entering = (cw and orange_seen) or (ccw and blue_seen)
    if entering:
        _last_line_t   = now
        _phase_start_t = now
        state = State.CORNER_TURNING
        print(f"ENTER corner  laps={laps}  corners={corners_completed}")


# ─────────────────────────────────────────────────────────────────────────────
# FSM  —  one tick at 50 Hz
# ─────────────────────────────────────────────────────────────────────────────

def fsm_step(motor, servo, tof_readings, vision_result, orange_seen, blue_seen):
    global state, corners_completed, laps
    global _phase_start_t, _pillar_phase, _pillar_dir

    now = time.monotonic()

    # ── FINISHED ──────────────────────────────────────────────────────────────
    if state == State.FINISHED:
        motor.stop()
        servo.center()
        return

    # ── STRAIGHT ──────────────────────────────────────────────────────────────
    if state == State.STRAIGHT:

        _check_line(orange_seen, blue_seen)

        if state == State.STRAIGHT:
            pillar = _nearest_pillar(vision_result)
            if pillar and pillar["dist_cm"] < PILLAR_TRIGGER_CM:
                _pillar_phase  = 0
                _phase_start_t = now
                # TUNE: flip signs if robot goes the wrong way around the pillar
                # Red   → must pass on LEFT of pillar  → steer LEFT  (dir = -1)
                # Green → must pass on RIGHT of pillar → steer RIGHT (dir = +1)
                _pillar_dir = -1 if pillar["color"] == "red" else +1
                state = State.PILLAR_AVOID
                print(f"PILLAR {pillar['color']}  dist={pillar['dist_cm']:.0f} cm")

        if state == State.STRAIGHT:
            wf    = _wall_follow_steer(tof_readings)
            vf    = _vision_frame_steer(vision_result)
            servo.set(SERVO_CENTER + wf + vf)
            motor.forward(DRIVE_SPEED)

    # ── CORNER_TURNING ────────────────────────────────────────────────────────
    elif state == State.CORNER_TURNING:

        elapsed = now - _phase_start_t
        cw = direction == Direction.CW

        if elapsed < CORNER_TURN_S:
            angle = SERVO_CENTER + (CORNER_LOCK_DEG if cw else -CORNER_LOCK_DEG)
            servo.set(angle)
            motor.forward(CORNER_SPEED)

        elif elapsed < CORNER_TURN_S + CORNER_SETTLE_S:
            servo.center()
            motor.forward(DRIVE_SPEED)

        else:
            # Corner complete
            corners_completed += 1
            print(f"Corner done  corners={corners_completed}/{CORNERS_PER_LAP}")

            if corners_completed >= CORNERS_PER_LAP:
                corners_completed = 0
                laps += 1
                print(f"Lap {laps}/{TOTAL_LAPS} complete")

            if laps >= TOTAL_LAPS:
                state = State.FINISHED
                print("Race complete!")
            else:
                state = State.STRAIGHT
                _phase_start_t = now

    # ── PILLAR_AVOID ──────────────────────────────────────────────────────────
    elif state == State.PILLAR_AVOID:

        elapsed = now - _phase_start_t

        if _pillar_phase == 0:
            servo.set(SERVO_CENTER + _pillar_dir * PILLAR_STEER_DEG)
            motor.forward(PILLAR_SPEED)
            if elapsed >= PILLAR_AVOID_S:
                _pillar_phase  = 1
                _phase_start_t = now

        else:
            servo.set(SERVO_CENTER - _pillar_dir * PILLAR_STEER_DEG)
            motor.forward(PILLAR_SPEED)
            if elapsed >= PILLAR_RECOVER_S:
                state = State.STRAIGHT
                print("Pillar cleared → STRAIGHT")


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP
# ─────────────────────────────────────────────────────────────────────────────

def _wait_for_start(button, leds):
    global state
    leds_on      = False
    last_toggle  = time.monotonic()
    print("Ready — press the start button.")
    while True:
        now = time.monotonic()
        if now - last_toggle >= 0.5:
            leds_on = not leds_on
            for led in leds:
                led.on() if leds_on else led.off()
            last_toggle = now
        if button.is_pressed:
            for led in leds:
                led.off()
            state = State.STRAIGHT
            print("GO!")
            break
        time.sleep(0.01)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    button = Button(PIN_START_BTN, pull_up=False, bounce_time=0.05)
    leds   = [LED(p) for p in PIN_LEDS]
    motor  = MotorCtrl()
    servo  = ServoCtrl()
    color  = ColorSensor(
        PIN_COLOR_S0, PIN_COLOR_S1, PIN_COLOR_S2,
        PIN_COLOR_S3, PIN_COLOR_OUT, PIN_COLOR_LED,
    )
    tof    = ToFReader()
    vision = VisionThread()
    vision.start()

    _wait_for_start(button, leds)

    try:
        while state != State.FINISHED:
            tick_start = time.monotonic()

            tof_readings  = tof.get()
            vision_result = vision.get_latest_result()

            fsm_step(
                motor,
                servo,
                tof_readings,
                vision_result,
                color.orange_seen,
                color.blue_seen,
            )

            print(
                f"state={state.name} | "
                f"dir={direction.name} | "
                f"lap={laps} | "
                f"corner={corners_completed} | "
                f"tof_L={tof_readings[TOF_IDX_LEFT]} "
                f"tof_R={tof_readings[TOF_IDX_RIGHT]} mm"
            )

            elapsed = time.monotonic() - tick_start
            if elapsed < DT_S:
                time.sleep(DT_S - elapsed)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        motor.stop()
        servo.center()
        servo.shutdown()
        color.stop()
        tof.stop()
        vision.stop()
        for led in leds:
            led.off()
        print("Stopped.")


if __name__ == "__main__":
    main()
