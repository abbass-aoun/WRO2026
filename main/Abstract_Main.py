import time
import csv
import math
import numpy as np
import cv2 as cv

from enum import Enum, auto

from config import DT_S 

from trajectory.builder import TrajectoryBuilder, RED, GREEN
from control.tof_sensor import ToFSensors

from main.support import calibrate_gyro
from main.vision_adapter import VisionThread, transform_to_global
from main.initialize_hardware import initialize_hardware

#-----------------------------------------------------------
# States of the robot
#-----------------------------------------------------------

class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

#-----------------------------------------------------------

class DrivingDirection(Enum):
    UNKNOWN = auto()
    CCW = auto()
    CW = auto()

#-----------------------------------------------------------

class SectionState(Enum):
    STRAIGHT = auto()
    CORNER = auto()

#-----------------------------------------------------------


driving_direction = DrivingDirection.UNKNOWN
state = State.WAITING
section_state = SectionState.STRAIGHT

corner_initialized = False
current_trajectory = None
near_s = 0.0

finish_entry_x = None
finish_entry_y = None

corners_completed = 0 # Number of 90-degree corners that have been COMPLETED.
laps = 0

INITIAL_THETA = 0.0  # radians (The robot starts facing along the global +X axis).
target_theta = 0.0 

HEADING_DEADBAND_RAD = math.radians(1.0) # minimum error in heading direction for correction (reduces switches and movement in servo)
STRAIGHT_KP = 0.27          # servo degrees per heading-error degree
STRAIGHT_CROSS_TRACK_KP = 0.20  # servo degrees per cm
STRAIGHT_MAX_DEG = 18.0
MAX_STEERING_CHANGE = 6.0   # maximum change per control update

STRAIGHT_ALIGNMENT_TOLERANCE_RAD = math.radians(3.0)
STRAIGHT_CROSS_TRACK_MAX_DEG = 2.0

straight_reference_ready = False

_previous_straight_steering = 0.0

DEBOUNCE_S = 0.3 # time between section transitions
_last_transition_time = 0.0 # last time where we transitioned between corner and straight section


# Minimum forward progress required before accepting the
# next STRAIGHT -> CORNER line detection.
MIN_STRAIGHT_PROGRESS_CM = 65.0
FIRST_LINE_MIN_PROGRESS_CM = 15.0
STRAIGHT_BACKUP_ENTRY_DISTANCE_CM = 120.0

# True only when the current corner was entered because
# the 140 cm backup activated.
corner_entered_by_backup = False

# Records whether the expected second line was seen.
# It confirms the corner but no longer controls the exit.
corner_exit_line_seen = False

# A backup-entered corner may exit automatically when the
# generated corner trajectory is nearly complete.
CORNER_AUTO_EXIT_REMAINING_CM = 8.0
CORNER_AUTO_EXIT_HEADING_TOLERANCE_RAD = math.radians(12.0)

FIRST_CORNER_BACKUP_DISTANCE_CM = 65.0

# Direction to assume when the first line is completely missed.
FIRST_CORNER_BACKUP_DIRECTION = DrivingDirection.CW


first_line_ref_x = None
first_line_ref_y = None

# EKF position where the current straight section began.
straight_ref_x = None
straight_ref_y = None

# Reference used for floor-line distance validation and
# the 140 cm backup. It is set immediately at corner exit.
straight_progress_ref_x = None
straight_progress_ref_y = None


pillar_initialized = False
pillar_trajectory = None
pillar_near_s = 0.0
_active_pillar_key = None


#-----------------------------------------------------------
# Initializing PID controller
#-----------------------------------------------------------

from control.pid_controller import PIDController
from config import(
    PID_KP,
    PID_KI,
    PID_KD,
    PID_WINDUP_LIM,
    PID_HEADING_W,
    SERVO_MAX_DEG,
    CORNER_RADIUS_CM,
    WHEELBASE_CM,

    PILLAR_CLEARANCE_CM,
    PILLAR_TRIGGER_CM,
    PILLAR_RECONNECT_CM,
    PILLAR_DONE_CM,
    PILLAR_DUTY,
    PILLAR_APPROACH_CM,
)

straight_pid = PIDController(
    Kp=PID_KP,
    Ki=PID_KI,
    Kd=PID_KD,
    output_limits=(-SERVO_MAX_DEG, SERVO_MAX_DEG),
    windup_limit=PID_WINDUP_LIM
)
#-----------------------------------------------------------

def wait_for_start(start_button, leds):
    global state

    leds_on = False
    last_toggle = time.monotonic()

    print("Ready — waiting for start button.")

    while state == State.WAITING:

        # Toggle all LEDs every 0.5 seconds
        if time.monotonic() - last_toggle >= 0.5:
            leds_on = not leds_on

            for led in leds:
                if leds_on:
                    led.on()
                else:
                    led.off()

            last_toggle = time.monotonic()

        # Start race when physical button is pressed
        if start_button.is_pressed:

            # Turn all LEDs off before starting
            for led in leds:
                led.off()

            state = State.RUNNING
            print("GO!")
            break

        time.sleep(0.01)


def read_sensors_and_update_ekf(encoders, color, ekf, robot, dt, gyro_bias):
    
    # 1. Encoders
    v_l, v_r = encoders.get_linear_speeds()
    speed    = 0.5 * (v_l + v_r)

    # 2. Gyro, bias-corrected
    omega = encoders.get_yaw_rate() - gyro_bias

    # 3. EKF — gyro supplies the rotation, steer angle is the fallback
    steer_rad = math.radians(robot.steer_angle)
    ekf.predict(speed, steer_rad, dt, omega_gyro=omega)

    # 4. Publish to shared state
    x, y, theta = ekf.state
    robot.update_pose(x, y, theta)
    robot.update_speed(speed)

    # 5. Color flags (background thread — instant read)
    return speed, v_l, v_r, omega, x, y, theta, color.orange_seen, color.blue_seen


def confirm_floor_line(orange_seen, blue_seen, robot_x, robot_y):
    """
    Validate a floor-line reading.

    Rules:
    1. Reject ambiguous readings.
    2. Accept the first line immediately because it determines direction.
    3. While inside a corner, allow the opposite-colored exit line.
    4. During later straight sections, reject detections until enough
       forward progress has been made.
    """

    # Both False means no line.
    # Both True means an ambiguous reading.
    if orange_seen == blue_seen:
        return False, False

    # The first valid line determines CW or CCW.
    # The starting position may be close to this line,
    # so no minimum-distance condition is applied.
    if driving_direction == DrivingDirection.UNKNOWN:

        # From here, the robot is in a straight section.
        # Use the reference recorded immediately at corner exit,
        # not the delayed steering reference.
        if (
            straight_progress_ref_x is None
            or straight_progress_ref_y is None
        ):
            return False, False

        dx = robot_x - straight_progress_ref_x
        dy = robot_y - straight_progress_ref_y

        first_line_progress_cm = (
            dx * math.cos(INITIAL_THETA)
            + dy * math.sin(INITIAL_THETA)
        )

        if first_line_progress_cm < FIRST_LINE_MIN_PROGRESS_CM:
            return False, False

        return orange_seen, blue_seen

    # While cornering, pass the detection to add_section().
    # add_section() will accept only the expected opposite color.
    if section_state == SectionState.CORNER:
        return orange_seen, blue_seen

    # From here, the robot is in a straight section.
    if straight_ref_x is None or straight_ref_y is None:
        return False, False

    dx = robot_x - straight_ref_x
    dy = robot_y - straight_ref_y

    # Measure progress along the intended straight direction.
    # Sideways movement contributes very little to this value.
    forward_progress_cm = (
        dx * math.cos(target_theta)
        + dy * math.sin(target_theta)
    )

    if forward_progress_cm < MIN_STRAIGHT_PROGRESS_CM:
        return False, False

    return orange_seen, blue_seen


def add_section(
    orange_seen,
    blue_seen,
    force_corner_exit=False,
):
    global driving_direction
    global section_state
    global corners_completed
    global laps
    global corner_initialized
    global current_trajectory
    global near_s
    global _last_transition_time
    global corner_exit_line_seen

    if state != State.RUNNING:
        return None

    now = time.monotonic()

    # =====================================================
    # Forced CORNER -> STRAIGHT transition
    # =====================================================
    # This is called only after corner_step() reports that
    # the corner trajectory has been completed.
    if force_corner_exit:

        if section_state != SectionState.CORNER:
            return None

        # Safety fallback. Normally the first-corner backup
        # already assigns this direction before entering.
        if driving_direction == DrivingDirection.UNKNOWN:
            driving_direction = FIRST_CORNER_BACKUP_DIRECTION

        # Retry on later loops if the transition is still
        # inside the debounce period.
        if now - _last_transition_time < DEBOUNCE_S:
            return None

        corners_completed += 1
        section_state = SectionState.STRAIGHT

        corner_initialized = False
        current_trajectory = None
        near_s = 0.0
        corner_exit_line_seen = False

        _last_transition_time = now

        # Four completed corners form one lap.
        if corners_completed >= 4:
            corners_completed = 0
            laps += 1

            print(
                f"Lap completed: {laps}/3"
            )
        else:
            print(
                f"Corner completed: "
                f"{corners_completed}/4"
            )

        return "EXIT_CORNER"

    # A normal sensor call requires exactly one color.
    if not (orange_seen or blue_seen):
        return None

    if orange_seen and blue_seen:
        return None

    # =====================================================
    # First line determines direction
    # =====================================================
    if driving_direction == DrivingDirection.UNKNOWN:

        if blue_seen:
            driving_direction = DrivingDirection.CCW
            print("Direction detected: CCW")

        elif orange_seen:
            driving_direction = DrivingDirection.CW
            print("Direction detected: CW")

    cw = driving_direction == DrivingDirection.CW
    ccw = driving_direction == DrivingDirection.CCW

    # =====================================================
    # STRAIGHT -> CORNER
    # =====================================================
    if section_state == SectionState.STRAIGHT:

        entering_corner = (
            (cw and orange_seen)
            or
            (ccw and blue_seen)
        )

        if not entering_corner:
            return None

        if now - _last_transition_time < DEBOUNCE_S:
            return None

        section_state = SectionState.CORNER

        corner_initialized = False
        current_trajectory = None
        near_s = 0.0

        # The second line has not been seen yet.
        corner_exit_line_seen = False

        _last_transition_time = now

        print("Entering corner")

        return "ENTER_CORNER"

    # =====================================================
    # Exit-line confirmation while cornering
    # =====================================================
    if section_state == SectionState.CORNER:

        expected_exit_line = (
            (cw and blue_seen)
            or
            (ccw and orange_seen)
        )

        if (
            expected_exit_line
            and not corner_exit_line_seen
        ):
            corner_exit_line_seen = True

            print(
                "[CORNER] Expected exit line detected. "
                "Continuing until the corner trajectory "
                "is complete."
            )

            return "EXIT_LINE_SEEN"

    return None


def initialize_straight_reference(x, y):
    """
    Called ONCE when the robot enters a new straight section.

    The current EKF position becomes a point on the desired
    straight reference line.
    """
    global straight_ref_x, straight_ref_y, target_theta

    straight_ref_x = x
    straight_ref_y = y

    target_theta = get_target_theta(
        driving_direction,
        corners_completed
    )


def begin_new_straight(x, y):
    """
    Prepare navigation state after completing a corner.

    The progress reference is stored immediately.
    The steering reference remains unset until heading alignment.
    """
    global target_theta
    global straight_ref_x
    global straight_ref_y
    global straight_progress_ref_x
    global straight_progress_ref_y
    global straight_reference_ready
    global _previous_straight_steering
    global corner_entered_by_backup

    # add_section() has already updated corners_completed.
    target_theta = get_target_theta(
        driving_direction,
        corners_completed,
    )

    # Used immediately for line validation and backup distance.
    straight_progress_ref_x = x
    straight_progress_ref_y = y

    # Steering reference will be created later, after alignment.
    straight_ref_x = None
    straight_ref_y = None
    straight_reference_ready = False

    _previous_straight_steering = 0.0
    corner_entered_by_backup = False

    straight_pid.reset()

    print(
        "New straight started: "
        f"x={x:.2f}, y={y:.2f}, "
        f"target={math.degrees(target_theta):+.2f}°"
    )


def normalize_angle(angle_rad: float) -> float:
    """
    Normalize an angle to the range [-pi, +pi).

    Example:
        270 degrees  -> -90 degrees
        360 degrees  ->   0 degrees
    """
    return math.atan2(
        math.sin(angle_rad),
        math.cos(angle_rad)
    )


def get_target_theta(direction, corners_done):

    if corners_done == 0:
        return INITIAL_THETA

    if direction == DrivingDirection.CCW:
        turn_sign = +1

    elif direction == DrivingDirection.CW:
        turn_sign = -1

    else:
        raise ValueError("Driving direction is still unknown.")

    target_theta = (
        INITIAL_THETA
        + turn_sign * corners_done * (math.pi / 2)
    )

    return normalize_angle(target_theta)


def calculate_straight_steering(x, y, theta):
    global _previous_straight_steering
    global straight_ref_x
    global straight_ref_y
    global straight_reference_ready

    heading_error = normalize_angle(
        theta - target_theta
    )

    heading_error_deg = math.degrees(
        heading_error
    )

    if abs(heading_error) < HEADING_DEADBAND_RAD:
        heading_correction = 0.0
    else:
        heading_correction = (
            STRAIGHT_KP * heading_error_deg
        )

    # After a corner, first align with target_theta.
    # Do not use cross-track correction yet.
    if not straight_reference_ready:

        cross_track_correction = 0.0

        if (
            abs(heading_error)
            <= STRAIGHT_ALIGNMENT_TOLERANCE_RAD
        ):
            straight_ref_x = x
            straight_ref_y = y
            straight_reference_ready = True

            print(
                "Straight reference initialized: "
                f"x={x:.2f}, y={y:.2f}"
            )

    else:
        dx = x - straight_ref_x
        dy = y - straight_ref_y

        cross_track_error_cm = (
            -dx * math.sin(target_theta)
            + dy * math.cos(target_theta)
        )

        cross_track_correction = (
            STRAIGHT_CROSS_TRACK_KP
            * cross_track_error_cm
        )

        # Prevent lateral correction from forcing
        # a large heading overshoot.
        cross_track_correction = max(
            -STRAIGHT_CROSS_TRACK_MAX_DEG,
            min(
                STRAIGHT_CROSS_TRACK_MAX_DEG,
                cross_track_correction,
            ),
        )

    desired_steering = (
        heading_correction
        + cross_track_correction
    )

    desired_steering = max(
        -STRAIGHT_MAX_DEG,
        min(
            STRAIGHT_MAX_DEG,
            desired_steering,
        ),
    )

    minimum = (
        _previous_straight_steering
        - MAX_STEERING_CHANGE
    )

    maximum = (
        _previous_straight_steering
        + MAX_STEERING_CHANGE
    )

    steering_deg = max(
        minimum,
        min(
            maximum,
            desired_steering,
        ),
    )

    _previous_straight_steering = steering_deg

    return steering_deg


def _pillars_ahead(pillars, x, y, theta):
    """
    Return visible pillars that are ahead of the robot,
    sorted nearest first.
    """

    ahead = []

    forward_x = math.cos(theta)
    forward_y = math.sin(theta)

    for pillar in pillars:

        px = pillar.get("global_x_cm")
        py = pillar.get("global_y_cm")
        color = pillar.get("color")

        if px is None or py is None:
            continue

        if color not in ("red", "green"):
            continue

        dx = px - x
        dy = py - y

        # Dot product with robot forward direction.
        # Positive means the pillar is ahead.
        forward_distance = (
            dx * forward_x
            + dy * forward_y
        )

        if forward_distance <= 0:
            continue

        distance = math.hypot(dx, dy)

        ahead.append(
            (distance, pillar)
        )

    ahead.sort(key=lambda item: item[0])

    return [
        pillar
        for _, pillar in ahead
    ]


def pillar_in_range(pillars, x, y, theta):
    """
    Check whether the nearest pillar ahead is close enough
    to begin avoidance.
    """

    ahead = _pillars_ahead(
        pillars,
        x,
        y,
        theta,
    )

    if not ahead:
        return False

    pillar = ahead[0]

    distance = math.hypot(
        pillar["global_x_cm"] - x,
        pillar["global_y_cm"] - y,
    )

    return distance <= PILLAR_TRIGGER_CM


def calculate_trajectory_to_pillar(
    pillars,
    x,
    y,
    theta,
):
    """
    Build a two-segment Bezier avoidance trajectory
    around the nearest pillar ahead.
    """

    ahead = _pillars_ahead(
        pillars,
        x,
        y,
        theta,
    )

    if not ahead:
        return None

    # -------------------------------
    # Nearest pillar
    # -------------------------------

    pillar = ahead[0]

    pillar_x = pillar["global_x_cm"]
    pillar_y = pillar["global_y_cm"]
    color = pillar["color"]

    pillar_color = (
        GREEN
        if color == "green"
        else RED
    )

    # Direction of the current straight.
    nx = math.cos(target_theta)
    ny = math.sin(target_theta)

    # -------------------------------
    # Second pillar already visible
    # -------------------------------

    if len(ahead) >= 2:

        next_pillar = ahead[1]

        next_x = next_pillar["global_x_cm"]
        next_y = next_pillar["global_y_cm"]
        next_color = next_pillar["color"]

        # Green → approach on left.
        # Red   → approach on right.
        side = (
            +1.0
            if next_color == "green"
            else -1.0
        )

        end_x = (
            next_x
            + side * (-ny) * PILLAR_CLEARANCE_CM
            - PILLAR_APPROACH_CM * nx
        )

        end_y = (
            next_y
            + side * nx * PILLAR_CLEARANCE_CM
            - PILLAR_APPROACH_CM * ny
        )

    # -------------------------------
    # Only one pillar visible
    # -------------------------------

    else:

        # Reconnect PILLAR_RECONNECT_CM AFTER the pillar.
        end_x = (
            pillar_x
            + PILLAR_RECONNECT_CM * nx
        )

        end_y = (
            pillar_y
            + PILLAR_RECONNECT_CM * ny
        )

    return TrajectoryBuilder.pillar_swerve(
        start_x=x,
        start_y=y,
        start_theta=theta,

        pillar_x=pillar_x,
        pillar_y=pillar_y,
        pillar_color=pillar_color,

        end_x=end_x,
        end_y=end_y,
        end_theta=target_theta,

        clearance=PILLAR_CLEARANCE_CM,
    )


def pillar_step(
    car,
    robot,
    pillars,
    x,
    y,
    theta,
    steering_pid,
):
    """
    Build a pillar-avoidance trajectory once,
    then follow it until complete.

    Returns:
        steering_deg, done
    """

    global pillar_trajectory
    global pillar_near_s
    global pillar_initialized
    global _active_pillar_key

    # ========================================
    # Build trajectory once
    # ========================================

    if not pillar_initialized:

        ahead = _pillars_ahead(
            pillars,
            x,
            y,
            theta,
        )

        if not ahead:
            return 0.0, True

        active = ahead[0]

        pillar_trajectory = (
            calculate_trajectory_to_pillar(
                pillars,
                x,
                y,
                theta,
            )
        )

        if pillar_trajectory is None:
            return 0.0, True

        pillar_near_s = 0.0
        pillar_initialized = True

        _active_pillar_key = (
            round(active["global_x_cm"], 1),
            round(active["global_y_cm"], 1),
            active["color"],
        )

        steering_pid.reset()

        print(
            f"Pillar swerve created: "
            f"{_active_pillar_key}"
        )

    # ========================================
    # Find position along trajectory
    # ========================================

    pillar_near_s = (
        pillar_trajectory.find_closest(
            x,
            y,
            pillar_near_s,
        )
    )

    px, py = pillar_trajectory.get_point(
        pillar_near_s
    )

    tx, ty = pillar_trajectory.get_tangent(
        pillar_near_s
    )

    path_theta = math.atan2(ty, tx)

    # ========================================
    # Calculate tracking errors
    # ========================================

    cross_track_error = (
        -math.sin(path_theta) * (x - px)
        + math.cos(path_theta) * (y - py)
    )

    heading_error = normalize_angle(
        theta - path_theta
    )

    combined_error = (
        cross_track_error
        + PID_HEADING_W * heading_error
    )

    # ========================================
    # Steering
    # ========================================

    steering_deg = steering_pid._compute(
        combined_error
    )

    steering_deg = max(
        -SERVO_MAX_DEG,
        min(
            SERVO_MAX_DEG,
            steering_deg,
        ),
    )

    robot.update_steering(steering_deg)

    car.set_all(
        direction="f",
        speed=PILLAR_DUTY,
        angle=steering_deg,
    )

    # ========================================
    # Check trajectory completion
    # ========================================

    remaining = (
        pillar_trajectory.total_length
        - pillar_near_s
    )

    done = remaining <= PILLAR_DONE_CM

    if done:
        clear_pillar()
        print("Pillar cleared")

    return steering_deg, done


def clear_pillar():
    """Reset state for the next pillar."""

    global pillar_initialized
    global pillar_trajectory
    global pillar_near_s
    global _active_pillar_key

    pillar_initialized = False
    pillar_trajectory = None
    pillar_near_s = 0.0
    _active_pillar_key = None


def corner_step(
    car,
    robot,
    x,
    y,
    theta,
    steering_pid,
    motor_duty
):
    global current_trajectory
    global near_s
    global corner_initialized


    # --------------------------------------------
    # Build corner trajectory ONCE
    # --------------------------------------------

    if not corner_initialized:

        if driving_direction == DrivingDirection.CCW:
            turn_dir = +1
        else:
            turn_dir = -1

        current_trajectory = TrajectoryBuilder.corner(
            start_x=x,
            start_y=y,
            start_theta=theta,
            turn_direction=turn_dir,
            radius=CORNER_RADIUS_CM
        )

        near_s = 0.0
        corner_initialized = True

        steering_pid.reset()

        print("Corner trajectory created")


    # --------------------------------------------
    # Find robot's closest point on curve
    # --------------------------------------------

    near_s = current_trajectory.find_closest(
        x,
        y,
        near_s
    )

    px, py = current_trajectory.get_point(near_s)

    tx, ty = current_trajectory.get_tangent(near_s)


    # --------------------------------------------
    # Calculate trajectory errors
    # --------------------------------------------

    path_theta = math.atan2(ty, tx)

    cross_track_error = (
        -math.sin(path_theta) * (x - px)
        + math.cos(path_theta) * (y - py)
    )

    heading_error = normalize_angle(
        theta - path_theta
    )

    combined_error = (
        cross_track_error
        + PID_HEADING_W * heading_error
    )


    # --------------------------------------------
    # Steering
    # --------------------------------------------

    # PID corrects position and heading errors.
    pid_correction_deg = steering_pid._compute(
        combined_error
    )

    # Base steering required by the corner geometry.
    curvature = current_trajectory.get_curvature(
        near_s
    )

    maximum_curvature = 1.0 / CORNER_RADIUS_CM

    curvature = max(
        -maximum_curvature,
        min(maximum_curvature, curvature),
    )

    # Positive curvature means a left/CCW curve.
    # Negative logical steering means left.
    feedforward_deg = -math.degrees(
        math.atan(
            WHEELBASE_CM * curvature
        )
    )

    steering_deg = (
        feedforward_deg
        + pid_correction_deg
    )

    steering_deg = max(
        -SERVO_MAX_DEG,
        min(
            SERVO_MAX_DEG,
            steering_deg,
        ),
    )

    

    robot.update_steering(steering_deg)

    car.set_all(
        direction="f",
        speed=motor_duty,
        angle=steering_deg,
    )

    # --------------------------------------------
    # Corner trajectory completion
    # --------------------------------------------
    total_length = current_trajectory.total_length

    remaining_cm = max(
        0.0,
        total_length - near_s,
    )

    progress_fraction = (
        near_s / total_length
        if total_length > 0.0
        else 1.0
    )

    # Use the trajectory's own final tangent. This is better
    # than assuming the robot entered the corner at an exact
    # cardinal heading.
    end_tx, end_ty = current_trajectory.get_tangent(
        total_length
    )

    trajectory_end_theta = math.atan2(
        end_ty,
        end_tx,
    )

    end_heading_error = abs(
        normalize_angle(
            theta - trajectory_end_theta
        )
    )

    corner_done = (
        (
            remaining_cm <= CORNER_AUTO_EXIT_REMAINING_CM
            and end_heading_error
            <= CORNER_AUTO_EXIT_HEADING_TOLERANCE_RAD
        )
        or
        (
            progress_fraction >= 0.75
            and end_heading_error <= math.radians(5.0)
        )
    )

    return steering_deg, corner_done


def reset_race():
    global driving_direction
    global section_state
    global state
    global corners_completed
    global laps
    global target_theta
    global corner_initialized
    global corner_exit_line_seen
    global current_trajectory
    global near_s
    global _last_transition_time
    global finish_entry_x
    global finish_entry_y
    global straight_ref_x
    global straight_ref_y
    global first_line_ref_x
    global first_line_ref_y
    global straight_progress_ref_x
    global straight_progress_ref_y
    global straight_reference_ready
    global corner_entered_by_backup
    global _previous_straight_steering

    driving_direction = DrivingDirection.UNKNOWN

    section_state = SectionState.STRAIGHT

    state = State.WAITING

    corners_completed = 0
    laps = 0

    target_theta = INITIAL_THETA

    corner_initialized = False
    current_trajectory = None
    near_s = 0.0
    corner_exit_line_seen = False

    _last_transition_time = 0.0

    finish_entry_x = None
    finish_entry_y = None

    straight_ref_x = None
    straight_ref_y = None

    straight_progress_ref_x = None
    straight_progress_ref_y = None

    first_line_ref_x = None
    first_line_ref_y = None

    straight_reference_ready = False
    corner_entered_by_backup = False
    _previous_straight_steering = 0.0


    clear_pillar()
    straight_pid.reset()


def main():
    global state
    global target_theta
    global driving_direction
    global _previous_straight_steering
    global straight_ref_x
    global straight_ref_y
    global first_line_ref_x
    global first_line_ref_y
    global straight_reference_ready
    global straight_progress_ref_x
    global straight_progress_ref_y
    global corner_entered_by_backup

    STRAIGHT_DUTY = 0.65
    CORNER_DUTY = 0.55
    

    LOOP_PERIOD_S = 0.02
    PRINT_PERIOD_S = 0.10

    reset_race()

    (
        start_button,
        leds,
        encoders,
        color,
        car,
        robot,
        ekf,
    ) = initialize_hardware()

    print("\n=== FULL LAP TEST ===")
    print("The robot will:")
    print("1. Detect direction from the first floor line.")
    print("2. Complete four straight sections.")
    print("3. Complete four 90-degree corners.")
    print("4. Stop after one full lap.")
    print("5. Use the 140 cm entry backup when needed.")
    
    print("Press Ctrl+C immediately if the robot behaves incorrectly.")

    try:
        # -----------------------------------------
        # Safe setup
        # -----------------------------------------
        car.stop()
        car.set_steering(0)

        robot.update_steering(0.0)

        print("\nKeep the robot stationary during gyro calibration.")
        gyro_bias = calibrate_gyro(encoders)

        wait_for_start(start_button, leds)

        # Wait until the start button is released.
        while start_button.is_pressed:
            time.sleep(0.05)

        # -----------------------------------------
        # Reset navigation state
        # -----------------------------------------
        encoders.reset()
        robot.reset()

        ekf.initialize(
            x0=0.0,
            y0=0.0,
            theta0=0.0,
        )

        straight_ref_x = 0.0
        straight_ref_y = 0.0

        first_line_ref_x = 0.0
        first_line_ref_y = 0.0

        straight_progress_ref_x = 0.0
        straight_progress_ref_y = 0.0

        straight_reference_ready = True
        corner_entered_by_backup = False

        target_theta = INITIAL_THETA
        _previous_straight_steering = 0.0

        # corner_step() sets this when the target heading is reached.
        # The forced EXIT_CORNER is processed on the next loop.
        corner_exit_pending = False

        robot.update_pose(
            0.0,
            0.0,
            0.0,
        )
        robot.update_speed(0.0)
        robot.update_steering(0.0)

        straight_pid.reset()

        left_cm, right_cm = encoders.get_distances()

        previous_average_cm = 0.5 * (
            left_cm + right_cm
        )

        previous_time = time.monotonic()
        previous_print_time = previous_time

        color_log_file = open(
            "color_readings.csv",
            "w",
            newline="",
        )

        color_log = csv.writer(color_log_file)

        color_log.writerow([
            "time",
            "x",
            "y",
            "section",
            "r",
            "g",
            "b",
            "orange",
            "blue",
        ])
        
        print("\nStarting movement...\n")

        # =========================================
        # Main control loop
        # =========================================
        while state == State.RUNNING:

            now = time.monotonic()
            dt = now - previous_time

            if dt < LOOP_PERIOD_S:
                time.sleep(
                    LOOP_PERIOD_S - dt
                )
                continue

            previous_time = now

            # -------------------------------------
            # Encoder distance and measured speed
            # -------------------------------------
            left_cm, right_cm = encoders.get_distances()

            average_cm = 0.5 * (
                left_cm + right_cm
            )

            delta_distance_cm = (
                average_cm
                - previous_average_cm
            )

            previous_average_cm = average_cm

            if dt > 0.0:
                speed_cm_s = (
                    delta_distance_cm / dt
                )
            else:
                speed_cm_s = 0.0

            # -------------------------------------
            # Corrected gyro yaw rate
            # -------------------------------------
            omega_rad_s = (
                encoders.get_yaw_rate()
                - gyro_bias
            )

            # Steering command that was active
            # during the previous movement interval.
            previous_steering_rad = math.radians(
                robot.steer_angle
            )

            # -------------------------------------
            # EKF prediction
            # -------------------------------------
            ekf.predict(
                speed=speed_cm_s,
                steer_angle=previous_steering_rad,
                dt=dt,
                omega_gyro=omega_rad_s,
            )

            x, y, theta = ekf.state

            robot.update_pose(
                x,
                y,
                theta,
            )

            robot.update_speed(
                speed_cm_s
            )

            # -------------------------------------
            # Floor-line detection
            # -------------------------------------
            orange_seen = color.orange_seen
            blue_seen = color.blue_seen

            confirmed_orange, confirmed_blue = (
                confirm_floor_line(
                    orange_seen=orange_seen,
                    blue_seen=blue_seen,
                    robot_x=x,
                    robot_y=y,
                )
            )

            r, g, b = color.rgb

            color_log.writerow([
                time.monotonic(),
                x,
                y,
                section_state.name,
                r,
                g,
                b,
                orange_seen,
                blue_seen,
            ])

            if orange_seen or blue_seen:
                print(
                    f"RAW COLOR | "
                    f"RGB={color.rgb} | "
                    f"orange={orange_seen} | "
                    f"blue={blue_seen}"
                )

            # -------------------------------------------------
            # Process corner completion or current color reading
            # -------------------------------------------------
            if corner_exit_pending:

                transition = add_section(
                    orange_seen=False,
                    blue_seen=False,
                    force_corner_exit=True,
                )

                # If debounce rejected the forced exit, keep
                # corner_exit_pending True and retry next loop.
                if transition == "EXIT_CORNER":
                    corner_exit_pending = False

            else:
                transition = add_section(
                    confirmed_orange,
                    confirmed_blue,
                )

                # This entry came from a real floor-line reading.
                if transition == "ENTER_CORNER":
                    corner_entered_by_backup = False

            # First-corner backup: the first color line was missed.
            if (
                transition is None
                and section_state == SectionState.STRAIGHT
                and driving_direction == DrivingDirection.UNKNOWN
            ):
                dx = x - first_line_ref_x
                dy = y - first_line_ref_y

                first_straight_progress_cm = (
                    dx * math.cos(INITIAL_THETA)
                    + dy * math.sin(INITIAL_THETA)
                )

                if (
                    first_straight_progress_cm
                    >= FIRST_CORNER_BACKUP_DISTANCE_CM
                ):
                    print(
                        "\n[BACKUP] First corner line missed. "
                        f"Progress={first_straight_progress_cm:.2f} cm"
                    )

                    if (
                        FIRST_CORNER_BACKUP_DIRECTION
                        == DrivingDirection.CW
                    ):
                        # Orange sets CW and enters the corner.
                        transition = add_section(
                            orange_seen=True,
                            blue_seen=False,
                        )
                    else:
                        # Blue sets CCW and enters the corner.
                        transition = add_section(
                            orange_seen=False,
                            blue_seen=True,
                        )

                    if transition == "ENTER_CORNER":
                        corner_entered_by_backup = True

            # A normal sensor transition always has priority.
            if transition == "ENTER_CORNER":
                corner_entered_by_backup = False

            # -------------------------------------------------
            # Backup entry into the next corner
            # -------------------------------------------------
            # This applies after the first corner. The first
            # floor line is still required to determine direction.
            if (
                transition is None
                and section_state == SectionState.STRAIGHT
                and driving_direction != DrivingDirection.UNKNOWN
                and straight_progress_ref_x is not None
                and straight_progress_ref_y is not None
            ):
                dx = x - straight_progress_ref_x
                dy = y - straight_progress_ref_y

                straight_forward_progress_cm = (
                    dx * math.cos(target_theta)
                    + dy * math.sin(target_theta)
                )

                if (
                    straight_forward_progress_cm
                    >= STRAIGHT_BACKUP_ENTRY_DISTANCE_CM
                ):
                    print(
                        "\n[BACKUP] Corner entry line missed. "
                        f"Forward progress="
                        f"{straight_forward_progress_cm:.2f} cm"
                    )

                    if driving_direction == DrivingDirection.CW:
                        # Orange normally enters a CW corner.
                        transition = add_section(
                            orange_seen=True,
                            blue_seen=False,
                        )

                    elif driving_direction == DrivingDirection.CCW:
                        # Blue normally enters a CCW corner.
                        transition = add_section(
                            orange_seen=False,
                            blue_seen=True,
                        )

                    if transition == "ENTER_CORNER":
                        corner_entered_by_backup = True

                        print(
                            "[BACKUP] Corner entered through "
                            "distance backup."
                        )

            # -------------------------------------
            # Handle section transitions
            # -------------------------------------
            if transition == "ENTER_CORNER":
                corner_exit_pending = False
                print("\n=== ENTERING CORNER ===")
                print(
                    f"Direction: {driving_direction}"
                )
                print(
                    f"Entry source: "
                    f"{'BACKUP' if corner_entered_by_backup else 'COLOR SENSOR'}"
                )
                print(
                    f"Entry position: "
                    f"x={x:.2f}, y={y:.2f}"
                )
                print(
                    f"Entry heading: "
                    f"{math.degrees(theta):+.2f}°"
                )

            elif transition == "EXIT_CORNER":

                print("\n=== CORNER EXITED ===")
                print(
                    f"Corners completed in current lap: "
                    f"{corners_completed}"
                )
                print(
                    f"Laps completed: {laps}"
                )
                print(
                    f"Exit position: "
                    f"x={x:.2f}, y={y:.2f}"
                )
                print(
                    f"Current heading: "
                    f"{math.degrees(theta):+.2f}°"
                )

                # add_section() resets corners_completed to 0
                # and increments laps after the fourth corner.
                if laps >= 1:
                    car.stop()
                    state = State.FINISHED

                    print("\n=== FULL LAP COMPLETED ===")
                    break

                # add_section() has already incremented
                # corners_completed before returning.
                target_theta = get_target_theta(
                    driving_direction,
                    corners_completed,
                )

                # Immediately store where the new straight begins.
                # This reference is used for:
                # 1. Floor-line distance validation.
                # 2. The 140 cm backup corner entry.
                straight_progress_ref_x = x
                straight_progress_ref_y = y

                # The direction-correction reference remains delayed
                # until the robot aligns with target_theta.
                straight_ref_x = None
                straight_ref_y = None
                straight_reference_ready = False

                _previous_straight_steering = 0.0

                # The previous corner is now fully finished.
                corner_entered_by_backup = False

                straight_pid.reset()

                print(
                    f"New target heading: "
                    f"{math.degrees(target_theta):+.2f}°"
                )
                print(
                    "[STRAIGHT] Progress reference set: "
                    f"x={straight_progress_ref_x:.2f}, "
                    f"y={straight_progress_ref_y:.2f}"
                )

            # -------------------------------------
            # Select the active controller
            # -------------------------------------
            if section_state == SectionState.CORNER:

                steering_deg, corner_done = corner_step(
                    car=car,
                    robot=robot,
                    x=x,
                    y=y,
                    theta=theta,
                    steering_pid=straight_pid,
                    motor_duty=CORNER_DUTY,
                )

                # Every corner exits when its trajectory is complete,
                # regardless of whether entry used color or backup distance.
                if corner_done and not corner_exit_pending:
                    corner_exit_pending = True

                    if corner_exit_line_seen:
                        print(
                            "\n[CORNER] Trajectory completed. "
                            "The expected exit line was detected."
                        )
                    else:
                        print(
                            "\n[CORNER BACKUP] Trajectory completed "
                            "without detecting the exit line."
                        )



            else:
                steering_deg = (
                    calculate_straight_steering(
                        x, y, theta
                    )
                )

                robot.update_steering(
                    steering_deg
                )

                car.set_all(
                    direction="f",
                    speed=STRAIGHT_DUTY,
                    angle=steering_deg,
                )

            
            # -------------------------------------
            # Terminal output
            # -------------------------------------
            if (
                now - previous_print_time
                >= PRINT_PERIOD_S
            ):
                print(
                    f"section={section_state.name:<8} | "
                    f"L={left_cm:6.2f} | "
                    f"R={right_cm:6.2f} | "
                    f"x={x:7.2f} | "
                    f"y={y:+7.2f} | "
                    f"theta="
                    f"{math.degrees(theta):+7.2f}° | "
                    f"target="
                    f"{math.degrees(target_theta):+7.2f}° | "
                    f"steer={steering_deg:+6.2f}° | "
                    f"orange={orange_seen} | "
                    f"blue={blue_seen} | "
                    
                    
                )

                previous_print_time = now

    
        # =========================================
        # Final result
        # =========================================
        car.stop()

        left_cm, right_cm = encoders.get_distances()
        x, y, theta = ekf.state

        print("\n=== TEST RESULT ===")
        print(
            f"Direction:             "
            f"{driving_direction}"
        )
        print(
            f"Section:               "
            f"{section_state}"
        )
        print(
            f"Corners completed:      "
            f"{corners_completed}"
        )
        print(
            f"Left encoder:           "
            f"{left_cm:.2f} cm"
        )
        print(
            f"Right encoder:          "
            f"{right_cm:.2f} cm"
        )
        print(
            f"EKF X:                  "
            f"{x:.2f} cm"
        )
        print(
            f"EKF Y:                  "
            f"{y:+.2f} cm"
        )
        print(
            f"Final heading:          "
            f"{math.degrees(theta):+.2f}°"
        )
        print(
            f"Target heading:         "
            f"{math.degrees(target_theta):+.2f}°"
        )
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    except Exception as error:
        print(
            f"\nTest stopped because of an error: "
            f"{error}"
        )
        raise

    finally:
        car.stop()
        car.set_steering(0)

        try:
            color.stop()
        except Exception:
            pass

        for led in leds:
            led.off()

        state = State.FINISHED
        color_log_file.close()
        print("Hardware stopped safely.")

    
if __name__ == "__main__":
    main()

            

    

