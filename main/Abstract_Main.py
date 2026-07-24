import time
import math
import numpy as np
import cv2 as cv

from enum import Enum, auto

from config import DT_S

from trajectory.builder import TrajectoryBuilder, RED, GREEN

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

TEST_SPEED = 0.35
CORNER_DUTY = 0.30

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

DEBOUNCE_S = 0.3 # time between section transitions
_last_transition_time = 0.0 # last time where we transitioned between corner and straight section

pending_line = None
pending_line_x = None
pending_line_y = None

# Maximum distance robot may travel between camera confirmation
# and floor-sensor detection.
#
# CALIBRATE based on:
# - camera position
# - color sensor position
# - where TRACK_LINE_NEAR_Y_RATIO is set.
LINE_ARM_MAX_TRAVEL_CM = 50.0


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


def update_pending_line_from_camera(
    vision_result,
    robot_x,
    robot_y,
):
    """
    Arm a line once the camera has reliably confirmed that
    the line is close ahead.

    The line remains armed even after it disappears from
    the camera because it may become hidden by the robot.
    """

    global pending_line
    global pending_line_x
    global pending_line_y

    track_lines = vision_result.get("track_lines")

    if not track_lines:
        return

    orange_confirmed = (
        track_lines["orange"]
        .get("confirmed_close", False)
    )

    blue_confirmed = (
        track_lines["blue"]
        .get("confirmed_close", False)
    )

    # Ambiguous camera result.
    if orange_confirmed and blue_confirmed:
        return

    if orange_confirmed:

        # Arm only if nothing is already pending.
        if pending_line is None:
            pending_line = "orange"
            pending_line_x = robot_x
            pending_line_y = robot_y

            print("Camera armed ORANGE line")

    elif blue_confirmed:

        if pending_line is None:
            pending_line = "blue"
            pending_line_x = robot_x
            pending_line_y = robot_y

            print("Camera armed BLUE line")


def confirm_floor_line(
    orange_seen,
    blue_seen,
    robot_x,
    robot_y,
):
    """
    Confirm a TCS3200 line detection only if the camera
    previously armed the same line nearby.

    Returns:
        confirmed_orange
        confirmed_blue
    """

    global pending_line
    global pending_line_x
    global pending_line_y

    if pending_line is None:
        return False, False

    distance_travelled = math.hypot(
        robot_x - pending_line_x,
        robot_y - pending_line_y,
    )

    # Camera confirmation is now too old spatially.
    if distance_travelled > LINE_ARM_MAX_TRAVEL_CM:

        print(
            f"Expired pending {pending_line} line "
            f"after {distance_travelled:.1f} cm"
        )

        pending_line = None
        pending_line_x = None
        pending_line_y = None

        return False, False

    confirmed_orange = (
        orange_seen
        and not blue_seen
        and pending_line == "orange"
    )

    confirmed_blue = (
        blue_seen
        and not orange_seen
        and pending_line == "blue"
    )

    if confirmed_orange or confirmed_blue:

        print(
            f"Confirmed {pending_line} line "
            f"after {distance_travelled:.1f} cm"
        )

        # Consume the pending line.
        pending_line = None
        pending_line_x = None
        pending_line_y = None

    return confirmed_orange, confirmed_blue


def add_section(orange_seen, blue_seen):
    global driving_direction
    global section_state
    global corners_completed
    global laps
    global corner_initialized
    global current_trajectory
    global near_s
    global _last_transition_time

    if state != State.RUNNING:
        return None

    if not (orange_seen or blue_seen):
        return None

    now = time.monotonic()

    if now - _last_transition_time < DEBOUNCE_S:
        return None


    # --------------------------------------------
    # First line determines driving direction
    # --------------------------------------------

    if driving_direction == DrivingDirection.UNKNOWN:

        if blue_seen and not orange_seen:
            driving_direction = DrivingDirection.CCW
            print("Direction detected: CCW")

        elif orange_seen and not blue_seen:
            driving_direction = DrivingDirection.CW
            print("Direction detected: CW")

        else:
            return None


    cw = driving_direction == DrivingDirection.CW
    ccw = driving_direction == DrivingDirection.CCW


    # --------------------------------------------
    # STRAIGHT -> CORNER
    # --------------------------------------------

    if section_state == SectionState.STRAIGHT:

        entering_corner = (
            (cw and orange_seen)
            or
            (ccw and blue_seen)
        )

        if entering_corner:

            section_state = SectionState.CORNER

            corner_initialized = False
            current_trajectory = None
            near_s = 0.0

            _last_transition_time = now

            print("Entering corner")

            return "ENTER_CORNER"


    # --------------------------------------------
    # CORNER -> STRAIGHT
    # --------------------------------------------

    elif section_state == SectionState.CORNER:

        exiting_corner = (
            (cw and blue_seen)
            or
            (ccw and orange_seen)
        )

        if exiting_corner:

            corners_completed += 1

            section_state = SectionState.STRAIGHT

            corner_initialized = False
            current_trajectory = None
            near_s = 0.0

            _last_transition_time = now


            # Four corners = one complete lap
            if corners_completed >= 4:

                corners_completed = 0
                laps += 1

                print(f"Lap completed: {laps}/3")

            else:

                print(
                    f"Corner completed: "
                    f"{corners_completed}/4"
                )

            return "EXIT_CORNER"

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


def update_target_theta():
    global target_theta

    target_theta = get_target_theta(
        driving_direction,
        corners_completed
    )


def calculate_straight_steering(theta):
    """
    Keep the robot aligned with the target heading.

    theta:
        Current robot heading from the EKF, in radians.

    Returns:
        Steering correction in degrees.
    """

    heading_error = normalize_angle(
        theta - target_theta
    )

    # Ignore very small heading fluctuations
    if abs(heading_error) < HEADING_DEADBAND_RAD:
        return 0.0

    weighted_error = PID_HEADING_W * heading_error

    steering_deg = straight_pid._compute(
        weighted_error
    )

    return steering_deg

def take_step(car, robot, theta):

    steering_deg = calculate_straight_steering(theta)

    robot.update_steering(steering_deg)

    car.set_all(
        direction='f',
        speed=TEST_SPEED,
        angle=steering_deg
    )

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
    steering_pid
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

    steering_deg = steering_pid._compute(
        combined_error
    )

    steering_deg = max(
        -SERVO_MAX_DEG,
        min(SERVO_MAX_DEG, steering_deg)
    )


    robot.update_steering(steering_deg)

    car.set_all(
        direction='f',
        speed=CORNER_DUTY,
        angle=steering_deg
    )

    return steering_deg


def reset_race():
    global driving_direction
    global section_state
    global state
    global corners_completed
    global laps
    global target_theta
    global corner_initialized
    global current_trajectory
    global near_s
    global _last_transition_time
    global finish_entry_x
    global finish_entry_y
    global pending_line
    global pending_line_x
    global pending_line_y

    driving_direction = DrivingDirection.UNKNOWN

    section_state = SectionState.STRAIGHT

    state = State.WAITING

    corners_completed = 0
    laps = 0

    target_theta = INITIAL_THETA

    corner_initialized = False
    current_trajectory = None
    near_s = 0.0

    _last_transition_time = 0.0

    finish_entry_x = None
    finish_entry_y = None

    pending_line = None
    pending_line_x = None
    pending_line_y = None

    clear_pillar()
    straight_pid.reset()


def main():
    global state
    global finish_entry_x
    global finish_entry_y

    reset_race()

    (
        start_button,
        leds,
        encoders,
        color,
        car,
        robot,
        ekf
    ) = initialize_hardware()

    vision = VisionThread()
    vision.start()

    # Robot must remain still
    gyro_bias = calibrate_gyro(encoders)


    wait_for_start(
        start_button,
        leds
    )


    # Global coordinate origin
    ekf.initialize(
        x0=0.0,
        y0=0.0,
        theta0=0.0
    )

    encoders.reset()
    robot.reset()

    # First straight is +X
    update_target_theta()

    last_time = time.monotonic()


    try:

        while state == State.RUNNING:

            now = time.monotonic()

            dt = now - last_time
            last_time = now


            # ==========================================
            # 1. SENSORS + EKF
            # ==========================================

            (
                speed,
                v_l,
                v_r,
                omega,
                x,
                y,
                theta,
                orange_seen,
                blue_seen

            ) = read_sensors_and_update_ekf(
                encoders,
                color,
                ekf,
                robot,
                dt,
                gyro_bias
            )

            # ==========================================
            # 2. UPDATE TRACK SECTION
            # ==========================================
            
            
            vision_result = vision.get_latest_result()

            vision_result = transform_to_global(
                vision_result,
                x,
                y,
                theta,
            )

            debug_frame = vision.get_latest_frame()

            if debug_frame is not None:

                cv.imshow(
                    "WRO Vision",
                    debug_frame,
                )

                cv.waitKey(1)

            pillars = vision_result.get(
                "pillars",
                [],
            )



            # Camera PRE-CONFIRMATION
            update_pending_line_from_camera(
                vision_result,
                x,
                y,
            )
            
            # FLOOR SENSOR + CAMERA FUSION
            confirmed_orange, confirmed_blue = (
                confirm_floor_line(
                    orange_seen,
                    blue_seen,
                    x,
                    y,
                )
            )

            # SECTION UPDATE
            transition = add_section(
                confirmed_orange,
                confirmed_blue
            )


            # New corner started
            if transition == "ENTER_CORNER":
                clear_pillar()
                straight_pid.reset()


            # Corner finished -> new straight
            elif transition == "EXIT_CORNER":

                update_target_theta()

                straight_pid.reset()


                # Three laps complete:
                # we have just entered the finish straight.
                if laps >= 3:

                    finish_entry_x = x
                    finish_entry_y = y


            # ==========================================
            # 3. FINISH LOGIC
            # ==========================================

            if (
                laps >= 3
                and finish_entry_x is not None
            ):

                distance_into_finish = math.hypot(
                    x - finish_entry_x,
                    y - finish_entry_y
                )

                if distance_into_finish >= 40.0:

                    print("Finish section reached.")

                    state = State.FINISHED

                    break


            # ==========================================
            # 4. CHOOSE CONTROLLER
            # ==========================================

            if section_state == SectionState.STRAIGHT:

                # Continue an existing swerve even if the pillar
                # has disappeared from the camera.
    
                if (
                    pillar_initialized
                    or pillar_in_range(
                        pillars,
                        x,
                        y,
                        theta,
                    )
                ):

                    steering, pillar_done = pillar_step(
                        car,
                        robot,
                        pillars,
                        x,
                        y,
                        theta,
                        straight_pid,
                    )

                    if pillar_done:
                        straight_pid.reset()

                else:

                    steering = take_step(
                        car,
                        robot,
                        theta,
                    )

            elif section_state == SectionState.CORNER:

                steering = corner_step(
                    car,
                    robot,
                    x,
                    y,
                    theta,
                    straight_pid
                )


            # ==========================================
            # 5. DEBUG
            # ==========================================

            print(
                f"section={section_state.name} | "
                f"dir={driving_direction.name} | "
                f"lap={laps} | "
                f"corner={corners_completed} | "
                f"theta={math.degrees(theta):+.1f}° | "
                f"target={math.degrees(target_theta):+.1f}° | "
                f"steer={steering:+.1f}°"
            )


            # ==========================================
            # 6. CONTROL LOOP RATE
            # ==========================================

            elapsed = time.monotonic() - now

            if elapsed < DT_S:
                time.sleep(DT_S - elapsed)


    finally:

        car.stop()
        vision.stop()
        color.stop()
        cv.destroyAllWindows()
        
        for led in leds:
            led.off()

        state = State.FINISHED
    
if __name__ == "__main__":
    main()

            
#This method calculates the nearest distance from the robot to the curve

#decides if the robot should stop in this section or not, based on the number of sections and laps
#def stop_in_this_section():        
  
#def parking():
    
#def exit_parking_lot():
    
#def special_trick():
    

    

