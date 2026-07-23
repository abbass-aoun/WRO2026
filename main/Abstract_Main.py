import time
import math
import numpy as np

from enum import Enum, auto

from trajectory.builder import TrajectoryBuilder

from main.support import calibrate_gyro
from config import DT_S


TEST_DUTY = 0.30
TEST_SPEED = 0.7
CORNER_DUTY = 0.30

steer_path_s = 0.0

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

class SectionState(Enum):
    STRAIGHT = auto()
    CORNER = auto()


driving_direction = DrivingDirection.UNKNOWN
state = State.WAITING
section_state = SectionState.STRAIGHT

corner_initialized = False
current_trajectory = None
near_s = 0.0

_last_transition_time = 0.0

finish_entry_x = None
finish_entry_y = None

corners_completed = 0 # Number of 90-degree corners that have been COMPLETED.
laps = 0
INITIAL_THETA = 0.0  # radians (The robot starts facing along the global +X axis).
target_theta = 0.0

HEADING_DEADBAND_RAD = math.radians(1.0)

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
    CORNER_RADIUS_CM
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


DEBOUNCE_S = 0.3


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

            transition = add_section(
                orange_seen,
                blue_seen
            )


            # New corner started
            if transition == "ENTER_CORNER":

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

                steering = take_step(
                    car,
                    robot,
                    theta
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

        color.stop()

        for led in leds:
            led.off()

        state = State.FINISHED
    
if __name__ == "__main__":
    main()

#bezier curve to move towards the pillar, also take a distance from the pillar to avoid collision.  
#def calculate_trajectory_to_pillar(): 
        
#def see_wall():
    
#This method calculates the nearest distance from the robot to the curve

#this method will update the local reference to the current position
#def  update_reference():#transformation matrix
    
#This method should drive the robot 1 step according to the saved trajectory
#def take_step():
       
       
#def see_pillar():
    
#this is the a cv part,maybe more than one function.
#def process_data(vision):           

#this is to count, sections, laps and corners.
#def add_Section():    
    
#decides if the robot should stop in this section or not, based on the number of sections and laps
#def stop_in_this_section():    
    
  
#def parking():
    
#if we start with the parking lot, we need to drive out

#def exit_parking_lot():
    

#the trick after lap 2
#def special_trick():
    

    

