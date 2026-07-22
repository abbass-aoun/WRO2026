import time
import math
from enum import Enum, auto

from trajectory.builder import TrajectoryBuilder
from control.steering_controller import SteeringPIDController
from control.pid_controller import PIDController
from control.car_controller import CarController

from config import (
    SERVO_MAX_DEG,
    PID_KP,
    PID_KI,
    PID_KD,
    PID_HEADING_W,
    PID_WINDUP_LIM,
)

from gpiozero import Button
from gpiozero import LED

import math
import time
import numpy as np

from config import (
    WHEELBASE_CM, DT_S,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2, EKF_R_GYRO_R2,
)
from control.allEncodersClass import RobotEncoders
from control.color_sensor     import ColorSensor
from control.robot            import Robot
from estimation.ekf           import EKF
from main.support             import calibrate_gyro
from control.car_controller import CarController

PIN_MOTOR_IN1 = 18
PIN_MOTOR_IN2 = 13
PIN_MOTOR_ENA = 19
PIN_SERVO = 12
PIN_START_BUTTON = 8 # Start button is at GPIO 8
TEST_DUTY = 0.30
TEST_SPEED = 0.25

straight_pid = PIDController(
    Kp=PID_KP,
    Ki=PID_KI,
    Kd=PID_KD,
    output_limits=(-SERVO_MAX_DEG, SERVO_MAX_DEG),
    windup_limit=PID_WINDUP_LIM
)

steer_path_s = 0.0

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

driving_direction = DrivingDirection.UNKNOWN

# Number of 90-degree corners that have been COMPLETED.
corners_completed = 0

# The robot starts facing along the global +X axis.
INITIAL_THETA = 0.0  # radians

straight_ref_x = None
straight_ref_y = None
target_theta = 0.0

state = State.WAITING
start_button = None # placeholder for the button object
led1 = None
led2 = None
led3 = None
led4 = None


def initialize_start_hardware():
    global start_button, led1, led2, led3, led4

    # Start button initialization
    start_button = Button(
        PIN_START_BUTTON,
        pull_up=False,
        bounce_time=0.05
    )

    # Initialization of LEDs
    led1 = LED(16)
    led2 = LED(20)
    led3 = LED(21)
    led4 = LED(26)
    

    # Safe initial condition
    led1.off()
    led2.off()
    led3.off()
    led4.off()

    time.sleep(0.1)

PIN_ENC_LEFT     = 7
PIN_ENC_RIGHT    = 5
PIN_COLOR_S0     = 17
PIN_COLOR_S1     = 27
PIN_COLOR_S2     = 22
PIN_COLOR_S3     = 23
PIN_COLOR_OUT    = 24
PIN_COLOR_LED    = 25


def wait_for_start():
    global state

    leds = [led1, led2, led3, led4]

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


# ── EKF noise matrices ────────────────────────────────────────────────────────
_Q = np.diag([EKF_Q_XY_CM2, EKF_Q_XY_CM2, EKF_Q_THETA_R2])


def wait_for_start():
    global state

    leds = [led1, led2, led3, led4]

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

#read from sensors and update the EKF

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


def compute_tracking_error(x, y, theta):
    """
    Calculate how far the robot differs from the desired straight path.

    Inputs:
        x, y  : current absolute EKF position in cm
        theta : current absolute EKF heading in radians

    Returns:
        combined_error
        cross_track_error
        heading_error
    """

    if straight_ref_x is None or straight_ref_y is None:
        raise RuntimeError("Straight reference has not been initialized.")

    # Robot displacement from a known point on the desired line
    dx = x - straight_ref_x
    dy = y - straight_ref_y

    # Signed perpendicular distance from the desired straight line.
    # Units: cm
    cross_track_error = (
        -math.sin(target_theta) * dx
        + math.cos(target_theta) * dy
    )

    # Difference between current robot heading and desired heading.
    # Normalized to [-pi, +pi].
    heading_error = math.atan2(
        math.sin(theta - target_theta),
        math.cos(theta - target_theta)
    )

    # Combine position and heading errors.
    combined_error = (
        cross_track_error
        + PID_HEADING_W * heading_error
    )

    return combined_error, cross_track_error, heading_error


def calculate_straight_steering(x, y, theta):

    error, cte, heading_error = compute_tracking_error(
        x,
        y,
        theta
    )

    steering_deg = straight_pid._compute(error)

    return steering_deg


def take_step(car, robot, x, y, theta):
    """
    Move the robot one control-loop step according to
    the current straight reference trajectory.
    """

    # Calculate steering needed to return to / stay on target path
    steering_deg = calculate_straight_steering(
        x,
        y,
        theta
    )

    # Save steering so EKF knows the latest commanded steering angle
    robot.update_steering(steering_deg)

    # Apply steering + forward motor command
    car.set_all(
        direction='f',
        speed=TEST_SPEED,
        angle=steering_deg
    )

    return steering_deg


def main():
    global state, driving_direction

    # Hardware
    initialize_start_hardware()

    car = CarController(
        PIN_MOTOR_IN1,
        PIN_MOTOR_IN2,
        PIN_MOTOR_ENA,
        PIN_SERVO
    )

    encoders = RobotEncoders(
        PIN_ENC_LEFT,
        PIN_ENC_RIGHT
    )

    color = ColorSensor(
        PIN_COLOR_S0,
        PIN_COLOR_S1,
        PIN_COLOR_S2,
        PIN_COLOR_S3,
        PIN_COLOR_OUT,
        PIN_COLOR_LED
    )

    robot = Robot()

    ekf = EKF(
        wheelbase=WHEELBASE_CM,
        Q=_Q,
        R_imu=EKF_R_GYRO_R2
    )

    # Robot must stay still here
    car.stop()
    gyro_bias = calibrate_gyro(encoders)

    # Wait for physical button
    wait_for_start()

    # Start coordinate system at (0, 0), facing +X
    encoders.reset()
    robot.reset()
    ekf.initialize(
        x0=0.0,
        y0=0.0,
        theta0=0.0
    )

    straight_initialized = False

    try:

        while state == State.RUNNING:

            # Sensors → EKF
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
                DT_S,
                gyro_bias
            )

            # Define the target straight only once
            if not straight_initialized:
                initialize_straight_reference(x, y)
                straight_initialized = True

            # Move one step according to target trajectory
            steering = take_step(
                car,
                robot,
                x,
                y,
                theta
            )

            print(
                f"x={x:.1f} "
                f"y={y:.1f} "
                f"theta={math.degrees(theta):+.1f}° "
                f"steer={steering:+.1f}°"
            )

            # Corner trajectory is not implemented yet:
            # safely stop at first coloured line.
            if orange_seen:

                driving_direction = DrivingDirection.CW
                print("Orange → CW. First corner reached.")
                break

            elif blue_seen:

                driving_direction = DrivingDirection.CCW
                print("Blue → CCW. First corner reached.")
                break

            time.sleep(DT_S)

    finally:
        car.stop()
        color.stop()
        state = State.FINISHED
    
    
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
    
    
    

