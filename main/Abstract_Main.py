import time
import math
import numpy as np

from enum import Enum, auto

from trajectory.builder import TrajectoryBuilder

from main.support import calibrate_gyro
from config import DT_S


TEST_DUTY = 0.30
TEST_SPEED = 0.25


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


driving_direction = DrivingDirection.UNKNOWN
state = State.WAITING

corners_completed = 0 # Number of 90-degree corners that have been COMPLETED.
INITIAL_THETA = 0.0  # radians (The robot starts facing along the global +X axis).

straight_ref_x = None
straight_ref_y = None
target_theta = 0.0

#-----------------------------------------------------------
# Initializing PID controller
#-----------------------------------------------------------

from control.pid_controller import PIDController
from config import(
    PID_KP,
    PID_KI,
    PID_KD,
    PID_WINDUP_LIM,
    SERVO_MAX_DEG
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

    steering_deg = straight_pid._compute(
        heading_error
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

def main():
    global state


    # --------------------------------------------------------
    # Initialize everything
    # --------------------------------------------------------

    (
        start_button,
        leds,
        encoders,
        color,
        car,
        robot,
        ekf

    ) = initialize_hardware()


    # --------------------------------------------------------
    # Calibrate gyro while robot is stationary
    # --------------------------------------------------------

    gyro_bias = calibrate_gyro(encoders)

    # --------------------------------------------------------
    # Wait for physical start button
    # --------------------------------------------------------

    wait_for_start(
        start_button,
        leds
    )

    # --------------------------------------------------------
    # Start coordinate system
    # --------------------------------------------------------

    ekf.initialize(
        x0=0.0,
        y0=0.0,
        theta0=0.0
    )

    encoders.reset()

    last_time = time.monotonic()

    while state == State.RUNNING:
        
        now = time.monotonic()
        dt = now - last_time
        last_time = now

        speed, v_l, v_r, omega, x, y, theta, orange_seen, blue_seen = read_sensors_and_update_ekf(
            encoders,
            color,
            ekf,
            robot,
            dt,
            gyro_bias
        )

        steering = take_step(car, robot, theta)
        print(
            f"theta={math.degrees(theta):+6.2f}° | "
            f"target={math.degrees(target_theta):+6.2f}° | "
            f"steering={steering:+6.2f}°"
        )
    
    

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
    
if __name__ == "__main__":
    main()
    

