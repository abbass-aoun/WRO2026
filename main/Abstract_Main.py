import time

from enum import Enum, auto

from gpiozero import Button
from gpiozero import LED

import math
import time
import numpy as np
from gpiozero import Button

from config import (
    WHEELBASE_CM, DT_S,
    EKF_Q_XY_CM2, EKF_Q_THETA_R2, EKF_R_GYRO_R2,
)
from control.allEncodersClass import RobotEncoders
from control.color_sensor     import ColorSensor
from control.robot            import Robot
from estimation.ekf           import EKF
from main.support             import calibrate_gyro


PIN_START_BUTTON = 8 # Start button is at GPIO 8

#-----------------------------------------------------------
# States of the robot
#-----------------------------------------------------------
class State(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()
#-----------------------------------------------------------

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




    
#bezier curve to move towards the pillar, also take a distance from the pillar to avoid collision.  
#def calculate_trajectory_to_pillar(): 
    
#bezier curve to move , but without a pillar 
#def calculate_trajectory_no_pillar():       
    
#bezier curve to move towards the corner, but without a pillar    
#def calculate_trajectory_to_corner(): 
    
    
#def see_wall():
    
#This method calculates the nearest distance from the robot to the curve
#def compute_tracking_error():#used in get tarjectory methods
    
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
    
    
    
def main():
    initialize_start_hardware()
    
    wait_for_start()
    print(state)
    
if __name__ == "__main__":
    main()
    
    
    
    
    
    

