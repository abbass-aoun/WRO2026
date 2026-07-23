import numpy as np

from control.allEncodersClass import RobotEncoders
from control.color_sensor import ColorSensor
from control.car_controller import CarController
from control.robot import Robot
from estimation.ekf import EKF
from gpiozero import Button, LED
from config import *

from config import (
    WHEELBASE_CM,
    EKF_Q_XY_CM2,
    EKF_Q_THETA_R2,
    EKF_R_GYRO_R2,
)

_hardware = {
    "start_button": None,
    "leds": [],
    "encoders": None,
    "color": None,
    "car": None,
}


def cleanup_hardware():
    """Stop actuators/threads and release every initialized GPIO resource."""
    car = _hardware["car"]
    if car is not None:
        car.close()
        _hardware["car"] = None

    color = _hardware["color"]
    if color is not None:
        color.stop()
        _hardware["color"] = None

    encoders = _hardware["encoders"]
    if encoders is not None:
        encoders.close()
        _hardware["encoders"] = None

    for led in _hardware["leds"]:
        led.off()
        led.close()
    _hardware["leds"] = []

    start_button = _hardware["start_button"]
    if start_button is not None:
        start_button.close()
        _hardware["start_button"] = None


# ============================================================
# GPIO PINS
# ============================================================

# Start button
 
PIN_START_BUTTON = 8

# LEDs
PIN_LED_1 = 16
PIN_LED_2 = 20
PIN_LED_3 = 21
PIN_LED_4 = 26

# Encoders
PIN_ENC_LEFT = 7
PIN_ENC_RIGHT = 5

# TCS3200 color sensor
PIN_COLOR_S0 = 17
PIN_COLOR_S1 = 27
PIN_COLOR_S2 = 22
PIN_COLOR_S3 = 23
PIN_COLOR_OUT = 24
PIN_COLOR_LED = 25

# Motor + servo
PIN_MOTOR_IN1 = 18
PIN_MOTOR_IN2 = 13
PIN_MOTOR_ENA = 19
PIN_SERVO = 12


def initialize_hardware():
    """
    Initialize all hardware currently needed by Abstract_Main.

    Returns:
        start_button
        leds
        encoders
        color
        car
        robot
        ekf
    """

    cleanup_hardware()

    # --------------------------------------------------------
    # Start button
    # --------------------------------------------------------

    start_button = Button(
        PIN_START_BUTTON,
        pull_up=False,
        bounce_time=0.05
    )
    _hardware["start_button"] = start_button


    # --------------------------------------------------------
    # LEDs
    # --------------------------------------------------------

    led1 = LED(PIN_LED_1)
    led2 = LED(PIN_LED_2)
    led3 = LED(PIN_LED_3)
    led4 = LED(PIN_LED_4)

    leds = [led1, led2, led3, led4]
    _hardware["leds"] = leds

    for led in leds:
        led.off()

    
    # --------------------------------------------------------
    # Encoders + IMU
    # --------------------------------------------------------

    encoders = RobotEncoders(
        PIN_ENC_LEFT,
        PIN_ENC_RIGHT
    )
    _hardware["encoders"] = encoders


    # --------------------------------------------------------
    # Floor color sensor
    # --------------------------------------------------------

    color = ColorSensor(
        PIN_COLOR_S0,
        PIN_COLOR_S1,
        PIN_COLOR_S2,
        PIN_COLOR_S3,
        PIN_COLOR_OUT,
        PIN_COLOR_LED
    )
    _hardware["color"] = color


    # --------------------------------------------------------
    # Motor + steering servo
    # --------------------------------------------------------

    car = CarController(
        PIN_MOTOR_IN1,
        PIN_MOTOR_IN2,
        PIN_MOTOR_ENA,
        PIN_SERVO
    )
    _hardware["car"] = car

    # Explicit safe state
    car.stop()


    # --------------------------------------------------------
    # Shared robot state
    # --------------------------------------------------------

    robot = Robot()


    # --------------------------------------------------------
    # EKF
    # --------------------------------------------------------

    Q = np.diag([
        EKF_Q_XY_CM2,
        EKF_Q_XY_CM2,
        EKF_Q_THETA_R2
    ])

    ekf = EKF(
        wheelbase=WHEELBASE_CM,
        Q=Q,
        R_imu=EKF_R_GYRO_R2
    )


    return (
        start_button,
        leds,
        encoders,
        color,
        car,
        robot,
        ekf
    )
