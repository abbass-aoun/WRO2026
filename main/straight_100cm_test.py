"""
main/motor_ground_test.py

Simple motor-on-ground test.

- No EKF
- No encoders
- No steering commands
- Runs the motor for a fixed duration
- Always stops the motor on exit

Run from the repository root:

    python3 -m main.motor_ground_test
"""

from __future__ import annotations

import time

from config import MAX_DUTY_SAFE
from control.car_controller import CarController
from main.initialize_hardware import (
    PIN_MOTOR_IN1,
    PIN_MOTOR_IN2,
    PIN_MOTOR_ENA,
    PIN_SERVO,
)


# ============================================================
# Test settings
# ============================================================

MOTOR_DIRECTION = "f"   # "f" for forward, "b" for backward
MOTOR_DUTY = 0.5   # PWM duty from 0.0 to 1.0
RUN_TIME_S = 3.0        # How long the motor remains on


def create_car() -> CarController:
    """Initialize the existing motor controller."""

    return CarController(
        PIN_MOTOR_IN1,
        PIN_MOTOR_IN2,
        PIN_MOTOR_ENA,
        PIN_SERVO,
    )


def run_motor_test(car: CarController) -> None:
    """Run the motor at constant duty for the configured time."""

    applied_duty = min(
        MOTOR_DUTY,
        MAX_DUTY_SAFE,
    )

    print("\nPlace the robot on the ground.")
    print("Make sure there is clear space in front of it.")
    print(
        f"Direction: {MOTOR_DIRECTION} | "
        f"Duty: {applied_duty:.2f} | "
        f"Duration: {RUN_TIME_S:.1f} s"
    )

    input("Press Enter to start...")

    print("Starting in:")

    for count in (3, 2, 1):
        print(count)
        time.sleep(1.0)

    print("Motor ON")

    car.set_motor(
        direction=MOTOR_DIRECTION,
        speed=applied_duty,
    )

    time.sleep(RUN_TIME_S)

    car.stop()

    print("Motor OFF")
    print("Test completed.")


def main() -> None:
    car = create_car()

    try:
        run_motor_test(car)

    except KeyboardInterrupt:
        print("\nTest interrupted.")

    finally:
        car.stop()
        print("Motor safely stopped.")


if __name__ == "__main__":
    main()