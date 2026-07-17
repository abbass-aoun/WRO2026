"""
control/motor_test.py — Standalone motor bring-up test.

Run from the project root:
    python3 -m control.motor_test

What it does:
    Drives the motor forward at TEST_DUTY for TEST_SEC seconds, then backward
    for the same duration, then stops.  Run with the robot lifted off the
    ground to confirm which direction is physically forward.

If forward command spins the wheels backward:
    Open config.py and set  MOTOR_INVERTED = True  then re-run.
"""

import time
from control.car_controller import CarController

# ── Pin numbers (must match main/main.py) ─────────────────────────────────
PIN_MOTOR_IN1 = 18
PIN_MOTOR_IN2 = 13
PIN_MOTOR_ENA = 19
PIN_SERVO     = 12

# ── Test parameters ───────────────────────────────────────────────────────
TEST_DUTY = 0.30   # 30 % PWM — safe bench speed; raise only after confirming direction
TEST_SEC  = 2.0    # seconds per direction


def run_test():
    print("Motor test — robot should be OFF THE GROUND.")
    print(f"  duty={TEST_DUTY:.0%}  duration={TEST_SEC}s each direction\n")

    car = CarController(PIN_MOTOR_IN1, PIN_MOTOR_IN2, PIN_MOTOR_ENA, PIN_SERVO)

    try:
        print("FORWARD …")
        car.set_motor('f', TEST_DUTY)
        time.sleep(TEST_SEC)

        print("STOP (0.5 s)")
        car.stop()
        time.sleep(0.5)

        print("BACKWARD …")
        car.set_motor('b', TEST_DUTY)
        time.sleep(TEST_SEC)

    finally:
        car.stop()
        print("\nStopped.  Check config.py → MOTOR_INVERTED if directions were wrong.")


if __name__ == "__main__":
    run_test()
