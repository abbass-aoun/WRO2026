"""
control/servo_test.py — Servo hardware test.

HOW TO RUN (from project root on the Raspberry Pi):
    python3 -m control.servo_test
"""

import time
from control.servoClass import myServo

PIN_SERVO = 32

def main():
    s = myServo(PIN_SERVO)
    print("Servo test — watch the wheels.")

    try:
        print("CENTER (0°)"); s.center_servo();       time.sleep(2)
        print("LEFT  (+27°)"); s.set_servo_angle(27); time.sleep(2)
        print("CENTER (0°)"); s.center_servo();       time.sleep(2)
        print("RIGHT (-27°)"); s.set_servo_angle(-27); time.sleep(2)
        print("CENTER (0°)"); s.center_servo();       time.sleep(1)
        print("Done.")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        s.cleanup()

if __name__ == "__main__":
    main()
