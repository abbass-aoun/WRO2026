"""
control/servo_calibrate.py — Find the correct servo center angle.

HOW TO RUN:
    python3 -m control.servo_calibrate

Type a number (offset in degrees) and press Enter.
Watch the wheels. When they point straight, that offset is your correction.
Add the offset to center_angle=78 in config.py.

Examples:
    Type  +5  -> servo moves to 83 degrees
    Type  -5  -> servo moves to 73 degrees
    Type   0  -> back to current center (78)
    Type   q  -> quit
"""

from control.servoClass import myServo

CENTER = 78   # current center_angle from config — adjust here if needed
PIN_SERVO = 12

def main():
    s = myServo(PIN_SERVO, center_angle=CENTER, max_deviation=45)
    print(f"Servo calibration. Current center_angle = {CENTER}")
    print("Type an offset (e.g. +5, -3, 0) and press Enter. 'q' to quit.\n")
    s.center_servo()

    try:
        while True:
            raw = input("Offset degrees: ").strip()
            if raw.lower() == 'q':
                break
            try:
                offset = float(raw)
                absolute = CENTER + offset
                s.set_servo_angle(offset)
                print(f"  -> absolute angle = {absolute:.1f}°  (offset={offset:+.1f})")
            except ValueError:
                print("  Enter a number or 'q'.")
    except KeyboardInterrupt:
        pass
    finally:
        s.cleanup()
        print(f"\nIf wheels are straight at offset X, set center_angle = {CENTER} + X in config.py")

if __name__ == "__main__":
    main()
