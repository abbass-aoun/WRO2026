"""
control/sensor_test.py — Level 0 sensor hardware test.
=======================================================

Tests ALL sensors with NO robot movement (motors stay off).

HOW TO RUN (from project root):
    python3 -m control.sensor_test

WHAT TO DO WHILE IT RUNS:
    Encoders  : spin each rear wheel slowly by hand → pulse count rises, speed shows cm/s
    Gyro      : rotate the whole robot by hand      → yaw_rate changes sign and magnitude
    Color     : slide the sensor over orange tape   → orange=True
                slide the sensor over blue tape     → blue=True
                floor/other surfaces                → both False

Press Ctrl+C to stop.

WHAT COUNTS AS PASSING:
    Encoders  : L and R pulse counts increase when you spin each wheel
    Gyro      : yaw_rate is non-zero when rotating; ~0 when still; sign flips with direction
    Color     : orange=True / blue=True trigger correctly on their target surface

If any sensor reads 0 / False at all times despite stimulation, check wiring or pin number.
"""

import math
import time

from control.allEncodersClass import RobotEncoders
from control.color_sensor     import ColorSensor

# ── Pin numbers (must match main/main.py and PinsUpdated.txt) ────────────────
PIN_ENC_LEFT  = 7    # IR encoder left  — GPIO 7  / Pin 26
PIN_ENC_RIGHT = 5    # IR encoder right — GPIO 5  / Pin 29

PIN_COLOR_S0  = 17   # TCS3200 S0 — GPIO 17 / Pin 11
PIN_COLOR_S1  = 27   # TCS3200 S1 — GPIO 27 / Pin 13
PIN_COLOR_S2  = 22   # TCS3200 S2 — GPIO 22 / Pin 15
PIN_COLOR_S3  = 23   # TCS3200 S3 — GPIO 23 / Pin 16
PIN_COLOR_OUT = 24   # TCS3200 OUT — GPIO 24 / Pin 18
PIN_COLOR_LED = 25   # TCS3200 LED — GPIO 25 / Pin 22

PRINT_HZ = 5   # refresh rate (prints per second)


def main() -> None:
    print("Initialising sensors …")

    encoders = RobotEncoders(PIN_ENC_LEFT, PIN_ENC_RIGHT)
    color    = ColorSensor(PIN_COLOR_S0, PIN_COLOR_S1,
                           PIN_COLOR_S2, PIN_COLOR_S3,
                           PIN_COLOR_OUT, PIN_COLOR_LED)

    print("Ready. Stimulate each sensor and watch the values change.")
    print("Ctrl+C to stop.\n")

    print(
        f"{'L_pulses':>9} {'R_pulses':>9} "
        f"{'vL cm/s':>9} {'vR cm/s':>9} "
        f"{'distL cm':>9} {'distR cm':>9}  "
        f"{'yaw °/s':>8}  "
        f"{'R':>6} {'G':>6} {'B':>6}  "
        f"{'orange':>6} {'blue':>6}"
    )
    print("-" * 100)

    try:
        while True:
            # ── Encoders ──────────────────────────────────────────────────────
            v_l, v_r       = encoders.get_linear_speeds()
            dist_l, dist_r = encoders.get_distances()
            l_pulses       = encoders._left_count
            r_pulses       = encoders._right_count

            # ── Gyro ──────────────────────────────────────────────────────────
            omega_rad = encoders.get_yaw_rate()       # rad/s
            omega_deg = math.degrees(omega_rad)       # deg/s (easier to read)

            # ── Colour sensor ─────────────────────────────────────────────────
            r_raw = color._read_channel(*ColorSensor._RED)
            g_raw = color._read_channel(*ColorSensor._GREEN)
            b_raw = color._read_channel(*ColorSensor._BLUE)

            print(
                f"{l_pulses:>9d} {r_pulses:>9d} "
                f"{v_l:>9.1f} {v_r:>9.1f} "
                f"{dist_l:>9.1f} {dist_r:>9.1f}  "
                f"{omega_deg:>+8.1f}  "
                f"{r_raw:>6.0f} {g_raw:>6.0f} {b_raw:>6.0f}  "
                f"{'YES' if color.orange_seen else 'no':>6} "
                f"{'YES' if color.blue_seen   else 'no':>6}"
            )

            time.sleep(1.0 / PRINT_HZ)

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        color.stop()


if __name__ == "__main__":
    main()
