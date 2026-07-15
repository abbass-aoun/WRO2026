"""
Servo calibration table — control/servo_cal.py
===============================================

Maps commanded steering angle (degrees) to servo PWM pulse width (microseconds).

HOW TO CALIBRATE (do this on the real robot):
    1. Place the car on a flat surface with wheels free.
    2. Run:  python -m control.servo_cal  (prints a check table and optional plot)
    3. For each angle in CALIBRATION_TABLE:
       - Command that angle via the servo driver.
       - Measure the actual wheel angle with a protractor.
       - Adjust the PWM value until commanded == measured.
    4. Re-run the script to verify the fit is smooth.

SIGN CONVENTION  (must match steering_controller.py):
    Positive steer_deg = car turns LEFT  (CCW viewed from above)
    Negative steer_deg = car turns RIGHT (CW  viewed from above)
    Verify on real robot: positive command → left turn.

TYPICAL SERVO RANGE:
    Most hobby servos: 1000 µs (one extreme) to 2000 µs (other extreme), neutral 1500 µs.
    The direction (which extreme = left vs right) depends on how the servo is mounted.
    TUNE ON REAL ROBOT.
"""

import math
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Calibration data
# Format: (commanded_deg, pwm_us)
# Negative deg = right turn, positive deg = left turn.
# All PWM values below are PLACEHOLDERS — TUNE ON REAL ROBOT.
# ─────────────────────────────────────────────────────────────────────────────
CALIBRATION_TABLE: list[tuple[float, int]] = [
    (-27.0, 2000),   # hard right  # TUNE ON REAL ROBOT
    (-20.0, 1840),   #             # TUNE ON REAL ROBOT
    (-10.0, 1620),   #             # TUNE ON REAL ROBOT
    ( -5.0, 1560),   #             # TUNE ON REAL ROBOT
    (  0.0, 1500),   # centre      # TUNE ON REAL ROBOT
    (  5.0, 1440),   #             # TUNE ON REAL ROBOT
    ( 10.0, 1380),   #             # TUNE ON REAL ROBOT
    ( 20.0, 1160),   #             # TUNE ON REAL ROBOT
    ( 27.0, 1000),   # hard left   # TUNE ON REAL ROBOT
]

# Build sorted numpy arrays for interpolation
_angles = np.array([r[0] for r in CALIBRATION_TABLE], dtype=float)
_pwms   = np.array([r[1] for r in CALIBRATION_TABLE], dtype=float)

# Derived constants (read-only after import)
STEER_MIN_DEG  = float(_angles[0])    # = -27.0
STEER_MAX_DEG  = float(_angles[-1])   # = +27.0
PWM_MIN_US     = int(_pwms.min())     # = 1000
PWM_MAX_US     = int(_pwms.max())     # = 2000
PWM_NEUTRAL_US = int(round(float(np.interp(0.0, _angles, _pwms))))  # = 1500


def steer_deg_to_pwm(steer_deg: float) -> int:
    """
    Convert a commanded steering angle to a servo PWM value.

    Clamps to the calibration range and linearly interpolates between
    the nearest two table entries.

    Args:
        steer_deg: desired steering angle in degrees.
                   Positive = left, negative = right.

    Returns:
        PWM pulse width in microseconds (integer).
    """
    clamped = float(np.clip(steer_deg, STEER_MIN_DEG, STEER_MAX_DEG))
    return int(round(float(np.interp(clamped, _angles, _pwms))))


def pwm_to_steer_deg(pwm_us: int) -> float:
    """
    Inverse lookup: PWM value → steering angle in degrees.
    Useful for sanity-checking what angle a given PWM actually commands.

    Args:
        pwm_us: servo PWM pulse width in microseconds.

    Returns:
        Equivalent steering angle in degrees.
    """
    # _pwms is monotone decreasing (right→left = high→low PWM).
    # np.interp needs xp increasing, so flip both arrays.
    clamped = float(np.clip(pwm_us, PWM_MIN_US, PWM_MAX_US))
    return float(np.interp(clamped, _pwms[::-1], _angles[::-1]))


# =============================================================================
# Self-test  —  run from project root:  python -m control.servo_cal
# =============================================================================
if __name__ == "__main__":
    print("Servo calibration check")
    print(f"  Neutral PWM : {PWM_NEUTRAL_US} us")
    print(f"  Range       : [{STEER_MIN_DEG}, {STEER_MAX_DEG}] deg  "
          f"-> [{PWM_MIN_US}, {PWM_MAX_US}] us")
    print()
    print(f"  {'Angle (deg)':>12}  {'PWM (us)':>10}  {'Round-trip (deg)':>18}")
    for deg in range(-27, 28, 3):
        pwm = steer_deg_to_pwm(float(deg))
        rt  = pwm_to_steer_deg(pwm)
        print(f"  {deg:>12.1f}  {pwm:>10d}  {rt:>18.2f}")

    # Optional: plot calibration curve if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        degs = np.linspace(STEER_MIN_DEG, STEER_MAX_DEG, 200)
        pwms = [steer_deg_to_pwm(d) for d in degs]
        plt.figure(figsize=(8, 4))
        plt.plot(degs, pwms, 'b-', lw=1.5, label='Interpolated')
        plt.plot(_angles, _pwms, 'ro', ms=7, label='Cal points')
        plt.axhline(PWM_NEUTRAL_US, color='gray', ls='--', lw=0.8, label='Neutral')
        plt.xlabel('Steering command (deg)')
        plt.ylabel('Servo PWM (us)')
        plt.title('Servo calibration curve  (TUNE ON REAL ROBOT)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('servo_cal.png', dpi=120)
        print("\nSaved -> servo_cal.png")
    except ImportError:
        pass
