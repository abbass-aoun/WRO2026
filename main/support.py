import math
import time


def calibrate_gyro(encoders, samples=200):
    """Robot MUST be perfectly still during this."""
    print("Calibrating gyro — do NOT move the robot...")
    total = 0.0
    for _ in range(samples):
        total += encoders.get_yaw_rate()
        time.sleep(0.01)
    bias = total / samples
    print(f"Gyro bias = {math.degrees(bias):+.2f} deg/s\n")
    return bias
