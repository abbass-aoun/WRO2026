#!/usr/bin/env python3
"""
Don't forget to install:

pip install adafruit-circuitpython-vl53l0x adafruit-blinka mpu6050-raspberrypi RPi.GPIO --break-system-packages


Optimized Combined Sensor Script - Raspberry Pi 5
- MPU6050: Accelerometer + Gyroscope (I2C, 0x68)
- VL53L0X: Time-of-Flight Distance (I2C, 0x29)
- TCS3200: Color Sensor (GPIO, pulse-counting method)

Optimizations:
- TCS3200 uses interrupt-based pulse counting (non-blocking, faster, more stable)
  instead of GPIO.wait_for_edge per channel
- Single I2C bus instance shared cleanly
- Graceful degradation if a sensor is missing (others keep running)
- CSV logging with timestamps
- Configurable sample rate
"""

import time
import csv
import signal
import sys
import board
import busio
import RPi.GPIO as GPIO
from mpu6050 import mpu6050
import adafruit_vl53l0x

# ---------------- Configuration ----------------
SAMPLE_INTERVAL = 0.5          # seconds between readings
COLOR_SAMPLE_TIME = 0.05       # window (s) to count pulses per channel
LOG_TO_CSV = True
CSV_FILENAME = "sensor_log.csv"

# TCS3200 pins
S0, S1, S2, S3, OUT = 17, 27, 22, 23, 24

# ---------------- TCS3200 (pulse-counting) ----------------
class TCS3200:
    def __init__(self, s0, s1, s2, s3, out_pin, sample_time=0.05):
        self.s2, self.s3, self.out_pin = s2, s3, out_pin
        self.sample_time = sample_time
        self._count = 0

        GPIO.setup(s0, GPIO.OUT)
        GPIO.setup(s1, GPIO.OUT)
        GPIO.setup(s2, GPIO.OUT)
        GPIO.setup(s3, GPIO.OUT)
        GPIO.setup(out_pin, GPIO.IN)

        # 20% frequency scaling — good speed/accuracy tradeoff
        GPIO.output(s0, GPIO.HIGH)
        GPIO.output(s1, GPIO.LOW)

    def _pulse_counter(self, channel):
        self._count += 1

    def _read_channel(self, s2_state, s3_state):
        GPIO.output(self.s2, s2_state)
        GPIO.output(self.s3, s3_state)
        time.sleep(0.002)  # let filter settle after mux switch

        self._count = 0
        GPIO.add_event_detect(self.out_pin, GPIO.FALLING, callback=self._pulse_counter)
        time.sleep(self.sample_time)
        GPIO.remove_event_detect(self.out_pin)

        return self._count / self.sample_time  # pulses/sec = Hz

    def read_rgb(self):
        r = self._read_channel(GPIO.LOW, GPIO.LOW)
        g = self._read_channel(GPIO.HIGH, GPIO.HIGH)
        b = self._read_channel(GPIO.LOW, GPIO.HIGH)
        return r, g, b


# ---------------- Sensor Manager ----------------
class SensorSuite:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        self.i2c = busio.I2C(board.SCL, board.SDA)

        self.imu = self._safe_init("MPU6050", lambda: mpu6050(0x68))
        self.tof = self._safe_init("VL53L0X", self._init_tof)
        self.color = self._safe_init(
            "TCS3200", lambda: TCS3200(S0, S1, S2, S3, OUT, COLOR_SAMPLE_TIME)
        )

    def _init_tof(self):
        sensor = adafruit_vl53l0x.VL53L0X(self.i2c)
        sensor.measurement_timing_budget = 200000
        return sensor

    def _safe_init(self, name, init_fn):
        try:
            obj = init_fn()
            print(f"[OK]   {name} initialized")
            return obj
        except Exception as e:
            print(f"[FAIL] {name} init failed: {e}")
            return None

    def read_all(self):
        data = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

        if self.imu:
            try:
                accel = self.imu.get_accel_data()
                gyro = self.imu.get_gyro_data()
                data.update({
                    "accel_x": round(accel["x"], 3), "accel_y": round(accel["y"], 3), "accel_z": round(accel["z"], 3),
                    "gyro_x": round(gyro["x"], 3), "gyro_y": round(gyro["y"], 3), "gyro_z": round(gyro["z"], 3),
                })
            except Exception as e:
                print(f"IMU read error: {e}")

        if self.tof:
            try:
                data["distance_mm"] = self.tof.range
            except Exception as e:
                print(f"ToF read error: {e}")

        if self.color:
            try:
                r, g, b = self.color.read_rgb()
                data.update({"color_r": round(r, 1), "color_g": round(g, 1), "color_b": round(b, 1)})
            except Exception as e:
                print(f"Color read error: {e}")

        return data

    def cleanup(self):
        GPIO.cleanup()


# ---------------- Main ----------------
def main():
    suite = SensorSuite()

    csv_writer = None
    csv_file = None
    if LOG_TO_CSV:
        csv_file = open(CSV_FILENAME, "w", newline="")
        fieldnames = ["timestamp", "accel_x", "accel_y", "accel_z",
                      "gyro_x", "gyro_y", "gyro_z", "distance_mm",
                      "color_r", "color_g", "color_b"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        print(f"Logging to {CSV_FILENAME}")

    def handle_exit(sig, frame):
        print("\nShutting down...")
        if csv_file:
            csv_file.close()
        suite.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print("\nStarting sensor loop. Press Ctrl+C to stop.\n")

    while True:
        start = time.time()
        data = suite.read_all()

        print("-" * 60)
        if "accel_x" in data:
            print(f"Accel (g):    X={data['accel_x']:.2f}  Y={data['accel_y']:.2f}  Z={data['accel_z']:.2f}")
            print(f"Gyro (deg/s): X={data['gyro_x']:.2f}  Y={data['gyro_y']:.2f}  Z={data['gyro_z']:.2f}")
        if "distance_mm" in data:
            print(f"Distance:     {data['distance_mm']} mm ({data['distance_mm']/10:.1f} cm)")
        if "color_r" in data:
            print(f"Color (Hz):   R={data['color_r']:.0f}  G={data['color_g']:.0f}  B={data['color_b']:.0f}")

        if csv_writer:
            csv_writer.writerow(data)
            csv_file.flush()

        elapsed = time.time() - start
        time.sleep(max(0, SAMPLE_INTERVAL - elapsed))


if __name__ == "__main__":
    main()
