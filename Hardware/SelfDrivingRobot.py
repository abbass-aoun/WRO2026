#!/usr/bin/env python3
"""
Self-Driving Robot - Raspberry Pi 5
=====================================
Sensors:
  - VL53L0X   : Time-of-Flight distance (I2C, 0x29)   -> obstacle avoidance
  - MPU6050   : Accel + Gyro (I2C, 0x68)               -> tilt / collision safety cutoff
  - TCS3200   : Color sensor (GPIO)                    -> stop-line / marker detection
  - Push button (GPIO)                                 -> toggles robot ON/OFF

Actuators (ASSUMED - adjust to your hardware):
  - L298N H-bridge driving main drive motor(s)
  - Steering servo (PWM) for direction

If your motor driver is different (TB6612FNG, ESCs, etc.), only the
MotorController class needs to change - everything else stays the same.

Wiring summary (BCM numbering):
  I2C (shared)      : SDA=GPIO2, SCL=GPIO3
  TCS3200           : S0=17 S1=27 S2=22 S3=23 OUT=24
  Button            : GPIO25 (button to GND, internal pull-up)
  Status LED (opt)  : GPIO26
  L298N             : IN1=5 IN2=6 ENA(PWM)=12 | IN3=19 IN4=16 ENB(PWM)=13
  Steering servo    : GPIO20 (PWM)
"""

import time
import signal
import sys
import board
import busio
import RPi.GPIO as GPIO
from mpu6050 import mpu6050
import adafruit_vl53l0x

# ================= CONFIG =================
# --- Button / power ---
BUTTON_PIN = 25
POWER_LED_PIN = 26
DEBOUNCE_TIME = 0.3

# --- TCS3200 color sensor ---
S0, S1, S2, S3, OUT = 17, 27, 22, 23, 24
COLOR_SAMPLE_TIME = 0.03

# --- L298N motor driver ---
IN1, IN2, ENA = 5, 6, 12     # Left / main drive motor
IN3, IN4, ENB = 19, 16, 13   # Right motor (omit if single-motor chassis)
PWM_FREQ = 1000

# --- Steering servo ---
SERVO_PIN = 20
SERVO_FREQ = 50
SERVO_CENTER = 7.5   # duty cycle % for straight
SERVO_LEFT = 5.0
SERVO_RIGHT = 10.0

# --- Navigation thresholds ---
OBSTACLE_STOP_MM = 150       # stop/turn if closer than this
OBSTACLE_SLOW_MM = 400       # slow down if closer than this
DRIVE_SPEED = 60             # 0-100 % normal cruising speed
TURN_SPEED = 45
IMPACT_ACCEL_THRESHOLD = 2.0   # g's of sudden change -> treat as collision
STOP_COLOR_R_THRESHOLD = 300   # tune to your red marker/line under your lighting

LOOP_INTERVAL = 0.1  # main loop period (s)

# ================= STATE =================
robot_on = False
last_press_time = 0


# ================= TCS3200 =================
class TCS3200:
    def __init__(self, s0, s1, s2, s3, out_pin, sample_time=0.03):
        self.s2, self.s3, self.out_pin = s2, s3, out_pin
        self.sample_time = sample_time
        self._count = 0

        GPIO.setup(s0, GPIO.OUT)
        GPIO.setup(s1, GPIO.OUT)
        GPIO.setup(s2, GPIO.OUT)
        GPIO.setup(s3, GPIO.OUT)
        GPIO.setup(out_pin, GPIO.IN)

        GPIO.output(s0, GPIO.HIGH)  # 20% frequency scaling
        GPIO.output(s1, GPIO.LOW)

    def _pulse_counter(self, channel):
        self._count += 1

    def _read_channel(self, s2_state, s3_state):
        GPIO.output(self.s2, s2_state)
        GPIO.output(self.s3, s3_state)
        time.sleep(0.002)

        self._count = 0
        GPIO.add_event_detect(self.out_pin, GPIO.FALLING, callback=self._pulse_counter)
        time.sleep(self.sample_time)
        GPIO.remove_event_detect(self.out_pin)

        return self._count / self.sample_time

    def read_rgb(self):
        r = self._read_channel(GPIO.LOW, GPIO.LOW)
        g = self._read_channel(GPIO.HIGH, GPIO.HIGH)
        b = self._read_channel(GPIO.LOW, GPIO.HIGH)
        return r, g, b


# ================= Motor + Servo =================
class MotorController:
    """L298N H-bridge control. Adjust here if using a different driver."""

    def __init__(self):
        for pin in (IN1, IN2, ENA, IN3, IN4, ENB):
            GPIO.setup(pin, GPIO.OUT)

        self.pwm_a = GPIO.PWM(ENA, PWM_FREQ)
        self.pwm_b = GPIO.PWM(ENB, PWM_FREQ)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

    def _set_left(self, speed):  # speed: -100..100
        GPIO.output(IN1, GPIO.HIGH if speed >= 0 else GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW if speed >= 0 else GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(min(abs(speed), 100))

    def _set_right(self, speed):
        GPIO.output(IN3, GPIO.HIGH if speed >= 0 else GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW if speed >= 0 else GPIO.HIGH)
        self.pwm_b.ChangeDutyCycle(min(abs(speed), 100))

    def forward(self, speed=DRIVE_SPEED):
        self._set_left(speed)
        self._set_right(speed)

    def backward(self, speed=DRIVE_SPEED):
        self._set_left(-speed)
        self._set_right(-speed)

    def turn_left(self, speed=TURN_SPEED):
        self._set_left(-speed * 0.3)
        self._set_right(speed)

    def turn_right(self, speed=TURN_SPEED):
        self._set_left(speed)
        self._set_right(-speed * 0.3)

    def stop(self):
        self._set_left(0)
        self._set_right(0)

    def cleanup(self):
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()


class SteeringServo:
    def __init__(self, pin=SERVO_PIN):
        GPIO.setup(pin, GPIO.OUT)
        self.pwm = GPIO.PWM(pin, SERVO_FREQ)
        self.pwm.start(SERVO_CENTER)

    def center(self):
        self.pwm.ChangeDutyCycle(SERVO_CENTER)

    def left(self):
        self.pwm.ChangeDutyCycle(SERVO_LEFT)

    def right(self):
        self.pwm.ChangeDutyCycle(SERVO_RIGHT)

    def cleanup(self):
        self.pwm.stop()


# ================= Sensor Suite =================
class SensorSuite:
    def __init__(self):
        self.i2c = busio.I2C(board.SCL, board.SDA)
        self.imu = self._safe_init("MPU6050", lambda: mpu6050(0x68))
        self.tof = self._safe_init("VL53L0X", self._init_tof)
        self.color = self._safe_init(
            "TCS3200", lambda: TCS3200(S0, S1, S2, S3, OUT, COLOR_SAMPLE_TIME)
        )
        self._last_accel = None

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

    def get_distance_mm(self):
        if not self.tof:
            return None
        try:
            return self.tof.range
        except Exception as e:
            print(f"ToF read error: {e}")
            return None

    def detect_impact(self):
        """Returns True if sudden acceleration change suggests a collision/tip."""
        if not self.imu:
            return False
        try:
            accel = self.imu.get_accel_data()
            vec = (accel["x"], accel["y"], accel["z"])
            if self._last_accel is not None:
                delta = sum(abs(a - b) for a, b in zip(vec, self._last_accel))
                self._last_accel = vec
                return delta > IMPACT_ACCEL_THRESHOLD
            self._last_accel = vec
            return False
        except Exception as e:
            print(f"IMU read error: {e}")
            return False

    def detect_stop_color(self):
        """Returns True if a strong red marker/line is seen (tune threshold)."""
        if not self.color:
            return False
        try:
            r, g, b = self.color.read_rgb()
            return r > STOP_COLOR_R_THRESHOLD and r > g and r > b
        except Exception as e:
            print(f"Color read error: {e}")
            return False


# ================= Button Handling =================
def button_callback(channel):
    global robot_on, last_press_time
    now = time.time()
    if now - last_press_time < DEBOUNCE_TIME:
        return
    last_press_time = now
    robot_on = not robot_on
    GPIO.output(POWER_LED_PIN, GPIO.HIGH if robot_on else GPIO.LOW)
    print("ROBOT ON" if robot_on else "ROBOT OFF")


# ================= Main =================
def main():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)

    GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(POWER_LED_PIN, GPIO.OUT)
    GPIO.output(POWER_LED_PIN, GPIO.LOW)
    GPIO.add_event_detect(BUTTON_PIN, GPIO.FALLING, callback=button_callback, bouncetime=300)

    sensors = SensorSuite()
    motors = MotorController()
    servo = SteeringServo()

    def handle_exit(sig, frame):
        print("\nShutting down...")
        motors.cleanup()
        servo.cleanup()
        GPIO.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_exit)

    print("\nReady. Press the button to start driving. Ctrl+C to quit.\n")

    was_on = False
    while True:
        loop_start = time.time()

        if not robot_on:
            if was_on:
                motors.stop()
                servo.center()
                was_on = False
            time.sleep(LOOP_INTERVAL)
            continue
        was_on = True

        # --- Safety check first: collision / tip-over ---
        if sensors.detect_impact():
            print("Impact/tilt detected -> emergency stop")
            motors.stop()
            servo.center()
            time.sleep(0.5)
            continue

        # --- Stop-line / marker check ---
        if sensors.detect_stop_color():
            print("Stop marker detected -> halting")
            motors.stop()
            servo.center()
            time.sleep(LOOP_INTERVAL)
            continue

        # --- Obstacle avoidance via ToF ---
        distance = sensors.get_distance_mm()
        if distance is not None:
            if distance < OBSTACLE_STOP_MM:
                print(f"Obstacle at {distance}mm -> stop & turn")
                motors.stop()
                servo.left()
                motors.turn_left()
                time.sleep(0.4)
                servo.center()
            elif distance < OBSTACLE_SLOW_MM:
                servo.center()
                motors.forward(speed=TURN_SPEED)
            else:
                servo.center()
                motors.forward(speed=DRIVE_SPEED)
        else:
            # No ToF reading available - drive cautiously
            servo.center()
            motors.forward(speed=TURN_SPEED)

        elapsed = time.time() - loop_start
        time.sleep(max(0, LOOP_INTERVAL - elapsed))


if __name__ == "__main__":
    main()
