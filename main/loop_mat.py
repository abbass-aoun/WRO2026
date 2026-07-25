import time
from gpiozero import Motor
from rpi_hardware_pwm import HardwarePWM


# ---------------- Drive Motor ----------------
class MotorController:
    def __init__(self, forward_pin, backward_pin, enable_pin):
        self._motor = Motor(forward=forward_pin, backward=backward_pin, enable=enable_pin)

    def drive(self, speed: float, direction: int):
        speed = max(0.0, min(1.0, speed))  # clamp instead of raising errors
        if speed == 0:
            self._motor.stop()
        elif direction == 1:
            self._motor.forward(speed)
        else:
            self._motor.backward(speed)

    def stop(self):
        self._motor.stop()


# ---------------- Steering Servo ----------------
class SteeringController:
    def __init__(self, pwm_channel=0, chip=0, hz=50):
        self._pwm = HardwarePWM(pwm_channel=pwm_channel, hz=hz, chip=chip)
        self._pwm.start(0)

    def set_angle(self, angle):
        angle = max(0, min(180, angle))  # clamp to safe servo range
        duty = 2.5 + (angle / 180.0) * 10.0
        self._pwm.change_duty_cycle(duty)

    def straight(self):
        self.set_angle(STRAIGHT_ANGLE)

    def left(self):
        self.set_angle(LEFT_ANGLE)

    def right(self):
        self.set_angle(RIGHT_ANGLE)

    def stop(self):
        self._pwm.stop()


# ---------------- CALIBRATION (safe defaults, tune later) ----------------
STRAIGHT_ANGLE = 90
LEFT_ANGLE = 60          # conservative, not max-lock, to avoid strain
RIGHT_ANGLE = 120

DRIVE_SPEED = 0.35       # low speed for safety
SIDE_TIME = 3.0          # conservative estimate, tune to real speed
TURN_TIME = 1.0
SETTLE_TIME = 0.3        # pause between moves to reduce jerk


def safe_stop(motor: MotorController, steer: SteeringController):
    motor.stop()
    steer.straight()
    time.sleep(0.2)


def loop_mat(motor: MotorController, steer: SteeringController, laps=1):
    steer.straight()
    time.sleep(SETTLE_TIME)

    for _ in range(laps):
        for side in range(4):
            # straight side
            steer.straight()
            time.sleep(SETTLE_TIME)
            motor.drive(DRIVE_SPEED, 1)
            time.sleep(SIDE_TIME)
            motor.stop()
            time.sleep(SETTLE_TIME)

            # turn corner
            steer.right()
            time.sleep(SETTLE_TIME)
            motor.drive(DRIVE_SPEED, 1)
            time.sleep(TURN_TIME)
            motor.stop()
            steer.straight()
            time.sleep(SETTLE_TIME)


def main():
    motor = MotorController(forward_pin=18, backward_pin=13, enable_pin=19)
    steer = SteeringController(pwm_channel=0, chip=0)

    try:
        loop_mat(motor, steer, laps=1)
    except KeyboardInterrupt:
        print("Interrupted — stopping safely")
    except Exception as e:
        print(f"Error occurred: {e} — stopping safely")
    finally:
        safe_stop(motor, steer)
        steer.stop()


if __name__ == "__main__":
    main()