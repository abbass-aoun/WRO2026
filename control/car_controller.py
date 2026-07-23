from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import time, sleep
from control.servoClass import myServo
from control.allEncodersClass import RobotEncoders
from control.brake_controller import BrakePIDController
from config import MAX_DUTY_SAFE, MOTOR_INVERTED


class CarController:
    def __init__(self, in1_pin, in2_pin, ena_pin, servo_pin):
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)
        self.ena = PWMOutputDevice(ena_pin)
        self.servo = myServo(servo_pin, center_angle=78, max_deviation=27)
        self._last_steering_angle = None
        self._last_motor_command = None
        self._closed = False
        self.stop()   # explicit safe state — pins are defined but motor is off

    def set_motor(self, direction, speed=1.0):
        # Normalized PWM duty, not physical speed in cm/s.
        speed = max(0.0, min(MAX_DUTY_SAFE, float(speed)))
        # Swap direction if motor wires are physically reversed
        if MOTOR_INVERTED and direction in ('f', 'b'):
            direction = 'b' if direction == 'f' else 'f'
        command = (direction, speed)
        if self._last_motor_command == command:
            return
        if direction == 'f':
            self.in1.on()
            self.in2.off()
            self.ena.value = speed
        elif direction == 'b':
            self.in1.off()
            self.in2.on()
            self.ena.value = speed
        else:
            self.stop()
            return
        self._last_motor_command = command

    def set_steering(self, angle):
        angle = float(angle)
        if self._last_steering_angle == angle:
            return
        self.servo.set_servo_angle(angle)
        self._last_steering_angle = angle
     

    def set_all(self, direction, speed, angle):
        self.set_steering(angle)
        self.set_motor(direction, speed)

    def stop(self):
        self.in1.off()
        self.in2.off()
        self.ena.value = 0.0
        self._last_motor_command = None

    def close(self):
        """Stop the car and release all motor/servo GPIO resources."""
        if self._closed:
            return
        self.stop()
        self.servo.cleanup()
        self.ena.close()
        self.in1.close()
        self.in2.close()
        self._closed = True

    def brake(self, encoders, kp=15, ki=0.4, kd=4, tolerance=0.05, log_fn=None):
        pid = BrakePIDController(kp=kp, ki=ki, kd=kd)
        last_time = time()
        count = 0

        while True:
            v_l, v_r = encoders.get_linear_speeds()
            avg_speed = (v_l + v_r) / 2.0

            now = time()
            dt = now - last_time
            last_time = now

            if dt == 0:
                continue

            control = pid.compute(v_l, v_r, dt)
            control = max(0.0, min(1.0, abs(control)))

            if count % 2 == 0:
                self.set_motor("b" if avg_speed > 0 else "f", control)
            else:
                self.set_motor("f" if avg_speed > 0 else "b", 0)

            count += 1

            if log_fn:
                log_fn(now, v_l, v_r)

            if abs(avg_speed) < tolerance:
                break

            sleep(0.01)

        self.stop()
