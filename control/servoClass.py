"""
control/servoClass.py — Hardware servo driver (lgpio PWM).
===========================================================

Ported from last year's VSCode/sensors/servoClass.py.
Drives the steering servo via lgpio hardware PWM on the Raspberry Pi.

WIRING:
    Servo signal wire → GPIO pin (defined in main.py as PIN_SERVO)
    Servo power       → 5 V rail (NOT GPIO 3.3 V)
    Servo ground      → common GND

TUNING:
    center_angle  : PWM angle that makes the wheels point straight. TUNE ON REAL ROBOT.
    max_deviation : maximum steering deflection each side in degrees. TUNE ON REAL ROBOT.
"""

import lgpio
import time


class myServo:
    """
    Steering servo driver.

    Usage:
        servo = myServo(servo_pin=13, center_angle=78, max_deviation=27)
        servo.set_servo_angle(+15)   # steer 15 degrees right
        servo.set_servo_angle(-15)   # steer 15 degrees left
        servo.set_servo_angle(0)     # wheels straight
        servo.cleanup()              # release GPIO when done
    """

    def __init__(self, servo_pin: int,
                 center_angle: int = 78,
                 max_deviation: int = 27):
        """
        Args:
            servo_pin     : GPIO BCM pin number for the servo signal wire.
            center_angle  : PWM duty angle (0-180) that steers straight.  TUNE ON REAL ROBOT.
            max_deviation : max steering deflection each side (degrees).   TUNE ON REAL ROBOT.
        """
        self.chip      = lgpio.gpiochip_open(0)
        self.servo_pin = servo_pin
        self.center    = center_angle
        self.deviation = max_deviation
        self.pwm_freq  = 50   # standard servo PWM: 50 Hz
        lgpio.gpio_claim_output(self.chip, self.servo_pin)

    # ------------------------------------------------------------------

    def _angle_to_pulse_us(self, angle: float) -> int:
        """
        Convert a 0–180 degree angle to a PWM pulse width in microseconds.
        Standard hobby servo: 500 µs = 0°, 2500 µs = 180°.
        """
        angle = max(0.0, min(180.0, angle))
        return int(500 + (angle / 180.0) * 2000)

    def set_servo_angle(self, relative_angle: float) -> None:
        """
        Command the servo to a relative steering angle.

        Args:
            relative_angle : degrees from straight.
                             Positive = one side, negative = other side.
                             Clamped to ±max_deviation.
                             TUNE THE SIGN ON THE REAL ROBOT — if the car
                             steers the wrong way, flip the sign of the
                             angle you pass in from the steering controller.
        """
        absolute_angle = self.center + relative_angle
        absolute_angle = max(self.center - self.deviation,
                             min(self.center + self.deviation, absolute_angle))
        pulse_us = self._angle_to_pulse_us(absolute_angle)
        # Convert pulse width to duty cycle percentage for lgpio
        duty_pct = pulse_us * 1e-6 * self.pwm_freq * 100.0
        lgpio.tx_pwm(self.chip, self.servo_pin, self.pwm_freq, duty_pct)

    def center_servo(self) -> None:
        """Drive wheels straight (relative angle = 0)."""
        self.set_servo_angle(0.0)

    def cleanup(self) -> None:
        """Release the GPIO chip.  Call when the program exits."""
        lgpio.gpiochip_close(self.chip)
