"""
control/servoClass.py — Hardware servo driver (gpiozero PWM).
=============================================================

Drives the steering servo via gpiozero PWMOutputDevice.
gpiozero automatically selects the correct GPIO chip on all
Raspberry Pi models including RPi 5 (gpiochip4 / RP1).

WIRING:
    Servo signal wire → GPIO pin (defined in main.py as PIN_SERVO)
    Servo power       → 5 V rail (NOT GPIO 3.3 V)
    Servo ground      → common GND

TUNING:
    center_angle  : angle (0-180) that makes wheels point straight. TUNE ON REAL ROBOT.
    max_deviation : maximum steering deflection each side in degrees. TUNE ON REAL ROBOT.
"""

from gpiozero import PWMOutputDevice


class myServo:
    """
    Steering servo driver using gpiozero PWM.

    Usage:
        servo = myServo(servo_pin=12, center_angle=78, max_deviation=27)
        servo.set_servo_angle(+15)   # steer 15 degrees one way
        servo.set_servo_angle(-15)   # steer 15 degrees other way
        servo.set_servo_angle(0)     # wheels straight
        servo.cleanup()              # release GPIO when done
    """

    def __init__(self, servo_pin: int,
                 center_angle: int = 80,
                 max_deviation: int = 27):
        """
        Args:
            servo_pin     : GPIO BCM pin number for the servo signal wire.
            center_angle  : angle (0-180) that steers straight. TUNE ON REAL ROBOT.
            max_deviation : max steering deflection each side (degrees). TUNE ON REAL ROBOT.
        """
        self._pwm      = PWMOutputDevice(servo_pin, frequency=50)
        self.center    = center_angle
        self.deviation = max_deviation

    def _angle_to_duty(self, angle: float) -> float:
        """Convert 0-180 degree angle to gpiozero duty cycle (0.0-1.0)."""
        angle = max(0.0, min(180.0, angle))
        pulse_us = 500.0 + (angle / 180.0) * 2000.0   # 500-2500 µs
        return pulse_us * 1e-6 * 50.0                  # duty = pulse * freq

    def set_servo_angle(self, relative_angle: float) -> None:
        """
        Command the servo to a relative steering angle.

        Args:
            relative_angle : degrees from straight, clamped to ±max_deviation.
                             TUNE THE SIGN ON THE REAL ROBOT — if the car
                             steers the wrong way, flip the sign passed in.
        """
        absolute_angle = self.center + relative_angle
        absolute_angle = max(float(self.center - self.deviation),
                             min(float(self.center + self.deviation), absolute_angle))
        self._pwm.value = self._angle_to_duty(absolute_angle)

    def center_servo(self) -> None:
        """Drive wheels straight (relative angle = 0)."""
        self.set_servo_angle(0.0)

    def cleanup(self) -> None:
        """Release the GPIO pin."""
        self._pwm.close()
