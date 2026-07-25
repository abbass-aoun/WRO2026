"""
control/servoClass.py — Hardware servo driver (rpi_hardware_pwm).
=================================================================

WIRING:
    Servo signal wire → GPIO 12 (hardware PWM channel 0)
    Servo power       → 5 V rail (NOT GPIO 3.3 V)
    Servo ground      → common GND

TUNING:
    center_angle  : angle (0-180) that makes wheels point straight. TUNE ON REAL ROBOT.
    max_deviation : maximum steering deflection each side in degrees. TUNE ON REAL ROBOT.
"""

from rpi_hardware_pwm import HardwarePWM


class myServo:
    """
    Steering servo driver using hardware PWM.

    Usage:
        servo = myServo(pwm_channel=0, center_angle=80, max_deviation=27)
        servo.set_servo_angle(+15)   # steer 15 degrees one way
        servo.set_servo_angle(-15)   # steer 15 degrees other way
        servo.set_servo_angle(0)     # wheels straight
        servo.cleanup()              # release PWM when done
    """

    def __init__(self, pwm_channel: int = 0,
                 center_angle: int = 80,
                 max_deviation: int = 27):
        """
        Args:
            pwm_channel   : hardware PWM channel (0 = GPIO 12, 1 = GPIO 13).
            center_angle  : angle (0-180) that steers straight. TUNE ON REAL ROBOT.
            max_deviation : max steering deflection each side (degrees). TUNE ON REAL ROBOT.
        """
        self._pwm      = HardwarePWM(pwm_channel=pwm_channel, hz=50, chip=0)
        self._pwm.start(0)
        self.center    = center_angle
        self.deviation = max_deviation

    def _angle_to_duty(self, angle: float) -> float:
        """Convert 0-180 degree angle to duty cycle % (2.5–12.5)."""
        angle = max(0.0, min(180.0, angle))
        return 2.5 + (angle / 180.0) * 10.0

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
        self._pwm.change_duty_cycle(self._angle_to_duty(absolute_angle))

    def center_servo(self) -> None:
        """Drive wheels straight (relative angle = 0)."""
        self.set_servo_angle(0.0)

    def cleanup(self) -> None:
        """Release the PWM channel."""
        self._pwm.stop()
