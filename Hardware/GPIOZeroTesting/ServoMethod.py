import time
from gpiozero import AngularServo
 
 
class ServoController:
    def __init__(self, pin, min_angle=-90, max_angle=90, center_angle=0):
        """
        pin:          GPIO pin the servo signal wire is connected to
        min_angle:    minimum allowed angle (degrees)
        max_angle:    maximum allowed angle (degrees)
        center_angle: reference "centered" angle, used on startup
        """
        if min_angle >= max_angle:
            raise ValueError("min_angle must be less than max_angle")
        if not min_angle <= center_angle <= max_angle:
            raise ValueError("center_angle must be between min_angle and max_angle")
 
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.center_angle = center_angle
 
        self._servo = AngularServo(
            pin,
            min_angle=min_angle,
            max_angle=max_angle,
            initial_angle=center_angle,
            min_pulse_width=0.0005,
            max_pulse_width=0.0025,
        )
 
    def drive(self, angle: float):
        """Move the servo to `angle`, clamped to [min_angle, max_angle]."""
        clamped = max(self.min_angle, min(self.max_angle, angle))
        self._servo.angle = clamped
        return clamped
 
    def center(self):
        """Return the servo to its reference center angle."""
        self.drive(self.center_angle)
 
    def detach(self):
        """Stop sending pulses (lets the servo go limp)."""
        self._servo.detach()
 
 
def main():
    # Change pin/limits/center to match your servo and setup
    servo = ServoController(pin=12, min_angle=-90, max_angle=90, center_angle=0)
 
    try:
        print("Centered")
        time.sleep(1)
 
        print("Driving to 45")
        servo.drive(45)
        time.sleep(1)
 
        print("Driving to -60")
        servo.drive(-60)
        time.sleep(1)
 
        print("Driving to 200 (will clamp to max_angle=90)")
        servo.drive(200)
        time.sleep(1)
 
        print("Back to center")
        servo.center()
        time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        servo.detach()
 
 
if __name__ == "__main__":
    main()
