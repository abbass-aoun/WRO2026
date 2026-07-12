import time
from gpiozero import Motor
 
 
class MotorController:
    def __init__(self, forward_pin, backward_pin):
        self._motor = Motor(forward=forward_pin, backward=backward_pin)
 
    def drive(self, speed: float, direction: int):
        """
        speed:     0.0 - 1.0
        direction: 1 = forward, 0 = backward
        """
        if not 0.0 <= speed <= 1.0:
            raise ValueError("speed must be between 0.0 and 1.0")
        if direction not in (0, 1):
            raise ValueError("direction must be 0 (backward) or 1 (forward)")
 
        if speed == 0:
            self._motor.stop()
        elif direction == 1:
            self._motor.forward(speed)
        else:
            self._motor.backward(speed)
 
    def stop(self):
        self._motor.stop()
 
 
def main():
    # Change these to match your wiring
    motor = MotorController(forward_pin=17, backward_pin=18)
 
    try:
        print("Forward at 50% speed")
        motor.drive(0.5, 1)
        time.sleep(2)
 
        print("Backward at 100% speed")
        motor.drive(1.0, 0)
        time.sleep(2)
 
        print("Stop")
        motor.stop()
    except KeyboardInterrupt:
        motor.stop()
 
 
if __name__ == "__main__":
    main()
 
