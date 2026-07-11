class SteeringServo:
    def __init__(self, pin, freq=50, center=7.5, left=5.0, right=10.0):
        self.center_duty = center
        self.left_duty = left
        self.right_duty = right
 
        GPIO.setup(pin, GPIO.OUT)
        self.pwm = GPIO.PWM(pin, freq)
        self.pwm.start(center)
        print("[OK]   SteeringServo initialized")
 
    def center(self):
        self.pwm.ChangeDutyCycle(self.center_duty)
 
    def left(self):
        self.pwm.ChangeDutyCycle(self.left_duty)
 
    def right(self):
        self.pwm.ChangeDutyCycle(self.right_duty)
 
    def cleanup(self):
        self.pwm.stop()
 
