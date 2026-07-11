class MotorController:
    """L298N H-bridge, two independently driven motors. Swap this class
    out if you're using a different driver (TB6612FNG, ESCs, etc.)."""
 
    def __init__(self, in1, in2, ena, in3, in4, enb, pwm_freq=1000):
        self.in1, self.in2, self.in3, self.in4 = in1, in2, in3, in4
 
        for pin in (in1, in2, ena, in3, in4, enb):
            GPIO.setup(pin, GPIO.OUT)
 
        self.pwm_a = GPIO.PWM(ena, pwm_freq)
        self.pwm_b = GPIO.PWM(enb, pwm_freq)
        self.pwm_a.start(0)
        self.pwm_b.start(0)
        print("[OK]   MotorController (L298N) initialized")
 
    def _set_left(self, speed):  # -100..100
        GPIO.output(self.in1, GPIO.HIGH if speed >= 0 else GPIO.LOW)
        GPIO.output(self.in2, GPIO.LOW if speed >= 0 else GPIO.HIGH)
        self.pwm_a.ChangeDutyCycle(min(abs(speed), 100))
 
    def _set_right(self, speed):
        GPIO.output(self.in3, GPIO.HIGH if speed >= 0 else GPIO.LOW)
        GPIO.output(self.in4, GPIO.LOW if speed >= 0 else GPIO.HIGH)
        self.pwm_b.ChangeDutyCycle(min(abs(speed), 100))
 
    def forward(self, speed):
        self._set_left(speed)
        self._set_right(speed)
 
    def backward(self, speed):
        self._set_left(-speed)
        self._set_right(-speed)
 
    def turn_left(self, speed):
        self._set_left(-speed * 0.3)
        self._set_right(speed)
 
    def turn_right(self, speed):
        self._set_left(speed)
        self._set_right(-speed * 0.3)
 
    def stop(self):
        self._set_left(0)
        self._set_right(0)
 
    def cleanup(self):
        self.stop()
        self.pwm_a.stop()
        self.pwm_b.stop()
 
