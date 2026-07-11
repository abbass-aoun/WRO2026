class ColorSensor:
    """TCS3200 color sensor using pulse counting over a fixed sample window."""
 
    def __init__(self, s0, s1, s2, s3, out_pin, sample_time=0.03,
                 stop_r_threshold=300):
        self.s2, self.s3, self.out_pin = s2, s3, out_pin
        self.sample_time = sample_time
        self.stop_r_threshold = stop_r_threshold
        self._count = 0
 
        GPIO.setup(s0, GPIO.OUT)
        GPIO.setup(s1, GPIO.OUT)
        GPIO.setup(s2, GPIO.OUT)
        GPIO.setup(s3, GPIO.OUT)
        GPIO.setup(out_pin, GPIO.IN)
 
        GPIO.output(s0, GPIO.HIGH)  # 20% frequency scaling
        GPIO.output(s1, GPIO.LOW)
        self._ok = True
        print("[OK]   TCS3200 initialized")
 
    @property
    def available(self):
        return self._ok
 
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
        try:
            r = self._read_channel(GPIO.LOW, GPIO.LOW)
            g = self._read_channel(GPIO.HIGH, GPIO.HIGH)
            b = self._read_channel(GPIO.LOW, GPIO.HIGH)
            return r, g, b
        except Exception as e:
            print(f"Color read error: {e}")
            return 0, 0, 0
 
    def detect_stop_marker(self):
        r, g, b = self.read_rgb()
        return r > self.stop_r_threshold and r > g and r > b
 
 
