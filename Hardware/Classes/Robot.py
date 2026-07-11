class Robot:
    def __init__(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
 
        i2c = busio.I2C(board.SCL, board.SDA)
 
        # --- Sensors ---
        self.imu = IMUSensor(i2c, address=0x68, impact_threshold=2.0)
        self.tof = ToFSensor(i2c)
        self.color = ColorSensor(s0=17, s1=27, s2=22, s3=23, out_pin=24,
                                  stop_r_threshold=300)
 
        # --- Actuators ---
        self.motors = MotorController(in1=5, in2=6, ena=12,
                                       in3=19, in4=16, enb=13)
        self.servo = SteeringServo(pin=20)
 
        # --- Button (drives self.running) ---
        self.running = False
        self.button = PushButton(pin=25, led_pin=26, on_toggle=self._on_power_toggle)
 
        # --- Navigation thresholds ---
        self.obstacle_stop_mm = 150
        self.obstacle_slow_mm = 400
        self.drive_speed = 60
        self.turn_speed = 45
        self.loop_interval = 0.1
 
        self._was_running = False
 
    def _on_power_toggle(self, new_state):
        self.running = new_state
 
    def _drive_step(self):
        # Safety first: collision / tip-over
        if self.imu.detect_impact():
            print("Impact/tilt detected -> emergency stop")
            self.motors.stop()
            self.servo.center()
            time.sleep(0.5)
            return
 
        # Stop-line / marker check
        if self.color.detect_stop_marker():
            print("Stop marker detected -> halting")
            self.motors.stop()
            self.servo.center()
            return
 
        # Obstacle avoidance via ToF
        distance = self.tof.read_mm()
        if distance is not None:
            if distance < self.obstacle_stop_mm:
                print(f"Obstacle at {distance}mm -> stop & turn")
                self.motors.stop()
                self.servo.left()
                self.motors.turn_left(self.turn_speed)
                time.sleep(0.4)
                self.servo.center()
            elif distance < self.obstacle_slow_mm:
                self.servo.center()
                self.motors.forward(self.turn_speed)
            else:
                self.servo.center()
                self.motors.forward(self.drive_speed)
        else:
            self.servo.center()
            self.motors.forward(self.turn_speed)
 
    def run(self):
        print("\nReady. Press the button to start driving. Ctrl+C to quit.\n")
        while True:
            loop_start = time.time()
 
            if not self.running:
                if self._was_running:
                    self.motors.stop()
                    self.servo.center()
                    self._was_running = False
                time.sleep(self.loop_interval)
                continue
 
            self._was_running = True
            self._drive_step()
 
            elapsed = time.time() - loop_start
            time.sleep(max(0, self.loop_interval - elapsed))
 
    def cleanup(self):
        self.motors.cleanup()
        self.servo.cleanup()
        GPIO.cleanup()
 
