class IMUSensor:
    """MPU6050 accelerometer + gyroscope, and simple impact detection."""
 
    def __init__(self, i2c, address=0x68, impact_threshold=2.0):
        self.impact_threshold = impact_threshold
        self._last_accel = None
        self._ok = False
        try:
            self.device = mpu6050(address)
            self._ok = True
            print("[OK]   MPU6050 initialized")
        except Exception as e:
            print(f"[FAIL] MPU6050 init failed: {e}")
            self.device = None
 
    @property
    def available(self):
        return self._ok
 
    def read(self):
        """Returns (accel_dict, gyro_dict) or (None, None) on failure."""
        if not self._ok:
            return None, None
        try:
            return self.device.get_accel_data(), self.device.get_gyro_data()
        except Exception as e:
            print(f"IMU read error: {e}")
            return None, None
 
    def detect_impact(self):
        """True if acceleration changed sharply since the last read (collision/tip)."""
        accel, _ = self.read()
        if accel is None:
            return False
        vec = (accel["x"], accel["y"], accel["z"])
        if self._last_accel is None:
            self._last_accel = vec
            return False
        delta = sum(abs(a - b) for a, b in zip(vec, self._last_accel))
        self._last_accel = vec
        return delta > self.impact_threshold
 
