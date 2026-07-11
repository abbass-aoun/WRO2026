class ToFSensor:
    """VL53L0X time-of-flight distance sensor."""
 
    def __init__(self, i2c, timing_budget_us=200000):
        self._ok = False
        try:
            self.device = adafruit_vl53l0x.VL53L0X(i2c)
            self.device.measurement_timing_budget = timing_budget_us
            self._ok = True
            print("[OK]   VL53L0X initialized")
        except Exception as e:
            print(f"[FAIL] VL53L0X init failed: {e}")
            self.device = None
 
    @property
    def available(self):
        return self._ok
 
    def read_mm(self):
        if not self._ok:
            return None
        try:
            return self.device.range
        except Exception as e:
            print(f"ToF read error: {e}")
            return None
 
