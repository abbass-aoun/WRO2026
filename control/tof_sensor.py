"""
control/tof_sensor.py — VL53L0X time-of-flight distance sensors (×3).
=======================================================================

Ported from Hardware/ToF LastYear.py, updated for the 2026 pinout and
rewritten to use gpiozero for the XSHUT pins (consistent with the rest
of the project).

Three VL53L0X sensors share the same I2C bus. They all boot at address
0x29, so we must power them up one at a time during __init__ and assign
each a unique address before turning the next one on.

WIRING (from Pins for Sensors.txt):
    SDA      → Pin 3  (GPIO 2)   shared I2C bus
    SCL      → Pin 5  (GPIO 3)   shared I2C bus
    Vin (×3) → Pin 1  (3.3 V)
    GND (×3) → Pin 6  (GND)
    XSHUT 1  → Pin 7  (GPIO 4)
    XSHUT 2  → Pin 19 (GPIO 10)
    XSHUT 3  → Pin 21 (GPIO 9)

INSTALL:
    pip install smbus2

USAGE:
    tof = ToFSensors()
    d1, d2, d3 = tof.read_all_mm()
    # Each value is distance in mm, or None if that sensor failed.
    # Sensor order matches xshut_pins: (GPIO 4, GPIO 10, GPIO 9).

TUNING:
    Which sensor is left/front/right depends on how you mounted them.
    Map the indices in main.py once you've confirmed which is which.
    Run  python -m control.tof_sensor  to read live distances.
"""

from gpiozero import DigitalOutputDevice
from time import sleep

try:
    from smbus2 import SMBus
    _SMBUS_AVAILABLE = True
except ImportError:
    _SMBUS_AVAILABLE = False


class ToFSensors:
    """
    Three VL53L0X time-of-flight sensors with automatic address reassignment.

    Usage:
        tof = ToFSensors()
        d1, d2, d3 = tof.read_all_mm()
    """

    _DEFAULT_ADDR = 0x29   # VL53L0X I2C address at power-on
    _ADDR_REG     = 0x8A   # register that holds the I2C address
    _RANGE_START  = 0x00   # write 0x01 here to start a single-shot range
    _RANGE_RESULT = 0x14   # base of the 20-byte result block
    _INT_CLEAR    = 0x0B   # write 0x01 here to clear the data-ready interrupt

    def __init__(
        self,
        xshut_pins: tuple = (4, 10, 9),           # BCM GPIO (Pins for Sensors.txt)
        addresses:  tuple = (0x30, 0x31, 0x32),   # addresses assigned at startup
    ):
        """
        Args:
            xshut_pins : BCM GPIO pin for each sensor's XSHUT line.
                         Sensor 1 = GPIO 4, Sensor 2 = GPIO 10, Sensor 3 = GPIO 9.
                         TUNE ON REAL ROBOT if a sensor does not respond.
            addresses  : unique I2C addresses to assign (must not be 0x29 or clash).
        """
        self._addresses = list(addresses)

        if not _SMBUS_AVAILABLE:
            print("[ToF] smbus2 not installed — sensors disabled.  "
                  "Run: pip install smbus2")
            self._bus    = None
            self._xshuts = []
            return

        self._bus    = SMBus(1)
        self._xshuts = [DigitalOutputDevice(p) for p in xshut_pins]

        # 1. Pull all XSHUT pins LOW → every sensor is off and I2C bus is clear
        for x in self._xshuts:
            x.off()
        sleep(0.1)

        # 2. Power up each sensor one at a time and assign its unique address
        for xshut, new_addr in zip(self._xshuts, self._addresses):
            xshut.on()
            sleep(0.1)   # VL53L0X boots in ~1.2 ms; 100 ms is a safe margin
            try:
                self._bus.write_byte_data(
                    self._DEFAULT_ADDR, self._ADDR_REG, new_addr
                )
            except OSError as e:
                print(f"[ToF] Could not assign address 0x{new_addr:X}: {e}")

        # All three sensors are now ON at their unique addresses.

    # ------------------------------------------------------------------

    def read_mm(self, index: int):
        """
        Read distance from one sensor.

        Args:
            index : 0, 1, or 2 — matches the order of xshut_pins.

        Returns:
            Distance in mm (int), or None if the sensor is unavailable or fails.
        """
        if self._bus is None or index >= len(self._addresses):
            return None
        addr = self._addresses[index]
        try:
            self._bus.write_byte_data(addr, self._RANGE_START, 0x01)
            sleep(0.05)   # wait for measurement to complete (~33 ms typical)
            data = self._bus.read_i2c_block_data(
                addr, self._RANGE_RESULT + 10, 2
            )
            dist_mm = (data[0] << 8) | data[1]
            self._bus.write_byte_data(addr, self._INT_CLEAR, 0x01)
            return dist_mm
        except OSError:
            return None

    def read_all_mm(self) -> tuple:
        """
        Read all three sensors sequentially.

        Returns:
            (d1_mm, d2_mm, d3_mm) — each value is mm or None on failure.

        NOTE: Each read takes ~55 ms (50 ms measurement + overhead), so
        reading all three takes ~165 ms. Do NOT call this inside the 50 Hz
        main loop. Either call it in a background thread or only use it
        for diagnostic purposes.
        """
        return tuple(self.read_mm(i) for i in range(len(self._addresses)))


# =============================================================================
# Self-test — run from project root:  python -m control.tof_sensor
# Prints live distance from all three sensors.
# =============================================================================
if __name__ == "__main__":
    import time

    print("VL53L0X distance test.  Press Ctrl+C to stop.\n")
    tof = ToFSensors()

    try:
        while True:
            d1, d2, d3 = tof.read_all_mm()
            print(f"  Sensor 1 (GPIO 4) : {d1} mm    "
                  f"Sensor 2 (GPIO 10): {d2} mm    "
                  f"Sensor 3 (GPIO 9) : {d3} mm")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped.")
