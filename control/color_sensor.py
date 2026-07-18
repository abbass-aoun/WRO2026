"""
control/color_sensor.py — TCS3200 color sensor for floor-line detection.
=========================================================================

Detects the orange and blue floor lines used by the WRO race manager to
know when a straight section ends and which lap direction the car is going.

WIRING (from Pins for Sensors.txt):
    S0  → GPIO 17  (Pin 11)
    S1  → GPIO 27  (Pin 13)
    S2  → GPIO 22  (Pin 15)
    S3  → GPIO 23  (Pin 16)
    OUT → GPIO 24  (Pin 18)
    LED → GPIO 25  (Pin 22)  — sensor's own illumination LEDs
    VCC → 3.3 V    (Pin 1)
    GND → GND      (Pin 9)

HOW THE TCS3200 WORKS:
    S0/S1 set the output frequency scaling (HIGH/LOW = 20%).
    S2/S3 select which color filter (see _RED/_GREEN/_BLUE below).
    OUT pulses at a rate proportional to light intensity of the chosen color.
    We count pulses over a short window to get a frequency (= brightness).

BACKGROUND THREAD:
    A full R+G+B read takes ~33 ms (3 × 10 ms sample + settling).
    To avoid blocking the 50 Hz main loop we run the sensor in a
    daemon thread that updates two boolean flags continuously.
    The main loop just reads:  color.orange_seen  and  color.blue_seen

TUNING (do this on the real robot):
    Run:  python -m control.color_sensor
    Hold the sensor over orange tape, then blue tape, then the plain floor.
    Note the printed R/G/B counts for each surface, then adjust the
    thresholds in detect_lines() accordingly.
"""

import threading
from gpiozero import DigitalOutputDevice, DigitalInputDevice
from time import sleep


class ColorSensor:
    """
    TCS3200 color sensor — detects orange and blue floor lines.

    Usage:
        sensor = ColorSensor()
        # inside the 50 Hz main loop:
        orange, blue = sensor.orange_seen, sensor.blue_seen
        sensor.stop()   # call once when the program exits
    """

    # S2/S3 truth table for the TCS3200 color filter:
    #   (S2=LOW,  S3=LOW)  → Red   filter
    #   (S2=HIGH, S3=HIGH) → Green filter
    #   (S2=LOW,  S3=HIGH) → Blue  filter
    _RED   = (False, False)
    _GREEN = (True,  True)
    _BLUE  = (False, True)

    def __init__(
        self,
        s0:            int   = 17,    # GPIO pin — frequency scaling high bit
        s1:            int   = 27,    # GPIO pin — frequency scaling low bit
        s2:            int   = 22,    # GPIO pin — color filter select
        s3:            int   = 23,    # GPIO pin — color filter select
        out_pin:       int   = 24,    # GPIO pin — pulse output from sensor
        led_pin:       int   = 25,    # GPIO pin — onboard illumination LEDs
        sample_time:   float = 0.010, # seconds per channel reading
        poll_interval: float = 0.050, # pause between complete RGB reads
    ):
        """
        Args:
            s0..s3       : GPIO BCM pin numbers matching the wiring above.
            out_pin      : GPIO BCM pin for the TCS3200 OUT signal.
            led_pin      : GPIO BCM pin for the sensor's onboard LEDs (GPIO 25).
                           Turning these on gives consistent lighting for readings.
            sample_time  : how long (s) to count pulses per colour channel.
                           10 ms is fast enough for line detection.
            poll_interval: pause between full RGB reads in the background loop.
                           50 ms → ~12 Hz update rate for the boolean flags.
        """
        self._s2  = DigitalOutputDevice(s2)
        self._s3  = DigitalOutputDevice(s3)
        self._out = DigitalInputDevice(out_pin, pull_up=False)

        # S0=HIGH, S1=LOW → 20% output frequency scaling (good mid-range choice)
        self._s0 = DigitalOutputDevice(s0)
        self._s1 = DigitalOutputDevice(s1)
        self._s0.on()
        self._s1.off()

        # Turn on the sensor's own illumination LEDs for consistent readings
        self._led = DigitalOutputDevice(led_pin)
        self._led.on()

        self._sample_time   = sample_time
        self._poll_interval = poll_interval
        self._pulse_count   = 0

        # Flags written by background thread, read by main loop
        self.orange_seen: bool = False
        self.blue_seen:   bool = False

        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_pulse(self) -> None:
        self._pulse_count += 1

    def _read_channel(self, s2_on: bool, s3_on: bool) -> float:
        """Count pulses on one colour channel. Returns pulses per second."""
        if s2_on:
            self._s2.on()
        else:
            self._s2.off()
        if s3_on:
            self._s3.on()
        else:
            self._s3.off()

        sleep(0.001)                           # let filter settle
        self._pulse_count = 0
        self._out.when_activated = self._on_pulse
        sleep(self._sample_time)
        self._out.when_activated = None
        return self._pulse_count / self._sample_time

    def _run(self) -> None:
        """Background loop — updates orange_seen / blue_seen continuously."""
        while not self._stop_event.is_set():
            r = self._read_channel(*self._RED)
            g = self._read_channel(*self._GREEN)
            b = self._read_channel(*self._BLUE)

            # Orange: R strongly dominant over B and G.
            # Ratio-based so ambient light level doesn't matter.
            # TUNE ON REAL ROBOT if false positives occur.
            self.orange_seen = (r > 2000 and r > b * 1.5 and r > g * 0.7)

            # Blue: B strongly dominant over R and G.
            # TUNE ON REAL ROBOT if false positives occur.
            self.blue_seen   = (b > 2000 and b > r * 1.5 and b > g * 0.7)

            sleep(self._poll_interval)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def stop(self) -> None:
        """Stop the background thread and turn off the illumination LEDs."""
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._led.off()


# =============================================================================
# Self-test — run from project root:  python -m control.color_sensor
# Prints live R, G, B counts so you can tune the orange/blue thresholds.
# =============================================================================
if __name__ == "__main__":
    import time

    print("TCS3200 live reading — hold sensor over different surfaces.")
    print("Press Ctrl+C to stop.\n")

    sensor = ColorSensor()

    # Give the background thread one full cycle to warm up
    time.sleep(0.2)

    try:
        while True:
            # Read raw counts directly (bypass the boolean flags)
            r = sensor._read_channel(*ColorSensor._RED)
            g = sensor._read_channel(*ColorSensor._GREEN)
            b = sensor._read_channel(*ColorSensor._BLUE)
            o = sensor.orange_seen
            bl = sensor.blue_seen
            print(f"R={r:6.0f}  G={g:6.0f}  B={b:6.0f}  "
                  f"orange={o}  blue={bl}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        sensor.stop()
        print("Stopped.")
