#!/usr/bin/env python3
"""
TCS3200 / TCS230 Color Sensor test script for Raspberry Pi 5
Reads Red, Green, Blue and Clear frequency values and prints them.

Wiring:
  VCC (both) -> Pin 1  (3.3V)   <-- Use 3.3V, NOT 5V
  GND (both) -> Pin 9  (GND)
  S0         -> Pin 11 (GPIO17)
  S1         -> Pin 13 (GPIO27)
  S2         -> Pin 15 (GPIO22)
  S3         -> Pin 16 (GPIO23)
  OUT        -> Pin 18 (GPIO24)
  LED        -> Pin 22 (GPIO25)  (optional, onboard LEDs)

Requires: gpiozero (works on Pi 5 via the lgpio backend automatically)
    pip install gpiozero --break-system-packages
"""

from gpiozero import DigitalOutputDevice, DigitalInputDevice
import time

# --- Pin setup ---
S0 = DigitalOutputDevice(17)
S1 = DigitalOutputDevice(27)
S2 = DigitalOutputDevice(22)
S3 = DigitalOutputDevice(23)
OUT = DigitalInputDevice(24)
LED = DigitalOutputDevice(25)

SAMPLE_TIME = 0.1  # seconds to count pulses for each color

# --- Frequency scaling: 20% (S0=H, S1=L) ---
S0.on()
S1.off()

# --- Filter select combinations ---
FILTERS = {
    "Red":   (0, 0),  # S2=L, S3=L
    "Blue":  (0, 1),  # S2=L, S3=H
    "Clear": (1, 0),  # S2=H, S3=L
    "Green": (1, 1),  # S2=H, S3=H
}


def set_filter(s2_val, s3_val):
    S2.value = s2_val
    S3.value = s3_val
    time.sleep(0.02)  # small settle time after switching filter


def count_pulses(duration):
    count = 0

    def on_pulse():
        nonlocal count
        count += 1

    OUT.when_activated = on_pulse
    count = 0
    start = time.time()
    while time.time() - start < duration:
        pass
    OUT.when_activated = None
    return count / duration  # pulses per second = frequency (Hz)


def read_color():
    readings = {}
    for name, (s2_val, s3_val) in FILTERS.items():
        set_filter(s2_val, s3_val)
        freq = count_pulses(SAMPLE_TIME)
        readings[name] = freq
    return readings


def main():
    print("Initializing TCS3200...")
    LED.on()  # turn on onboard LEDs for consistent lighting
    time.sleep(0.2)
    print("Reading colors (Ctrl+C to stop)\n")

    try:
        while True:
            data = read_color()
            print(f"Red={data['Red']:7.1f} Hz   "
                  f"Green={data['Green']:7.1f} Hz   "
                  f"Blue={data['Blue']:7.1f} Hz   "
                  f"Clear={data['Clear']:7.1f} Hz")
            time.sleep(0.3)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        LED.off()
        S0.close()
        S1.close()
        S2.close()
        S3.close()
        OUT.close()
        LED.close()


if __name__ == "__main__":
    main()
