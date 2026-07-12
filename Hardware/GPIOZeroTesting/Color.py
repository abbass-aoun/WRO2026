import time
from gpiozero import DigitalOutputDevice, DigitalInputDevice
 
# --- Pin setup (BCM numbering) - change to match your wiring ---
S0 = DigitalOutputDevice(5)
S1 = DigitalOutputDevice(6)
S2 = DigitalOutputDevice(13)
S3 = DigitalOutputDevice(19)
OUT = DigitalInputDevice(26)
 
# Set frequency scaling to 20%
S0.on()
S1.off()
 
 
def count_pulses(sample_time=0.1):
    """Count OUT pin pulses over sample_time seconds and return frequency (Hz)."""
    count = 0
 
    def _increment():
        nonlocal count
        count += 1
 
    OUT.when_activated = _increment
    start = time.time()
    while time.time() - start < sample_time:
        time.sleep(0.001)
    OUT.when_activated = None
 
    return count / sample_time
 
 
def read_color():
    # Red
    S2.off()
    S3.off()
    time.sleep(0.05)
    red = count_pulses()
 
    # Green
    S2.on()
    S3.on()
    time.sleep(0.05)
    green = count_pulses()
 
    # Blue
    S2.off()
    S3.on()
    time.sleep(0.05)
    blue = count_pulses()
 
    return red, green, blue
 
 
def main():
    print("Reading TCS3200 color sensor... Press Ctrl+C to stop.\n")
    try:
        while True:
            r, g, b = read_color()
            print(f"R:{r:7.1f} Hz   G:{g:7.1f} Hz   B:{b:7.1f} Hz")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
 
 
if __name__ == "__main__":
    main()
