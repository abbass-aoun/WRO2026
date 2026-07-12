import time
import board
import busio
import adafruit_vl53l0x
 
def main():
    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = adafruit_vl53l0x.VL53L0X(i2c)
 
    print("Reading VL53L0X distance... Press Ctrl+C to stop.\n")
    try:
        while True:
            distance_mm = sensor.range
            print(f"Distance: {distance_mm} mm")
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\nStopped by user.")
 
if __name__ == "__main__":
    main()
