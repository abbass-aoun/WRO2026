import time
import adafruit_vl53l0x
import busio
import board

i2c = busio.I2C(board.SCL, board.SDA)

sensor = adafruit_vl53l0x.VL53L0X(i2c)

while True:
    print(sensor.range, "mm")
    time.sleep(0.3)

