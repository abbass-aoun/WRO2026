import time
from mpu6050 import mpu6050
 
MPU_ADDRESS = 0x68  # default I2C address (0x69 if AD0 pin is pulled high)
 
def main():
    sensor = mpu6050(MPU_ADDRESS)
 
    print("Reading MPU6050... Press Ctrl+C to stop.\n")
    try:
        while True:
            accel = sensor.get_accel_data()   # m/s^2
            gyro = sensor.get_gyro_data()      # deg/s
            temp = sensor.get_temp()           # Celsius
 
            print(f"Accel [g]:  x={accel['x']:6.2f}  y={accel['y']:6.2f}  z={accel['z']:6.2f}")
            print(f"Gyro [dps]: x={gyro['x']:6.2f}  y={gyro['y']:6.2f}  z={gyro['z']:6.2f}")
            print(f"Temp [C]:   {temp:5.2f}")
            print("-" * 40)
 
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped by user.")
 
if __name__ == "__main__":
    main()
 
