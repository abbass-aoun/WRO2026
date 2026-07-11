
from smbus2 import SMBus
import time
import math

bus = SMBus(1)
address = 0x68

# Wake up sensor
bus.write_byte_data(address, 0x6B, 0)

# ---------- smoothing setup ----------
x_buffer = []
y_buffer = []
window_size = 10

def read_word(reg):
    high = bus.read_byte_data(address, reg)
    low = bus.read_byte_data(address, reg + 1)
    value = (high << 8) | low
    if value >= 0x8000:
        value -= 65536
    return value

def moving_avg(buffer, value):
    buffer.append(value)
    if len(buffer) > window_size:
        buffer.pop(0)
    return sum(buffer) / len(buffer)

while True:
    # 📊 Accelerometer (g)
    ax = read_word(0x3B) / 16384.0
    ay = read_word(0x3D) / 16384.0
    az = read_word(0x3F) / 16384.0

    # 🔄 Gyroscope (°/s)
    gx = read_word(0x43) / 131.0
    gy = read_word(0x45) / 131.0
    gz = read_word(0x47) / 131.0

    # 📐 Raw tilt angles
    x_raw = math.degrees(math.atan2(ay, az))
    y_raw = math.degrees(math.atan2(ax, az))
    z_raw = math.degrees(math.atan2(ay, ax))  # approximate

    # 🧹 Smoothed tilt
    x_angle = moving_avg(x_buffer, x_raw)
    y_angle = moving_avg(y_buffer, y_raw)

    print("===== MPU6050 STABLE DATA =====")
    print(f"Ax: {ax:6.2f} g   Ay: {ay:6.2f} g   Az: {az:6.2f} g")
    print(f"Gx: {gx:6.2f} °/s Gy: {gy:6.2f} °/s Gz: {gz:6.2f} °/s")
    print(f"X tilt: {x_angle:6.2f}°")
    print(f"Y tilt: {y_angle:6.2f}°")
    print(f"Z tilt: {z_raw:6.2f}° (raw)")
    print("------------------------------")

    time.sleep(0.1)

