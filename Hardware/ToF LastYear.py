import smbus2
from gpiozero import DigitalOutputDevice
import time

class DistanceSensor:
    SYSRANGE_START = 0x00
    RESULT_RANGE_STATUS = 0x14
    SYSTEM_INTERRUPT_CLEAR = 0x0B
    I2C_ADDR_REG = 0x8A      # Register to change address
    def __init__(self,xshut_pin):
        
        self.bus = smbus2.SMBus(1)
        self.address = 0x29
        self.xshut = DigitalOutputDevice(xshut_pin) # BCM (GPIO) numbering
        self.turn_on()
        st=1 #holds the status of the sensor if on
        time.sleep(0.1)
        self.turn_off()
        st=0

    def turn_off(self):
        self.xshut.off()
        st=0
        time.sleep(0.1)
    
    def turn_on(self):
        self.xshut.on()
        st=1
        time.sleep(0.1)    
    
    def change_address(self,new_Add):
        if(new_Add!=self.address):
            if not st:
                self.turn_on()
            st=1
            self.bus.write_byte_data(self.address, self.I2C_ADDR_REG, new_Add)
            self.address=new_Add
        else:
            if not st:
                self.turn_on()
            st=1
            time.sleep(0.1)
            return

    def read(self):
        try:
            self.bus.write_byte_data(self.address, self.SYSRANGE_START, 0x01)
            time.sleep(0.05)
            data = self.bus.read_i2c_block_data(self.address, self.RESULT_RANGE_STATUS + 10, 2)
            distance = (data[0] << 8) + data[1]
            self.bus.write_byte_data(self.address, self.SYSTEM_INTERRUPT_CLEAR, 0x01)
            return distance
        except OSError as e:
            print(f"Sensor at 0x{self.address:X} failed: {e}")
            return None



dist_sensors=[]
dist_sensors_pins=[4,10,11,6] #GPIo numbering
dist_sensors_Addresses=[0x30,0x31,0x32,0x33]  # I2C addresses
for i in range(4):
    dist_sensors.append(DistanceSensor(dist_sensors_pins[i]))#create a sensor object and intlize it and add it to the array
    dist_sensors[i].change_address(dist_sensors_Addresses[i])#give each sensor a costum address

for j in range(10):
    for i in range(4):
        print("reading from sensor ",i)
        print(dist_sensors[i].read())
        time.sleep(0.3)
