#this one works!!
from gpiozero import PWMOutputDevice, DigitalOutputDevice
from time import time, sleep
from servoClass import myServo
from allEncodersClass import RobotEncoders

class CarController:
    def __init__(self, in1_pin, in2_pin, ena_pin, servo_pin):
        # Motor control pins
        self.in1 = DigitalOutputDevice(in1_pin)
        self.in2 = DigitalOutputDevice(in2_pin)
        self.ena = PWMOutputDevice(ena_pin)

        # Steering servo (adjust angles and pulse widths if needed)
        self.servo = myServo(servo_pin, center_angle=78, max_deviation=27)

    def set_motor(self, direction, speed=1.0):
        speed = max(0.0, min(1.0, speed))
        if direction == 'f':
            self.in1.on()
            self.in2.off()
            self.ena.value = speed
        elif direction == 'b':
            self.in1.off()
            self.in2.on()
            self.ena.value = speed
        else:
            self.stop()

    def set_steering(self, angle):
        self.servo.set_servo_angle(angle)
        print(f"Steering set to {angle} degrees")
        sleep(0.05)

    def stop(self):

        self.in1.off()
        self.in2.off()
        self.ena.value = 0.0

    def setAll(self,direction,speed,angle):
        self.set_steering(angle)
        self.set_motor(direction,speed)
    
    def brake(self, encoders, kp=15, ki=0.4, kd=4, tolerance=0.05, log_fn=None):
        integral = 0
        last_error = 0
        last_time = time()
        count = 0

        while True:
            v_l, v_r = encoders.get_linear_speeds()
            avg_speed = (v_l + v_r) / 2.0

            error = -avg_speed
            now = time()
            dt = now - last_time
            last_time = now

            if dt == 0:
                continue

            integral += error * dt
            derivative = (error - last_error) / dt
            last_error = error

            control = kp * error + ki * integral + kd * derivative
            control = max(0.0, min(1.0, abs(control)))

            if count % 2 == 0:
                self.set_motor("b" if avg_speed > 0 else "f", control)
            else:
                self.set_motor("f" if avg_speed > 0 else "b", 0)  # release

            count += 1

            if log_fn:
                log_fn(now, v_l, v_r)

            if abs(avg_speed) < tolerance:
                break

            sleep(0.01)

        self.stop()






        

                                
# # === Demo usage ===                                                                                                                                                                                                                
# if False:
    # try:
        # car = CarController(in1_pin=14, in2_pin=15, ena_pin=21, servo_pin=4)

        # steering = 0
        # for i in range(18):
            # car.set_steering(10)
            # sleep(0.5)    
        # #car.setAll("b", 0.6, 10)

        # car.stop()
       

    # except KeyboardInterrupt:
        # print("Interrupted")

    # finally:
        # car.stop()
