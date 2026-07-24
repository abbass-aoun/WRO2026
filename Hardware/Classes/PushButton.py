import gpiozero as GPIO
import time
class PushButton:
    """Interrupt-driven toggle button with debounce and optional status LED."""
 
    def __init__(self, pin, led_pin=None, debounce_time=0.3, on_toggle=None):
        self.pin = pin
        self.led_pin = led_pin
        self.debounce_time = debounce_time
        self.on_toggle = on_toggle  # callback(new_state: bool)
        self.state = False
        self._last_press = 0
 
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        if self.led_pin is not None:
            GPIO.setup(self.led_pin, GPIO.OUT)
            GPIO.output(self.led_pin, GPIO.LOW)
 
        GPIO.add_event_detect(self.pin, GPIO.FALLING,
                               callback=self._callback, bouncetime=300)
 
    def _callback(self, channel):
        now = time.time()
        if now - self._last_press < self.debounce_time:
            return
        self._last_press = now
 
        self.state = not self.state
        if self.led_pin is not None:
            GPIO.output(self.led_pin, GPIO.HIGH if self.state else GPIO.LOW)
 
        print("ROBOT ON" if self.state else "ROBOT OFF")
        if self.on_toggle:
            self.on_toggle(self.state)
