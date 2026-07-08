import time


class PIDController:
    def __init__(self, Kp, Ki, Kd, output_limits, windup_limit):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.windup_limit = windup_limit
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def _compute(self, error):
        current_time = time.time()
        dt = current_time - self.last_time if self.last_time else 0.01
        self.last_time = current_time

        P = self.Kp * error

        self.integral += error * dt
        self.integral = max(min(self.integral, self.windup_limit), -self.windup_limit)
        I = self.Ki * self.integral

        derivative = (error - self.last_error) / dt if dt > 0 else 0.0
        D = self.Kd * derivative
        self.last_error = error

        output = P + I + D
        return max(self.output_limits[0], min(output, self.output_limits[1]))
