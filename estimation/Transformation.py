import math

class Robot:
    # ... existing x, y, theta, speed, steer_angle attributes ...

    def local_to_world(self, x_rel_cm: float, y_rel_cm: float) -> tuple[float, float]:
        """
        Convert a point given relative to the robot (x_rel = forward,
        y_rel = left, both in cm, robot-frame) into world coordinates,
        using the robot's current pose (self.x, self.y, self.theta).

        This is the SE(2) homogeneous transform:

            [x_world]   [cos(theta)  -sin(theta)   x][x_rel]
            [y_world] = [sin(theta)   cos(theta)   y][y_rel]
            [   1   ]   [    0            0        1][  1  ]

        i.e. rotate the relative offset by the robot's heading, then
        translate by the robot's world position.
        """
        cos_t = math.cos(self.theta)
        sin_t = math.sin(self.theta)

        x_world = self.x + x_rel_cm * cos_t - y_rel_cm * sin_t
        y_world = self.y + x_rel_cm * sin_t + y_rel_cm * cos_t

        return x_world, y_world
