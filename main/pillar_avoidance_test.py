"""
main/pillar_avoidance_test.py

Standalone physical test for pillar avoidance.

Test 1 uses a simulated pillar position. The camera is not used.
The robot follows the avoidance path, reconnects to the original
straight line, and stops.

Run from the repository root:

    python3 -m main.pillar_avoidance_test
"""

from __future__ import annotations
from enum import Enum, auto
import math
import time

from config import (
    DT_S,
    PID_KD,
    PID_KI,
    PID_KP,
    PID_WINDUP_LIM,
    SERVO_MAX_DEG,
)
from control.pid_controller import PIDController
from control.pillar_avoidance import (
    PillarAvoidanceController,
    PillarTarget,
)
from control.straight_reference_controller import (
    StraightReferenceController,
)
from main.initialize_hardware import initialize_hardware
from main.support import calibrate_gyro

class TestPhase(Enum):
    IDLE = auto()
    AVOIDING = auto()
    RECOVERING = auto()
    FINISHED = auto()

class PillarAvoidanceTest:
    """
    Standalone pillar-avoidance test program.

    The run() method remains small because each robot operation
    is contained in a dedicated method.
    """

    # ============================================================
    # Test configuration
    # ============================================================

    TEST_START_X_CM = 0.0
    TEST_START_Y_CM = 0.0
    TEST_START_THETA_RAD = 0.0

    # Pillar 1
    TEST_PILLAR_1_FORWARD_CM = 70.0
    TEST_PILLAR_1_LATERAL_CM = 0.0
    TEST_PILLAR_1_COLOR = "red"

    # Pillar 2
    TEST_PILLAR_2_FORWARD_CM = 140.0
    TEST_PILLAR_2_LATERAL_CM = 0.0

    # Begin with "red" for an easier same-side test.
    # Change to "green" afterward for the harder S-turn test.
    TEST_PILLAR_2_COLOR = "green"

    # The standalone test does not need perfect recovery.
    # After this distance, a real corner could take priority.
    TEST_RECOVERY_DISTANCE_CM = 45.0

    TEST_TIMEOUT_S = 35.0
    LOG_INTERVAL_S = 0.20

    # Use the same tested duty during this standalone recovery.
    TEST_STRAIGHT_DUTY = 0.45

    def __init__(self):
        self.start_button = None
        self.leds = []
        self.encoders = None
        self.color_sensor = None
        self.car = None
        self.robot = None
        self.ekf = None

        self.gyro_bias = 0.0

        self.steering_pid = PIDController(
            Kp=PID_KP,
            Ki=PID_KI,
            Kd=PID_KD,
            output_limits=(
                -SERVO_MAX_DEG,
                SERVO_MAX_DEG,
            ),
            windup_limit=PID_WINDUP_LIM,
        )

        self.avoidance = PillarAvoidanceController(
            steering_pid=self.steering_pid,
        )

        self.straight_recovery = (
            StraightReferenceController(
                motor_duty=self.TEST_STRAIGHT_DUTY,

                # Correct heading error while recovering.
                heading_kp=0.30,

                # Stronger lateral correction because pillar avoidance
                # can leave the robot 20–35 cm away from the line.
                cross_track_kp=0.30,

                heading_deadband_rad=math.radians(1.0),

                # The normal straight controller uses a small 2° cap.
                # Recovery needs more authority to return in less distance.
                cross_track_max_deg=6.0,

                # Full steering remains available when necessary.
                max_steering_deg=18.0,

                # Preserve the existing command-transition smoothing.
                max_steering_change_deg=6.0,

                line_tolerance_cm=2.0,
                heading_tolerance_rad=math.radians(3.0),
                done_confirmations=5,
            )
        )
        self.phase = TestPhase.IDLE

        self.simulated_pillars: list[PillarTarget] = []

        self.current_pillar_index = 0

        self.recovery_start_distance_cm = 0.0

    # ============================================================
    # Overall test sequence
    # ============================================================

    def run(self) -> None:
        try:
            self.initialize_robot()
            self.wait_for_start_button()
            self.prepare_test_run()

            self.simulated_pillars = self.create_simulated_pillars()
            self.current_pillar_index = 0

            self.start_current_pillar_avoidance(
                robot_x=self.robot.x,
                robot_y=self.robot.y,
                robot_theta=self.robot.theta,
            )

            self.phase = TestPhase.AVOIDING

            self.run_control_loop()

        except KeyboardInterrupt:
            print("\nTest interrupted by user.")

        except Exception as exc:
            print(
                "\nPillar avoidance test failed: "
                f"{type(exc).__name__}: {exc}"
            )
            raise

        finally:
            self.stop_robot()

    # ============================================================
    # Hardware initialization
    # ============================================================

    def initialize_robot(self) -> None:
        print("Initializing hardware...")

        (
            self.start_button,
            self.leds,
            self.encoders,
            self.color_sensor,
            self.car,
            self.robot,
            self.ekf,
        ) = initialize_hardware()

        self.car.stop()

        print(
            "Hardware initialized. Keep the robot still "
            "for gyro calibration."
        )

        self.gyro_bias = calibrate_gyro(
            self.encoders
        )

    # ============================================================
    # Start button
    # ============================================================

    def wait_for_start_button(self) -> None:
        print(
            "\nPlace the robot at the test start position."
        )
        print(
            "Robot must face straight along global +X."
        )
        print(
            "Press the physical start button to begin."
        )

        leds_on = False
        last_toggle = time.monotonic()

        while not self.start_button.is_pressed:
            now = time.monotonic()

            if now - last_toggle >= 0.5:
                leds_on = not leds_on
                self.set_leds(leds_on)
                last_toggle = now

            time.sleep(0.01)

        self.set_leds(False)

        # Prevent one physical press from being interpreted repeatedly.
        while self.start_button.is_pressed:
            time.sleep(0.01)

        print("GO!")

    def set_leds(self, enabled: bool) -> None:
        for led in self.leds:
            if enabled:
                led.on()
            else:
                led.off()

    # ============================================================
    # Test preparation
    # ============================================================

    def prepare_test_run(self) -> None:
        self.car.stop()
        self.car.set_steering(0.0)

        self.robot.reset()
        self.encoders.reset()
        self.steering_pid.reset()
        self.avoidance.reset()

        self.straight_recovery.reset()
        self.phase = TestPhase.IDLE

        self.ekf.initialize(
            x0=self.TEST_START_X_CM,
            y0=self.TEST_START_Y_CM,
            theta0=self.TEST_START_THETA_RAD,
        )

        self.robot.update_pose(
            self.TEST_START_X_CM,
            self.TEST_START_Y_CM,
            self.TEST_START_THETA_RAD,
        )

        self.robot.update_speed(0.0)
        self.robot.update_steering(0.0)

    # ============================================================
    # Simulated pillar
    # ============================================================

    def create_simulated_pillar(
        self,
        forward_cm: float,
        lateral_cm: float,
        color: str,
    ) -> PillarTarget:
        """
        Convert one simulated pillar from local forward/lateral
        coordinates into the global EKF coordinate system.
        """

        forward_x = math.cos(
            self.TEST_START_THETA_RAD
        )

        forward_y = math.sin(
            self.TEST_START_THETA_RAD
        )

        left_x = -forward_y
        left_y = forward_x

        pillar_x = (
            self.TEST_START_X_CM
            + forward_cm * forward_x
            + lateral_cm * left_x
        )

        pillar_y = (
            self.TEST_START_Y_CM
            + forward_cm * forward_y
            + lateral_cm * left_y
        )

        return PillarTarget(
            global_x_cm=pillar_x,
            global_y_cm=pillar_y,
            color=color,
        )


    def create_simulated_pillars(
        self,
    ) -> list[PillarTarget]:
        """Create the two pillars used by this test."""

        return [
            self.create_simulated_pillar(
                forward_cm=(
                    self.TEST_PILLAR_1_FORWARD_CM
                ),
                lateral_cm=(
                    self.TEST_PILLAR_1_LATERAL_CM
                ),
                color=self.TEST_PILLAR_1_COLOR,
            ),
            self.create_simulated_pillar(
                forward_cm=(
                    self.TEST_PILLAR_2_FORWARD_CM
                ),
                lateral_cm=(
                    self.TEST_PILLAR_2_LATERAL_CM
                ),
                color=self.TEST_PILLAR_2_COLOR,
            ),
        ]


    def start_current_pillar_avoidance(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> None:
        """
        Start the trajectory for the current pillar.

        The controller receives the following pillar as context,
        allowing it to place the current trajectory endpoint in
        the correct approach position.
        """

        current_pillar = (
            self.simulated_pillars[
                self.current_pillar_index
            ]
        )

        next_index = (
            self.current_pillar_index + 1
        )

        if next_index < len(
            self.simulated_pillars
        ):
            next_pillar = (
                self.simulated_pillars[
                    next_index
                ]
            )
        else:
            next_pillar = None

        print(
            "\nStarting pillar "
            f"{self.current_pillar_index + 1}/"
            f"{len(self.simulated_pillars)}"
        )

        self.avoidance.start(
            pillar=current_pillar,
            next_pillar=next_pillar,

            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,

            target_theta=(
                self.TEST_START_THETA_RAD
            ),

            straight_ref_x=(
                self.TEST_START_X_CM
            ),

            straight_ref_y=(
                self.TEST_START_Y_CM
            ),
        )

    # ============================================================
    # Sensor and EKF update
    # ============================================================

    def update_robot_state(
        self,
    ) -> tuple[float, float, float, float]:
        """
        Update the EKF using wheel distance covered during the
        actual elapsed control-loop interval.
        """

        (
            v_left,
            v_right,
            actual_dt,
        ) = self.encoders.get_distance_based_speeds()

        speed = 0.5 * (
            v_left + v_right
        )

        yaw_rate = (
            self.encoders.get_yaw_rate()
            - self.gyro_bias
        )

        steering_rad = math.radians(
            self.robot.steer_angle
        )

        self.ekf.predict(
            speed=speed,
            steer_angle=steering_rad,
            dt=actual_dt,
            omega_gyro=yaw_rate,
        )

        x, y, theta = self.ekf.state

        self.robot.update_pose(
            x,
            y,
            theta,
        )

        self.robot.update_speed(speed)

        return x, y, theta, speed

    # ============================================================
    # Control loop
    # ============================================================

    def run_control_loop(self) -> None:
        test_start_time = time.monotonic()
        last_log_time = 0.0

        while True:
            loop_start_time = time.monotonic()

            x, y, theta, speed = (
                self.update_robot_state()
            )

            steering_deg, done = (
                self.run_active_controller_step(
                    x=x,
                    y=y,
                    theta=theta,
                )
            )

            now = time.monotonic()

            if (
                now - last_log_time
                >= self.LOG_INTERVAL_S
            ):
                self.print_status(
                    x=x,
                    y=y,
                    theta=theta,
                    speed=speed,
                    steering_deg=steering_deg,
                )

                last_log_time = now

            if done:
                self.car.stop()

                print(
                    "\nStandalone pillar avoidance "
                    "completed successfully."
                )

                print(
                    f"Final pose: "
                    f"x={x:.1f} cm, "
                    f"y={y:.1f} cm, "
                    f"theta={math.degrees(theta):+.1f}°"
                )

                break

            if (
                now - test_start_time
                >= self.TEST_TIMEOUT_S
            ):
                raise TimeoutError(
                    "Avoidance did not finish before "
                    f"{self.TEST_TIMEOUT_S:.1f} seconds."
                )

            self.sleep_until_next_tick(
                loop_start_time
            )


    def run_active_controller_step(
        self,
        x: float,
        y: float,
        theta: float,
    ) -> tuple[float, bool]:
        """
        Run whichever controller currently owns steering.

        Returns:
            steering_deg
            entire_test_done
        """

        if self.phase == TestPhase.AVOIDING:

            steering_deg, current_trajectory_done = (
                self.avoidance.step(
                    car=self.car,
                    robot=self.robot,
                    robot_x=x,
                    robot_y=y,
                    robot_theta=theta,
                    allow_handoff=True,
                )
            )

            # Continue following the current trajectory.
            if not current_trajectory_done:
                return steering_deg, False

            final_pillar_index = (
                len(self.simulated_pillars) - 1
            )

            # ========================================================
            # Pillar 1 completed: immediately start pillar 2
            # ========================================================
            if (
                self.current_pillar_index
                < final_pillar_index
            ):
                self.current_pillar_index += 1

                self.start_current_pillar_avoidance(
                    robot_x=x,
                    robot_y=y,
                    robot_theta=theta,
                )

                # Remain in avoidance mode.
                self.phase = TestPhase.AVOIDING

                return steering_deg, False

            # ========================================================
            # Final pillar completed: begin straight recovery
            # ========================================================
            self.start_straight_recovery()

            return steering_deg, False

        if self.phase == TestPhase.RECOVERING:

            steering_deg, aligned = (
                self.straight_recovery.step(
                    car=self.car,
                    robot=self.robot,
                    robot_x=x,
                    robot_y=y,
                    robot_theta=theta,
                )
            )

            recovered_distance_cm = (
                self.get_average_encoder_distance()
                - self.recovery_start_distance_cm
            )

            recovery_window_finished = (
                recovered_distance_cm
                >= self.TEST_RECOVERY_DISTANCE_CM
            )

            if aligned or recovery_window_finished:
                self.phase = TestPhase.FINISHED

                print(
                    "\n=== TWO-PILLAR TEST COMPLETED ==="
                )
                print(
                    f"Recovery distance: "
                    f"{recovered_distance_cm:.1f} cm"
                )
                print(
                    f"Final y: {y:+.2f} cm"
                )
                print(
                    "Final heading: "
                    f"{math.degrees(theta):+.2f}°"
                )

                return steering_deg, True

            return steering_deg, False


    def get_average_encoder_distance(
        self,
    ) -> float:
        """Return average cumulative wheel distance."""

        left_cm, right_cm = (
            self.encoders.get_distances()
        )

        return 0.5 * (
            left_cm + right_cm
        )


    def start_straight_recovery(self) -> None:
        """
        Give steering control to the original straight reference.

        The avoidance trajectory is finished, but the original
        reference itself has never been changed.
        """

        self.straight_recovery.start(
            reference_x=self.TEST_START_X_CM,
            reference_y=self.TEST_START_Y_CM,
            target_theta=self.TEST_START_THETA_RAD,

            # Begin from the steering angle that was active at handoff,
            # preventing an abrupt jump back to zero.
            initial_steering_deg=self.robot.steer_angle,
        )

        self.recovery_start_distance_cm = (
            self.get_average_encoder_distance()
        )

        self.phase = TestPhase.RECOVERING


    def sleep_until_next_tick(
        self,
        loop_start_time: float,
    ) -> None:
        elapsed = (
            time.monotonic()
            - loop_start_time
        )

        remaining = DT_S - elapsed

        if remaining > 0.0:
            time.sleep(remaining)

    # ============================================================
    # Logging
    # ============================================================

    def print_status(
        self,
        x: float,
        y: float,
        theta: float,
        speed: float,
        steering_deg: float,
    ) -> None:

        left_distance_cm, right_distance_cm = (
            self.encoders.get_distances()
        )

        average_distance_cm = 0.5 * (
            left_distance_cm
            + right_distance_cm
        )

        pillar_text = (
            f"{self.current_pillar_index + 1}/"
            f"{len(self.simulated_pillars)}"
            if self.simulated_pillars
            else "0/0"
        )

        print(
            f"phase={self.phase.name:<10} | "
            f"pillar={pillar_text} | "
            f"x={x:7.2f} cm | "
            f"y={y:7.2f} cm | "
            f"theta={math.degrees(theta):+7.2f}° | "
            f"speed={speed:6.2f} cm/s | "
            f"encoder_dist={average_distance_cm:7.2f} cm | "
            f"steer={steering_deg:+6.2f}° | "
            f"remaining={self.avoidance.remaining_cm:6.2f} cm"
        )

    # ============================================================
    # Cleanup
    # ============================================================

    def stop_robot(self) -> None:
        if self.car is not None:
            self.car.stop()
            self.car.set_steering(0.0)

        self.set_leds(False)

        if self.color_sensor is not None:
            self.color_sensor.stop()

        print("Robot stopped.")


if __name__ == "__main__":
    PillarAvoidanceTest().run()