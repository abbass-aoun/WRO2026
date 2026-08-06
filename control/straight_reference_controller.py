"""
Straight-line reference controller used by standalone tests.

It follows an infinite reference line described by:
- one point on the line;
- the desired line heading.

This mirrors the steering logic currently used in Abstract_Main.py,
but keeps all controller state inside one reusable object.
"""

from __future__ import annotations

import math


class StraightReferenceController:
    def __init__(
        self,
        motor_duty: float,
        heading_kp: float = 0.27,
        cross_track_kp: float = 0.20,
        heading_deadband_rad: float = math.radians(1.0),
        cross_track_max_deg: float = 2.0,
        max_steering_deg: float = 18.0,
        max_steering_change_deg: float = 6.0,
        line_tolerance_cm: float = 2.0,
        heading_tolerance_rad: float = math.radians(3.0),
        done_confirmations: int = 5,
    ):
        self.motor_duty = float(
            motor_duty
        )

        self.heading_kp = float(
            heading_kp
        )

        self.cross_track_kp = float(
            cross_track_kp
        )

        self.heading_deadband_rad = float(
            heading_deadband_rad
        )

        self.cross_track_max_deg = float(
            cross_track_max_deg
        )

        self.max_steering_deg = float(
            max_steering_deg
        )

        self.max_steering_change_deg = float(
            max_steering_change_deg
        )

        self.line_tolerance_cm = float(
            line_tolerance_cm
        )

        self.heading_tolerance_rad = float(
            heading_tolerance_rad
        )

        self.done_confirmations = int(
            done_confirmations
        )

        self.reference_x = 0.0
        self.reference_y = 0.0
        self.target_theta = 0.0

        self._previous_steering_deg = 0.0
        self._alignment_count = 0
        self.active = False

    # ============================================================
    # Start and reset
    # ============================================================

    def start(
        self,
        reference_x: float,
        reference_y: float,
        target_theta: float,
        initial_steering_deg: float = 0.0,
    ) -> None:
        """
        Begin following the specified infinite straight line.
        """

        self.reference_x = float(
            reference_x
        )

        self.reference_y = float(
            reference_y
        )

        self.target_theta = self._normalize_angle(
            target_theta
        )

        self._previous_steering_deg = max(
            -self.max_steering_deg,
            min(
                self.max_steering_deg,
                initial_steering_deg,
            ),
        )

        self._alignment_count = 0
        self.active = True

        print("\n=== STRAIGHT RECOVERY STARTED ===")
        print(
            f"Reference point: "
            f"x={self.reference_x:.2f}, "
            f"y={self.reference_y:.2f}"
        )
        print(
            f"Target heading: "
            f"{math.degrees(self.target_theta):+.2f}°"
        )

    def reset(self) -> None:
        self._previous_steering_deg = 0.0
        self._alignment_count = 0
        self.active = False

    # ============================================================
    # Control
    # ============================================================

    def step(
        self,
        car,
        robot,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> tuple[float, bool]:
        """
        Execute one straight-recovery control step.

        Returns:
            steering_deg
            aligned
        """

        if not self.active:
            return 0.0, True

        (
            line_error_cm,
            heading_error_rad,
        ) = self.get_errors(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
        )

        steering_deg = self._calculate_steering(
            line_error_cm=line_error_cm,
            heading_error_rad=heading_error_rad,
        )

        self._command_robot(
            car=car,
            robot=robot,
            steering_deg=steering_deg,
        )

        aligned_now = (
            abs(line_error_cm)
            <= self.line_tolerance_cm
            and
            abs(heading_error_rad)
            <= self.heading_tolerance_rad
        )

        if aligned_now:
            self._alignment_count += 1
        else:
            self._alignment_count = 0

        aligned = (
            self._alignment_count
            >= self.done_confirmations
        )

        if aligned:
            self.active = False

            print("\n=== STRAIGHT RECOVERY COMPLETED ===")
            print(
                f"Final line error: "
                f"{line_error_cm:+.2f} cm"
            )
            print(
                f"Final heading error: "
                f"{math.degrees(heading_error_rad):+.2f}°"
            )

        return steering_deg, aligned

    def _calculate_steering(
        self,
        line_error_cm: float,
        heading_error_rad: float,
    ) -> float:
        """
        Combine heading correction and lateral correction.
        """

        if (
            abs(heading_error_rad)
            < self.heading_deadband_rad
        ):
            heading_correction_deg = 0.0
        else:
            heading_correction_deg = (
                self.heading_kp
                * math.degrees(
                    heading_error_rad
                )
            )

        cross_track_correction_deg = (
            self.cross_track_kp
            * line_error_cm
        )

        cross_track_correction_deg = max(
            -self.cross_track_max_deg,
            min(
                self.cross_track_max_deg,
                cross_track_correction_deg,
            ),
        )

        desired_steering_deg = (
            heading_correction_deg
            + cross_track_correction_deg
        )

        desired_steering_deg = max(
            -self.max_steering_deg,
            min(
                self.max_steering_deg,
                desired_steering_deg,
            ),
        )

        minimum_steering_deg = (
            self._previous_steering_deg
            - self.max_steering_change_deg
        )

        maximum_steering_deg = (
            self._previous_steering_deg
            + self.max_steering_change_deg
        )

        steering_deg = max(
            minimum_steering_deg,
            min(
                maximum_steering_deg,
                desired_steering_deg,
            ),
        )

        self._previous_steering_deg = (
            steering_deg
        )

        return steering_deg

    # ============================================================
    # Reference errors
    # ============================================================

    def get_errors(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> tuple[float, float]:
        """
        Return:
            signed lateral error in cm;
            signed heading error in radians.
        """

        dx = robot_x - self.reference_x
        dy = robot_y - self.reference_y

        line_error_cm = (
            -dx * math.sin(
                self.target_theta
            )
            + dy * math.cos(
                self.target_theta
            )
        )

        heading_error_rad = self._normalize_angle(
            robot_theta - self.target_theta
        )

        return (
            line_error_cm,
            heading_error_rad,
        )

    # ============================================================
    # Hardware command
    # ============================================================

    def _command_robot(
        self,
        car,
        robot,
        steering_deg: float,
    ) -> None:
        robot.update_steering(
            steering_deg
        )

        car.set_all(
            direction="f",
            speed=self.motor_duty,
            angle=steering_deg,
        )

    @staticmethod
    def _normalize_angle(
        angle_rad: float,
    ) -> float:
        return math.atan2(
            math.sin(angle_rad),
            math.cos(angle_rad),
        )