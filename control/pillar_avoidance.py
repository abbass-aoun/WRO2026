"""
control/pillar_avoidance.py

Reusable pillar-avoidance controller.

Responsibilities:
- Receive one pillar position and color.
- Generate a two-segment Bézier avoidance trajectory.
- Follow that trajectory using the existing steering PID.
- Report when the trajectory is complete.

This module does not:
- Read the camera.
- Read the encoders or IMU.
- Decide when the race is in a corner.
- Count laps.
- Change the normal straight reference.

Those responsibilities remain outside this controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from config import (
    PID_HEADING_W,
    PILLAR_CLEARANCE_CM,
    PILLAR_CLEAR_MARGIN_CM,
    PILLAR_DUTY,
    PILLAR_HANDOFF_CONFIRMATIONS,
    PILLAR_HANDOFF_MAX_HEADING_RAD,
    PILLAR_RECONNECT_CM,
    PILLAR_TRIGGER_CM,
    SERVO_MAX_DEG,
    PILLAR_APPROACH_CM,
    PILLAR_SWITCH_TO_NEXT_CM,
)

from trajectory.builder import (
    GREEN,
    RED,
    TrajectoryBuilder,
)


@dataclass(frozen=True)
class PillarTarget:
    """One pillar expressed in the global EKF coordinate system."""

    global_x_cm: float
    global_y_cm: float
    color: str

    def color_code(self) -> int:
        normalized_color = self.color.strip().lower()

        if normalized_color == "red":
            return RED

        if normalized_color == "green":
            return GREEN

        raise ValueError(
            f"Unsupported pillar color: {self.color!r}. "
            "Expected 'red' or 'green'."
        )


class PillarAvoidanceController:
    """
    Build and follow one pillar-avoidance trajectory.

    One instance can be reused for multiple pillars by calling
    start() again after the previous maneuver finishes.
    """

    def __init__(
        self,
        steering_pid,
        clearance_cm: float = PILLAR_CLEARANCE_CM,
        reconnect_cm: float = PILLAR_RECONNECT_CM,
        trigger_cm: float = PILLAR_TRIGGER_CM,
        motor_duty: float = PILLAR_DUTY,
    ):
        self.steering_pid = steering_pid

        self.clearance_cm = float(clearance_cm)
        self.reconnect_cm = float(reconnect_cm)
        self.trigger_cm = float(trigger_cm)
        self.motor_duty = float(motor_duty)

        self.trajectory = None
        self.near_s = 0.0
        self.active = False
        self._handoff_count = 0

        self.active_pillar: Optional[PillarTarget] = None
        self.next_pillar: Optional[PillarTarget] = None
        self.target_theta = 0.0
        self.straight_ref_x = 0.0
        self.straight_ref_y = 0.0

    # ============================================================
    # Public state
    # ============================================================

    @property
    def remaining_cm(self) -> float:
        """Return remaining trajectory distance."""

        if self.trajectory is None:
            return 0.0

        return max(
            0.0,
            self.trajectory.total_length - self.near_s,
        )

    # ============================================================
    # Starting-condition check
    # ============================================================

    def should_start(
        self,
        pillar: PillarTarget,
        robot_x: float,
        robot_y: float,
        target_theta: float,
    ) -> bool:
        """
        Return True when the pillar is ahead and within trigger range.

        This will be used during final integration with the camera.
        The standalone test starts the maneuver directly.
        """

        if self.active:
            return False

        dx = pillar.global_x_cm - robot_x
        dy = pillar.global_y_cm - robot_y

        forward_distance = (
            dx * math.cos(target_theta)
            + dy * math.sin(target_theta)
        )

        direct_distance = math.hypot(dx, dy)

        return (
            forward_distance > 0.0
            and direct_distance <= self.trigger_cm
        )

    # ============================================================
    # Start avoidance
    # ============================================================

    def start(
        self,
        pillar: PillarTarget,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        target_theta: float,
        straight_ref_x: float,
        straight_ref_y: float,
        next_pillar: Optional[PillarTarget] = None,
    ) -> None:
        """
        Generate a new avoidance trajectory.

        straight_ref_x/straight_ref_y identify the original desired
        straight line. The reconnect point is placed on that line,
        rather than simply using the pillar's lateral position.
        """

        self.reset()

        # Validate the color before beginning movement.
        pillar.color_code()

        self.active_pillar = pillar
        self.next_pillar = next_pillar

        self.target_theta = self._normalize_angle(
            target_theta
        )

        self.straight_ref_x = float(straight_ref_x)
        self.straight_ref_y = float(straight_ref_y)

        self.trajectory = self._build_trajectory(
            pillar=pillar,
            next_pillar=next_pillar,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
        )

        self.near_s = 0.0
        self._handoff_count = 0
        self.active = True

        self.steering_pid.reset()

        passing_side = (
            "RIGHT"
            if pillar.color.lower() == "red"
            else "LEFT"
        )

        print("\n=== PILLAR AVOIDANCE STARTED ===")
        print(
            f"Pillar: color={pillar.color}, "
            f"x={pillar.global_x_cm:.1f}, "
            f"y={pillar.global_y_cm:.1f}"
        )
        print(f"Passing side: {passing_side}")
        print(
            f"Trajectory length: "
            f"{self.trajectory.total_length:.1f} cm"
        )

        if next_pillar is not None:
            print(
                "Next pillar prepared: "
                f"color={next_pillar.color}, "
                f"x={next_pillar.global_x_cm:.1f}, "
                f"y={next_pillar.global_y_cm:.1f}"
            )

    # ============================================================
    # One control-loop step
    # ============================================================

    def step(
        self,
        car,
        robot,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        allow_handoff: bool = True,
    ) -> tuple[float, bool]:
        """
        Follow the active pillar trajectory for one control-loop step.

        Returns:
            steering_deg
            handoff_ready

        handoff_ready=True does not mean that the robot is already
        centered on the original straight.

        It means:
        - the active pillar is safely behind;
        - the robot heading is safe enough for the straight
        controller to take over.

        For multiple nearby pillars, allow_handoff can remain False
        until the final pillar has been cleared.
        """

        if not self.active or self.trajectory is None:
            return 0.0, True

        # --------------------------------------------------------
        # Find the nearest position on the active Bézier path
        # --------------------------------------------------------
        self.near_s = self.trajectory.find_closest(
            robot_x,
            robot_y,
            self.near_s,
        )

        path_x, path_y = self.trajectory.get_point(
            self.near_s
        )

        tangent_x, tangent_y = self.trajectory.get_tangent(
            self.near_s
        )

        path_theta = math.atan2(
            tangent_y,
            tangent_x,
        )

        # --------------------------------------------------------
        # Calculate path-following errors
        # --------------------------------------------------------
        cross_track_error = self._calculate_cross_track_error(
            robot_x=robot_x,
            robot_y=robot_y,
            path_x=path_x,
            path_y=path_y,
            path_theta=path_theta,
        )

        heading_error = self._normalize_angle(
            robot_theta - path_theta
        )

        combined_error = (
            cross_track_error
            + PID_HEADING_W * heading_error
        )

        # --------------------------------------------------------
        # Calculate and apply pillar-avoidance steering
        # --------------------------------------------------------
        steering_deg = self.steering_pid._compute(
            combined_error
        )

        steering_deg = self._clamp_steering(
            steering_deg
        )

        robot.update_steering(
            steering_deg
        )

        car.set_all(
            direction="f",
            speed=self.motor_duty,
            angle=steering_deg,
        )

        # ============================================================
        # Intermediate pillar completion
        # ============================================================

        # When another pillar is waiting, do not hand control back
        # to straight steering. Finish near the approach point and
        # let the test manager immediately start the next trajectory.
        if self.next_pillar is not None:

            forward_x = math.cos(self.target_theta)
            forward_y = math.sin(self.target_theta)

            pillar_behind_cm = (
                (robot_x - self.active_pillar.global_x_cm) * forward_x
                + (robot_y - self.active_pillar.global_y_cm) * forward_y
            )

            ready_for_next_pillar = (
                self.remaining_cm <= PILLAR_SWITCH_TO_NEXT_CM
                and pillar_behind_cm >= PILLAR_CLEAR_MARGIN_CM
            )

            if ready_for_next_pillar:
                self.active = False
                self.steering_pid.reset()

                print(
                    "\n=== CURRENT PILLAR TRAJECTORY COMPLETED ==="
                )
                print(
                    f"Pillar behind: {pillar_behind_cm:.1f} cm"
                )
                print(
                    f"Trajectory remaining: {self.remaining_cm:.1f} cm"
                )
                print(
                    "Starting the next pillar trajectory "
                    "without straight recovery."
                )

            return steering_deg, ready_for_next_pillar

        # --------------------------------------------------------
        # Check whether straight steering may take over
        # --------------------------------------------------------
        (
            safe_to_handoff,
            pillar_behind_cm,
            straight_heading_error_rad,
        ) = self._get_handoff_status(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_theta=robot_theta,
        )

        if allow_handoff and safe_to_handoff:
            self._handoff_count += 1
        else:
            self._handoff_count = 0

        handoff_ready = (
            self._handoff_count
            >= PILLAR_HANDOFF_CONFIRMATIONS
        )

        if handoff_ready:
            self.active = False
            self.steering_pid.reset()

            print("\n=== PILLAR SAFELY CLEARED ===")
            print(
                f"Pillar behind robot by: "
                f"{pillar_behind_cm:.2f} cm"
            )
            print(
                f"Heading error at handoff: "
                f"{math.degrees(straight_heading_error_rad):+.2f}°"
            )
            print(
                "Handing control back to straight steering."
            )

        return steering_deg, handoff_ready


    def _get_handoff_status(
        self,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ) -> tuple[bool, float, float]:
        """
        Check whether the active pillar is safely behind the robot.

        Returns:
            safe_to_handoff
            pillar_behind_cm
            heading_error_rad
        """

        if self.active_pillar is None:
            return False, 0.0, 0.0

        forward_x = math.cos(
            self.target_theta
        )

        forward_y = math.sin(
            self.target_theta
        )

        # Vector from robot to pillar.
        dx = (
            self.active_pillar.global_x_cm
            - robot_x
        )

        dy = (
            self.active_pillar.global_y_cm
            - robot_y
        )

        # Positive means pillar is ahead.
        # Negative means pillar is behind.
        pillar_forward_from_robot_cm = (
            dx * forward_x
            + dy * forward_y
        )

        pillar_behind_cm = (
            -pillar_forward_from_robot_cm
        )

        heading_error_rad = self._normalize_angle(
            robot_theta - self.target_theta
        )

        pillar_is_safely_behind = (
            pillar_behind_cm
            >= PILLAR_CLEAR_MARGIN_CM
        )

        heading_is_safe = (
            abs(heading_error_rad)
            <= PILLAR_HANDOFF_MAX_HEADING_RAD
        )

        safe_to_handoff = (
            pillar_is_safely_behind
            and heading_is_safe
        )

        return (
            safe_to_handoff,
            pillar_behind_cm,
            heading_error_rad,
        )


    # ============================================================
    # Reset
    # ============================================================

    def reset(self) -> None:
        """Clear the current maneuver."""

        self.trajectory = None
        self.near_s = 0.0
        self.active = False

        self.active_pillar = None
        self.next_pillar = None

        self._completion_count = 0
        
        self.steering_pid.reset()

    # ============================================================
    # Trajectory generation
    # ============================================================

    def _build_trajectory(
        self,
        pillar: PillarTarget,
        next_pillar: Optional[PillarTarget],
        robot_x: float,
        robot_y: float,
        robot_theta: float,
    ):
        """
        Build one pillar-avoidance trajectory.

        When next_pillar is None:
            reconnect to the original straight after the pillar.

        When next_pillar exists:
            end before the next pillar, already positioned on
            the side required to pass it.
        """

        forward_x = math.cos(
            self.target_theta
        )

        forward_y = math.sin(
            self.target_theta
        )

        # Left-facing normal of the original straight.
        left_x = -forward_y
        left_y = forward_x

        # ========================================================
        # Another pillar follows immediately
        # ========================================================
        if next_pillar is not None:

            next_color_code = next_pillar.color_code()

            # Green -> pass LEFT
            # Red   -> pass RIGHT
            next_side = (
                1.0
                if next_color_code == GREEN
                else -1.0
            )

            # End before pillar 2 and already on the side
            # required for avoiding it.
            end_x = (
                next_pillar.global_x_cm
                + next_side
                * left_x
                * self.clearance_cm
                - PILLAR_APPROACH_CM
                * forward_x
            )

            end_y = (
                next_pillar.global_y_cm
                + next_side
                * left_y
                * self.clearance_cm
                - PILLAR_APPROACH_CM
                * forward_y
            )

        # ========================================================
        # Final pillar in the sequence
        # ========================================================
        else:
            pillar_dx = (
                pillar.global_x_cm
                - self.straight_ref_x
            )

            pillar_dy = (
                pillar.global_y_cm
                - self.straight_ref_y
            )

            # Pillar's progress along the original straight.
            pillar_progress_cm = (
                pillar_dx * forward_x
                + pillar_dy * forward_y
            )

            reconnect_progress_cm = (
                pillar_progress_cm
                + self.reconnect_cm
            )

            # Final endpoint is back on the original reference line.
            end_x = (
                self.straight_ref_x
                + reconnect_progress_cm
                * forward_x
            )

            end_y = (
                self.straight_ref_y
                + reconnect_progress_cm
                * forward_y
            )

        return TrajectoryBuilder.pillar_swerve(
            start_x=robot_x,
            start_y=robot_y,
            start_theta=robot_theta,

            pillar_x=pillar.global_x_cm,
            pillar_y=pillar.global_y_cm,
            pillar_color=pillar.color_code(),

            end_x=end_x,
            end_y=end_y,
            end_theta=self.target_theta,

            clearance=self.clearance_cm,
        )

    
    # ============================================================
    # Error calculations
    # ============================================================

    @staticmethod
    def _calculate_cross_track_error(
        robot_x: float,
        robot_y: float,
        path_x: float,
        path_y: float,
        path_theta: float,
    ) -> float:
        """Signed perpendicular distance from robot to path."""

        return (
            -math.sin(path_theta)
            * (robot_x - path_x)
            + math.cos(path_theta)
            * (robot_y - path_y)
        )

    @staticmethod
    def _normalize_angle(angle_rad: float) -> float:
        """Normalize an angle to [-pi, +pi)."""

        return math.atan2(
            math.sin(angle_rad),
            math.cos(angle_rad),
        )

    @staticmethod
    def _clamp_steering(
        steering_deg: float,
    ) -> float:
        """Apply the physical steering limit."""

        return max(
            -SERVO_MAX_DEG,
            min(
                SERVO_MAX_DEG,
                steering_deg,
            ),
        )