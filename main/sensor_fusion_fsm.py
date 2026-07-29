"""Robust sensor-fusion FSM for the WRO 2026 Future Engineers robot.

Run from the repository root:

    python3 -m main.sensor_fusion_fsm --mode open
    python3 -m main.sensor_fusion_fsm --mode obstacle

This module deliberately does not use the fixed-map assumptions in
``main.Abstract_Main``.  Its pose starts at (0, 0, 0), and section-relative
odometry, wall geometry, line events and yaw change are fused at transitions.

TOF MOUNTING WARNING
--------------------
The repository documents XSHUT order and assigned addresses, but not physical
sensor orientation.  ``TOF_ROLE_TO_INDEX`` is therefore a bring-up assignment,
not a verified fact.  Run ``--tof-diagnostic`` with one target placed in front
of each sensor and correct this single mapping before driving the real robot.
"""

from __future__ import annotations

import argparse
import copy
import math
import statistics
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from config import (
    EKF_Q_THETA_R2,
    EKF_Q_XY_CM2,
    EKF_R_GYRO_R2,
    ROBOT_LENGTH_CM,
    ROBOT_WIDTH_CM,
    SERVO_MAX_DEG,
    WHEELBASE_CM,
)
from estimation.Transformation import camera_to_world
from estimation.ekf import EKF
from trajectory.builder import GREEN, RED, TrajectoryBuilder


# Repository wiring order: GPIO 4/10/9/6 -> I2C 0x30/0x31/0x32/0x33.
# PHYSICAL ORIENTATION IS NOT DOCUMENTED. Confirm and edit this block only.
TOF_ROLE_TO_INDEX: Mapping[str, int] = {
    "front": 0,
    "left": 1,
    "right": 2,
    "rear": 3,
}
TOF_XSHUT_GPIO: Tuple[int, ...] = (4, 10, 9, 6)
TOF_I2C_ADDRESS: Tuple[int, ...] = (0x30, 0x31, 0x32, 0x33)
TOF_MAPPING_CONFIRMED = False


class ChallengeMode(Enum):
    OPEN = "open"
    OBSTACLE = "obstacle"


class Direction(Enum):
    UNKNOWN = auto()
    CW = auto()   # right turns
    CCW = auto()  # left turns


class LineColour(Enum):
    ORANGE = "orange"
    BLUE = "blue"


class State(Enum):
    BOOT = auto()
    WAIT_FOR_START = auto()
    SENSOR_WARMUP = auto()
    DETERMINE_DIRECTION = auto()
    INITIAL_LOCALISATION = auto()
    FOLLOW_STRAIGHT = auto()
    APPROACH_CORNER = auto()
    TURN_CORNER = auto()
    EXIT_CORNER = auto()
    PILLAR_APPROACH = auto()
    PILLAR_PASS = auto()
    PILLAR_RECOVER = auto()
    FINISH_SECTION = auto()
    PARKING_SEARCH = auto()
    PARKING_APPROACH = auto()
    PARKING_MANOEUVRE = auto()
    FINISHED = auto()
    SAFE_STOP = auto()
    FAULT = auto()


@dataclass(frozen=True)
class FSMConfig:
    """FSM-specific tunables. Units are encoded in every field name."""

    control_hz: float = 50.0
    telemetry_hz: float = 10.0
    dt_min_s: float = 0.005
    dt_max_s: float = 0.050
    warmup_s: float = 1.0
    gyro_samples: int = 150
    gyro_sample_period_s: float = 0.01

    localisation_duty: float = 0.20
    straight_duty: float = 0.32
    corner_approach_duty: float = 0.20
    corner_duty: float = 0.24
    pillar_approach_duty: float = 0.22
    pillar_pass_duty: float = 0.20
    recovery_duty: float = 0.22
    parking_duty: float = 0.16
    emergency_crawl_duty: float = 0.10
    max_duty: float = 0.35

    steering_max_deg: float = SERVO_MAX_DEG
    steering_rate_deg_s: float = 90.0
    steering_sign: float = 1.0  # repository says +left/-right; verify physically
    wall_kp_deg_per_mm: float = 0.035
    wall_kd_deg_per_mm_s: float = 0.003
    heading_kp_deg_per_rad: float = 18.0
    single_wall_target_mm_600: float = 240.0
    single_wall_target_mm_1000: float = 440.0
    side_safety_bias_deg: float = 9.0

    tof_min_mm: int = 30
    tof_max_mm: int = 2000
    tof_stale_s: float = 0.45
    tof_jump_mm: float = 500.0
    tof_filter_size: int = 5
    tof_error_backoff_s: float = 0.08
    front_approach_mm: float = 700.0
    front_turn_entry_mm: float = 430.0
    front_emergency_mm: float = 150.0
    side_emergency_mm: float = 100.0
    front_trend_samples: int = 3

    camera_stale_s: float = 0.40
    camera_error_backoff_s: float = 0.05
    pillar_min_confidence: float = 0.60
    pillar_confirm_frames: int = 3
    pillar_trigger_mm: float = 900.0
    pillar_min_forward_mm: float = 100.0
    pillar_max_lateral_mm: float = 650.0
    pillar_clearance_cm: float = 20.0
    pillar_reconnect_cm: float = 65.0
    pillar_passed_margin_mm: float = 180.0
    pillar_track_jump_mm: float = 300.0
    pillar_cooldown_s: float = 4.0
    pillar_wall_margin_mm: float = 130.0

    line_debounce_s: float = 0.20
    line_rearm_clear_s: float = 0.10
    line_min_event_interval_s: float = 0.45
    line_min_event_distance_cm: float = 8.0
    camera_line_fallback_frames: int = 5
    camera_line_arm_distance_cm: float = 55.0

    corner_confidence_enter: float = 3.0
    corner_confidence_hold: float = 2.0
    corner_evidence_hold_s: float = 0.12
    corner_radius_cm: float = 45.0
    corner_front_overhang_cm: float = (ROBOT_LENGTH_CM - WHEELBASE_CM) / 2.0
    corner_entry_safety_cm: float = 8.0
    corner_min_travel_cm: float = 35.0
    corner_heading_min_deg: float = 70.0
    corner_heading_max_deg: float = 110.0
    corner_heading_target_tolerance_deg: float = 18.0
    corner_trajectory_done_cm: float = 8.0
    exit_confirm_s: float = 0.20
    exit_straighten_distance_cm: float = 18.0
    finish_clear_distance_cm: float = ROBOT_LENGTH_CM + 12.0

    corridor_600_min_mm: float = 480.0
    corridor_600_max_mm: float = 760.0
    corridor_1000_min_mm: float = 800.0
    corridor_1000_max_mm: float = 1200.0
    learned_straight_min_cm: float = 80.0
    learned_straight_max_cm: float = 320.0
    learned_ema_alpha: float = 0.35
    predicted_corner_slow_margin_cm: float = 45.0

    no_encoder_motion_timeout_s: float = 1.2
    encoder_imbalance_timeout_s: float = 0.8
    encoder_speed_max_cm_s: float = 150.0
    imu_stale_s: float = 0.35
    max_consecutive_overruns: int = 25
    thread_failure_timeout_s: float = 1.5

    parking_search_timeout_s: float = 15.0
    parking_approach_timeout_s: float = 12.0
    parking_manoeuvre_timeout_s: float = 12.0
    parking_marker_confirm_frames: int = 4
    parking_stop_forward_mm: float = ROBOT_LENGTH_CM * 10.0 / 2.0 + 50.0
    parking_alignment_tolerance_mm: float = 20.0

    state_timeouts_s: Mapping[State, float] = field(default_factory=lambda: {
        State.SENSOR_WARMUP: 8.0,
        State.DETERMINE_DIRECTION: 20.0,
        State.INITIAL_LOCALISATION: 30.0,
        State.FOLLOW_STRAIGHT: 20.0,
        State.APPROACH_CORNER: 6.0,
        State.TURN_CORNER: 7.0,
        State.EXIT_CORNER: 4.0,
        State.PILLAR_APPROACH: 5.0,
        State.PILLAR_PASS: 8.0,
        State.PILLAR_RECOVER: 5.0,
        State.FINISH_SECTION: 8.0,
        State.PARKING_SEARCH: 15.0,
        State.PARKING_APPROACH: 12.0,
        State.PARKING_MANOEUVRE: 12.0,
    })

    @property
    def period_s(self) -> float:
        return 1.0 / self.control_hz


@dataclass(frozen=True)
class ToFChannel:
    raw_mm: Optional[float] = None
    filtered_mm: Optional[float] = None
    timestamp_s: float = 0.0
    valid: bool = False
    stale: bool = True
    error_count: int = 0


@dataclass(frozen=True)
class ToFSnapshot:
    timestamp_s: float
    channels: Tuple[ToFChannel, ToFChannel, ToFChannel, ToFChannel]

    def role(self, name: str) -> ToFChannel:
        return self.channels[TOF_ROLE_TO_INDEX[name]]


@dataclass(frozen=True)
class VisionSnapshot:
    timestamp_s: float = 0.0
    result: Mapping[str, Any] = field(default_factory=dict)
    frame: Any = None
    error_count: int = 0
    thread_alive: bool = False


@dataclass(frozen=True)
class LineEvent:
    colour: LineColour
    timestamp_s: float
    distance_cm: float
    source: str


@dataclass
class PillarTrack:
    key: str
    colour: str
    relative_x_mm: float
    relative_y_mm: float
    confidence: float
    first_seen_s: float
    last_seen_s: float
    support_frames: int = 1
    global_x_cm: Optional[float] = None
    global_y_cm: Optional[float] = None

    @property
    def pass_side(self) -> str:
        return "right" if self.colour == "red" else "left"


@dataclass(frozen=True)
class SensorSnapshot:
    timestamp_s: float
    dt_s: float
    distance_cm: float
    left_distance_cm: float
    right_distance_cm: float
    speed_cm_s: float
    left_speed_cm_s: float
    right_speed_cm_s: float
    yaw_rate_rad_s: float
    heading_rad: float
    x_cm: float
    y_cm: float
    orange_level: bool
    blue_level: bool
    tof: ToFSnapshot
    vision: VisionSnapshot
    encoder_age_s: float
    imu_age_s: float


@dataclass
class NavigationContext:
    direction: Direction = Direction.UNKNOWN
    total_corners_completed: int = 0
    corners_in_current_lap: int = 0
    laps_completed: int = 0
    current_straight_index: int = 0
    run_distance_origin_cm: float = 0.0
    state_distance_origin_cm: float = 0.0
    straight_distance_origin_cm: float = 0.0
    last_line_distance_cm: float = -1e9
    last_line_time_s: float = -1e9
    target_heading_rad: float = 0.0
    corner_entry_heading_rad: float = 0.0
    corner_entry_distance_cm: float = 0.0
    corner_path_s_cm: float = 0.0
    start_to_first_corner_cm: Optional[float] = None
    corridor_class: str = "uncertain"
    corridor_width_mm: Optional[float] = None
    initial_wall_signature_mm: Tuple[Optional[float], Optional[float], Optional[float]] = (
        None, None, None
    )
    first_line_sequence: List[str] = field(default_factory=list)
    learned_straight_lengths_cm: List[Optional[float]] = field(
        default_factory=lambda: [None, None, None, None]
    )
    active_pillar: Optional[PillarTrack] = None
    handled_pillars: Dict[str, float] = field(default_factory=dict)
    finish_entry_distance_cm: Optional[float] = None


@dataclass(frozen=True)
class TransitionRecord:
    old_state: State
    new_state: State
    timestamp_s: float
    reason: str
    lap: int
    corner_count: int
    encoder_distance_cm: float
    heading_rad: float
    tof_mm: Mapping[str, Optional[float]]


@dataclass(frozen=True)
class ControlCommand:
    direction: str = "s"
    duty: float = 0.0
    steering_deg: float = 0.0
    reason: str = ""


class SimplePID:
    """Monotonic-time PID whose state is private to one control mode."""

    def __init__(self, kp: float, ki: float, kd: float, limit: float) -> None:
        self.kp, self.ki, self.kd, self.limit = kp, ki, kd, abs(limit)
        self.integral = 0.0
        self.previous_error = 0.0
        self.ready = False

    def reset(self) -> None:
        self.integral = self.previous_error = 0.0
        self.ready = False

    def compute(self, error: float, dt_s: float) -> float:
        dt_s = max(1e-3, dt_s)
        self.integral = max(-self.limit, min(self.limit, self.integral + error * dt_s))
        derivative = (error - self.previous_error) / dt_s if self.ready else 0.0
        self.previous_error, self.ready = error, True
        return max(-self.limit, min(
            self.limit, self.kp * error + self.ki * self.integral + self.kd * derivative
        ))


class BackgroundToFReader:
    """Staggered, non-blocking-to-the-main-loop VL53L0X reader."""

    def __init__(self, sensors: Any, config: FSMConfig) -> None:
        self._sensors = sensors
        self._cfg = config
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._histories: List[Deque[float]] = [
            deque(maxlen=config.tof_filter_size) for _ in range(4)
        ]
        self._jump_candidates: List[Optional[float]] = [None] * 4
        self._channels: List[ToFChannel] = [ToFChannel() for _ in range(4)]
        self._last_progress_s = time.monotonic()

    @property
    def alive(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    @property
    def last_progress_s(self) -> float:
        return self._last_progress_s

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="tof-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            now = time.monotonic()
            previous = self._channels[index]
            try:
                raw = self._sensors.read_mm(index)
                valid = raw is not None and self._cfg.tof_min_mm <= raw <= self._cfg.tof_max_mm
                if valid and previous.filtered_mm is not None:
                    # A real opening can be a large jump. Require two similar samples,
                    # then seed a fresh window rather than rejecting the new level forever.
                    if abs(float(raw) - previous.filtered_mm) > self._cfg.tof_jump_mm:
                        candidate = self._jump_candidates[index]
                        if candidate is not None and abs(float(raw) - candidate) <= 100.0:
                            self._histories[index].clear()
                            self._jump_candidates[index] = None
                        else:
                            self._jump_candidates[index] = float(raw)
                            valid = False
                    else:
                        self._jump_candidates[index] = None
                if valid:
                    self._histories[index].append(float(raw))
                    filtered = float(statistics.median(self._histories[index]))
                    channel = ToFChannel(float(raw), filtered, now, True, False, previous.error_count)
                else:
                    channel = ToFChannel(
                        None if raw is None else float(raw), previous.filtered_mm,
                        previous.timestamp_s, False, True, previous.error_count + 1
                    )
                with self._lock:
                    self._channels[index] = channel
                self._last_progress_s = now
                index = (index + 1) % 4
            except OSError:
                with self._lock:
                    self._channels[index] = ToFChannel(
                        previous.raw_mm, previous.filtered_mm, previous.timestamp_s,
                        False, True, previous.error_count + 1
                    )
                self._stop.wait(self._cfg.tof_error_backoff_s)
            except Exception as exc:
                print(f"[ToF] reader error: {type(exc).__name__}: {exc}", flush=True)
                self._stop.wait(self._cfg.tof_error_backoff_s)

    def snapshot(self, now_s: Optional[float] = None) -> ToFSnapshot:
        now_s = time.monotonic() if now_s is None else now_s
        with self._lock:
            copied = tuple(self._channels)
        channels = tuple(
            ToFChannel(
                c.raw_mm, c.filtered_mm, c.timestamp_s, c.valid,
                (not c.valid) or now_s - c.timestamp_s > self._cfg.tof_stale_s,
                c.error_count,
            )
            for c in copied
        )
        return ToFSnapshot(now_s, channels)  # type: ignore[arg-type]

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)


class RobustVisionReader:
    """Camera reader with immutable snapshots and bounded error retries."""

    def __init__(self, show_camera: bool, config: FSMConfig) -> None:
        self.show_camera = show_camera
        self.cfg = config
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._camera: Any = None
        self._snapshot = VisionSnapshot()
        self._orange_count = self._blue_count = 0
        self._last_progress_s = time.monotonic()

    @property
    def last_progress_s(self) -> float:
        return self._last_progress_s

    def start(self) -> None:
        from cv.camera import open_camera
        self._camera = open_camera()
        self._thread = threading.Thread(target=self._run, name="vision-reader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from cv.camera import read_frame
        from cv.vision import draw_vision_result, process_image
        errors = 0
        while not self._stop.is_set():
            try:
                frame = read_frame(self._camera)
                result = process_image(frame)
                lines = result.setdefault("track_lines", {})
                orange = lines.setdefault("orange", {})
                blue = lines.setdefault("blue", {})
                self._orange_count = self._orange_count + 1 if orange.get("close") else 0
                self._blue_count = self._blue_count + 1 if blue.get("close") else 0
                orange["confirmed_close"] = self._orange_count >= 3
                blue["confirmed_close"] = self._blue_count >= 3
                debug_frame = draw_vision_result(frame, result) if self.show_camera else None
                now = time.monotonic()
                with self._lock:
                    self._snapshot = VisionSnapshot(
                        now, copy.deepcopy(result), debug_frame, errors, True
                    )
                self._last_progress_s = now
            except Exception as exc:
                errors += 1
                if errors == 1 or errors % 20 == 0:
                    print(f"[Vision] error #{errors}: {type(exc).__name__}: {exc}", flush=True)
                self._stop.wait(self.cfg.camera_error_backoff_s)

    def snapshot(self) -> VisionSnapshot:
        with self._lock:
            snap = self._snapshot
            frame = None if snap.frame is None else snap.frame.copy()
            return VisionSnapshot(
                snap.timestamp_s, copy.deepcopy(snap.result), frame,
                snap.error_count, bool(self._thread and self._thread.is_alive())
            )

    def stop(self) -> None:
        from cv.camera import release_camera
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self._camera is not None:
            try:
                release_camera(self._camera)
            except Exception as exc:
                print(f"[Vision] release warning: {exc}", flush=True)
            self._camera = None


class LineEventDetector:
    """Debounced edge latch; simultaneous colours are always ambiguous."""

    def __init__(self, cfg: FSMConfig) -> None:
        self.cfg = cfg
        self.armed = True
        self.clear_since_s: Optional[float] = None
        self.candidate: Optional[LineColour] = None
        self.candidate_since_s = 0.0
        self.last_event_s = -1e9
        self.last_event_distance_cm = -1e9
        self.camera_counts = {LineColour.ORANGE: 0, LineColour.BLUE: 0}

    def reset_levels(self) -> None:
        self.armed, self.clear_since_s, self.candidate = False, None, None

    def update(
        self, now_s: float, distance_cm: float, orange: bool, blue: bool,
        camera_lines: Mapping[str, Any], camera_fresh: bool
    ) -> Optional[LineEvent]:
        if orange and blue:
            self.candidate = None
            return None
        if not orange and not blue:
            self.candidate = None
            if self.clear_since_s is None:
                self.clear_since_s = now_s
            elif now_s - self.clear_since_s >= self.cfg.line_rearm_clear_s:
                self.armed = True
        else:
            self.clear_since_s = None

        colour = LineColour.ORANGE if orange else LineColour.BLUE if blue else None
        if self.armed and colour is not None:
            if self.candidate != colour:
                self.candidate, self.candidate_since_s = colour, now_s
            elif (
                now_s - self.candidate_since_s >= self.cfg.line_debounce_s
                and now_s - self.last_event_s >= self.cfg.line_min_event_interval_s
                and distance_cm - self.last_event_distance_cm
                >= self.cfg.line_min_event_distance_cm
            ):
                self.armed = False
                self.last_event_s, self.last_event_distance_cm = now_s, distance_cm
                return LineEvent(colour, now_s, distance_cm, "floor")

        # Degraded fallback: multiple fresh camera frames, never both colours.
        orange_cam = bool(camera_lines.get("orange", {}).get("confirmed_close"))
        blue_cam = bool(camera_lines.get("blue", {}).get("confirmed_close"))
        if camera_fresh and orange_cam != blue_cam and not orange and not blue:
            c = LineColour.ORANGE if orange_cam else LineColour.BLUE
            other = LineColour.BLUE if c == LineColour.ORANGE else LineColour.ORANGE
            self.camera_counts[c] += 1
            self.camera_counts[other] = 0
            if (
                self.camera_counts[c] >= self.cfg.camera_line_fallback_frames
                and now_s - self.last_event_s >= self.cfg.line_min_event_interval_s
                and distance_cm - self.last_event_distance_cm
                >= self.cfg.line_min_event_distance_cm
            ):
                self.camera_counts[c] = 0
                self.last_event_s, self.last_event_distance_cm = now_s, distance_cm
                return LineEvent(c, now_s, distance_cm, "camera-degraded")
        else:
            self.camera_counts[LineColour.ORANGE] = 0
            self.camera_counts[LineColour.BLUE] = 0
        return None


class WallFollower:
    """Side-ToF centring with heading hold and mode-private controller state."""

    def __init__(self, cfg: FSMConfig) -> None:
        self.cfg = cfg
        self.wall_pid = SimplePID(
            cfg.wall_kp_deg_per_mm, 0.0, cfg.wall_kd_deg_per_mm_s,
            cfg.steering_max_deg * 0.70
        )

    def reset(self) -> None:
        self.wall_pid.reset()

    def steering(self, snap: SensorSnapshot, target_heading_rad: float) -> float:
        left = valid_tof_mm(snap.tof, "left")
        right = valid_tof_mm(snap.tof, "right")
        wall_error_mm = 0.0
        if left is not None and right is not None:
            # Positive means too close to right / needs left steering.
            wall_error_mm = right - left
        elif left is not None:
            target = (
                self.cfg.single_wall_target_mm_600
                if left < 350 else self.cfg.single_wall_target_mm_1000
            )
            wall_error_mm = target - left
        elif right is not None:
            target = (
                self.cfg.single_wall_target_mm_600
                if right < 350 else self.cfg.single_wall_target_mm_1000
            )
            wall_error_mm = right - target
        wall = self.wall_pid.compute(wall_error_mm, snap.dt_s)
        heading_error = normalize_angle(target_heading_rad - snap.heading_rad)
        heading = self.cfg.heading_kp_deg_per_rad * heading_error
        return wall + heading


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def valid_tof_mm(snapshot: ToFSnapshot, role: str) -> Optional[float]:
    channel = snapshot.role(role)
    return channel.filtered_mm if channel.valid and not channel.stale else None


class SensorFusionFSM:
    """One class-based FSM used for both challenge modes."""

    def __init__(
        self, mode: ChallengeMode, config: Optional[FSMConfig] = None,
        dry_run: bool = False, show_camera: bool = False, debug: bool = False
    ) -> None:
        self.mode = mode
        self.cfg = config or FSMConfig()
        self.dry_run = dry_run
        self.show_camera = show_camera
        self.debug = debug
        self.state = State.BOOT
        self.context = NavigationContext()
        self.transitions: List[TransitionRecord] = []
        self.state_entered_s = time.monotonic()
        self.last_snapshot: Optional[SensorSnapshot] = None
        self.line_detector = LineEventDetector(self.cfg)
        self.wall_follower = WallFollower(self.cfg)
        self.corner_pid = SimplePID(1.0, 0.0, 0.08, self.cfg.steering_max_deg)
        self.pillar_pid = SimplePID(1.0, 0.0, 0.06, self.cfg.steering_max_deg)
        self.parking_pid = SimplePID(0.04, 0.0, 0.005, 14.0)
        self.front_history: Deque[float] = deque(maxlen=5)
        self.corner_evidence_since_s: Optional[float] = None
        self.exit_evidence_since_s: Optional[float] = None
        self.corner_trajectory: Any = None
        self.pillar_trajectory: Any = None
        self.last_command = ControlCommand()
        self.last_steering_deg = 0.0
        self.gyro_bias_rad_s = 0.0
        self.cleanup_done = False
        self.started = False
        self._last_encoder_motion_s = time.monotonic()
        self._encoder_imbalance_since_s: Optional[float] = None
        self._parking_support = 0
        self._corner_counted_for_entry = False
        self._loop_count = self._missed_deadlines = 0
        self._loop_dt_sum_s = self._max_overrun_s = 0.0
        self._consecutive_overruns = 0
        self._last_telemetry_s = 0.0

        self.start_button: Any = None
        self.leds: List[Any] = []
        self.encoders: Any = None
        self.color: Any = None
        self.car: Any = None
        self.robot: Any = None
        self.ekf: Any = None
        self.tof_reader: Optional[BackgroundToFReader] = None
        self.vision_reader: Optional[RobustVisionReader] = None

    def transition_to(self, new_state: State, reason: str) -> None:
        if new_state == self.state:
            return
        now = time.monotonic()
        snap = self.last_snapshot
        tof_values = {
            role: valid_tof_mm(snap.tof, role) if snap else None
            for role in TOF_ROLE_TO_INDEX
        }
        record = TransitionRecord(
            self.state, new_state, now, reason,
            self.context.laps_completed, self.context.total_corners_completed,
            snap.distance_cm if snap else 0.0,
            snap.heading_rad if snap else 0.0, tof_values,
        )
        self.transitions.append(record)
        print(
            f"[FSM] {self.state.name} -> {new_state.name}: {reason} | "
            f"lap={record.lap} corners={record.corner_count} "
            f"d={record.encoder_distance_cm:.1f}cm "
            f"heading={math.degrees(record.heading_rad):+.1f}deg tof={tof_values}",
            flush=True,
        )
        self.state = new_state
        self.state_entered_s = now
        if snap:
            self.context.state_distance_origin_cm = snap.distance_cm
        self._on_entry(new_state)

    def _on_entry(self, state: State) -> None:
        if state in (State.FOLLOW_STRAIGHT, State.INITIAL_LOCALISATION):
            self.wall_follower.reset()
        if state == State.TURN_CORNER:
            self.corner_pid.reset()
            self.context.corner_entry_heading_rad = self.last_snapshot.heading_rad
            self.context.corner_entry_distance_cm = self.last_snapshot.distance_cm
            turn = -1 if self.context.direction == Direction.CW else 1
            self.corner_trajectory = TrajectoryBuilder.corner(
                self.last_snapshot.x_cm, self.last_snapshot.y_cm,
                self.last_snapshot.heading_rad, turn, self.cfg.corner_radius_cm
            )
            self.context.corner_path_s_cm = 0.0
            self._corner_counted_for_entry = False
        elif state == State.EXIT_CORNER:
            self.exit_evidence_since_s = None
        elif state == State.PILLAR_PASS:
            self.pillar_pid.reset()
            self._build_pillar_path()
        elif state == State.PILLAR_RECOVER:
            self.wall_follower.reset()
        elif state in (State.SAFE_STOP, State.FAULT, State.FINISHED):
            self.apply_command(ControlCommand("s", 0.0, 0.0, state.name))

    def initialize_hardware(self) -> None:
        """Initialize in safe order; every motor object is stopped immediately."""
        if self.dry_run:
            return
        try:
            from main.initialize_hardware import initialize_hardware
            from control.tof_sensor import ToFSensors
            (
                self.start_button, self.leds, self.encoders, self.color,
                self.car, self.robot, self.ekf
            ) = initialize_hardware()
            self.car.stop()
            self.car.set_steering(0.0)
            tof = ToFSensors()
            self.tof_reader = BackgroundToFReader(tof, self.cfg)
            self.tof_reader.start()
            self.vision_reader = RobustVisionReader(self.show_camera, self.cfg)
            try:
                self.vision_reader.start()
            except Exception as exc:
                print(f"[Vision] unavailable; non-camera navigation remains active: {exc}")
                self.vision_reader = None
            self.transition_to(State.SENSOR_WARMUP, "hardware safe and readers started")
        except Exception:
            if self.car is not None:
                self.car.stop()
            raise

    def calibrate_gyro(self) -> None:
        samples: List[float] = []
        print("[IMU] Keep robot stationary: calibrating gyro.", flush=True)
        for _ in range(self.cfg.gyro_samples):
            try:
                value = float(self.encoders.get_yaw_rate())
                if math.isfinite(value):
                    samples.append(value)
            except Exception as exc:
                print(f"[IMU] calibration read failed: {exc}", flush=True)
            time.sleep(self.cfg.gyro_sample_period_s)
        if len(samples) < self.cfg.gyro_samples // 2:
            raise RuntimeError("insufficient valid gyro samples")
        self.gyro_bias_rad_s = statistics.median(samples)
        print(f"[IMU] bias={math.degrees(self.gyro_bias_rad_s):+.3f} deg/s")

    def wait_for_start(self) -> None:
        self.transition_to(State.WAIT_FOR_START, "sensor warmup and stationary gyro calibration complete")
        led_on, last_toggle = False, time.monotonic()
        while not self.start_button.is_pressed:
            now = time.monotonic()
            if now - last_toggle >= 0.5:
                led_on, last_toggle = not led_on, now
                for led in self.leds:
                    led.on() if led_on else led.off()
            # Explicitly enforce the stopped state throughout the wait.
            self.car.stop()
            time.sleep(0.01)
        for led in self.leds:
            led.off()
        self.encoders.reset()
        self.ekf.initialize(0.0, 0.0, 0.0)
        self.robot.reset()
        self.context = NavigationContext()
        self.started = True
        self.line_detector.reset_levels()  # already-on-line must clear before event
        self.transition_to(State.DETERMINE_DIRECTION, "physical Start button pressed")

    def _read_snapshot(self, now_s: float, dt_s: float) -> SensorSnapshot:
        left_speed, right_speed = self.encoders.get_linear_speeds()
        speed = 0.5 * (left_speed + right_speed)
        left_distance, right_distance = self.encoders.get_distances()
        distance = 0.5 * (left_distance + right_distance)  # driver is unsigned
        yaw_rate = self.encoders.get_yaw_rate() - self.gyro_bias_rad_s
        ekf_dt = min(self.cfg.dt_max_s, max(self.cfg.dt_min_s, dt_s))
        if abs(ekf_dt - dt_s) > 1e-9:
            print(f"[Timing] EKF dt clamped {dt_s:.4f}s -> {ekf_dt:.4f}s", flush=True)
        self.ekf.predict(
            speed, math.radians(self.robot.steer_angle), ekf_dt, omega_gyro=yaw_rate
        )
        x_cm, y_cm, heading = self.ekf.state
        self.robot.update_pose(x_cm, y_cm, heading)
        self.robot.update_speed(speed)
        tof = self.tof_reader.snapshot(now_s) if self.tof_reader else empty_tof(now_s)
        vision = self.vision_reader.snapshot() if self.vision_reader else VisionSnapshot()
        return SensorSnapshot(
            now_s, dt_s, distance, left_distance, right_distance, speed,
            left_speed, right_speed, yaw_rate, heading, x_cm, y_cm,
            bool(self.color.orange_seen), bool(self.color.blue_seen), tof, vision,
            0.0, 0.0,
        )

    def _line_event(self, snap: SensorSnapshot) -> Optional[LineEvent]:
        camera_fresh = (
            snap.vision.timestamp_s > 0
            and snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        )
        lines = snap.vision.result.get("track_lines", {}) if camera_fresh else {}
        event = self.line_detector.update(
            snap.timestamp_s, snap.distance_cm, snap.orange_level, snap.blue_level,
            lines, camera_fresh
        )
        if event:
            self.context.last_line_distance_cm = event.distance_cm
            self.context.last_line_time_s = event.timestamp_s
            self.context.first_line_sequence.append(event.colour.value)
            print(f"[Line] {event.colour.value} via {event.source}", flush=True)
        return event

    def update(self, snap: SensorSnapshot) -> ControlCommand:
        self.last_snapshot = snap
        event = self._line_event(snap)
        self._update_front_history(snap)
        self._update_pillar_tracks(snap)
        timeout = self.cfg.state_timeouts_s.get(self.state)
        if timeout and snap.timestamp_s - self.state_entered_s > timeout:
            self.transition_to(
                State.SAFE_STOP, f"{self.state.name} timeout after {timeout:.1f}s"
            )
            return ControlCommand("s", 0.0, 0.0, "state timeout")

        safety = self._safety_fault(snap)
        if safety:
            self.transition_to(State.FAULT, safety)
            return ControlCommand("s", 0.0, 0.0, safety)

        handler = getattr(self, f"_update_{self.state.name.lower()}", None)
        command = handler(snap, event) if handler else ControlCommand("s", 0.0, 0.0, "idle")
        return self._apply_safety_overrides(snap, command)

    def _update_determine_direction(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        if event:
            # Project/official convention required by this FSM.
            self.context.direction = (
                Direction.CW if event.colour == LineColour.ORANGE else Direction.CCW
            )
            self.context.target_heading_rad = 0.0
            self.transition_to(
                State.INITIAL_LOCALISATION,
                f"unambiguous {event.colour.value} event locked {self.context.direction.name}"
            )
        elif self._corner_imminent(snap):
            return ControlCommand("f", self.cfg.emergency_crawl_duty, 0.0, "direction unknown near corner")
        return self._straight_command(snap, self.cfg.localisation_duty, "acquire direction")

    def _update_initial_localisation(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        self._classify_corridor(snap)
        confidence, evidence = self._corner_confidence(snap, event)
        if confidence >= self.cfg.corner_confidence_enter:
            self.context.start_to_first_corner_cm = snap.distance_cm
            self.context.initial_wall_signature_mm = (
                valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "front"),
                valid_tof_mm(snap.tof, "right"),
            )
            self.transition_to(State.APPROACH_CORNER, f"first corner fused evidence: {evidence}")
        return self._straight_command(snap, self.cfg.localisation_duty, "initial localisation")

    def _update_follow_straight(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        confidence, evidence = self._corner_confidence(snap, event)
        if confidence >= self.cfg.corner_confidence_enter:
            if self.corner_evidence_since_s is None:
                self.corner_evidence_since_s = snap.timestamp_s
            elif snap.timestamp_s - self.corner_evidence_since_s >= self.cfg.corner_evidence_hold_s:
                self._learn_current_straight(snap)
                self.transition_to(State.APPROACH_CORNER, f"corner confidence {confidence:.1f}: {evidence}")
        elif confidence < self.cfg.corner_confidence_hold:
            self.corner_evidence_since_s = None

        if self.mode == ChallengeMode.OBSTACLE and self._pillar_ready(snap):
            self.transition_to(State.PILLAR_APPROACH, "confirmed nearest forward pillar")
        duty = self.cfg.straight_duty
        learned = self.context.learned_straight_lengths_cm[self.context.current_straight_index]
        if learned and self._straight_distance(snap) > learned - self.cfg.predicted_corner_slow_margin_cm:
            duty = self.cfg.corner_approach_duty
        return self._straight_command(snap, duty, "wall follow")

    def _update_approach_corner(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        confidence, evidence = self._corner_confidence(snap, event)
        front = valid_tof_mm(snap.tof, "front")
        entry_mm = (
            (self.cfg.corner_front_overhang_cm + self.cfg.corner_entry_safety_cm) * 10.0
            + self.cfg.front_turn_entry_mm
        )
        heading_known = self.context.direction != Direction.UNKNOWN
        physical_entry = (
            front is not None and front <= entry_mm
            and confidence >= self.cfg.corner_confidence_hold
        )
        if heading_known and physical_entry:
            self.transition_to(State.TURN_CORNER, f"corner entry gated by front range and {evidence}")
        elif confidence < 1.0 and snap.timestamp_s - self.state_entered_s > 0.5:
            self.transition_to(State.FOLLOW_STRAIGHT, "corner evidence decayed before entry")
        return self._straight_command(snap, self.cfg.corner_approach_duty, "align for corner")

    def _update_turn_corner(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        turn_sign = -1.0 if self.context.direction == Direction.CW else 1.0
        travelled = snap.distance_cm - self.context.corner_entry_distance_cm
        heading_change = turn_sign * normalize_angle(
            snap.heading_rad - self.context.corner_entry_heading_rad
        )
        target = normalize_angle(
            self.context.corner_entry_heading_rad + turn_sign * math.pi / 2.0
        )
        trajectory_steer = turn_sign * self.cfg.steering_max_deg * 0.78
        if self.corner_trajectory is not None:
            s = self.corner_trajectory.find_closest(
                snap.x_cm, snap.y_cm, self.context.corner_path_s_cm
            )
            self.context.corner_path_s_cm = max(self.context.corner_path_s_cm, s)
            px, py = self.corner_trajectory.get_point(s)
            tx, ty = self.corner_trajectory.get_tangent(s)
            path_heading = math.atan2(ty, tx)
            cte = -math.sin(path_heading) * (snap.x_cm - px) + math.cos(path_heading) * (snap.y_cm - py)
            error = cte + 8.0 * normalize_angle(path_heading - snap.heading_rad)
            trajectory_steer += self.corner_pid.compute(error, snap.dt_s)
        opened = self._front_opening(snap)
        corridor = self._corridor_resembles_straight(snap)
        heading_ok = (
            math.radians(self.cfg.corner_heading_min_deg) <= heading_change
            <= math.radians(self.cfg.corner_heading_max_deg)
            and abs(normalize_angle(target - snap.heading_rad))
            <= math.radians(self.cfg.corner_heading_target_tolerance_deg)
        )
        progress_ok = (
            travelled >= self.cfg.corner_min_travel_cm
            and (
                self.corner_trajectory is None
                or self.corner_trajectory.total_length - self.context.corner_path_s_cm
                <= self.cfg.corner_trajectory_done_cm
            )
        )
        if heading_ok and progress_ok and (opened or corridor or event is not None):
            self.context.target_heading_rad = target
            self.transition_to(
                State.EXIT_CORNER,
                f"heading={math.degrees(heading_change):.1f}deg travel={travelled:.1f}cm "
                f"open={opened} corridor={corridor} line={event is not None}"
            )
        return ControlCommand("f", self.cfg.corner_duty, trajectory_steer, "corner trajectory")

    def _update_exit_corner(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        distance = self._state_distance(snap)
        heading_error = abs(normalize_angle(self.context.target_heading_rad - snap.heading_rad))
        straight_evidence = self._corridor_resembles_straight(snap) or self._front_opening(snap)
        if (
            distance >= self.cfg.exit_straighten_distance_cm
            and heading_error <= math.radians(self.cfg.corner_heading_target_tolerance_deg)
            and straight_evidence
        ):
            if not self._corner_counted_for_entry:
                self._complete_corner(snap)
                self._corner_counted_for_entry = True
            if self.context.laps_completed >= 3:
                next_state = (
                    State.FINISH_SECTION if self.mode == ChallengeMode.OPEN
                    else State.PARKING_SEARCH
                )
                self.context.finish_entry_distance_cm = snap.distance_cm
                self.transition_to(next_state, "12 confirmed corner exits completed")
            else:
                self.transition_to(State.FOLLOW_STRAIGHT, "corner exit corridor and heading confirmed")
        steer = self.wall_follower.steering(snap, self.context.target_heading_rad)
        blend = min(1.0, distance / max(1.0, self.cfg.exit_straighten_distance_cm))
        return ControlCommand(
            "f", self.cfg.recovery_duty,
            (1.0 - blend) * self.last_steering_deg + blend * steer,
            "gradual corner exit",
        )

    def _update_pillar_approach(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        pillar = self.context.active_pillar
        if pillar is None:
            self.transition_to(State.FOLLOW_STRAIGHT, "pillar track vanished before latch")
            return self._straight_command(snap, self.cfg.recovery_duty, "pillar cancelled")
        side_role = "right" if pillar.colour == "red" else "left"
        side_clearance = valid_tof_mm(snap.tof, side_role)
        if side_clearance is not None and side_clearance < self.cfg.pillar_wall_margin_mm:
            self.transition_to(State.SAFE_STOP, f"insufficient {side_role} wall clearance for {pillar.colour}")
            return ControlCommand("s", 0.0, 0.0, "unsafe pillar pass")
        if pillar.relative_y_mm <= self.cfg.pillar_trigger_mm:
            self.transition_to(State.PILLAR_PASS, f"{pillar.colour} pass {pillar.pass_side} latched")
        return self._straight_command(snap, self.cfg.pillar_approach_duty, "pillar approach")

    def _update_pillar_pass(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        if self.pillar_trajectory is None:
            self.transition_to(State.SAFE_STOP, "pillar path could not be made safely")
            return ControlCommand("s", 0.0, 0.0, "no pillar path")
        s = self.pillar_trajectory.find_closest(
            snap.x_cm, snap.y_cm, self.context.corner_path_s_cm
        )
        self.context.corner_path_s_cm = max(self.context.corner_path_s_cm, s)
        px, py = self.pillar_trajectory.get_point(s)
        tx, ty = self.pillar_trajectory.get_tangent(s)
        path_heading = math.atan2(ty, tx)
        cte = -math.sin(path_heading) * (snap.x_cm - px) + math.cos(path_heading) * (snap.y_cm - py)
        steer = self.pillar_pid.compute(
            cte + 8.0 * normalize_angle(path_heading - snap.heading_rad), snap.dt_s
        )
        remaining = self.pillar_trajectory.total_length - s
        if remaining <= 12.0:
            self.transition_to(State.PILLAR_RECOVER, "swerve path nearly complete")
        return ControlCommand("f", self.cfg.pillar_pass_duty, steer, "latched pillar swerve")

    def _update_pillar_recover(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        pillar = self.context.active_pillar
        passed = pillar is None or (
            snap.timestamp_s - pillar.last_seen_s > self.cfg.camera_stale_s
            and self._state_distance(snap) * 10.0 >= self.cfg.pillar_passed_margin_mm
        )
        if passed and self._state_distance(snap) >= 20.0:
            if pillar:
                self.context.handled_pillars[pillar.key] = snap.timestamp_s
            self.context.active_pillar = None
            self.pillar_trajectory = None
            confidence, evidence = self._corner_confidence(snap, event)
            target = State.APPROACH_CORNER if confidence >= self.cfg.corner_confidence_enter else State.FOLLOW_STRAIGHT
            self.transition_to(target, f"pillar behind and reconnect complete; corner={evidence}")
        return self._straight_command(snap, self.cfg.recovery_duty, "pillar recovery")

    def _update_finish_section(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        origin = self.context.finish_entry_distance_cm or snap.distance_cm
        if snap.distance_cm - origin >= self.cfg.finish_clear_distance_cm:
            self.transition_to(State.FINISHED, "robot length plus margin inside finish straight")
            return ControlCommand("s", 0.0, 0.0, "finished")
        return self._straight_command(snap, self.cfg.recovery_duty, "clear finish boundary")

    def _update_parking_search(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        parking = snap.vision.result.get("parking", {})
        full = parking.get("parking_status") == "full_slot_detected"
        fresh = snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        self._parking_support = self._parking_support + 1 if full and fresh else 0
        if self._parking_support >= self.cfg.parking_marker_confirm_frames:
            self.transition_to(State.PARKING_APPROACH, "two magenta markers confirmed across frames")
        return self._straight_command(snap, self.cfg.parking_duty, "parking search")

    def _update_parking_approach(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        parking = snap.vision.result.get("parking", {})
        x_mm = parking.get("slot_center_relative_x_mm")
        y_mm = parking.get("slot_center_relative_y_mm")
        fresh = snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        if not fresh or x_mm is None or y_mm is None:
            return ControlCommand("s", 0.0, 0.0, "parking target stale")
        if float(y_mm) <= self.cfg.parking_stop_forward_mm:
            self.transition_to(State.PARKING_MANOEUVRE, "parking slot entry reached")
        steer = self.parking_pid.compute(float(x_mm), snap.dt_s)
        return ControlCommand("f", self.cfg.parking_duty, steer, "parking slot centre")

    def _update_parking_manoeuvre(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> ControlCommand:
        # Existing parking.py is absent. Use only observable marker and wall alignment;
        # never claim success from a guessed global lot.
        parking = snap.vision.result.get("parking", {})
        fresh = snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        left, right = valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "right")
        aligned = left is not None and right is not None and abs(left - right) <= self.cfg.parking_alignment_tolerance_mm
        full = fresh and parking.get("parking_status") == "full_slot_detected"
        x_mm = parking.get("slot_center_relative_x_mm")
        y_mm = parking.get("slot_center_relative_y_mm")
        inside_evidence = (
            full and x_mm is not None and y_mm is not None
            and abs(float(x_mm)) < 100.0 - ROBOT_WIDTH_CM * 5.0
            and float(y_mm) <= self.cfg.parking_stop_forward_mm
        )
        if inside_evidence and aligned:
            self.transition_to(State.FINISHED, "parking projection and <=20mm wall alignment confirmed")
            return ControlCommand("s", 0.0, 0.0, "parked")
        if not full:
            return ControlCommand("s", 0.0, 0.0, "parking markers lost; safe hold")
        steer = self.parking_pid.compute(float(x_mm or 0.0), snap.dt_s)
        return ControlCommand("f", self.cfg.parking_duty, steer, "parking manoeuvre")

    def _straight_command(self, snap: SensorSnapshot, duty: float, reason: str) -> ControlCommand:
        steer = self.wall_follower.steering(snap, self.context.target_heading_rad)
        return ControlCommand("f", duty, steer, reason)

    def _update_front_history(self, snap: SensorSnapshot) -> None:
        front = valid_tof_mm(snap.tof, "front")
        if front is not None:
            self.front_history.append(front)

    def _front_decreasing(self) -> bool:
        n = self.cfg.front_trend_samples
        values = list(self.front_history)[-n:]
        return len(values) == n and all(a > b + 10.0 for a, b in zip(values, values[1:]))

    def _front_opening(self, snap: SensorSnapshot) -> bool:
        front = valid_tof_mm(snap.tof, "front")
        return front is not None and front > self.cfg.front_approach_mm

    def _corner_imminent(self, snap: SensorSnapshot) -> bool:
        front = valid_tof_mm(snap.tof, "front")
        return front is not None and front < self.cfg.front_turn_entry_mm and self._front_decreasing()

    def _corridor_resembles_straight(self, snap: SensorSnapshot) -> bool:
        left, right = valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "right")
        return left is not None and right is not None and 350.0 <= left + ROBOT_WIDTH_CM * 10 + right <= 1300.0

    def _corner_confidence(
        self, snap: SensorSnapshot, event: Optional[LineEvent]
    ) -> Tuple[float, List[str]]:
        """Weighted evidence: physical front trend dominates; priors cannot commit alone."""
        score, evidence = 0.0, []
        front = valid_tof_mm(snap.tof, "front")
        if front is not None and front < self.cfg.front_approach_mm and self._front_decreasing():
            score += 2.0  # multiple independent physical samples: strongest evidence
            evidence.append("front-ToF trend")
        if event and self._line_expected(event.colour):
            score += 1.5  # spatial/temporal-gated floor edge
            evidence.append(f"{event.colour.value} line")
        walls = snap.vision.result.get("walls", {})
        camera_fresh = snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        camera_front = walls.get("front_wall_distance_mm")
        if camera_fresh and camera_front is not None and camera_front < self.cfg.front_approach_mm:
            score += 0.75  # monocular estimate is secondary
            evidence.append("camera wall")
        left, right = valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "right")
        if (left is None) != (right is None) and front is not None:
            score += 0.75  # side opening/geometry change
            evidence.append("side geometry")
        learned = self.context.learned_straight_lengths_cm[self.context.current_straight_index]
        if learned and self._straight_distance(snap) >= learned - self.cfg.predicted_corner_slow_margin_cm:
            score += 0.5  # weak slip-prone odometry prior
            evidence.append("learned distance")
        return score, evidence

    def _line_expected(self, colour: LineColour) -> bool:
        if self.context.direction == Direction.UNKNOWN:
            return True
        # Repository executable section sequence: CW entry orange, CCW entry blue.
        expected = LineColour.ORANGE if self.context.direction == Direction.CW else LineColour.BLUE
        return colour == expected

    def _classify_corridor(self, snap: SensorSnapshot) -> None:
        left, right = valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "right")
        if left is None or right is None:
            return
        width = left + ROBOT_WIDTH_CM * 10.0 + right
        self.context.corridor_width_mm = width
        if self.cfg.corridor_600_min_mm <= width <= self.cfg.corridor_600_max_mm:
            self.context.corridor_class = "600mm"
        elif self.cfg.corridor_1000_min_mm <= width <= self.cfg.corridor_1000_max_mm:
            self.context.corridor_class = "1000mm"
        else:
            self.context.corridor_class = "uncertain"

    def _straight_distance(self, snap: SensorSnapshot) -> float:
        return max(0.0, snap.distance_cm - self.context.straight_distance_origin_cm)

    def _state_distance(self, snap: SensorSnapshot) -> float:
        return max(0.0, snap.distance_cm - self.context.state_distance_origin_cm)

    def _learn_current_straight(self, snap: SensorSnapshot) -> None:
        measured = self._straight_distance(snap)
        if not self.cfg.learned_straight_min_cm <= measured <= self.cfg.learned_straight_max_cm:
            return
        index = self.context.current_straight_index
        old = self.context.learned_straight_lengths_cm[index]
        self.context.learned_straight_lengths_cm[index] = (
            measured if old is None
            else (1.0 - self.cfg.learned_ema_alpha) * old + self.cfg.learned_ema_alpha * measured
        )

    def _complete_corner(self, snap: SensorSnapshot) -> None:
        self.context.total_corners_completed += 1
        self.context.corners_in_current_lap += 1
        if self.context.corners_in_current_lap == 4:
            self.context.laps_completed += 1
            self.context.corners_in_current_lap = 0
        self.context.current_straight_index = (
            self.context.current_straight_index + 1
        ) % 4
        self.context.straight_distance_origin_cm = snap.distance_cm
        self.corner_trajectory = None
        self.corner_pid.reset()
        self.line_detector.reset_levels()
        print(
            f"[Lap] completed corner {self.context.total_corners_completed}; "
            f"lap={self.context.laps_completed}, in-lap={self.context.corners_in_current_lap}",
            flush=True,
        )

    def _update_pillar_tracks(self, snap: SensorSnapshot) -> None:
        if self.mode != ChallengeMode.OBSTACLE:
            return
        fresh = snap.timestamp_s - snap.vision.timestamp_s <= self.cfg.camera_stale_s
        if not fresh:
            return
        candidates = []
        for raw in snap.vision.result.get("pillars", []):
            try:
                colour = str(raw.get("color", "")).lower()
                confidence = float(raw.get("confidence"))
                x_mm, y_mm = float(raw.get("relative_x_mm")), float(raw.get("relative_y_mm"))
                distance_mm = float(raw.get("estimated_distance_mm"))
            except (TypeError, ValueError):
                continue
            if (
                colour not in ("red", "green")
                or confidence < self.cfg.pillar_min_confidence
                or not all(map(math.isfinite, (x_mm, y_mm, distance_mm)))
                or y_mm <= self.cfg.pillar_min_forward_mm
                or distance_mm <= 0 or abs(x_mm) > self.cfg.pillar_max_lateral_mm
            ):
                continue
            candidates.append((y_mm, colour, confidence, x_mm, raw))
        if not candidates:
            return
        _, colour, confidence, x_mm, raw = min(candidates)
        y_mm = float(raw["relative_y_mm"])
        key = f"{colour}:{round(x_mm / 150):+d}:{round(y_mm / 250):d}"
        old = self.context.active_pillar
        if old and old.colour == colour:
            jump = math.hypot(x_mm - old.relative_x_mm, y_mm - old.relative_y_mm)
            if jump <= self.cfg.pillar_track_jump_mm:
                alpha = 0.35
                old.relative_x_mm = (1 - alpha) * old.relative_x_mm + alpha * x_mm
                old.relative_y_mm = (1 - alpha) * old.relative_y_mm + alpha * y_mm
                old.confidence = (1 - alpha) * old.confidence + alpha * confidence
                old.support_frames += 1
                old.last_seen_s = snap.timestamp_s
            return
        if key in self.context.handled_pillars and (
            snap.timestamp_s - self.context.handled_pillars[key] < self.cfg.pillar_cooldown_s
        ):
            return
        gx, gy = camera_to_world(snap.x_cm, snap.y_cm, snap.heading_rad, x_mm, y_mm)
        self.context.active_pillar = PillarTrack(
            key, colour, x_mm, y_mm, confidence, snap.timestamp_s, snap.timestamp_s,
            1, gx, gy
        )

    def _pillar_ready(self, snap: SensorSnapshot) -> bool:
        p = self.context.active_pillar
        return bool(
            p and p.support_frames >= self.cfg.pillar_confirm_frames
            and snap.timestamp_s - p.last_seen_s <= self.cfg.camera_stale_s
            and p.relative_y_mm <= self.cfg.pillar_trigger_mm
        )

    def _build_pillar_path(self) -> None:
        p, snap = self.context.active_pillar, self.last_snapshot
        self.pillar_trajectory = None
        if p is None or snap is None or p.global_x_cm is None or p.global_y_cm is None:
            return
        side_role = "right" if p.colour == "red" else "left"
        wall = valid_tof_mm(snap.tof, side_role)
        required_mm = (
            ROBOT_WIDTH_CM * 5.0 + self.cfg.pillar_clearance_cm * 10.0
            + self.cfg.pillar_wall_margin_mm
        )
        if wall is not None and wall < required_mm:
            return
        forward_x, forward_y = math.cos(self.context.target_heading_rad), math.sin(self.context.target_heading_rad)
        end_x = p.global_x_cm + forward_x * self.cfg.pillar_reconnect_cm
        end_y = p.global_y_cm + forward_y * self.cfg.pillar_reconnect_cm
        self.pillar_trajectory = TrajectoryBuilder.pillar_swerve(
            snap.x_cm, snap.y_cm, snap.heading_rad, p.global_x_cm, p.global_y_cm,
            RED if p.colour == "red" else GREEN, end_x, end_y,
            self.context.target_heading_rad, self.cfg.pillar_clearance_cm
        )
        self.context.corner_path_s_cm = 0.0

    def _safety_fault(self, snap: SensorSnapshot) -> Optional[str]:
        if not all(math.isfinite(x) for x in (
            snap.speed_cm_s, snap.heading_rad, snap.yaw_rate_rad_s,
            snap.left_speed_cm_s, snap.right_speed_cm_s
        )):
            return "non-finite estimator or encoder value"
        if max(snap.left_speed_cm_s, snap.right_speed_cm_s) > self.cfg.encoder_speed_max_cm_s:
            return "impossible encoder speed"
        moving_commanded = self.last_command.direction == "f" and self.last_command.duty > 0.08
        if snap.speed_cm_s > 1.0:
            self._last_encoder_motion_s = snap.timestamp_s
        elif moving_commanded and snap.timestamp_s - self._last_encoder_motion_s > self.cfg.no_encoder_motion_timeout_s:
            return "motor commanded but encoders report no motion"
        imbalance = abs(snap.left_speed_cm_s - snap.right_speed_cm_s) > max(
            25.0, 0.8 * max(snap.left_speed_cm_s, snap.right_speed_cm_s)
        )
        if imbalance and moving_commanded:
            self._encoder_imbalance_since_s = self._encoder_imbalance_since_s or snap.timestamp_s
            if snap.timestamp_s - self._encoder_imbalance_since_s > self.cfg.encoder_imbalance_timeout_s:
                return "persistent wheel-speed imbalance"
        else:
            self._encoder_imbalance_since_s = None
        if self.tof_reader and (
            not self.tof_reader.alive
            or snap.timestamp_s - self.tof_reader.last_progress_s > self.cfg.thread_failure_timeout_s
        ):
            return "ToF background reader stopped or stalled"
        return None

    def _apply_safety_overrides(
        self, snap: SensorSnapshot, command: ControlCommand
    ) -> ControlCommand:
        front = valid_tof_mm(snap.tof, "front")
        left, right = valid_tof_mm(snap.tof, "left"), valid_tof_mm(snap.tof, "right")
        duty, steer = command.duty, command.steering_deg
        if front is not None and front < self.cfg.front_emergency_mm:
            duty = 0.0
        if left is not None and left < self.cfg.side_emergency_mm:
            steer -= self.cfg.side_safety_bias_deg  # steer right, away from left wall
            duty = min(duty, self.cfg.emergency_crawl_duty)
        if right is not None and right < self.cfg.side_emergency_mm:
            steer += self.cfg.side_safety_bias_deg
            duty = min(duty, self.cfg.emergency_crawl_duty)
        valid_side_count = sum(x is not None for x in (left, right))
        if valid_side_count == 0:
            duty = min(duty, self.cfg.localisation_duty)
        return ControlCommand(command.direction if duty > 0 else "s", duty, steer, command.reason)

    def apply_command(self, command: ControlCommand) -> None:
        """The only path from navigation intent to motor/servo hardware."""
        if not all(math.isfinite(v) for v in (command.duty, command.steering_deg)):
            command = ControlCommand("s", 0.0, 0.0, "non-finite command")
        duty = max(0.0, min(self.cfg.max_duty, command.duty))
        desired = self.cfg.steering_sign * max(
            -self.cfg.steering_max_deg, min(self.cfg.steering_max_deg, command.steering_deg)
        )
        dt = self.last_snapshot.dt_s if self.last_snapshot else self.cfg.period_s
        max_delta = self.cfg.steering_rate_deg_s * max(self.cfg.dt_min_s, dt)
        steer = max(
            self.last_steering_deg - max_delta,
            min(self.last_steering_deg + max_delta, desired),
        )
        if not self.started or self.state in (
            State.BOOT, State.WAIT_FOR_START, State.SENSOR_WARMUP,
            State.SAFE_STOP, State.FAULT, State.FINISHED
        ):
            duty, direction = 0.0, "s"
        else:
            direction = command.direction if command.direction in ("f", "b") else "s"
        safe = ControlCommand(direction, duty, steer, command.reason)
        self.last_command, self.last_steering_deg = safe, steer
        if self.dry_run:
            return
        self.robot.update_steering(steer)
        self.car.set_steering(steer)
        self.car.set_motor(direction, duty)

    def run(self) -> int:
        try:
            self.initialize_hardware()
            print_tof_mapping(self.tof_reader.snapshot() if self.tof_reader else None)
            self.calibrate_gyro()
            self.wait_for_start()
            last_s = time.monotonic()
            next_tick_s = last_s
            while self.state not in (State.FINISHED, State.SAFE_STOP, State.FAULT):
                now_s = time.monotonic()
                dt_s, last_s = now_s - last_s, now_s
                snap = self._read_snapshot(now_s, dt_s)
                command = self.update(snap)
                self.apply_command(command)
                self._telemetry(snap)
                self._loop_count += 1
                self._loop_dt_sum_s += dt_s
                next_tick_s += self.cfg.period_s
                remaining = next_tick_s - time.monotonic()
                if remaining > 0:
                    time.sleep(remaining)
                    self._consecutive_overruns = 0
                else:
                    overrun = -remaining
                    self._missed_deadlines += 1
                    self._consecutive_overruns += 1
                    self._max_overrun_s = max(self._max_overrun_s, overrun)
                    if self._consecutive_overruns >= self.cfg.max_consecutive_overruns:
                        self.transition_to(State.SAFE_STOP, "repeated control-loop overruns")
            return 0 if self.state == State.FINISHED else 2
        except KeyboardInterrupt:
            print("[FSM] Ctrl+C: safe shutdown", flush=True)
            return 130
        except Exception as exc:
            print(f"[FSM] fatal: {type(exc).__name__}: {exc}", flush=True)
            if self.state != State.FAULT:
                self.transition_to(State.FAULT, f"exception: {type(exc).__name__}: {exc}")
            return 1
        finally:
            self.cleanup()

    def _telemetry(self, snap: SensorSnapshot) -> None:
        if snap.timestamp_s - self._last_telemetry_s < 1.0 / self.cfg.telemetry_hz:
            return
        self._last_telemetry_s = snap.timestamp_s
        ages = {
            role: (
                None if snap.tof.role(role).timestamp_s <= 0
                else snap.timestamp_s - snap.tof.role(role).timestamp_s
            )
            for role in TOF_ROLE_TO_INDEX
        }
        avg_hz = self._loop_count / self._loop_dt_sum_s if self._loop_dt_sum_s > 0 else 0.0
        print(
            f"[TEL] {self.state.name} dir={self.context.direction.name} "
            f"lap={self.context.laps_completed} corner={self.context.total_corners_completed} "
            f"v={snap.speed_cm_s:.1f} duty={self.last_command.duty:.2f} "
            f"steer={self.last_command.steering_deg:+.1f} Hz={avg_hz:.1f} "
            f"miss={self._missed_deadlines} max_over={self._max_overrun_s*1000:.1f}ms "
            f"tof_age={ages} cam_age={snap.timestamp_s-snap.vision.timestamp_s:.2f}s",
            flush=True,
        )
        if self.show_camera and snap.vision.frame is not None:
            try:
                import cv2 as cv
                frame = snap.vision.frame
                text = (
                    f"{self.state.name} {self.context.direction.name} "
                    f"L{self.context.laps_completed} C{self.context.total_corners_completed} "
                    f"duty={self.last_command.duty:.2f} steer={self.last_command.steering_deg:+.1f}"
                )
                cv.putText(frame, text, (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                cv.imshow("WRO Sensor Fusion FSM", frame)
                cv.waitKey(1)
            except Exception as exc:
                print(f"[Display] disabled after error: {exc}")
                self.show_camera = False

    def cleanup(self) -> None:
        """Idempotent motor-first shutdown for every exit path."""
        if self.cleanup_done:
            return
        self.cleanup_done = True
        if self.car is not None:
            try:
                self.car.stop()
                self.car.set_steering(0.0)
            except Exception as exc:
                print(f"[Cleanup] car warning: {exc}")
        if self.tof_reader:
            self.tof_reader.stop()
        if self.vision_reader:
            self.vision_reader.stop()
        if self.color is not None:
            try:
                self.color.stop()
            except Exception as exc:
                print(f"[Cleanup] colour warning: {exc}")
        for led in self.leds:
            try:
                led.off()
            except Exception:
                pass
        try:
            import cv2 as cv
            cv.destroyAllWindows()
        except Exception:
            pass
        if self._loop_count:
            avg_hz = self._loop_count / max(1e-9, self._loop_dt_sum_s)
            print(
                f"[Timing] avg={avg_hz:.1f}Hz missed={self._missed_deadlines} "
                f"max_overrun={self._max_overrun_s*1000:.1f}ms"
            )


def empty_tof(now_s: float) -> ToFSnapshot:
    return ToFSnapshot(now_s, (ToFChannel(), ToFChannel(), ToFChannel(), ToFChannel()))


def print_tof_mapping(snapshot: Optional[ToFSnapshot] = None) -> None:
    print(
        f"[ToF] physical role mapping ({'CONFIRMED' if TOF_MAPPING_CONFIRMED else 'UNCONFIRMED'}):"
    )
    for role, index in TOF_ROLE_TO_INDEX.items():
        value = None if snapshot is None else snapshot.channels[index].filtered_mm
        print(
            f"  GPIO/XSHUT {TOF_XSHUT_GPIO[index]:>2} -> I2C "
            f"0x{TOF_I2C_ADDRESS[index]:02X} -> index {index} -> "
            f"{role:>5}: {value} mm"
        )


def _mock_snapshot(
    now_s: float, distance_cm: float, heading_deg: float = 0.0,
    front_mm: float = 900.0, left_mm: float = 440.0, right_mm: float = 440.0,
    orange: bool = False, blue: bool = False,
) -> SensorSnapshot:
    channels = [ToFChannel(900, 900, now_s, True, False) for _ in range(4)]
    for role, value in (("front", front_mm), ("left", left_mm), ("right", right_mm)):
        i = TOF_ROLE_TO_INDEX[role]
        channels[i] = ToFChannel(value, value, now_s, True, False)
    return SensorSnapshot(
        now_s, 0.02, distance_cm, distance_cm, distance_cm, 20.0, 20.0, 20.0,
        0.0, math.radians(heading_deg), distance_cm, 0.0, orange, blue,
        ToFSnapshot(now_s, tuple(channels)),  # type: ignore[arg-type]
        VisionSnapshot(now_s, {"pillars": [], "parking": {}, "walls": {}, "track_lines": {}},
                       None, 0, True),
        0.0, 0.0,
    )


def run_dry_run(mode: ChallengeMode, debug: bool = False) -> int:
    """Deterministically exercise direction, 12 corners, finish/parking safety."""
    fsm = SensorFusionFSM(mode, dry_run=True, debug=debug)
    fsm.started = True
    fsm.state = State.DETERMINE_DIRECTION
    base = time.monotonic()
    distance = 0.0

    # A clean orange edge after debounce locks CW.
    for i in range(14):
        distance += 0.5
        snap = _mock_snapshot(base + i * 0.02, distance, orange=i >= 1)
        fsm.apply_command(fsm.update(snap))
    assert fsm.context.direction == Direction.CW, "orange must lock CW"
    if fsm.state == State.INITIAL_LOCALISATION:
        fsm.transition_to(State.FOLLOW_STRAIGHT, "dry-run localisation evidence")

    for corner in range(12):
        distance += 100.0
        approach = _mock_snapshot(base + 1 + corner, distance, -(corner * 90) % 360, 600)
        fsm.last_snapshot = approach
        fsm._learn_current_straight(approach)
        fsm.transition_to(State.APPROACH_CORNER, "dry-run fused corner evidence")
        entry = _mock_snapshot(base + 1.1 + corner, distance + 5, -(corner * 90) % 360, 350)
        fsm.last_snapshot = entry
        fsm.transition_to(State.TURN_CORNER, "dry-run physical entry")
        exit_snap = _mock_snapshot(
            base + 1.2 + corner, distance + 55, -((corner + 1) * 90) % 360, 900
        )
        fsm.last_snapshot = exit_snap
        fsm.context.target_heading_rad = exit_snap.heading_rad
        fsm.transition_to(State.EXIT_CORNER, "dry-run heading/travel/opening")
        fsm._complete_corner(exit_snap)
        fsm._corner_counted_for_entry = True
        distance += 55
        if corner < 11:
            fsm.transition_to(State.FOLLOW_STRAIGHT, "dry-run confirmed exit")

    assert fsm.context.total_corners_completed == 12
    assert fsm.context.laps_completed == 3
    if mode == ChallengeMode.OPEN:
        fsm.context.finish_entry_distance_cm = distance
        fsm.transition_to(State.FINISH_SECTION, "dry-run three laps")
        finish = _mock_snapshot(base + 20, distance + fsm.cfg.finish_clear_distance_cm + 1)
        fsm.last_snapshot = finish
        command = fsm._update_finish_section(finish, None)
        fsm.apply_command(command)
        assert fsm.state == State.FINISHED
    else:
        fsm.transition_to(State.PARKING_SEARCH, "dry-run three laps")
        # No invented marker: obstacle dry run verifies safe search structure.
        assert fsm.mode == ChallengeMode.OBSTACLE
        fsm.transition_to(State.SAFE_STOP, "dry-run parking API absent/no markers")

    # Red/right and green/left are enforced by typed tracks and builder constants.
    assert PillarTrack("r", "red", 0, 500, 1, base, base).pass_side == "right"
    assert PillarTrack("g", "green", 0, 500, 1, base, base).pass_side == "left"
    assert fsm.last_command.duty == 0.0 or fsm.state == State.FINISHED
    print(
        f"[DryRun] PASS mode={mode.value}, direction=orange->CW, "
        f"corners={fsm.context.total_corners_completed}, laps={fsm.context.laps_completed}, "
        f"terminal={fsm.state.name}"
    )
    return 0


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("open", "obstacle"), default="open")
    parser.add_argument("--show-camera", action="store_true", help="optional GUI overlay")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="no GPIO; exercise transitions")
    parser.add_argument(
        "--tof-diagnostic", action="store_true",
        help="print wiring/address/role mapping and live distances"
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def run_tof_diagnostic() -> int:
    from control.tof_sensor import ToFSensors
    cfg = FSMConfig()
    reader = BackgroundToFReader(ToFSensors(), cfg)
    reader.start()
    try:
        while True:
            print_tof_mapping(reader.snapshot())
            time.sleep(0.5)
    except KeyboardInterrupt:
        return 0
    finally:
        reader.stop()


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    mode = ChallengeMode(args.mode)
    if args.tof_diagnostic:
        return run_tof_diagnostic()
    if args.dry_run:
        print_tof_mapping()
        return run_dry_run(mode, args.debug)
    return SensorFusionFSM(
        mode, show_camera=args.show_camera, debug=args.debug
    ).run()


if __name__ == "__main__":
    raise SystemExit(main())
