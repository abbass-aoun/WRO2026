"""
main/race_manager.py -- Top-level race state machine.

WHAT DOES THIS DO?
    This is the "brain" of the race.  Every 50 Hz tick, the main loop
    calls race_manager.update(robot, vision) and gets back the trajectory
    to follow.  The race manager:

        1. Detects direction (CCW or CW) from the first line the car crosses.
        2. Builds the trajectory for each section (straight / corner / swerve).
        3. Detects when a section ends (proximity to exit OR line crossing).
        4. Counts laps (each time section 7 completes, lap counter increments).
        5. After 3 laps, hands off to ParkingManager.

RACE STRUCTURE (WRO Future Engineers):
    3 laps x 8 sections = 24 sections total.
    Sections alternate: STRAIGHT, CORNER, STRAIGHT, CORNER ...
    Pillars appear only in STRAIGHT sections.

WRO 2026 SPECIAL RULE — LAP 3 DIRECTION:
    After lap 2 completes, the color of the LAST traffic sign seen determines
    whether lap 3 continues in the same direction or reverses:
        GREEN -> same direction as laps 1-2
        RED   -> reverse direction (CCW↔CW)
    This is tracked in _last_sign_color and applied at the lap-2->3 transition.

TRACK GEOMETRY:
    Outer boundary:  300 x 300 cm.
    Centerlines:     y=50 (bottom), y=250 (top), x=50 (left), x=250 (right).
    Corner radius:   50 cm (TUNE ON REAL ROBOT).
    All coordinates and angles are relative to this fixed world frame.

    Verified: each section's exit_xy == next section's entry_xy.
    CCW sections exit angles:  0 -> pi/2 -> pi -> -pi/2 -> 0 (one lap)
    CW  sections exit angles:  pi -> pi/2 -> 0 -> -pi/2 -> pi (one lap)

CONNECTIONS:
    Reads:    Robot.x, Robot.y, Robot.theta  (current car state, written by EKF)
    Builds:   TrajectoryBase objects via TrajectoryBuilder
    Returns:  TrajectoryBase --> consumed by SteeringPIDController + DrivingPIDController
    Uses:     ParkingManager (main/parking.py) for the post-race parking sequence

DIRECTION DETECTION:
    orange_line_seen = True  -->  CCW (counter-clockwise)
    blue_line_seen   = True  -->  CW  (clockwise)
    If direction is known at start(), this detection is skipped.

VISION INTERFACE (filled by the vision team every tick):
    VisionFrame.pillars          list of (x_cm, y_cm, color) for detected pillars
    VisionFrame.orange_line_seen True if orange start/finish line is under camera
    VisionFrame.blue_line_seen   True if blue section-divider line is under camera
    VisionFrame.parking_lot      (x, y, theta) of lot entry, or None
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple

from control.robot import Robot
from trajectory.base import TrajectoryBase
from trajectory.builder import TrajectoryBuilder, RED, GREEN
from main.parking import ParkingManager
from config import LOT_DEPTH_CM


# ===========================================================================
# Data types
# ===========================================================================

@dataclass
class VisionFrame:
    """
    Data from the vision system, expected every control loop tick.

    The vision team fills this and hands it to race_manager.update().
    None/empty/False means "not detected this tick".

    pillars:          list of (x_cm, y_cm, color) -- closest first.
                      color: RED=0 (pass right) or GREEN=1 (pass left).
    orange_line_seen: True if the orange start/finish line is under the camera.
    blue_line_seen:   True if a blue section-divider line is under the camera.
    parking_lot:      (x_cm, y_cm, theta_rad) of the lot ENTRY edge, or None.
    """
    pillars:          List[Tuple[float, float, int]] = field(default_factory=list)
    orange_line_seen: bool                           = False
    blue_line_seen:   bool                           = False
    parking_lot:      Optional[Tuple[float, float, float]] = None


class Direction(Enum):
    """Race direction, determined at the start of the race."""
    CCW     = "ccw"      # counter-clockwise -- turns are all left turns
    CW      = "cw"       # clockwise         -- turns are all right turns
    UNKNOWN = "unknown"  # waiting for first line detection


class RaceState(Enum):
    """Top-level state of the race manager."""
    WAITING = "waiting"   # waiting for first line to detect direction
    RACING  = "racing"    # actively following track sections
    PARKING = "parking"   # post-race parking sequence
    DONE    = "done"      # parked; race complete


# ===========================================================================
# Track geometry
# ===========================================================================

@dataclass(frozen=True)
class Section:
    """
    Describes one of the 8 sections that make up one lap.

    entry_xy and entry_theta: where and how the car enters this section.
    exit_xy and exit_theta:   where and how the car leaves.
    kind:           "straight" or "corner".
    turn_direction: +1 = left/CCW turn, -1 = right/CW turn (corners only).
    corner_radius:  radius of the circular arc approximation (cm).
    """
    kind:           str
    entry_x:        float
    entry_y:        float
    entry_theta:    float   # radians
    exit_x:         float
    exit_y:         float
    exit_theta:     float   # radians
    turn_direction: int   = 0
    corner_radius:  float = 50.0


class WROTrack:
    """
    Standard WRO 2026 Future Engineers track geometry.

    ALL COORDINATES IN CENTIMETRES.
    World frame: x = East, y = North.  Angles in radians, 0 = East.

    Track sections for CCW direction (all left turns, turn_direction=+1):
        0: straight  (150,50)  -> (250,50)   heading East
        1: corner    (250,50)  -> (250,150)  heading East  -> North
        2: straight  (250,150) -> (250,250)  heading North
        3: corner    (250,250) -> (150,250)  heading North -> West
        4: straight  (150,250) -> (50,250)   heading West
        5: corner    (50,250)  -> (50,150)   heading West  -> South
        6: straight  (50,150)  -> (50,50)    heading South
        7: corner    (50,50)   -> (150,50)   heading South -> East
        -> loops back to section 0 ✓

    Track sections for CW direction (all right turns, turn_direction=-1):
        0: straight  (150,50)  -> (50,50)    heading West
        1: corner    (50,50)   -> (50,150)   heading West  -> North
        2: straight  (50,150)  -> (50,250)   heading North
        3: corner    (50,250)  -> (150,250)  heading North -> East
        4: straight  (150,250) -> (250,250)  heading East
        5: corner    (250,250) -> (250,150)  heading East  -> South
        6: straight  (250,150) -> (250,50)   heading South
        7: corner    (250,50)  -> (150,50)   heading South -> West
        -> loops back to section 0 ✓

    These exit-entry connections are verified by computing TrajectoryBuilder.corner()
    P3 = entry + radius*perp + radius*exit_tangent for each corner.
    """

    @staticmethod
    def ccw_sections(corner_radius: float = 50.0) -> List[Section]:
        """Return the 8 sections for one CCW lap."""
        r = corner_radius
        PI  = math.pi
        PI2 = math.pi / 2.0
        return [
            # idx  kind        ex  ey    eth  xx   xy    xth  turn  r
            Section("straight", 150, 50,   0,   250, 50,   0,    0,  r),
            Section("corner",   250, 50,   0,   250, 150,  PI2,  +1, r),
            Section("straight", 250, 150,  PI2, 250, 250,  PI2,  0,  r),
            Section("corner",   250, 250,  PI2, 150, 250,  PI,   +1, r),
            Section("straight", 150, 250,  PI,   50, 250,  PI,   0,  r),
            Section("corner",    50, 250,  PI,   50, 150, -PI2,  +1, r),
            Section("straight",  50, 150, -PI2,  50,  50, -PI2,  0,  r),
            Section("corner",    50,  50, -PI2, 150,  50,  0,    +1, r),
        ]

    @staticmethod
    def cw_sections(corner_radius: float = 50.0) -> List[Section]:
        """Return the 8 sections for one CW lap."""
        r = corner_radius
        PI  = math.pi
        PI2 = math.pi / 2.0
        return [
            Section("straight", 150, 50,   PI,   50,  50,  PI,   0,  r),
            Section("corner",    50, 50,   PI,   50, 150,  PI2,  -1, r),
            Section("straight",  50, 150,  PI2,  50, 250,  PI2,  0,  r),
            Section("corner",    50, 250,  PI2, 150, 250,  0,    -1, r),
            Section("straight", 150, 250,  0,  250, 250,  0,    0,  r),
            Section("corner",   250, 250,  0,  250, 150, -PI2,  -1, r),
            Section("straight", 250, 150, -PI2, 250,  50, -PI2,  0,  r),
            Section("corner",   250,  50, -PI2, 150,  50,  PI,   -1, r),
        ]


# ===========================================================================
# Race Manager
# ===========================================================================

class RaceManager:
    """
    Top-level state machine for the 3-lap WRO race.

    Each call to update() returns the TrajectoryBase object that the
    steering and driving controllers should follow for that tick.

    TYPICAL MAIN LOOP USAGE:
        race = RaceManager()
        race.start(robot, direction=Direction.CCW)   # if direction known
        # or race.start(robot)                        # auto-detects from lines

        while True:
            # 1. read sensors, run EKF
            ekf.predict(robot.speed, robot.steer_angle, dt)
            ekf.update_imu(imu_theta)
            ekf.update_robot(robot)

            # 2. get vision data from partner code
            vision = get_vision_frame()     # VisionFrame from partner

            # 3. update race manager -- get current path
            trajectory = race.update(robot, vision)
            if race.is_done:
                stop_all_motors()
                break

            # 4. run controllers
            steer    = steer_ctrl.compute(robot.x, robot.y, robot.theta,
                                          trajectory, steer_ctrl.current_s)
            throttle = drive_ctrl.compute(robot.x, robot.y, robot.theta,
                                          trajectory, drive_ctrl.current_s,
                                          robot.speed)
            # 5. apply to hardware
            car_ctrl.set_steering(steer)
            car_ctrl.set_speed(throttle)
    """

    TOTAL_LAPS:        int   = 3
    SECTIONS_PER_LAP:  int   = 8
    SECTION_END_CM:    float = 15.0   # proximity to exit_xy that counts as "done"
    LINE_COOLDOWN_TICKS: int = 25     # ticks to ignore lines after a section advance

    def __init__(self, corner_radius: float = 50.0) -> None:
        """
        Args:
            corner_radius: radius used by TrajectoryBuilder.corner() in cm.
                           TUNE ON REAL ROBOT.
        """
        self._corner_radius: float = corner_radius
        self._sections: List[Section] = []
        self._direction: Direction = Direction.UNKNOWN
        self._section_idx: int = 0
        self._lap: int = 1
        self._state: RaceState = RaceState.WAITING
        self._current_trajectory: Optional[TrajectoryBase] = None
        self._swerve_active: bool = False
        self._line_cooldown: int = 0
        self._parking: ParkingManager = ParkingManager()
        self._last_sign_color: Optional[int] = None   # tracks last pillar color seen

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    def start(self, robot: Robot,
              direction: Direction = Direction.UNKNOWN) -> None:
        """
        Initialise the race manager and build the first section's trajectory.

        Args:
            robot:     current car state (used to build first trajectory)
            direction: CCW, CW, or UNKNOWN.
                       If UNKNOWN, the direction is detected from the first
                       orange (CCW) or blue (CW) line seen in update().
        """
        self._direction    = direction
        self._section_idx  = 0
        self._lap          = 1
        self._swerve_active = False
        self._line_cooldown = 0
        self._last_sign_color = None
        self._state = RaceState.RACING if direction != Direction.UNKNOWN else RaceState.WAITING

        if direction != Direction.UNKNOWN:
            self._sections = (WROTrack.ccw_sections(self._corner_radius)
                              if direction == Direction.CCW
                              else WROTrack.cw_sections(self._corner_radius))
            self._build_trajectory(robot, VisionFrame())

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def update(self, robot: Robot, vision: VisionFrame) -> Optional[TrajectoryBase]:
        """
        Call every 50 Hz tick.  Returns the trajectory to follow.

        Args:
            robot:  latest state from EKF (x, y, theta, speed)
            vision: latest data from the vision team (VisionFrame)

        Returns:
            TrajectoryBase  -- pass directly to steer_ctrl.compute() and
                               drive_ctrl.compute().
            None            -- if direction is still unknown and no line seen yet.
        """
        # ---- WAITING: try to detect direction from first line ----
        if self._state == RaceState.WAITING:
            return self._try_detect_direction(robot, vision)

        # ---- PARKING: delegate to ParkingManager ----
        if self._state == RaceState.PARKING:
            traj = self._parking.update(robot)
            if self._parking.state == "DONE":
                self._state = RaceState.DONE
            return traj

        # ---- DONE: nothing to do ----
        if self._state == RaceState.DONE:
            return self._current_trajectory

        # ---- RACING ----

        # Tick down the line-detection cooldown
        if self._line_cooldown > 0:
            self._line_cooldown -= 1

        # Rebuild trajectory if we somehow have none
        if self._current_trajectory is None:
            self._build_trajectory(robot, vision)
            return self._current_trajectory

        sec = self._current_section

        # Pillar swerve: if a pillar is reported on a straight section,
        # replace the current straight trajectory with a smooth bypass path.
        if (sec.kind == "straight" and
                not self._swerve_active and
                len(vision.pillars) > 0):
            self._current_trajectory = self._build_pillar_swerve(robot, vision)
            self._swerve_active = True

        # Track last sign color seen across laps 1 and 2 (WRO 2026 direction rule)
        if vision.pillars and self._lap <= 2:
            self._last_sign_color = vision.pillars[0][2]

        # Section completion check
        if self._is_section_complete(robot, vision):
            self._advance_section(robot, vision)

        return self._current_trajectory

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def current_trajectory(self) -> Optional[TrajectoryBase]:
        """The trajectory the controllers should follow right now."""
        return self._current_trajectory

    @property
    def section(self) -> int:
        """Current section index within the lap (0-7)."""
        return self._section_idx

    @property
    def lap(self) -> int:
        """Current lap number (1-based; becomes TOTAL_LAPS+1 when parking starts)."""
        return self._lap

    @property
    def state(self) -> RaceState:
        """Current race state."""
        return self._state

    @property
    def direction(self) -> Direction:
        """Race direction (UNKNOWN until first line detected)."""
        return self._direction

    @property
    def is_done(self) -> bool:
        """True when the car has finished parking."""
        return self._state == RaceState.DONE

    @property
    def _current_section(self) -> Optional[Section]:
        """The Section object for the current section_idx."""
        if not self._sections:
            return None
        return self._sections[self._section_idx]

    # ------------------------------------------------------------------
    # Private: trajectory building
    # ------------------------------------------------------------------

    def _build_trajectory(self, robot: Robot, vision: VisionFrame) -> None:
        """Build and store the trajectory for the current section."""
        sec = self._current_section
        if sec is None:
            return

        if sec.kind == "straight":
            self._current_trajectory = TrajectoryBuilder.straight(
                sec.entry_x, sec.entry_y,
                sec.exit_x,  sec.exit_y,
            )
        else:  # corner
            self._current_trajectory = TrajectoryBuilder.corner(
                sec.entry_x, sec.entry_y, sec.entry_theta,
                sec.turn_direction, sec.corner_radius,
            )

    def _build_pillar_swerve(self, robot: Robot,
                              vision: VisionFrame) -> TrajectoryBase:
        """
        Replace the current straight trajectory with a smooth bypass around
        the nearest reported pillar.

        Takes the pillar closest to the car (vision.pillars[0], already sorted
        by the vision team), builds a 2-segment BezierPath that:
            start  = robot's current position
            bypass = pillar_xy shifted sideways by clearance (RED=right, GREEN=left)
            end    = current section's exit_xy, heading = section's exit_theta
        """
        pillar_x, pillar_y, pillar_color = vision.pillars[0]
        sec = self._current_section
        return TrajectoryBuilder.pillar_swerve(
            robot.x,   robot.y,  robot.theta,
            pillar_x,  pillar_y, pillar_color,
            sec.exit_x, sec.exit_y, sec.exit_theta,
        )

    # ------------------------------------------------------------------
    # Private: state transitions
    # ------------------------------------------------------------------

    def _is_section_complete(self, robot: Robot, vision: VisionFrame) -> bool:
        """
        True when the car should advance to the next section.

        Primary trigger:  line detection (orange or blue), with a cooldown
                          to prevent re-triggering on the same crossing.
        Secondary trigger: Euclidean distance to the section's exit point
                          falls below SECTION_END_CM.
        """
        sec = self._current_section
        if sec is None:
            return False

        # Line detection (primary)
        if self._line_cooldown == 0:
            if vision.orange_line_seen or vision.blue_line_seen:
                return True

        # Proximity to exit point (secondary / fallback)
        dx = robot.x - sec.exit_x
        dy = robot.y - sec.exit_y
        return math.sqrt(dx * dx + dy * dy) < self.SECTION_END_CM

    def _advance_section(self, robot: Robot, vision: VisionFrame) -> None:
        """
        Move to the next section, rebuild the trajectory.

        If section 7 just completed: increment the lap counter.
        If lap counter exceeds TOTAL_LAPS: start parking.
        """
        old_idx = self._section_idx
        self._section_idx   = (self._section_idx + 1) % self.SECTIONS_PER_LAP
        self._swerve_active = False
        self._line_cooldown = self.LINE_COOLDOWN_TICKS

        # Completed a full lap when the section index wraps 7 -> 0
        if old_idx == self.SECTIONS_PER_LAP - 1:
            self._lap += 1

            # WRO 2026 special rule: last sign RED after lap 2 -> reverse direction
            if self._lap == 3 and self._last_sign_color == RED:
                if self._direction == Direction.CCW:
                    self._direction = Direction.CW
                    self._sections  = WROTrack.cw_sections(self._corner_radius)
                else:
                    self._direction = Direction.CCW
                    self._sections  = WROTrack.ccw_sections(self._corner_radius)

            if self._lap > self.TOTAL_LAPS:
                self._start_parking(robot, vision)
                return

        self._build_trajectory(robot, vision)

    def _start_parking(self, robot: Robot, vision: VisionFrame) -> None:
        """Transition from RACING to PARKING."""
        self._state = RaceState.PARKING

        if vision.parking_lot is not None:
            lot_x, lot_y, lot_theta = vision.parking_lot
        else:
            # Fallback: WRO 2026 rules — parking lot is ALWAYS on the bottom
            # starting straight, against the outer wall (y=0).
            # Entry is at y=LOT_DEPTH_CM from the outer wall (inner edge of lot).
            # Car enters heading South (-π/2) into the lot toward y=0.
            # X position is variable (set by judges); fallback guesses ahead of car:
            #   CCW car finishes heading East  → lot likely right of centre (x=200)
            #   CW  car finishes heading West  → lot likely left  of centre (x=100)
            lot_theta = -math.pi / 2          # always South (into outer wall)
            lot_y     = LOT_DEPTH_CM          # inner edge of lot ≈ 27 cm from wall
            lot_x     = 200.0 if self._direction == Direction.CCW else 100.0

        self._parking.set_lot(lot_x, lot_y, lot_theta)
        self._parking.build_approach(robot)

    def _try_detect_direction(self, robot: Robot,
                               vision: VisionFrame) -> Optional[TrajectoryBase]:
        """
        When direction is UNKNOWN, detect from first line crossing.
        Blue line -> CCW.   Orange line -> CW.
        """
        if vision.blue_line_seen:
            self.start(robot, Direction.CCW)
        elif vision.orange_line_seen:
            self.start(robot, Direction.CW)
        return self._current_trajectory

    def __repr__(self) -> str:
        return (f"RaceManager(lap={self._lap}/{self.TOTAL_LAPS}, "
                f"section={self._section_idx}, "
                f"state={self._state.value}, "
                f"dir={self._direction.value})")


# ===========================================================================
# TEST -- run from project root:  python -m main.race_manager
# ===========================================================================
if __name__ == "__main__":
    import math

    print("=" * 60)
    print("RACE MANAGER TEST SUITE")
    print("=" * 60)

    def _robot_at(x, y, theta=0.0, speed=0.0):
        r = Robot()
        r.update_pose(x, y, theta)
        r.update_speed(speed)
        return r

    # ------------------------------------------------------------------
    # Test 1: CCW track has 8 sections with correct geometry
    # ------------------------------------------------------------------
    print("\nTEST 1: CCW track -- 8 sections, correct entry/exit points")
    secs = WROTrack.ccw_sections()
    assert len(secs) == 8, f"Expected 8 sections, got {len(secs)}"

    # Section 0: straight East
    s0 = secs[0]
    assert s0.kind == "straight"
    assert abs(s0.entry_x - 150) < 0.1 and abs(s0.entry_y - 50) < 0.1
    assert abs(s0.exit_x  - 250) < 0.1 and abs(s0.exit_y  - 50) < 0.1
    assert abs(s0.entry_theta)   < 0.01   # heading East = 0

    # Section 7 exit should connect back to section 0 entry
    s7 = secs[7]
    assert abs(s7.exit_x - secs[0].entry_x) < 0.1
    assert abs(s7.exit_y - secs[0].entry_y) < 0.1
    print("  8 sections, section 0 East, loop closes  PASS")

    # ------------------------------------------------------------------
    # Test 2: CW track closes loop correctly
    # ------------------------------------------------------------------
    print("\nTEST 2: CW track -- 8 sections, section 0 West, loop closes")
    secs_cw = WROTrack.cw_sections()
    assert len(secs_cw) == 8
    s0_cw = secs_cw[0]
    assert s0_cw.kind == "straight"
    assert abs(s0_cw.entry_x - 150) < 0.1 and abs(s0_cw.entry_y - 50) < 0.1
    assert abs(abs(s0_cw.entry_theta) - math.pi) < 0.01  # heading West = pi
    s7_cw = secs_cw[7]
    assert abs(s7_cw.exit_x - secs_cw[0].entry_x) < 0.1
    assert abs(s7_cw.exit_y - secs_cw[0].entry_y) < 0.1
    print("  8 sections, section 0 West, loop closes  PASS")

    # ------------------------------------------------------------------
    # Test 3: CCW straight-corner exit/entry continuity (all 8 sections)
    # ------------------------------------------------------------------
    print("\nTEST 3: All CCW section exit_xy == next section entry_xy")
    for i in range(8):
        sec  = secs[i]
        nxt  = secs[(i + 1) % 8]
        assert abs(sec.exit_x - nxt.entry_x) < 0.1, \
            f"Section {i} exit_x={sec.exit_x} != section {(i+1)%8} entry_x={nxt.entry_x}"
        assert abs(sec.exit_y - nxt.entry_y) < 0.1, \
            f"Section {i} exit_y={sec.exit_y} != section {(i+1)%8} entry_y={nxt.entry_y}"
    print("  All 8 sections connect end-to-end  PASS")

    # ------------------------------------------------------------------
    # Test 4: start() builds trajectory for section 0
    # ------------------------------------------------------------------
    print("\nTEST 4: start(CCW) builds trajectory for section 0 (straight)")
    race = RaceManager()
    robot = _robot_at(150, 50, 0.0)
    race.start(robot, Direction.CCW)
    assert race.state   == RaceState.RACING
    assert race.section == 0
    assert race.lap     == 1
    assert race.direction == Direction.CCW
    traj = race.current_trajectory
    assert traj is not None
    assert traj.total_length > 0
    print(f"  section 0 trajectory length = {traj.total_length:.2f} cm  PASS")

    # ------------------------------------------------------------------
    # Test 5: update() when far from end -- no section advance
    # ------------------------------------------------------------------
    print("\nTEST 5: update() far from section end -- section stays at 0")
    vision = VisionFrame()
    traj_returned = race.update(_robot_at(150, 50, 0.0), vision)
    assert race.section == 0
    assert traj_returned is traj   # same trajectory object
    print("  section unchanged, same trajectory returned  PASS")

    # ------------------------------------------------------------------
    # Test 6: update() near section 0 exit -- advances to section 1
    # ------------------------------------------------------------------
    print("\nTEST 6: update() near section 0 exit -- advances to section 1")
    race2 = RaceManager()
    race2.start(_robot_at(150, 50, 0.0), Direction.CCW)
    assert race2.section == 0

    # Place robot within 15 cm of section 0 exit (250, 50)
    race2.update(_robot_at(240, 50, 0.0), VisionFrame())
    assert race2.section == 1, f"Expected section 1, got {race2.section}"
    assert race2.current_trajectory is not None
    print(f"  advanced to section 1 (corner)  PASS")

    # ------------------------------------------------------------------
    # Test 7: 8 section advances -> lap 2
    # ------------------------------------------------------------------
    print("\nTEST 7: 8 section advances complete lap 1 -> lap becomes 2")
    race3 = RaceManager()
    race3.start(_robot_at(150, 50, 0.0), Direction.CCW)
    assert race3.lap == 1

    # Simulate 8 section completions by putting robot near each exit
    for i in range(8):
        sec = WROTrack.ccw_sections()[i]
        # Inject position very close to exit (within SECTION_END_CM)
        race3._section_idx = i
        race3._swerve_active = False
        race3._line_cooldown = 0
        race3._build_trajectory(_robot_at(sec.entry_x, sec.entry_y), VisionFrame())
        race3.update(_robot_at(sec.exit_x, sec.exit_y, sec.exit_theta), VisionFrame())

    # After completing section 7, lap should be 2
    assert race3.lap == 2, f"Expected lap 2, got {race3.lap}"
    assert race3.section == 0  # wrapped back to start
    print(f"  lap counter = {race3.lap}  PASS")

    # ------------------------------------------------------------------
    # Test 8: pillar detected on straight -> swerve trajectory
    # ------------------------------------------------------------------
    print("\nTEST 8: pillar detected on straight section -> swerve trajectory")
    race4 = RaceManager()
    race4.start(_robot_at(150, 50, 0.0), Direction.CCW)
    assert race4._current_section.kind == "straight"

    # Report a GREEN pillar at (200, 50) -- car is at (160, 50)
    v_pillar = VisionFrame(pillars=[(200.0, 50.0, GREEN)])
    traj_before = race4.current_trajectory
    race4.update(_robot_at(160, 50, 0.0), v_pillar)
    traj_after  = race4.current_trajectory

    assert traj_after is not traj_before, "Expected new swerve trajectory"
    assert race4._swerve_active == True
    # Swerve for GREEN should go to the LEFT (+y side)
    s_vals = [traj_after.total_length * i / 50 for i in range(51)]
    ys = [traj_after.get_point(s)[1] for s in s_vals]
    assert max(ys) > 50.0, f"GREEN swerve should go above y=50, max_y={max(ys):.2f}"
    print(f"  swerve active, max_y = {max(ys):.2f} cm (expected > 50)  PASS")

    # ------------------------------------------------------------------
    # Test 9: direction detection from orange line
    # ------------------------------------------------------------------
    print("\nTEST 9: Direction.UNKNOWN + orange line -> CCW detected")
    race5 = RaceManager()
    race5.start(_robot_at(150, 50, 0.0), Direction.UNKNOWN)
    assert race5.state     == RaceState.WAITING
    assert race5.direction == Direction.UNKNOWN

    race5.update(_robot_at(150, 50, 0.0), VisionFrame(orange_line_seen=True))
    assert race5.direction == Direction.CCW, f"Expected CCW, got {race5.direction}"
    assert race5.state     == RaceState.RACING
    print(f"  orange line -> direction={race5.direction.value}  PASS")

    # ------------------------------------------------------------------
    # Test 10: 3 laps -> PARKING state
    # ------------------------------------------------------------------
    print("\nTEST 10: After 3 full laps -> transitions to PARKING")
    race6 = RaceManager()
    race6.start(_robot_at(150, 50, 0.0), Direction.CCW)

    # Simulate 3 full laps = 24 section completions
    for lap in range(3):
        for i in range(8):
            sec = WROTrack.ccw_sections()[i]
            race6._section_idx  = i
            race6._swerve_active = False
            race6._line_cooldown = 0
            race6._build_trajectory(_robot_at(sec.entry_x, sec.entry_y), VisionFrame())
            # Provide lot position on last section so parking can start
            lot = (200.0, 50.0, 0.0) if (lap == 2 and i == 7) else None
            race6.update(
                _robot_at(sec.exit_x, sec.exit_y, sec.exit_theta),
                VisionFrame(parking_lot=lot),
            )

    assert race6.state == RaceState.PARKING, f"Expected PARKING, got {race6.state}"
    assert race6._parking.state in ("APPROACH", "DRIVE_IN", "DONE")
    print(f"  state={race6.state.value}, parking_state={race6._parking.state}  PASS")

    # ------------------------------------------------------------------
    # Test 11: repr()
    # ------------------------------------------------------------------
    print("\nTEST 11: __repr__")
    race7 = RaceManager()
    race7.start(_robot_at(150, 50, 0.0), Direction.CCW)
    print(f"  {race7}")
    print("  PASS")

    # ------------------------------------------------------------------
    # Test 12: WRO 2026 special rule — RED sign at end of lap 2 reverses direction
    # ------------------------------------------------------------------
    print("\nTEST 12: Last sign RED after lap 2 -> lap 3 reverses direction")
    race8 = RaceManager()
    race8.start(_robot_at(150, 50, 0.0), Direction.CCW)

    # Simulate 2 full laps, reporting a RED pillar on the last straight of lap 2
    for lap in range(2):
        for i in range(8):
            sec = WROTrack.ccw_sections()[i]
            race8._section_idx  = i
            race8._swerve_active = False
            race8._line_cooldown = 0
            race8._build_trajectory(_robot_at(sec.entry_x, sec.entry_y), VisionFrame())
            # Report a RED pillar on the last straight of lap 2
            pillar = [(200.0, 50.0, RED)] if (lap == 1 and i == 6) else []
            race8.update(
                _robot_at(sec.exit_x, sec.exit_y, sec.exit_theta),
                VisionFrame(pillars=pillar),
            )

    assert race8.lap == 3, f"Expected lap 3, got {race8.lap}"
    assert race8.direction == Direction.CW, \
        f"Expected CW (reversed), got {race8.direction}"
    assert race8._sections == WROTrack.cw_sections(), \
        "Expected CW sections after direction flip"
    print(f"  lap={race8.lap}, direction={race8.direction.value} (flipped CCW->CW)  PASS")

    # GREEN sign should keep same direction
    print("\nTEST 13: Last sign GREEN after lap 2 -> lap 3 keeps same direction")
    race9 = RaceManager()
    race9.start(_robot_at(150, 50, 0.0), Direction.CCW)
    for lap in range(2):
        for i in range(8):
            sec = WROTrack.ccw_sections()[i]
            race9._section_idx  = i
            race9._swerve_active = False
            race9._line_cooldown = 0
            race9._build_trajectory(_robot_at(sec.entry_x, sec.entry_y), VisionFrame())
            pillar = [(200.0, 50.0, GREEN)] if (lap == 1 and i == 6) else []
            race9.update(
                _robot_at(sec.exit_x, sec.exit_y, sec.exit_theta),
                VisionFrame(pillars=pillar),
            )
    assert race9.lap == 3
    assert race9.direction == Direction.CCW, \
        f"Expected CCW (unchanged), got {race9.direction}"
    print(f"  lap={race9.lap}, direction={race9.direction.value} (kept CCW)  PASS")

    # ------------------------------------------------------------------
    # Test 14: _start_parking() fallback when vision has no lot position
    # ------------------------------------------------------------------
    print("\nTEST 14: Parking fallback when vision.parking_lot is None")
    race10 = RaceManager()
    race10.start(_robot_at(150, 50, 0.0), Direction.CCW)
    robot_end = _robot_at(150, 50, 0.0)
    race10._start_parking(robot_end, VisionFrame())   # no parking_lot in vision
    assert race10.state == RaceState.PARKING
    # CCW fallback: x=200, y=LOT_DEPTH_CM (inner edge of lot), theta=-π/2
    assert race10._parking.lot_position == (200.0, LOT_DEPTH_CM, -math.pi/2)
    print(f"  lot_position={race10._parking.lot_position}  PASS")

    print("\n" + "=" * 60)
    print("ALL 14 RACE MANAGER TESTS PASSED")
    print("=" * 60)
