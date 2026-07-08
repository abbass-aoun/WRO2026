"""
main/parking.py -- End-of-race parking sequence manager.

WHAT IS THIS?
    After 3 laps, the car must park inside a marked lot on the track.
    This module manages the two-phase parking sequence:

        Phase 1 -- APPROACH:
            A smooth Bezier curve from wherever the car is now to the
            lot entry point.  Built from TrajectoryBuilder.parking_approach().

        Phase 2 -- DRIVE_IN:
            A short straight path from the lot entry, driving 'lot_depth' cm
            into the lot.  Built from TrajectoryBuilder.straight().

    The caller (RaceManager) gives this module the lot position from vision,
    then calls update() every tick.  The module returns the correct trajectory
    and advances its own state.

INTERNAL STATES:
    IDLE       -- lot position not yet known; no path built
    APPROACH   -- following curve toward the lot entry
    DRIVE_IN   -- following straight into the lot
    DONE       -- car is inside the lot; safe to stop

CONNECTIONS:
    Reads:   Robot.x, Robot.y, Robot.theta  (current car state)
    Uses:    TrajectoryBuilder.parking_approach()  -- smooth entry curve
             TrajectoryBuilder.straight()          -- drive-in straight
    Returns: TrajectoryBase objects  -->  given to steering/driving controllers
    Called by: RaceManager once lap 3 is complete
"""

import math
from trajectory.base import TrajectoryBase
from trajectory.builder import TrajectoryBuilder
from control.robot import Robot


class ParkingManager:
    """
    Manages the two-phase end-of-race parking sequence.

    Usage:
        pm = ParkingManager()
        pm.set_lot(lot_x, lot_y, lot_theta)          # once, from vision data
        traj = pm.build_approach(robot)               # once, to start parking
        # every 50 Hz tick:
        traj = pm.update(robot)
        if pm.state == "DONE":
            stop_motors()
    """

    # WRO 2026 rules:
    #   Width is ALWAYS 20 cm (gap between the two magenta marker blocks).
    #   Depth = 1.5 × robot length — calculated once the physical robot is built.
    ROBOT_LENGTH_CM:  float = 18.0   # cm total car length     (TUNE ON REAL ROBOT)
    DEFAULT_LOT_WIDTH: float = 20.0  # cm — fixed by WRO 2026 rules
    DEFAULT_LOT_DEPTH: float = 27.0  # cm — 1.5 × ROBOT_LENGTH_CM
    APPROACH_DONE_CM: float = 15.0   # how close to lot entry counts as "arrived"

    def __init__(self) -> None:
        self._lot_x:     float = None
        self._lot_y:     float = None
        self._lot_theta: float = None   # heading to drive INTO the lot (radians)
        self._lot_width: float = self.DEFAULT_LOT_WIDTH
        self._lot_depth: float = self.DEFAULT_LOT_DEPTH

        self._approach_path: TrajectoryBase = None
        self._drive_in_path: TrajectoryBase = None
        self._state: str = "IDLE"

    # ------------------------------------------------------------------
    # Setup -- called once when vision confirms the lot
    # ------------------------------------------------------------------

    def set_lot(self, lot_x: float, lot_y: float, lot_theta: float,
                lot_width: float = DEFAULT_LOT_WIDTH,
                lot_depth: float = DEFAULT_LOT_DEPTH) -> None:
        """
        Register the parking lot position from the vision team.

        Args:
            lot_x, lot_y:  position of the lot ENTRY edge centre (cm)
            lot_theta:     heading to drive INTO the lot (radians).
                           E.g. lot is to the North  -> lot_theta = pi/2
            lot_width:     lateral width in cm       # TUNE ON REAL ROBOT
            lot_depth:     depth to drive in (cm)    # TUNE ON REAL ROBOT
        """
        self._lot_x     = float(lot_x)
        self._lot_y     = float(lot_y)
        self._lot_theta = float(lot_theta)
        self._lot_width = float(lot_width)
        self._lot_depth = float(lot_depth)
        self._state     = "IDLE"

    # ------------------------------------------------------------------
    # Path builders
    # ------------------------------------------------------------------

    def build_approach(self, robot: Robot) -> TrajectoryBase:
        """
        Build a smooth Bezier curve from the car's current position to the
        lot entry.  Call this ONCE when you decide to begin parking.

        Returns:
            BezierSegment to follow until approach_done() is True.

        Raises:
            RuntimeError: if set_lot() has not been called yet.
        """
        if self._lot_x is None:
            raise RuntimeError("ParkingManager: call set_lot() before build_approach().")

        self._approach_path = TrajectoryBuilder.parking_approach(
            robot.x, robot.y, robot.theta,
            self._lot_x, self._lot_y, self._lot_theta,
        )
        self._state = "APPROACH"
        return self._approach_path

    def build_drive_in(self) -> TrajectoryBase:
        """
        Build a short straight path from the lot entry into the lot.
        Called automatically by update() when approach_done() becomes True.

        Stop point: (lot_depth - ROBOT_LENGTH_CM/2) from the entry edge.
        This ensures the car's rear clears the entry and the front stays
        clear of the back wall when the car is fully inside.

        Returns:
            WaypointPath to follow until is_complete() returns True.
        """
        if self._lot_x is None:
            raise RuntimeError("ParkingManager: call set_lot() before build_drive_in().")

        # Drive in until the car centre is at (lot_depth - half_car) from entry.
        # Example: lot_depth=27 cm, half_car=9 cm → stop 18 cm in.
        # At that point: rear at 18-9=9 cm from entry (inside) ✓
        #                front at 18+9=27 cm from entry = exactly at back wall ✓
        stop_dist = self._lot_depth - self.ROBOT_LENGTH_CM / 2
        end_x = self._lot_x + stop_dist * math.cos(self._lot_theta)
        end_y = self._lot_y + stop_dist * math.sin(self._lot_theta)

        self._drive_in_path = TrajectoryBuilder.straight(
            self._lot_x, self._lot_y, end_x, end_y,
        )
        self._state = "DRIVE_IN"
        return self._drive_in_path

    # ------------------------------------------------------------------
    # Main tick function
    # ------------------------------------------------------------------

    def update(self, robot: Robot) -> TrajectoryBase:
        """
        Advance the state machine.  Call every 50 Hz tick.

        Returns the trajectory to follow this tick, or None if IDLE.
        Automatically transitions APPROACH -> DRIVE_IN -> DONE.
        """
        if self._state == "IDLE":
            return None

        if self._state == "APPROACH":
            if self.approach_done(robot):
                return self.build_drive_in()
            return self._approach_path

        if self._state == "DRIVE_IN":
            if self._is_inside_lot(robot):
                self._state = "DONE"
            return self._drive_in_path

        # DONE -- keep returning the drive-in path so controllers stay calm
        return self._drive_in_path

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def approach_done(self, robot: Robot) -> bool:
        """True when car is within APPROACH_DONE_CM of the lot entry."""
        if self._lot_x is None:
            return False
        return self._distance_to_entry(robot) < self.APPROACH_DONE_CM

    def is_complete(self, robot: Robot) -> bool:
        """True when the car is inside the parking lot boundaries."""
        if self._lot_x is None:
            return False
        return self._is_inside_lot(robot)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        """Current state: IDLE | APPROACH | DRIVE_IN | DONE."""
        return self._state

    @property
    def current_path(self) -> TrajectoryBase:
        """Active trajectory (None while IDLE)."""
        if self._state == "APPROACH":
            return self._approach_path
        if self._state in ("DRIVE_IN", "DONE"):
            return self._drive_in_path
        return None

    @property
    def lot_position(self) -> tuple:
        """(x, y, theta) of the lot entry, or None if not set."""
        if self._lot_x is None:
            return None
        return (self._lot_x, self._lot_y, self._lot_theta)

    def __repr__(self) -> str:
        return f"ParkingManager(state={self._state}, lot={self.lot_position})"

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _distance_to_entry(self, robot: Robot) -> float:
        """Straight-line distance from robot to lot entry centre (cm)."""
        dx = robot.x - self._lot_x
        dy = robot.y - self._lot_y
        return math.sqrt(dx * dx + dy * dy)

    def _is_inside_lot(self, robot: Robot) -> bool:
        """
        Check if the robot is inside the parking lot rectangle.

        Transforms the robot position into lot-local coordinates:
            local_x: lateral offset from lot centre-line  (should be +-lot_width/2)
            local_y: depth into the lot from entry edge   (should be 0..lot_depth)
        """
        dx = robot.x - self._lot_x
        dy = robot.y - self._lot_y
        ct = math.cos(self._lot_theta)
        st = math.sin(self._lot_theta)
        # along: distance driven INTO the lot (dot product with entry direction)
        along = ct * dx + st * dy
        # perp: lateral offset from the lot centre-line
        perp  = -st * dx + ct * dy
        half_w = self._lot_width / 2.0
        return ((-half_w <= perp  <= half_w) and
                (0.0 <= along <= self._lot_depth))


# ===========================================================================
# TEST -- run from project root:  python -m main.parking
# ===========================================================================
if __name__ == "__main__":
    import math

    print("=" * 60)
    print("PARKING MANAGER TEST SUITE")
    print("=" * 60)

    def _make_robot(x, y, theta=0.0):
        """Helper: build a Robot with given pose."""
        from control.robot import Robot
        r = Robot()
        r.update_pose(x, y, theta)
        return r

    # ------------------------------------------------------------------
    # Test 1: IDLE state before set_lot
    # ------------------------------------------------------------------
    print("\nTEST 1: IDLE state -- no path until set_lot() called")
    pm = ParkingManager()
    assert pm.state == "IDLE"
    assert pm.current_path is None
    assert pm.lot_position is None
    assert pm.update(_make_robot(0, 0)) is None
    print("  state=IDLE, current_path=None, lot_position=None  PASS")

    # ------------------------------------------------------------------
    # Test 2: set_lot stores values
    # ------------------------------------------------------------------
    print("\nTEST 2: set_lot() stores lot geometry")
    pm.set_lot(200.0, 50.0, 0.0, lot_width=20.0, lot_depth=27.0)
    assert pm.lot_position == (200.0, 50.0, 0.0)
    assert pm.state == "IDLE"
    print(f"  lot_position={pm.lot_position}  PASS")

    # ------------------------------------------------------------------
    # Test 3: build_approach returns a valid trajectory
    # ------------------------------------------------------------------
    print("\nTEST 3: build_approach() builds a BezierSegment path")
    robot = _make_robot(50.0, 50.0, 0.0)  # car far from lot
    path = pm.build_approach(robot)
    assert path is not None
    assert path.total_length > 0
    assert pm.state == "APPROACH"
    print(f"  approach path length = {path.total_length:.2f} cm  PASS")

    # ------------------------------------------------------------------
    # Test 4: approach_done -- far away = False, close = True
    # ------------------------------------------------------------------
    print("\nTEST 4: approach_done() distance threshold")
    far_robot  = _make_robot(50.0,  50.0, 0.0)   # 150 cm from lot entry
    near_robot = _make_robot(190.0, 50.0, 0.0)   # 10 cm from lot entry (< 15)
    assert pm.approach_done(far_robot)  == False
    assert pm.approach_done(near_robot) == True
    print("  far -> False, near -> True  PASS")

    # ------------------------------------------------------------------
    # Test 5: update transitions APPROACH -> DRIVE_IN when close
    # ------------------------------------------------------------------
    print("\nTEST 5: update() transitions APPROACH -> DRIVE_IN")
    pm2 = ParkingManager()
    pm2.set_lot(200.0, 50.0, 0.0, lot_width=20.0, lot_depth=27.0)
    pm2.build_approach(_make_robot(50.0, 50.0, 0.0))
    assert pm2.state == "APPROACH"

    drive_in_path = pm2.update(_make_robot(192.0, 50.0, 0.0))  # close to entry
    assert pm2.state == "DRIVE_IN", f"Expected DRIVE_IN, got {pm2.state}"
    assert drive_in_path is not None
    # stop_dist = 27 - 9 = 18 cm
    assert abs(drive_in_path.total_length - 18.0) < 0.5, \
        f"Expected drive-in length ~18 cm, got {drive_in_path.total_length:.2f}"
    print(f"  transitioned to DRIVE_IN, path length={drive_in_path.total_length:.2f} cm  PASS")

    # ------------------------------------------------------------------
    # Test 6: is_complete -- outside = False, inside = True
    # ------------------------------------------------------------------
    print("\nTEST 6: is_complete() checks car inside lot rectangle")
    pm3 = ParkingManager()
    pm3.set_lot(200.0, 50.0, 0.0, lot_width=20.0, lot_depth=27.0)
    # Lot: entry at (200,50), heading East (theta=0)
    # along=dx, perp=dy  (since theta=0, ct=1, st=0)
    # Inside: 0<=dx<=27 and -10<=dy<=10
    outside = _make_robot(100.0, 50.0, 0.0)   # far left of lot
    inside  = _make_robot(215.0, 50.0, 0.0)   # 15 cm into lot, on centreline
    assert pm3.is_complete(outside) == False
    assert pm3.is_complete(inside)  == True
    print("  outside -> False, inside -> True  PASS")

    # ------------------------------------------------------------------
    # Test 7: full sequence IDLE -> APPROACH -> DRIVE_IN -> DONE
    # ------------------------------------------------------------------
    print("\nTEST 7: Full state machine sequence")
    pm4 = ParkingManager()
    assert pm4.state == "IDLE"

    pm4.set_lot(100.0, 0.0, 0.0, lot_width=20.0, lot_depth=27.0)
    pm4.build_approach(_make_robot(0.0, 0.0, 0.0))
    assert pm4.state == "APPROACH"

    # Tick near entry -> transitions to DRIVE_IN
    pm4.update(_make_robot(92.0, 0.0, 0.0))
    assert pm4.state == "DRIVE_IN"

    # Tick inside lot (15 cm deep, on centreline) -> transitions to DONE
    pm4.update(_make_robot(115.0, 0.0, 0.0))
    assert pm4.state == "DONE"
    print("  IDLE -> APPROACH -> DRIVE_IN -> DONE  PASS")

    # ------------------------------------------------------------------
    # Test 8: drive-in path ends at stop_dist = lot_depth - ROBOT_LENGTH/2
    # ------------------------------------------------------------------
    print("\nTEST 8: drive-in path ends at lot_depth - robot_half_length inside lot")
    pm5 = ParkingManager()
    pm5.set_lot(100.0, 0.0, math.pi/2, lot_depth=27.0)  # lot facing North
    pm5.build_approach(_make_robot(50.0, -50.0, math.pi/2))
    pm5.update(_make_robot(100.0, -5.0, math.pi/2))     # trigger drive-in
    assert pm5.state == "DRIVE_IN"
    end = pm5.current_path.get_point(pm5.current_path.total_length)
    # stop_dist = 27 - 18/2 = 18 cm north of entry (entry at y=0, heading North)
    expected_end_y = 0.0 + 18.0
    print(f"  drive-in end: ({end[0]:.2f}, {end[1]:.2f})  expected (~100, ~18)")
    assert abs(end[0] - 100.0) < 0.1
    assert abs(end[1] - expected_end_y) < 0.1, \
        f"Expected end_y={expected_end_y}, got {end[1]:.2f}"
    print("  PASS")

    print("\n" + "=" * 60)
    print("ALL 8 PARKING TESTS PASSED")
    print("=" * 60)
