"""
main/vision_adapter.py — Bridge between partner's cv/ module and VisionFrame.
==============================================================================

The cv/ module (cv-development branch) outputs camera-relative positions in mm.
This module:
    1. Runs the full camera pipeline in a background thread (camera is ~30 fps,
       main loop is 50 Hz — we must not block on frame capture or OpenCV).
    2. Converts camera-relative (x_mm, y_mm) to world-frame (x_cm, y_cm) using
       the robot's current EKF pose each tick.
    3. Converts pillar color strings ("red"/"green") to the integer constants
       (RED=0 / GREEN=1) used by TrajectoryBuilder.
    4. Returns a ready-to-use VisionFrame for race_manager.update().

COORDINATE SYSTEMS:
    Camera frame (from cv/vision.py):
        relative_x_mm : horizontal offset from camera centre axis
                         positive = object is to the RIGHT of the camera
        relative_y_mm : forward distance from the camera (always positive)

    World frame (our coordinate system):
        x : East, y : North
        theta : 0 = facing East, positive = counter-clockwise

    Transformation (camera → world):
        Forward direction in world  = (cos θ,  sin θ)
        Right   direction in world  = (sin θ, -cos θ)

        world_x = robot.x + fwd_cm * cos(θ) + right_cm * sin(θ)
        world_y = robot.y + fwd_cm * sin(θ) - right_cm * cos(θ)

        where fwd_cm = relative_y_mm / 10, right_cm = relative_x_mm / 10

CAMERA MOUNT OFFSET:
    If the camera is mounted ahead of the robot centre, adjust CAMERA_OFFSET_CM.
    Positive = camera is in front of the robot centre (measured in cm).
    TUNE ON REAL ROBOT.

IMPORT NOTE:
    cv/vision.py does  `from config import ...`  expecting cv/config.py.
    This conflicts with our root config.py.  We temporarily swap sys.modules
    so cv/ modules find their own config, then restore ours.
"""

import math
import os
import sys
import threading

from trajectory.builder import RED, GREEN

# ─────────────────────────────────────────────────────────────────────────────
# Camera mount offset  (cm, measured forward from robot centre)  TUNE ON REAL ROBOT
# ─────────────────────────────────────────────────────────────────────────────
CAMERA_OFFSET_CM = 0.0   # TUNE ON REAL ROBOT

# NOTE FOR PARTNER: cv/config.py has ROBOT_LENGTH_MM = 300 (placeholder).
# Update it to the real robot length (180 mm = 18 cm) so that
# PARKING_LOT_LENGTH_MM = 1.5 × 180 = 270 mm is correct.


# ─────────────────────────────────────────────────────────────────────────────
# Import cv/ modules without polluting the root 'config' namespace
# ─────────────────────────────────────────────────────────────────────────────
def _import_cv():
    """
    Load cv/ modules while keeping our root config.py cached under 'config'.

    Strategy:
        1. Import our root config first → cached as sys.modules['config'].
        2. Temporarily hide it so cv/ finds cv/config.py.
        3. Import cv modules → they cache cv/config.py as 'config'.
        4. Restore our root config under 'config'.
        5. Cache cv/config.py under 'cv_config' for reference.
    """
    import config as _root_cfg          # 1. ensure root config is imported

    _cv_dir = os.path.join(os.path.dirname(__file__), '..', 'cv')
    sys.path.insert(0, _cv_dir)

    _saved = sys.modules.pop('config', None)  # 2. hide root config

    try:
        import importlib
        # 3. Import cv modules – they will pull in cv/config.py as 'config'
        _cam    = importlib.import_module('camera')
        _vis    = importlib.import_module('vision')
        _cv_cfg = sys.modules.get('config')   # this is now cv/config.py
    finally:
        # 4. Restore root config
        if _saved is not None:
            sys.modules['config'] = _saved
        # 5. Store cv/config under a separate name so it stays accessible
        if _cv_cfg is not None:
            sys.modules['cv_config'] = _cv_cfg
        sys.path.remove(_cv_dir)

    return _cam, _vis


try:
    _cam_mod, _vis_mod = _import_cv()

    _open_camera             = _cam_mod.open_camera
    _read_frame              = _cam_mod.read_frame
    _release_camera          = _cam_mod.release_camera
    _convert_to_hsv          = _vis_mod.convert_to_hsv
    _create_red_mask         = _vis_mod.create_red_mask
    _create_green_mask       = _vis_mod.create_green_mask
    _create_pink_mask        = _vis_mod.create_pink_mask
    _detect_pillars          = _vis_mod.detect_pillars
    _detect_parking_markers  = _vis_mod.detect_parking_markers
    _create_navigation_output = _vis_mod.create_navigation_output
    _create_parking_output   = _vis_mod.create_parking_output

    _CV_AVAILABLE = True

except Exception as _e:
    print(f"[VisionAdapter] cv/ module not available: {_e}")
    _CV_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Default outputs (used when camera is unavailable or thread has no result yet)
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULT_NAV = {
    "action": "continue_normal_driving",
    "pillar_color": None,
    "relative_x_mm": None,
    "relative_y_mm": None,
    "confidence": 0.0,
}
_DEFAULT_PARK = {
    "parking_detected": False,
    "parking_status": "not_detected",
    "slot_center_relative_x_mm": None,
    "slot_center_relative_y_mm": None,
}


# ─────────────────────────────────────────────────────────────────────────────
# VisionThread — runs the camera pipeline in a daemon background thread
# ─────────────────────────────────────────────────────────────────────────────

class VisionThread:
    """
    Camera pipeline running in a background daemon thread.

    The 50 Hz main loop reads .navigation_output and .parking_output
    (both thread-safe via a lock), then calls build_vision_frame() to
    convert them into a VisionFrame using the current robot pose.

    Usage:
        vision = VisionThread()
        # inside the 50 Hz loop:
        nav  = vision.navigation_output
        park = vision.parking_output
        vision.stop()   # call once at program exit
    """

    def __init__(self) -> None:
        self._lock        = threading.Lock()
        self._nav         = dict(_DEFAULT_NAV)
        self._park        = dict(_DEFAULT_PARK)
        self._debug_frame = None   # latest raw frame for the debug window
        self._stop_event  = threading.Event()

        if not _CV_AVAILABLE:
            print("[VisionThread] cv/ not available — returning empty outputs.")
            self._thread = None
            return

        self._cap = _open_camera()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------

    @property
    def navigation_output(self) -> dict:
        """Latest pillar detection result (thread-safe copy)."""
        with self._lock:
            return dict(self._nav)

    @property
    def parking_output(self) -> dict:
        """Latest parking lot detection result (thread-safe copy)."""
        with self._lock:
            return dict(self._park)

    @property
    def debug_frame(self):
        """Latest raw camera frame as a NumPy array, or None if not ready."""
        with self._lock:
            f = self._debug_frame
            return f.copy() if f is not None else None

    # ------------------------------------------------------------------

    def _run(self) -> None:
        _err_count = 0
        while not self._stop_event.is_set():
            try:
                frame = _read_frame(self._cap)
                hsv   = _convert_to_hsv(frame)
                h, w  = frame.shape[:2]

                red_det   = _detect_pillars(_create_red_mask(hsv),   "red",   w)
                green_det = _detect_pillars(_create_green_mask(hsv),  "green", w)
                park_det  = _detect_parking_markers(_create_pink_mask(hsv), w)

                nav  = _create_navigation_output(red_det + green_det)
                park = _create_parking_output(park_det)

                with self._lock:
                    self._nav         = nav
                    self._park        = park
                    self._debug_frame = frame

                _err_count = 0   # reset on successful frame

            except Exception as e:
                _err_count += 1
                # Print on first error and every 100 thereafter to avoid log spam
                if _err_count == 1 or _err_count % 100 == 0:
                    print(f"[VisionThread] frame error (x{_err_count}): {e}")
                self._stop_event.wait(0.1)   # brief back-off before retrying

    def stop(self) -> None:
        """Stop the background thread and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if _CV_AVAILABLE and hasattr(self, '_cap'):
            _release_camera(self._cap)


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate transform helper
# ─────────────────────────────────────────────────────────────────────────────

def _cam_to_world(rel_x_mm: float, rel_y_mm: float,
                  robot_x: float, robot_y: float, robot_theta: float) -> tuple:
    """
    Convert a camera-relative position (mm) to world-frame (cm).

    Args:
        rel_x_mm: horizontal offset from camera axis (+= camera right)
        rel_y_mm: forward distance from camera    (+= ahead)
        robot_x, robot_y: robot position in world frame (cm)
        robot_theta:      robot heading in radians (0 = East)

    Returns:
        (world_x_cm, world_y_cm)
    """
    fwd_cm   = rel_y_mm / 10.0
    right_cm = rel_x_mm / 10.0

    cos_t = math.cos(robot_theta)
    sin_t = math.sin(robot_theta)

    # Camera position accounts for its forward offset from robot centre
    cam_x = robot_x + CAMERA_OFFSET_CM * cos_t
    cam_y = robot_y + CAMERA_OFFSET_CM * sin_t

    world_x = cam_x + fwd_cm * cos_t + right_cm * sin_t
    world_y = cam_y + fwd_cm * sin_t - right_cm * cos_t
    return world_x, world_y


# ─────────────────────────────────────────────────────────────────────────────
# VisionFrame builder  (called every 50 Hz tick by the main loop)
# ─────────────────────────────────────────────────────────────────────────────

def build_vision_frame(navigation_output: dict, parking_output: dict,
                       robot, color_sensor):
    """
    Convert the CV thread's outputs into a VisionFrame for race_manager.update().

    Args:
        navigation_output: dict from VisionThread.navigation_output
        parking_output:    dict from VisionThread.parking_output
        robot:             Robot object with current EKF pose (x, y, theta)
        color_sensor:      ColorSensor object for orange_seen / blue_seen

    Returns:
        VisionFrame ready to pass to race_manager.update()
    """
    from main.race_manager import VisionFrame   # imported here to avoid circular import

    rx = robot.x
    ry = robot.y
    rt = robot.theta

    # ── Pillars ───────────────────────────────────────────────────────────────
    pillars = []

    if navigation_output.get("action") == "avoid_pillar":
        rel_x = navigation_output.get("relative_x_mm")
        rel_y = navigation_output.get("relative_y_mm")
        color_str = navigation_output.get("pillar_color")

        if rel_x is not None and rel_y is not None and color_str is not None:
            px, py = _cam_to_world(rel_x, rel_y, rx, ry, rt)
            color  = RED if color_str == "red" else GREEN
            pillars = [(px, py, color)]

    # ── Parking lot ───────────────────────────────────────────────────────────
    # Only trust a full detection (both markers seen).
    # The slot centre detected by CV = the lot ENTRY centre in world frame.
    # Heading is always -π/2 (South, into the outer wall) per WRO 2026 spec.
    parking_lot = None

    if (parking_output.get("parking_detected") and
            parking_output.get("parking_status") == "full_slot_detected"):
        sx = parking_output.get("slot_center_relative_x_mm")
        sy = parking_output.get("slot_center_relative_y_mm")

        if sx is not None and sy is not None:
            lx, ly = _cam_to_world(sx, sy, rx, ry, rt)
            parking_lot = (lx, ly, -math.pi / 2)

    return VisionFrame(
        orange_line_seen = color_sensor.orange_seen,
        blue_line_seen   = color_sensor.blue_seen,
        pillars          = pillars,
        parking_lot      = parking_lot,
    )
