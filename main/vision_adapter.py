import copy
import threading

from cv.camera import open_camera, read_frame, release_camera
from cv.vision import process_image
from cv.config import TRACK_LINE_CONFIRM_FRAMES
from cv.vision import process_image, draw_vision_result

from estimation.Transformation import camera_to_world

class VisionThread:

    def __init__(self):
        self._camera = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._debug_frame = None

        self._orange_line_count = 0
        self._blue_line_count = 0

        self._result = {
            "pillars": [],
            "parking": {},
            "walls": {},
        }

    def start(self):
        self._camera = open_camera()

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self._thread.start()
    
    def _update_line_confirmation(self, result):
        """
        Require a nearby line to appear in several consecutive
        processed camera frames before confirming it.
        """

        track_lines = result["track_lines"]

        if track_lines["orange"]["close"]:
            self._orange_line_count += 1
        else:
            self._orange_line_count = 0

        if track_lines["blue"]["close"]:
            self._blue_line_count += 1
        else:
            self._blue_line_count = 0

        track_lines["orange"]["confirmed_close"] = (
            self._orange_line_count
            >= TRACK_LINE_CONFIRM_FRAMES
        )

        track_lines["blue"]["confirmed_close"] = (
            self._blue_line_count
            >= TRACK_LINE_CONFIRM_FRAMES
        )

    def _run(self):
        while not self._stop_event.is_set():

            try:
                frame = read_frame(self._camera)

                result = process_image(frame)

                self._update_line_confirmation(result)

                debug_frame = draw_vision_result(
                    frame,
                    result,
                )
                                
                with self._lock:
                    self._result = result
                    self._debug_frame = debug_frame

            except Exception as e:
                print(f"[Vision] Error: {e}")

    def get_latest_result(self):
        with self._lock:
            return copy.deepcopy(self._result)

    def get_latest_frame(self):
        with self._lock:

            if self._debug_frame is None:
                return None

            return self._debug_frame.copy()

    def stop(self):

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)

        if self._camera is not None:
            release_camera(self._camera)


def transform_to_global(
    vision_result,
    robot_x,
    robot_y,
    robot_theta,
):
    """
    Add global coordinates to detected pillars and parking.

    Original camera-relative coordinates are preserved.
    """

    result = copy.deepcopy(vision_result)

    # ----------------------------------------
    # Pillars
    # ----------------------------------------

    for pillar in result["pillars"]:

        relative_x_mm = pillar.get("relative_x_mm")
        relative_y_mm = pillar.get("relative_y_mm")

        if relative_x_mm is None or relative_y_mm is None:
            pillar["global_x_cm"] = None
            pillar["global_y_cm"] = None
            continue

        
        global_x, global_y = camera_to_world(
            robot_x,
            robot_y,
            robot_theta,
            relative_x_mm,
            relative_y_mm
        )

        pillar["global_x_cm"] = global_x
        pillar["global_y_cm"] = global_y

    # ----------------------------------------
    # Parking slot center
    # ----------------------------------------

    parking = result.get("parking", {})

    relative_x_mm = parking.get(
        "slot_center_relative_x_mm"
    )

    relative_y_mm = parking.get(
        "slot_center_relative_y_mm"
    )

    if relative_x_mm is not None and relative_y_mm is not None:

        global_x, global_y = camera_to_world(
            robot_x,
            robot_y,
            robot_theta,
            relative_x_mm,
            relative_y_mm
        )

        parking["slot_center_global_x_cm"] = global_x
        parking["slot_center_global_y_cm"] = global_y

    else:
        parking["slot_center_global_x_cm"] = None
        parking["slot_center_global_y_cm"] = None

    return result