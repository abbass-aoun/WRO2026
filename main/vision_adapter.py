import copy
import threading

from cv.camera import open_camera, read_frame, release_camera
from cv.vision import process_image


class VisionThread:

    def __init__(self):
        self._camera = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

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

    def _run(self):
        while not self._stop_event.is_set():

            try:
                frame = read_frame(self._camera)

                result = process_image(frame)

                with self._lock:
                    self._result = result

            except Exception as e:
                print(f"[Vision] Error: {e}")

    def get_latest_result(self):
        with self._lock:
            return copy.deepcopy(self._result)

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
    Add global coordinates to vision detections.

    The actual local_to_global() transformation
    will be provided separately.
    """

    result = copy.deepcopy(vision_result)

    # Transformation will be added here.

    return result