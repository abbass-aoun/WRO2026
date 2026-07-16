# cv/camera.py
import time

try:
    from picamera2 import Picamera2
    _PICAMERA2_AVAILABLE = True
except ImportError:
    _PICAMERA2_AVAILABLE = False

_FRAME_WIDTH  = 640
_FRAME_HEIGHT = 480


def open_camera():
    """
    Opens the Raspberry Pi Camera Module using Picamera2 and returns the
    camera object.  BGR888 format produces a NumPy array in BGR order,
    which is directly compatible with all OpenCV functions.
    """
    if not _PICAMERA2_AVAILABLE:
        raise RuntimeError(
            "picamera2 is not installed. Run: pip install picamera2"
        )
    cam = Picamera2()
    config = cam.create_preview_configuration(
        main={"format": "BGR888", "size": (_FRAME_WIDTH, _FRAME_HEIGHT)}
    )
    cam.configure(config)
    cam.start()
    time.sleep(0.5)   # warm-up: allow auto-exposure to stabilise
    return cam


def read_frame(cam):
    """
    Captures one frame and returns it as a (H, W, 3) uint8 NumPy array
    in BGR order, identical in layout to what cv2.VideoCapture.read()
    would have returned.
    """
    return cam.capture_array()


def release_camera(cam):
    """Stops and closes the camera cleanly."""
    cam.stop()
    cam.close()
