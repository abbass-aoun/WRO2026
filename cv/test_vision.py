import cv2 as cv

from camera import open_camera, read_frame, release_camera
from config import WINDOW_NAME, RED_MASK_WINDOW, GREEN_MASK_WINDOW
from vision import convert_to_hsv, create_red_mask, create_green_mask


def main():
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        hsv_frame = convert_to_hsv(frame)

        red_mask = create_red_mask(hsv_frame)
        green_mask = create_green_mask(hsv_frame)

        cv.imshow(WINDOW_NAME, frame)
        cv.imshow(RED_MASK_WINDOW, red_mask)
        cv.imshow(GREEN_MASK_WINDOW, green_mask)

        # Press q to close the camera window
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(cap)


if __name__ == "__main__":
    main()