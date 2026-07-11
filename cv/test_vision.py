import cv2 as cv

from camera import open_camera, read_frame, release_camera
from config import WINDOW_NAME, RED_MASK_WINDOW, GREEN_MASK_WINDOW, PINK_MASK_WINDOW
from vision import (
    convert_to_hsv,
    create_red_mask, 
    create_green_mask,
    create_pink_mask,
    draw_detections,
    draw_parking_markers,
    detect_pillars,
    detect_parking_markers,
    create_navigation_output,
    create_parking_output    
)


def main():
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        hsv_frame = convert_to_hsv(frame)

        red_mask = create_red_mask(hsv_frame)
        green_mask = create_green_mask(hsv_frame)
        pink_mask = create_pink_mask(hsv_frame)

        frame_height, frame_width = frame.shape[:2]

        red_detections = detect_pillars(red_mask, "red", frame_width)
        green_detections = detect_pillars(green_mask, "green", frame_width)

        all_detections = red_detections + green_detections

        navigation_output = create_navigation_output(all_detections)
        print("Navigation:", navigation_output)

        parking_markers = detect_parking_markers(pink_mask, frame_width)
        parking_output = create_parking_output(parking_markers)
        print("Parking:", parking_output)

        output_frame = draw_detections(frame, all_detections)
        output_frame = draw_parking_markers(
            output_frame,
            parking_markers,
            parking_output
    )

        cv.imshow(WINDOW_NAME, output_frame)
        cv.imshow(RED_MASK_WINDOW, red_mask)
        cv.imshow(GREEN_MASK_WINDOW, green_mask)
        cv.imshow(PINK_MASK_WINDOW, pink_mask)

        # Press q to close the camera window
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    release_camera(cap)

if __name__ == "__main__":
    main()