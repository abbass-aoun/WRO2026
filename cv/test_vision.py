import cv2 as cv

from camera import open_camera, read_frame, release_camera
from config import WINDOW_NAME, RED_MASK_WINDOW, GREEN_MASK_WINDOW, PINK_MASK_WINDOW, BLACK_WALL_MASK_WINDOW
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
    create_parking_output,
    create_black_wall_mask,
    detect_wall_slices,
    create_wall_output,
    draw_wall_slices   
)


def main():
    cap = open_camera()

    while True:
        frame = read_frame(cap)

        hsv_frame = convert_to_hsv(frame)


        # 1. Masking
        red_mask = create_red_mask(hsv_frame)
        green_mask = create_green_mask(hsv_frame)
        pink_mask = create_pink_mask(hsv_frame)
        black_wall_mask = create_black_wall_mask(hsv_frame)

        frame_height, frame_width = frame.shape[:2]


        # 2. Detecting
        red_detections = detect_pillars(red_mask, "red", frame_width)
        green_detections = detect_pillars(green_mask, "green", frame_width)
        parking_markers = detect_parking_markers(pink_mask, frame_width)
        wall_slices = detect_wall_slices(
            black_wall_mask,
            frame_width,
            frame_height,
        )

        all_detections = red_detections + green_detections

        
        # 3. Information outputs
        navigation_output = create_navigation_output(all_detections)
        parking_output = create_parking_output(parking_markers)
        wall_output = create_wall_output(wall_slices)


        # 4. Drawing
        output_frame = draw_detections(frame, all_detections)
        output_frame = draw_parking_markers(
            output_frame,
            parking_markers
        )
        output_frame = draw_wall_slices(
            output_frame,
            wall_slices,
            wall_output,
        )

        # 5. Displaying windows
        cv.imshow(WINDOW_NAME, output_frame)
        cv.imshow(RED_MASK_WINDOW, red_mask)
        cv.imshow(GREEN_MASK_WINDOW, green_mask)
        cv.imshow(PINK_MASK_WINDOW, pink_mask)
        cv.imshow(BLACK_WALL_MASK_WINDOW, black_wall_mask)

        key = cv.waitKey(1) & 0xFF

        if key == ord("p"):
            print("\n========== CURRENT VISION OUTPUT ==========")
            print("Navigation:", navigation_output)
            print()
            print("Parking:", parking_output)
            print()
            print("Walls:", wall_output)
            print("===========================================\n")

        # Press q to close the camera window
        if key == ord("q"):
            break

    release_camera(cap)

if __name__ == "__main__":
    main()