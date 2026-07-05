# Computer Vision Module
This folder contains the computer vision code for the WRO 2026 Future Engineers self-driving car.

## Currently
The first version will use Python, OpenCV, and HSV color thresholding to detect:

- red traffic signs
- green traffic signs
- magenta parking markers
- orange and blue field lines

## Files
- `camera.py`: handles camera setup and frame capture.
- `vision.py`: processes frames and detects colored objects.
- `config.py`: stores HSV ranges and tuning constants.
- `test_vision.py`: tests the vision system using the live camera.

## Notes
This module is still under development. The first goal is to make reliable red and green pillar detection using OpenCV.


## Feature 1: Camera Feed Test

### Purpose
The purpose of this feature is to verify that the robot’s camera can be accessed using OpenCV and that live frames can be captured and displayed correctly. This is the first step in developing the computer vision system because all later detection functions depend on a stable camera input.

### Design Logic
The camera system was separated into a dedicated `camera.py` file to keep the code organized and reusable. Instead of writing camera code directly inside the testing script, we created functions for opening the camera, reading frames, and releasing the camera. This makes the project easier to expand when color detection and obstacle detection are added later.

The camera index is stored in `config.py` so it can be changed easily when we change to the Rasberry pi camera of the robot.

### Algorithm Steps
1. Load the camera index from the configuration file.
2. Open the camera using OpenCV.
4. Continuously read frames from the camera.
5. Display each frame in a live window.
6. Stop the program when the user presses the `q` key.
7. Release the camera and close all OpenCV windows.

### Files Added or Modified
* `config.py`: stores the camera index and display window name.
* `camera.py`: contains reusable camera functions.
* `test_vision.py`: runs a live camera test.

### Testing Method
The feature was tested by running `python cv/test_vision.py` from the project folder. A live camera window should appear. The camera feed confirms that OpenCV can access the camera successfully. The program should close safely when the `q` key is pressed.

### Result
The camera feed was successfully displayed using OpenCV. This confirms that the basic camera pipeline is ready for the next stage, which is HSV color masking for red and green pillar detection.

### Next Step
The next feature will add HSV color masking to isolate red and green regions from the camera frame.


## Feature 2: HSV Color Masking for Red and Green Pillars

### Purpose
The purpose of this feature is to begin detecting the red and green pillars used in the WRO challenge. The robot must identify the color of each pillar because the navigation rule depends on the pillar color. A red pillar means the robot should pass on the pillar's left side, while a green pillar means the robot should pass on the pillar's right side.

### Design Logic
The camera provides frames in BGR format, which is the default color format used by OpenCV. However, BGR values are strongly affected by brightness and shadows, so they are not ideal for color filtering. To improve color separation, each frame is converted to HSV color space.

HSV separates the image into hue, saturation, and value. Hue represents the main color, saturation represents how strong the color is, and value represents brightness. This makes HSV more suitable for detecting red and green objects under changing lighting conditions.

Red detection requires two HSV ranges because red appears at both ends of the OpenCV hue scale. OpenCV represents hue from 0 to 179, and red exists near both 0 and 179. Therefore, two red masks are created and combined into one final red mask.

### Algorithm Steps
1. Capture a frame from the camera.
2. Convert the frame from BGR color space to HSV color space.
3. Apply the first HSV threshold range for red.
4. Apply the second HSV threshold range for red.
5. Combine both red masks into one red mask.
6. Apply the HSV threshold range for green.
7. Display the original camera frame, the red mask, and the green mask.
8. Use visual testing to check whether red and green objects are correctly isolated.

### Files Added or Modified
* `config.py`: Added initial HSV threshold ranges for red and green, along with window names for mask display.
* `vision.py`: Added functions for HSV conversion, red mask creation, and green mask creation.
* `test_vision.py`: Updated the test script to display the original frame, red mask, and green mask.

### Testing Method
The feature was tested using a live camera feed. Red and green objects were placed in front of the camera while observing the mask windows. A correct result occurs when the target color appears white in its corresponding mask and most of the background remains black.

Testing should be repeated under different lighting conditions because HSV values can change depending on shadows, reflections, and camera exposure.

### Result
The system can now convert live camera frames to HSV and create separate masks for red and green objects. This prepares the project for the next step, which is detecting contours and extracting pillar position information.

### Limitations
The HSV threshold values are initial estimates and may not perfectly match the official WRO pillars or the final robot camera. The masks may also detect other red or green objects in the background. Lighting changes, reflections, and shadows affected the quality of the mask and created noise.

### Next Step
The next feature will use the red and green masks to find object contours, filter small noisy regions, and draw bounding boxes around detected pillars.


