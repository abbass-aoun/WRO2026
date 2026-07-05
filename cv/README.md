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
