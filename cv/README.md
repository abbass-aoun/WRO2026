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


## Feature 3: Red and Green Pillar Contour Detection

### Purpose
The purpose of this feature is to detect the location and size of red and green pillar-like objects in the camera image.

This feature builds on the previous HSV masking. Instead of only showing red and green pixels, the system now finds connected colored regions and extracts useful information such as the bounding box, center point, and contour area.

### Design Logic
The HSV masks from the previous feature produce binary images, where the target color appears white and the rest of the image appears black. Contour detection is used to find connected white regions in these masks.

Small detected regions are ignored using a minimum area threshold. This reduces false detections caused by noise, reflections, shadows, or small colored objects in the background. For each valid contour, a bounding rectangle is calculated. The center of this rectangle is also calculated because it will later help determine whether the pillar is on the left, center, or right side of the camera view.

The detection result is stored as a list of dictionaries containing the color, bounding box coordinates, center point, and area. This makes the output easier to use later in the robot’s navigation logic. However, we might later create a general detections class, so that the detected pillars become objects of this class. That way, it becomes better for more complex pillar features.

### Algorithm Steps
After creating red and green binary masks using HSV thresholds:
1. Find contours in the red mask.
2. Find contours in the green mask.
3. Calculate the area of each contour.
4. Ignore contours smaller than the minimum area threshold.
5. Calculate a bounding rectangle around each valid contour.
6. Calculate the center point of each bounding rectangle.
7. Store the detection information in a dictionary.
8. Draw bounding boxes, center points, and labels on the camera frame for testing.

### Files Added or Modified
* `config.py`: Added the minimum pillar area and drawing settings.
* `vision.py`: Added pillar detection using contours and a function for drawing detections.
* `test_vision.py`: Updated the testing script to detect red and green objects and display bounding boxes.

### Testing Method
The feature was tested using a live camera feed. Red and green objects were placed in front of the camera and observed in both the mask windows and the main camera window. Correct detection occurs when the object appears white in the correct mask and a bounding box is drawn around it in the camera frame.

The minimum area threshold can be tuned to remove noise. If small noise is detected as a pillar, the threshold can be increased. If valid pillars are missed, the threshold can be decreased.

### Result
The computer vision system can now detect red and green colored regions, filter out small noisy regions, and return structured information about each detected object. The system also displays bounding boxes and center points for visual debugging.

### Limitations
This feature still depends on the quality of the HSV masks. If the HSV thresholds are not tuned correctly, the contour detection may miss a pillar or detect background objects with similar colors. The system also does not yet classify whether the pillar is on the left, center, or right side of the image. It only returns the object center coordinates. Area threshold should be retuned under different lighting conditions.

### Next Step
The next feature will classify the detected pillar position in the camera frame, such as left, center, or right. This will help connect computer vision output to the robot’s navigation decisions.



