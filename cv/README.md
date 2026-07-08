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
This feature still depends on the quality of the HSV masks. If the HSV thresholds are not tuned correctly, the contour detection may miss a pillar or detect background objects with similar colors. 

### Next Step
The next step is to add restraints and condition related to the object's shape, area, height, and width in order to rule out detections of noise and objects that are not pillars.


## Feature 4: Object Filtering and Confidence Logic

### Purpose
The purpose of this feature is to improve the reliability of red and green pillar detection. The previous detection system could find colored regions using contours, but not every colored region is a real WRO pillar. This feature adds filtering rules and a confidence score to reduce false detections caused by noise, reflections, shadows, or unrelated colored objects.

### Design Logic
The system now checks whether each detected contour has properties that are consistent with a pillar-like object. A valid pillar should have enough area, a reasonable bounding box size, a suitable height-to-width ratio, and a contour that fills its bounding box well.

The height-to-width ratio is used because a pillar is expected to appear taller than it is wide in the camera image. The extent value compares the contour area to the bounding box area. A solid rectangular object should fill a reasonable portion of its bounding box, while noisy or broken shapes usually have lower extent.

A rule-based confidence score is calculated for each detection. This score is based on area, height, extent, and aspect ratio. The confidence value is not produced by a machine learning model; it is a manually designed reliability score used to compare and filter detections.

### Algorithm Steps
1. Calculate the contour area.
2. Calculate the bounding rectangle around the contour.
3. Calculate the bounding box width and height.
4. Calculate the aspect ratio using height divided by width.
5. Calculate the extent using contour area divided by bounding box area.
6. Calculate a confidence score from area, height, extent, and aspect ratio.
7. Reject detections that do not pass the minimum filtering thresholds.
8. Store accepted detections with their color, bounding box, center point, area, aspect ratio, extent, and confidence.
9. Draw bounding boxes and confidence labels on the camera frame for testing.

### Files Added or Modified
* `config.py`: Added filtering thresholds for minimum area, minimum width, minimum height, aspect ratio, extent, and confidence.
* `vision.py`: Added confidence calculation and validation logic for pillar detections.
* `test_vision.py`: No major structural change was required because it already calls the detection and drawing functions.

### Testing Method
The feature was tested using a live camera feed with red and green objects. The system was checked to confirm that large pillar-like objects were accepted while small noisy regions and non-pillar-like shapes were rejected. The confidence label displayed on the camera frame was used to compare stronger and weaker detections.

Testing include similar pillar objects at different distances and under different lighting conditions. WHen correct pillars are rejected, the thresholds are relaxed. When false detections are accepted, the thresholds are made stricter.

### Result
The vision system can now filter colored contours more intelligently and assign a confidence score to each accepted detection. This improves the reliability of the computer vision pipeline before it is connected to navigation decisions.

### Limitations
The confidence score is rule-based and depends on manually selected thresholds. These thresholds may need to be changed when testing with the robot camera, official pillar colors, or different lighting conditions. The system still depends on the quality of the HSV masks, so poor threshold values can still cause missed or false detections.

During testing, red false detections appeared on skin under certain lighting conditions. This happened because skin tones can partially overlap with the red HSV range, especially under warm lighting. To reduce this issue, the red saturation threshold was increased so that only stronger red regions are accepted (we changed the lower saturation bound from 100 to 140, and the lower decreased the lower bound of value from 100 to 80). The aspect ratio filter was also adjusted (MIN_ASPECT_RATIO 1.2 -> 1.4) using the official pillar dimensions of 50 mm × 50 mm × 100 mm, giving an expected height-to-width ratio of approximately 2.0.

### Next Step
The next feature will classify the accepted pillar position relative to the camera view, such as left, center, or right. This will make the detection output more useful for navigation decisions.

