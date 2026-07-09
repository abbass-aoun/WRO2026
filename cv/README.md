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

A rule-based confidence score is calculated for each detection. This score is based on area, height, extent, and aspect ratio.

The confidence score is a manually designed rule-based score, not a machine learning probability. The weights were selected based on which properties are expected to best indicate a real WRO pillar. Contour area was given the highest weight because a real pillar should occupy a noticeable region in the image, while very small regions are often noise. Height and extent were also given significant weight because the official pillar is taller than it is wide and should appear as a relatively solid colored shape. Aspect ratio was included with a smaller weight because the real pillar has an expected height-to-width ratio of about 2.0, but camera angle and perspective can change the apparent ratio. These weights are initial values and should be tuned after testing with the final robot camera and official pillar objects.

Before calculating the final confidence, each measurement is converted into a normalized score between 0 and 1:

area_score = min(area / 3000, 1.0)
height_score = min(height / 120, 1.0)
extent_score = min(extent / 0.8, 1.0)

The area_score compares the contour area to a target area of 3000 pixels. A larger colored region is more likely to be a real pillar, while very small regions are more likely to be noise. If the area reaches or exceeds 3000 pixels, the score is limited to 1.0.

The height_score compares the bounding box height to a target height of 120 pixels. Since the official pillar is taller than it is wide, taller detections are considered more reliable than small blobs. If the height reaches or exceeds 120 pixels, the score is limited to 1.0.

The extent_score compares the contour extent to a target value of 0.8. Extent is calculated as the contour area divided by the bounding box area. A solid rectangular pillar should fill a large portion of its bounding box, while broken, noisy, or irregular detections usually have lower extent. If the extent reaches or exceeds 0.8, the score is limited to 1.0.

The min(..., 1.0) function is used to keep each score within the range of 0 to 1. This prevents one very large measurement from dominating the final confidence score.

These target values are initial estimates selected for testing. They depend on the camera resolution, camera angle, distance from the pillar, lighting conditions, and the final Raspberry Pi camera setup. Therefore, they should be tuned a

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


## Feature 5: Camera-Relative Pillar Position

### Purpose
The purpose of this feature is to classify the horizontal position of each detected pillar relative to the camera view. After detecting a red or green pillar, the robot needs to know whether the pillar appears on the left, center, or right side of the image. This information is important for later navigation decisions.

### Design Logic
The camera image is divided into three horizontal regions: left, center, and right. The center x-coordinate of each detected pillar is compared with two boundaries based on the frame width. If the center of the pillar is in the left third of the image, the detection is classified as left. If it is in the right third, it is classified as right. Otherwise, it is classified as center.

This method is simple and fast, which makes it suitable for real-time robot vision. It does not require a trained model and can run on lower-power hardware such as a Raspberry Pi.

### Algorithm Steps
Starting from the filtered detections produced in Feature 4:

1. Read the width of the current camera frame.
2. Calculate the left boundary as 33% of the frame width.
3. Calculate the right boundary as 66% of the frame width.
4. Compare the pillar’s center_x value with these boundaries.
5. Classify the pillar as left, right, or center depending if its center is left of the left bound, right of the right bound, or none respectively.
6. Store this result inside the detection dictionary as horizontal_position.
7. Display the position label on the camera feed for testing and debugging.

### Files Added or Modified
* `config.py`: Added left and right region ratios for dividing the image into horizontal zones.
* `vision.py`: Added a function for classifying horizontal position and updated the detection output to include this position.
* `test_vision.py`: Updated the detection calls to pass the camera frame width.

### Testing Method
The feature was tested using a live camera feed. A red or green object was moved across the left, center, and right parts of the camera frame. The detection label was checked to confirm that the displayed position matched the object's actual location in the image.

### Result
The vision system can now classify each accepted red or green detection as left, center, or right relative to the camera frame. This makes the detection output more useful for later navigation logic.

### Limitations
The classification is based only on the image position (2D), not the real-world position of the object (3D). Camera angle, lens distortion, and perspective can affect where an object appears in the image. The region boundaries may need tuning after the camera is mounted on the robot.

### Next Step
The next feature will determine the relative distance of a pillar away from the camera.


## Feature 6: Approximate Distance Estimation

### Purpose
The purpose of this feature is to estimate how close a detected pillar is to the camera. This is important because the robot should not react to every detected pillar in the same way. A far pillar may only require preparation, while a close pillar may require the robot to commit to an avoidance maneuver.

### Design Logic
The official pillar height is 100 mm. Since the physical height of the pillar is fixed, its height in the camera image can be used as an approximate indication of distance. A pillar that is far from the camera appears shorter in pixels, while a pillar that is close to the camera appears taller in pixels.

This feature uses the bounding box height from the filtered detection and classifies the pillar distance as `far`, `medium`, or `close`. This is not an exact distance in centimeters. It is a practical distance category that can be used by later navigation logic.

The thresholds are stored in `config.py` so they can be adjusted after testing with the final robot camera. This is important because pixel height depends on camera resolution, camera angle, lens, and mounting position.

### Algorithm Steps
Starting from the filtered and position-classified detections produced in the previous feature:

1. Take each accepted pillar detection.
2. Read the bounding box height of the detection.
3. Compare the height with the far-distance threshold.
4. If the height is below the far threshold, classify the pillar as `far`.
5. Compare the height with the close-distance threshold.
6. If the height is above the close threshold, classify the pillar as `close`.
7. If the height is between both thresholds, classify the pillar as `medium`.
8. Store the result in the detection dictionary as `distance_level`.
9. Display the distance level in the detection label for testing and debugging.

### Files Added or Modified
* `config.py`: Added distance threshold values based on bounding box height in pixels.
* `vision.py`: Added a function for estimating distance level and updated detections to include `distance_level`.
* `test_vision.py`: No major structural change was required because it already displays the updated detection labels.

### Testing Method
The feature was tested using a live camera feed. A red or green object was moved closer to and farther from the camera. The displayed label was checked to confirm that the distance level changed between `far`, `medium`, and `close` as the object size changed in the image.

The thresholds should be adjusted if the distance label changes too early or too late. Final tuning should be performed using the robot’s actual camera and the official pillar dimensions.

### Result
The vision system can now estimate whether a detected pillar is far, medium, or close based on its bounding box height. This adds useful information for future navigation decisions.

### Limitations
This feature does not calculate exact real-world distance. It only gives an approximate distance category. The result depends on camera resolution, camera angle, lens distortion, and mask quality. Accurate distance in centimeters will require camera calibration using the final Raspberry Pi camera setup.

### Next Step
The next feature will convert detection information into navigation output. It will use pillar color, confidence, horizontal position, and approximate distance to produce robot-useful commands such as passing to the left of a red pillar or to the right of a green pillar.


## Feature 7: Navigation Output

### Purpose
The purpose of this feature is to convert computer vision detections into a structured navigation output. Previous features detected red and green pillars and extracted information such as color, position, distance level, and confidence. This feature uses that information to produce a robot-useful output that describes whether the robot should continue normal driving or prepare to avoid a pillar.

### Design Logic
The WRO rule states that the robot must pass on the left side of a red pillar and on the right side of a green pillar. This rule is implemented as a function that converts pillar color into the required passing side.

The system may detect multiple colored objects in one frame. Therefore, the navigation output should not blindly use every detection. The system first filters detections using a minimum navigation confidence and allowed distance levels. For the current version, far pillars are ignored for immediate action, while medium and close pillars are considered relevant.

If multiple usable detections exist, the system selects one primary detection. The priority is to prefer close pillars first, then higher-confidence detections, and then larger-area detections. This helps the robot focus on the pillar most likely to affect its path.

### Algorithm Steps
Starting from the accepted detections produced in the previous feature:

1. Take the list of accepted red and green pillar detections.
2. Remove detections with confidence below the navigation confidence threshold.
3. Remove detections whose distance level is not important for immediate navigation.
4. If no usable detection remains, output `continue_normal_driving`.
5. If usable detections remain, select the primary detection.
6. Give priority to close detections, then higher confidence, then larger area.
7. Read the color of the selected primary detection.
8. Convert the pillar color into the required passing side.
9. For a red pillar, set the required passing side to `left`.
10. For a green pillar, set the required passing side to `right`.
11. Create a navigation output dictionary containing the action, pillar color, required passing side, horizontal position, distance level, confidence, center point, and area.
12. Display or print the navigation output for testing and debugging.

### Files Added or Modified
* `config.py`: Added navigation confidence and action distance thresholds.
* `vision.py`: Added functions for selecting the primary detection and creating navigation output.
* `test_vision.py`: Updated the testing script to print the navigation output during live detection.

### Testing Method
The feature was tested using the live camera feed. When no reliable pillar-like object was visible, the output should be `continue_normal_driving`. When a reliable red object was detected at a medium or close distance, the output should recommend avoiding the pillar and passing on its left side. When a reliable green object was detected at a medium or close distance, the output should recommend avoiding the pillar and passing on its right side.

Testing included cases with no object, one red object, one green object. Further tesing will be conducted in the next steps.

### Result
The vision system can now produce a structured navigation output from the detected pillars. This output connects the perception system to the future robot control system while keeping the CV module separate from motor control.

### Limitations
This feature does not yet give an accurate position of the pillar with respect to the robot, like distance and angle away from the axis of the camera.

### Next Step
The next feature will improve the position estimation of objects.


## Feature 8: Camera-Relative Distance and Angle Estimation

### Purpose
The purpose of this feature is to estimate the position of a detected pillar relative to the camera. Previous features classified the pillar using simple labels such as left, center, right, far, medium, and close. This feature adds numerical estimates for distance, horizontal angle, and camera-relative position.

This information is essential to plan smoother avoidance movements and understand where the pillar is located relative to the robot.

### Design Logic
The official pillar height is 100 mm. Since the real height of the pillar is known, the detected height of the pillar in pixels can be used to estimate the distance from the camera. A closer pillar appears taller in the image, while a farther pillar appears shorter.

The system uses a focal length value in pixels. This value depends on the camera, lens, resolution, and image scaling. Therefore, it must be calibrated for the current camera setup. The laptop webcam focal length can be used for laptop testing, but the value must be recalibrated when switching to the Raspberry Pi camera.

The horizontal angle is estimated by comparing the pillar center x-coordinate with the image center. If the pillar center is left of the image center, the angle is negative. If it is right of the image center, the angle is positive. This angle is then used with the estimated distance to calculate an approximate x and y position relative to the camera.

### Algorithm Steps
Starting from the accepted detections produced in the previous feature:

1. Take each accepted pillar detection.
2. Read the bounding box height in pixels.
3. Use the known real pillar height and focal length in pixels to estimate the distance from the camera.
4. Read the pillar center x-coordinate.
5. Calculate the image center x-coordinate.
6. Calculate the horizontal pixel offset between the pillar center and the image center.
7. Use the pixel offset and focal length to estimate the horizontal angle from the camera axis.
8. Convert the angle from radians to degrees for easier debugging.
9. Use the estimated distance and angle to calculate the pillar’s approximate x-position relative to the camera.
10. Use the estimated distance and angle to calculate the pillar’s approximate y-position relative to the camera.
11. Store the estimated distance, angle, relative x-position, and relative y-position inside the detection dictionary.
12. Display the distance and angle on the camera frame for testing and debugging.
13. Include the new values in the navigation output dictionary.

### Files Added or Modified
* `config.py`: Added the official pillar height and focal length value in pixels.
* `vision.py`: Added functions for estimating distance, horizontal angle, and camera-relative x/y position.
* `test_vision.py`: No major structural change was required because it already displays detections and prints navigation output.

### Testing Method
The feature is tested using a pillar-like object of known height. The object is placed at a measured distance from the camera. The detected pixel height is then used to calibrate the focal length in pixels.

After calibration, the object is moved closer and farther from the camera to check whether the estimated distance changes correctly. The object is also moved left and right in the frame to check whether the angle becomes negative on the left, near zero in the center, and positive on the right.

### Result
The vision system can now estimate a detected pillar’s approximate distance from the camera, horizontal angle from the camera axis, and relative x/y position. This provides more detailed spatial information than simple left, center, right, far, medium, and close labels.

### Limitations
The estimates are approximate. They depend on accurate focal length calibration, stable camera resolution, correct pillar detection, and full visibility of the pillar. If the pillar is partially hidden, tilted, detected poorly, or located near the edge of the camera image, the estimate may be less accurate.

### Next Step
The next feature will improve HSV masking and mask cleaning. This will make the detection pipeline more reliable under different lighting conditions and with the final robot camera.






