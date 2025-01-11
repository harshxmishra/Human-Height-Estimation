# Height Estimation Project

This project contains Python scripts for estimating the distance and height of a person using computer vision techniques with OpenCV and MediaPipe.

## Overview

The project includes the following scripts:

- **`distance_measure.py`**: Estimates the horizontal offset and distance of a person from the camera in real-time using video feed. It utilizes MediaPipe Pose to detect shoulder landmarks and calculates the distance based on the known shoulder width.
- **`height_measure.py`**: Estimates the height of a person in an image. It uses MediaPipe Pose to detect pose landmarks and segmentation masks to determine the highest and lowest points of the person. It requires a reference image with a person of known height to calibrate the estimation.

## Files

- **`haarcascade_frontalface_default.xml`**: A Haar Cascade classifier for face detection, used in `distance_measure.py`.
- **`pose_landmarker_heavy.task`**: A MediaPipe model file for accurate pose landmark detection, used in `height_measure.py`.

## `distance_measure.py`

### Description

This script captures video from the camera and uses MediaPipe Pose to detect the left and right shoulder landmarks. It calculates the distance to the person based on an assumed average shoulder width and the focal length of the camera. It also calculates the horizontal offset of the person from the center of the frame.

### Usage

Run the script directly:

```bash
python distance_measure.py
```

Ensure you have the necessary libraries installed (`cv2`, `mediapipe`, `numpy`).

### Constants

- `FOV`: Field of View of the camera in degrees (default: 60).
- `RW`: Resolution Width of the camera in pixels.
- `KNOWN_WIDTH`: Approximate shoulder width of an adult in centimeters (default: 51).

## `height_measure.py`

### Description

This script estimates the height of a person in an image by comparing it to a reference image with a person of known height. It uses MediaPipe Pose to get the segmentation mask of the person in both images and calculates the pixel difference between the highest and lowest points.

### Usage

Modify the script to include the paths to the reference and test images:

```python
refImage = mp.Image.create_from_file("path-to-reference-image")  # Add the path to the reference image here
image = mp.Image.create_from_file("path-to-test-image") # Add the path to the test image here
```

Run the script:

```bash
python height_measure.py
```

Ensure you have the necessary libraries installed (`cv2`, `mediapipe`, `numpy`).

## Dependencies

- OpenCV (`cv2`)
- MediaPipe (`mediapipe`)
- NumPy (`numpy`)

You can install the dependencies using pip:

```bash
pip install opencv-python mediapipe numpy
```

## Setup

1. Install the required dependencies.
2. Ensure that the `pose_landmarker_heavy.task` file is in the same directory as `height_measure.py` or provide the correct path in the script.
3. For `height_measure.py`, replace `"path-to-reference-image"` and `"path-to-test-image"` with the actual paths to your images.

## Notes

- The distance measurement in `distance_measure.py` is an approximation based on an assumed average shoulder width.
- The height estimation in `height_measure.py` requires a reference image with a person of known height for calibration.

## Form OCR

The project also includes a `Form OCR` directory, which contains scripts for Optical Character Recognition on forms using YOLOv8.

## Handwritten Texts

The `Handwritten Texts` directory contains scripts for processing and augmenting handwritten text images, likely also using YOLOv8.

Further details on the `Form OCR` and `Handwritten Texts` components can be added here.
