from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                                          for landmark in pose_landmarks])
    solutions.drawing_utils.draw_landmarks(annotated_image, pose_landmarks_proto, solutions.pose.POSE_CONNECTIONS,
                                           solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def main():
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    # STEP 3: Load the input image.
    refImage = mp.Image.create_from_file("path-to-reference-image")  #add the path to the reference image here

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(refImage)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(refImage.numpy_view(), detection_result)
    cv2.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    #Visualise the pose segmentation mask
    segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
    visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2)*255

    visualized_mask = cv2.cvtColor(visualized_mask.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    visualized_mask = cv2.threshold(visualized_mask, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imshow(visualized_mask)
    flag = 0
    highest, lowest = [0,0], [0, 0]
    for i in range(len(visualized_mask)):
        for j in range(len(visualized_mask[i])):
            if visualized_mask[i][j] == 255:
                if flag == 0:
                    highest = [i, j]
                    flag = 1
                    break
                else:
                    if lowest < [i, j]:
                        lowest = [i, j]
                        break
    print(lowest, highest)


    appHeight = lowest[0] - highest[0]
    actHeight = 169 #in cms

    pixLen = actHeight/appHeight

    # STEP 3: Load the input image.
    image = mp.Image.create_from_file("path-to-test-image") #add the path to test image

    # STEP 4: Detect pose landmarks from the input image.
    detection_result = detector.detect(image)

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)

    test_segmentation = detection_result.segmentation_masks[0].numpy_view()
    test_visualized = np.repeat(test_segmentation[:, :, np.newaxis], 3, axis=2)*255
    # cv2_imshow(test_visualized)
    # print('hi')
    test_visualized = cv2.cvtColor(test_visualized.astype(np.uint8),cv2.COLOR_BGR2GRAY)
    test_visualized = cv2.threshold(test_visualized, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    cv2.imshow(test_visualized)
    flag = 0
    new_High, new_Low = [0,0], [0, 0]
    for i in range(len(test_visualized)):

        for j in range(len(test_visualized[i])):
            if test_visualized[i][j] == 255:
                if flag == 0:
                    new_High = [i, j]
                    flag = 1
                    break
                else:
                    if new_Low < [i, j]:
                        new_Low = [i, j]
                        break

    print((new_Low[0] - new_High[0])*pixLen)


