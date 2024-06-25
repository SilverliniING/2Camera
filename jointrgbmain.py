import sys
import cv2 as cv
import numpy as np
from ultralytics import YOLO as yolo
from sort import Sort
from joingrgbheader import *

bbox_color = generate_rgb_array(10)

# Initialize the results variables
results1 = None
results2 = None

model = yolo("yolov8n-pose.pt", task="pose")

# Provide the paths to your video files
video_path1 = '/Users/aaryakawalay/Desktop/STOCK.mp4'
video_path2 = '/Users/aaryakawalay/Desktop/STOCK.mp4'

# Uncomment these lines if you want to use video files
# cap1 = cv.VideoCapture(video_path1)
# cap2 = cv.VideoCapture(video_path2)

# Use these lines if you want to use live camera feed
cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

ret1, prev1 = cap1.read()
ret2, prev2 = cap2.read()

thresh = 5

if not ret1 or not ret2:
    print("Error: Could not read frames from cameras.")
    cap1.release()
    cap2.release()
    cv.destroyAllWindows()
    exit()

# Initialize SORT trackers for each camera
tracker1 = Sort()
tracker2 = Sort()

def diff(img1, img2):
    return np.sum(np.abs(img1.astype(np.int16) - img2.astype(np.int16)))

# Main loop
while cap1.isOpened() and cap2.isOpened():
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    if ret1 and ret2:
        if diff(prev1, frame1) > thresh or diff(prev2, frame2) > thresh:
            results1 = model(source=frame1)
            results2 = model(source=frame2)
        else:
            if results1 is None:
                results1 = model(source=frame1)
            if results2 is None:
                results2 = model(source=frame2)
        
        prev1 = frame1
        prev2 = frame2

        joints1 = np.array(results1[0].keypoints.xy).astype(int)
        joints2 = np.array(results2[0].keypoints.xy).astype(int)

        bboxes1 = np.array(results1[0].boxes.xyxy).astype(int)
        bboxes2 = np.array(results2[0].boxes.xyxy).astype(int)

        # Update SORT trackers
        tracked_objects1 = tracker1.update(bboxes1)
        tracked_objects2 = tracker2.update(bboxes2)

        # Example usage
        cameraval, bbox_labels = label_same_person(joints1, joints2, frame1, frame2, bboxes1, bboxes2)
        frame1_annotated = frame1.copy()
        frame2_annotated = frame2.copy()
        
        if cameraval == 1:
            for i, obj in enumerate(tracked_objects1):
                x1, y1, x2, y2, id1 = obj
                cv.rectangle(frame1_annotated, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color[int(id1) % len(bbox_color)], 2)
                custom_text = f"Person {int(id1)}"
                bboxtuple = (int(x1), int(y1), int(x2), int(y2))
                camera2matchup = bbox_labels.get(bboxtuple)
                if camera2matchup is not None:
                    X1, Y1, X2, Y2 = camera2matchup
                    cv.rectangle(frame2_annotated, (X1, Y1), (X2, Y2), bbox_color[int(id1) % len(bbox_color)], 2)
                    text_X = X1
                    text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                    cv.putText(frame2_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color[int(id1) % len(bbox_color)], 2, cv.LINE_AA)
                text_x = int(x1)
                text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10
                cv.putText(frame1_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color[int(id1) % len(bbox_color)], 2, cv.LINE_AA)
                
                # Draw keypoints
                for joint in joints1[i]:
                    x, y = joint
                    if x != 0 or y != 0:
                        cv.circle(frame1_annotated, (x, y), 5, (0, 255, 0), -1)
                
        if cameraval == 2:
            for i, obj in enumerate(tracked_objects2):
                x1, y1, x2, y2, id2 = obj
                cv.rectangle(frame2_annotated, (int(x1), int(y1)), (int(x2), int(y2)), bbox_color[int(id2) % len(bbox_color)], 2)
                custom_text = f"Person {int(id2)}"
                bboxtuple = (int(x1), int(y1), int(x2), int(y2))
                camera2matchup = bbox_labels.get(bboxtuple)
                if camera2matchup is not None:
                    X1, Y1, X2, Y2 = camera2matchup
                    cv.rectangle(frame1_annotated, (X1, Y1), (X2, Y2), bbox_color[int(id2) % len(bbox_color)], 2)
                    text_X = X1
                    text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                    cv.putText(frame1_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color[int(id2) % len(bbox_color)], 2, cv.LINE_AA)
                text_x = int(x1)
                text_y = int(y1) - 10 if int(y1) - 10 > 10 else int(y1) + 10
                cv.putText(frame2_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color[int(id2) % len(bbox_color)], 2, cv.LINE_AA)
                
                # Draw keypoints
                for joint in joints2[i]:
                    x, y = joint
                    if x != 0 or y != 0:
                        cv.circle(frame2_annotated, (x, y), 5, (0, 255, 0), -1)

        cv.imshow("Camera 1", frame1_annotated)
        cv.imshow("Camera 2", frame2_annotated)

        # Break the loop on 'q' key press
        if cv.waitKey(20) & 0xFF == ord("q"):
            break
    else:
        break

cap1.release()
cap2.release()
cv.destroyAllWindows()
