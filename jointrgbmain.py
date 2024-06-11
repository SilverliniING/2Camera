import cv2 as cv
import numpy as np
from ultralytics import YOLO as yolo
from joingrgbheader import *
#???
# code fixing goals:
# make sure boxes show up on each person detected
# reduce frame processing by logging best fits
# set up threshold

bbox_color = generate_rgb_array(colors,10)

# Initialize the results variables
results1 = None
results2 = None

model = yolo("yolov8n-pose.pt", task="pose")

# Provide the paths to your video files '''
video_path1 = '/Users/aaryakawalay/Desktop/STOCK.mp4'
video_path2 = '/Users/aaryakawalay/Desktop/STOCK.mp4'

#cap1 = cv.VideoCapture(video_path1)
#cap2 = cv.VideoCapture(video_path2)



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

        # Print key point matrices
        print("Key Points Matrix - Camera 1:")
        print(joints1)
        print("\nKey Points Matrix - Camera 2:")
        print(joints2)
        print("RGB DIFF:")
        MATRIX = get_joint_rgb_diff(joints1, joints2, frame1, frame2)
        print(MATRIX)

        # Annotate the frame with custom colors and text
        frame1_annotated = frame1.copy()
        frame2_annotated = frame2.copy()
        
        bboxes1 = np.array(results1[0].boxes.xyxy).astype(int)  # Assuming boxes.xyxy provides (x1, y1, x2, y2)
        bboxes2 = np.array(results2[0].boxes.xyxy).astype(int)
       
        # Example usage
        cameraval, bbox_labels = label_same_person(joints1, joints2, frame1, frame2,bboxes1,bboxes2)
        print(bbox_labels)
        #bbox_color = generate_rgb_array(max(len(joints1),len(joints2)))
        if cameraval == 1: 
          for i, bbox in enumerate(bboxes1):
            x1, y1, x2, y2 = bbox
            cv.rectangle(frame1_annotated, (x1, y1), (x2, y2), bbox_color[i], 2)  # Draw the bounding box with custom color
            custom_text = f"Person {i + 1}"
            if i < len(bbox_labels):
               bboxtuple = tuple(bbox)
               camera2matchup = bbox_labels[bboxtuple]
               print("MATCH UP BOX")
               print(camera2matchup)
               if camera2matchup is not None:
                  X1, Y1, X2, Y2 = camera2matchup
                  cv.rectangle(frame2_annotated, (X1, Y1), (X2, Y2), bbox_color[i], 2) 
                  text_X = X1
                  text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                  cv.putText(frame2_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 2, bbox_color[i], 2, cv.LINE_AA)

            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(frame1_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, bbox_color[i], 2, cv.LINE_AA)
            
            # Draw keypoints
            for joint in joints1[i]:
                x, y = joint
                if x != 0 or y != 0:
                    cv.circle(frame1_annotated, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint location
                # Draw keypoi
            
        if cameraval == 2: 
          for i, bbox in enumerate(bboxes2):
            x1, y1, x2, y2 = bbox
            cv.rectangle(frame2_annotated, (x1, y1), (x2, y2), bbox_color[i], 2)  # Draw the bounding box with custom color
            custom_text = f"Person {i + 1}"
            if i < len(bbox_labels):
               bboxtuple = tuple(bbox)
               camera2matchup = bbox_labels[bboxtuple]
               print("MATCH UP BOX")
               print(camera2matchup)
               if camera2matchup is not None:
                  X1, Y1, X2, Y2 = camera2matchup
                  cv.rectangle(frame1_annotated, (X1, Y1), (X2, Y2), bbox_color[i], 2) 
                  text_X = X1
                  text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                  cv.putText(frame1_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 2, bbox_color[i], 2, cv.LINE_AA)

            text_x = x1
            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(frame2_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, bbox_color[i], 2, cv.LINE_AA)
            
            # Draw keypoints
            for joint in joints2[i]:
                x, y = joint
                if x != 0 or y != 0:
                    cv.circle(frame2_annotated, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint location
                # Draw keypoints
            
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
