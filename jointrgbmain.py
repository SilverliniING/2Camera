import sys
import cv2 as cv
import numpy as np
from ultralytics import YOLO as yolo
from sort import Sort
from joingrgbheader import *
import threading

bbox_color = generate_rgb_array(10)

# Initialize the results variables
results1 = None
results2 = None

model = yolo("yolov8n-pose.pt", task="pose")



# Uncomment these lines if you want to use video files
# cap1 = cv.VideoCapture(video_path1)
# cap2 = cv.VideoCapture(video_path2)

# Use these lines if you want to use live camera feed
cap1 = cv.VideoCapture(0)
cap2 = cv.VideoCapture(1)

# Set frame resolution
frame_width = 640
frame_height = 480
cap1.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap1.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
cap2.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
cap2.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)

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

def process_frame(frame, tracker, model, bbox_color, window_name):
    results = model(source=frame)
    joints = np.array(results[0].keypoints.xy).astype(int)
    
    # Convert keypoints to SORT format
    sort_input = keypoints_to_sort_format(joints)
    
    # Update SORT tracker
    tracked_objects = tracker.update(sort_input)
    
    # Annotate frame with tracked keypoints
    frame_annotated = frame.copy()
    for i, obj in enumerate(tracked_objects):
        x, y = int(obj[0]), int(obj[1])
        cv.circle(frame_annotated, (x, y), 5, bbox_color[i % len(bbox_color)], -1)
        cv.putText(frame_annotated, f"Person {int(obj[4])}", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color[i % len(bbox_color)], 2, cv.LINE_AA)
    
    cv.imshow(window_name, frame_annotated)

def main_loop():
    while cap1.isOpened() and cap2.isOpened():
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if ret1 and ret2:
            if diff(prev1, frame1) > thresh or diff(prev2, frame2) > thresh:
                thread1 = threading.Thread(target=process_frame, args=(frame1, tracker1, model, bbox_color, "Camera 1"))
                thread2 = threading.Thread(target=process_frame, args=(frame2, tracker2, model, bbox_color, "Camera 2"))
                
                thread1.start()
                thread2.start()
                
                thread1.join()
                thread2.join()

                prev1 = frame1
                prev2 = frame2

            # Break the loop on 'q' key press
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap1.release()
    cap2.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
