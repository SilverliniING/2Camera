The issue with the bounding boxes having the same color in your code is likely due to how you're managing and assigning colors to the bounding boxes. Specifically, the logic within the loop for assigning colors may not be ensuring that each bounding box gets a unique color. Additionally, the logic for updating the `people` dictionary and the color assignment could be flawed.

Here's a revised version of your code with adjustments to ensure unique colors for each bounding box:

```python
from joingrgbheader import *

bbox_color = generate_rgb_array(colors, 10)

# Initialize the results variables
results1 = None
results2 = None

model = yolo("yolov8n-pose.pt", task="pose")

# Provide the paths to your video files
video_path1 = '/Users/aaryakawalay/Desktop/STOCK.mp4'
video_path2 = '/Users/aaryakawalay/Desktop/STOCK.mp4'

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
people = {}
prevbboxes1 = {}
prevbboxes2 = {}

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

        frame1_annotated = frame1.copy()
        frame2_annotated = frame2.copy()

        for jointset in joints2:
            for joint in jointset:
                x, y = joint
                if x != 0 or y != 0:
                    cv.circle(frame2_annotated, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint location

        for jointset in joints1:
            for joint in jointset:
                x, y = joint
                if x != 0 or y != 0:
                    cv.circle(frame1_annotated, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle at each keypoint location

        if len(joints1) > 0 and len(joints2) > 0:
            MATRIX = get_joint_rgb_diff(joints1, joints2, frame1, frame2)
            cameraval, bbox_labels = label_same_person2(joints1, joints2, frame1, frame2)

            if cameraval == 1:
                for i, joint2 in enumerate(joints2):
                    if joint2.all != None and np.any(joint2 != [0, 0]):
                        x_centre, y_centre, w, h = find_bounding_box_center_width_height(joint2)
                        bbox = x_centre, y_centre, w, h
                        decimaltuple = convert_bbox_to_x1y1x2y2(bbox, True)
                        x1, y1, x2, y2 = tuple(math.floor(num) for num in decimaltuple)

                        # Ensure unique colors for each bounding box
                        if i < len(bbox_color):
                            color = bbox_color[i]
                        else:
                            color = (0, 255, 0)  # Default color if out of colors

                        if diff2(joint2, prevbboxes2):
                            update_key_by_rgb(people, bbox, color)
                            cv.rectangle(frame2_annotated, (x1, y1), (x2, y2), color, 2)  # Draw the bounding box with custom color
                        else:
                            findclosest, matchframe = find_closest_element(prevbboxes2, joints2)
                            if findclosest != (0, 0, 0, 0):
                                rgbofOG = people[findclosest]
                                update_key_by_rgb(people, bbox, rgbofOG)
                                color = rgbofOG
                        cv.rectangle(frame2_annotated, (x1, y1), (x2, y2), color, 2)  # Draw the bounding box with custom color

                        custom_text = f"Person {i + 1}"
                        if i < len(bbox_labels):
                            bboxtuple = tuple(bbox)
                            camera2matchup = convert_bbox_to_x1y1x2y2((bbox_labels[bboxtuple]), True)  # yolo format
                            if camera2matchup is not None:
                                X1, Y1, X2, Y2 = tuple(math.floor(num) for num in camera2matchup)
                                cv.rectangle(frame1_annotated, (X1, Y1), (X2, Y2), color, 2)
                                text_X = X1
                                text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                                cv.putText(frame1_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv.LINE_AA)

                        newx, newy, _, _ = tuple(math.floor(num) for num in (convert_bbox_to_x1y1x2y2(bbox, True)))
                        text_x = newx
                        text_y = newy - 10 if newy - 10 > 10 else newy + 10
                        cv.putText(frame2_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv.LINE_AA)

            if cameraval == 2:
                for i, joint1 in enumerate(joints1):
                    if joint1.all != None and np.any(joint1 != [0, 0]):
                        x_centre, y_centre, w, h = find_bounding_box_center_width_height(joint1)
                        bbox = x_centre, y_centre, w, h
                        decimaltuple = convert_bbox_to_x1y1x2y2(bbox, True)
                        x1, y1, x2, y2 = tuple(math.floor(num) for num in decimaltuple)

                        # Ensure unique colors for each bounding box
                        if i < len(bbox_color):
                            color = bbox_color[i]
                        else:
                            color = (0, 255, 0)  # Default color if out of colors

                        if diff2(joint1, prevbboxes1):
                            update_key_by_rgb(people, bbox, color)
                            cv.rectangle(frame1_annotated, (x1, y1), (x2, y2), color, 2)  # Draw the bounding box with custom color
                        else:
                            findclosest, matchframe = find_closest_element(prevbboxes1, joints1)
                            if findclosest != (0, 0, 0, 0):
                                rgbofOG = people[findclosest]
                                update_key_by_rgb(people, bbox, rgbofOG)
                                color = rgbofOG
                        cv.rectangle(frame1_annotated, (x1, y1), (x2, y2), color, 2)  # Draw the bounding box with custom color

                        custom_text = f"Person {i + 1}"
                        if i < len(bbox_labels):
                            bboxtuple = tuple(bbox)
                            camera2matchup = convert_bbox_to_x1y1x2y2((bbox_labels[bboxtuple]), True)  # yolo format
                            if camera2matchup is not None:
                                X1, Y1, X2, Y2 = tuple(math.floor(num) for num in camera2matchup)
                                cv.rectangle(frame2_annotated, (X1, Y1), (X2, Y2), color, 2)
                                text_X = X1
                                text_Y = Y1 - 10 if Y1 - 10 > 10 else Y1 + 10
                                cv.putText(frame2_annotated, custom_text, (text_X, text_Y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv.LINE_AA)

                        newx, newy, _, _ = tuple(math.floor(num) for num in (convert_bbox_to_x1y1x2y2(bbox, True)))
                        text_x = newx
                        text_y = newy - 10 if newy - 10 > 10 else newy + 10
                        cv.putText(frame1_annotated, custom_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv.LINE_AA)

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