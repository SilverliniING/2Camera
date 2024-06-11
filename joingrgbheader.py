import sys
import cv2 as cv
import numpy as np
sys.path.append('/Users/aaryakawalay/Desktop/IITB/IITB_2Cameras/sort')
import numpy as np
from sort import sort
from ultralytics import YOLO as yolo
import random

"""Input Data Types
prev (Previous Frame):

Type: NumPy ndarray
Shape: (height, width, channels) where channels is typically 3 for a BGR (Blue, Green, Red) image.
Description: This is the previous frame from the video, which is a color image.
frame (Current Frame):

Type: NumPy ndarray
Shape: (height, width, channels) where channels is typically 3 for a BGR image.
Description: This is the current frame from the video, which is also a color image.
Output Data Type
mse (Mean Squared Error):
Type: float
Description: This is a single floating-point number representing the average squared differences between the pixel values of the two frames. It quantifies the difference between the frames, with higher values indicating more significant changes."""


def get_min_indices(array):
    # Determine the longest dimension
    m, n = array.shape
    longest_dim = max(m, n)
    
    # Determine the other dimension
    other_dim = min(m, n)
    
    # Initialize an array to store the indices of the minimum values
    min_indices = np.zeros(other_dim, dtype=int)
    
    # Determine CAMERAVALUE
    if m > n:
        CAMERAVALUE = 1
    else:
        CAMERAVALUE = 2
    
    # Iterate over the longest dimension
    for i in range(longest_dim):
        # Extract the row or column depending on the longest dimension
        if m > n:  # Rows are longer
            row = array[i, :]
        else:  # Columns are longer
            row = array[:, i]
        
        # Find the index of the minimum value in the row or column
        min_index = np.argmin(row)
        
        # Update the min_indices array with the index of the minimum value
        min_indices[i % other_dim] = min_index
    
    return min_indices, CAMERAVALUE

def diff(prev, frame):
    #Converts frames to grayscale and computes the Mean Squared Error (MSE) to detect significant changes between consecutive frames.
    print("Previous frame shape:", prev.shape)  # Add this line
    print("Current frame shape:", frame.shape)  # Add this line

    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    print("Previous frame shape after conversion:", prev.shape)  # Add this line
    print("Current frame shape after conversion:", frame.shape)  # Add this line

    # Compute the Mean Squared Error (MSE)
    mse = ((prev - frame) ** 2).mean()
    return mse


def angle_calc(p1, p2, p3):
    #Computes the angle between three points representing joints.
    if (p1.all() == 0) or (p2.all() == 0) or (p3.all() == 0):
        return -1
    v1 = p1 - p2
    v2 = p3 - p2
    cos_theta = (np.dot(v1, v2)) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    angle = abs(theta * 180.0 / np.pi)
    if angle > 180:
        angle = 360 - angle
    return int(angle)


def joint_angles(joint: list):
    #Calculates angles for specific joints and returns a list of these angles.
    joints = np.zeros((17, 2))
    joints[: joint.shape[0], : joint.shape[1]] = joint
    out = []
    out.append(angle_calc(joints[9], joints[7], joints[5]))
    out.append(angle_calc(joints[10], joints[8], joints[6]))
    out.append(angle_calc(joints[7], joints[5], joints[11]))
    out.append(angle_calc(joints[8], joints[6], joints[12]))
    out.append(angle_calc(joints[5], joints[11], joints[13]))
    out.append(angle_calc(joints[6], joints[12], joints[14]))
    out.append(angle_calc(joints[11], joints[13], joints[15]))
    out.append(angle_calc(joints[12], joints[14], joints[16]))
    return out

def pose(flexion_angles: list):
    l = len(flexion_angles)
    up = flexion_angles[: int(l / 2)]
    down = flexion_angles[int(l / 2) :]
    final = []
    text = 0

    def is_between(value, min_val, max_val):
        return all(min_val[i] <= value[i] <= max_val[i] for i in range(len(value)))

    match up:
        case _ if is_between(up, [0, 0, 160, 160], [0, 0, 180, 180]):
            text = "Both Hands Up"
        case _ if is_between(up, [0, 0, 160, 0], [0, 0, 180, 0]):
            text = "Right Hand up"
        case _ if is_between(up, [0, 0, 0, 160], [0, 0, 0, 180]):
            text = "Left Hand Up"

    if text != 0:
        final.append(text)
        text = 0

    match up:
        case _ if is_between(up, [80, 80, 80, 80], [100, 100, 100, 100]):
            text = "Both Hands Raised Up"
        case _ if is_between(up, [80, 0, 80, 0], [100, 0, 100, 0]):
            text = "Right Hand Raised up"
        case _ if is_between(up, [0, 80, 0, 80], [0, 100, 0, 100]):
            text = "Left Hand Raised Up"

    if text != 0:
        final.append(text)
        text = 0

    match up:
        case _ if is_between(up, [160, 160, 80, 80], [180, 180, 100, 100]):
            text = "Both hands are horizontal"
        case _ if is_between(up, [160, 160, 0, 80], [180, 180, 20, 100]):
            text = "Right hand is horizontal"
        case _ if is_between(up, [160, 160, 80, 0], [180, 180, 100, 20]):
            text = "Left hand is horizontal"

    if text != 0:
        final.append(text)
        text = 0

    match up:
        case _ if is_between(up, [130, 130, 0, 0], [180, 180, 30, 30]):
            text = "Both Hands Down"
        case _ if is_between(up, [0, 130, 0, 0], [0, 180, 30, 30]):
            text = "Left Hand Down"
        case _ if is_between(up, [130, 0, 0, 0], [180, 30, 30, 30]):
            text = "Right Hand Down"

    if text != 0:
        final.append(text)
        text = 0

    match down:
        case _ if is_between(down, [80, 80, -1, -1], [100, 100, 180, 180]):
            text = "Sitting Down"
        case _ if is_between(down, [160, 160, -1, -1], [180, 180, 180, 180]):
            text = "Standing pose"
        case _ if is_between(down, [80, 160, -1, 160], [100, 180, 180, 180]):
            text = "Standing on left leg"
        case _ if is_between(down, [160, 80, 160, -1], [180, 100, 180, 180]):
            text = "Standing on Right leg"

    if text != 0:
        final.append(text)
        text = 0
    return final

colors = [
    (0, 0, 255),
    (0, 255, 0),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (128, 128, 128),
    (50, 50, 50),
    (200, 200, 200),
    (0, 0, 128),
    (0, 128, 128),
    (128, 0, 128),
    (128, 128, 0),
    (128, 0, 0),
    (0, 165, 255),
    (19, 69, 139),
    (203, 192, 255),
    (230, 216, 173)
]

##
import numpy as np

"""
def get_joint_rgb_diff(joints1, joints2, frame1, frame2):
    rgb_diff_matrix = np.zeros(((len(joints1)), (len(joints2))))
    for m in range(len(joints1)):
        jointset1 = joints1[m]
        for n in range(len(joints2)):
            jointset2 = joints2[n]
            count = 0
            sum_diff = 0
            for i in range(len(jointset1)):
                x, y = jointset1[i]
                if x != 0 or y != 0:
                    p, q = jointset2[i]
                    if p != 0 or q != 0:
                        count += 1
                        array1 = np.array([x, y])
                        array2 = np.array([p, q])
                        dot_product = np.dot(array1, array2)
                        magnitude1 = np.linalg.norm(array1)
                        magnitude2 = np.linalg.norm(array2)
                        cosine_similarity = dot_product / (magnitude1 * magnitude2)
                        sum_diff += cosine_similarity
            if count > 0:
                average_rgb_diff = sum_diff / count
                rgb_diff_matrix[m, n] = average_rgb_diff
    return rgb_diff_matrix
"""
    
    
# Function to calculate the average cosine similarity of RGB differences between corresponding joints in two arrays
import numpy as np

import numpy as np

import numpy as np

def get_joint_rgb_diff(joints1, joints2, frame1, frame2):
    rgb_diff_matrix = np.zeros((len(joints1), len(joints2)))

    for m in range(len(joints1)):
        jointset1 = joints1[m]
        
        if len(jointset1) == 0:
            continue
        
        for n in range(len(joints2)):
            jointset2 = joints2[n]
            
            if len(jointset2) == 0:
                continue
            
            count = 0
            mse_sum = 0

            for i in range(len(jointset1)):
                x, y = jointset1[i]
                
                if x <= 0 or y <= 0 or x >= frame1.shape[1] or y >= frame1.shape[0]:
                    continue
                
                p, q = jointset2[i]
                
                if p <= 0 or q <= 0 or p >= frame2.shape[1] or q >= frame2.shape[0]:
                    continue

                array1 = frame1[y-1, x-1]
                array2 = frame2[q-1, p-1]
                
                # Compute MSE
                mse = ((array1 - array2) ** 2).mean()
                mse_sum += mse

                count += 1

            if count != 0:
                avg_mse = mse_sum / count
                rgb_diff_matrix[m, n] = avg_mse
    
    return rgb_diff_matrix




def associate_joints_with_bboxes(joints, bboxes):
    """
    Associate each joint set with its corresponding bounding box based on their spatial overlap.
    
    Args:
    - joints (list of numpy arrays): List of joint sets detected in the frame.
    - bboxes (list of numpy arrays): List of bounding boxes detected in the frame.
    
    Returns:
    - dict: A dictionary where keys are bounding boxes (as tuples) and values are corresponding joint sets.
    """
    # Initialize an empty dictionary to store associations
    associations = {}

    # Iterate over each bounding box
    for bbox in bboxes:
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        
        # Find the center of the bounding box
        bbox_center_x = (bbox_x1 + bbox_x2) // 2
        bbox_center_y = (bbox_y1 + bbox_y2) // 2
        
        # Find the joint set closest to the center of the bounding box
        min_distance = float('inf')
        closest_joint_set = None
        for joint_set in joints:
            # Calculate the center of mass of the joint set
            joint_center_x = np.mean(joint_set[:, 0])
            joint_center_y = np.mean(joint_set[:, 1])
            
            # Calculate the distance between the center of the bounding box and the center of mass of the joint set
            distance = np.sqrt((bbox_center_x - joint_center_x)**2 + (bbox_center_y - joint_center_y)**2)
            
            # Update the closest joint set if the current one is closer
            if distance < min_distance:
                min_distance = distance
                closest_joint_set = joint_set
        
        # Add the association to the dictionary
        associations[(bbox_x1, bbox_y1, bbox_x2, bbox_y2)] = closest_joint_set

    return associations

def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if np.array_equal(value, target_value):
            return key
    return None  # Return None if the value is not found in the dictionary

import random



def generate_rgb_array(size):
    rgb_array = []
    for _ in range(size):
        rgb = random.choice(colors)
        rgb_array.append(rgb)
    return rgb_array


def label_same_person(joints1, joints2, frame1, frame2, bboxes1, bboxes2, thresh=0.5):
    """
    Label bounding boxes of joint sets belonging to the same person based on the lowest value differences
    between corresponding joint sets in two frames.

    Args:
    - joints1 (list of numpy arrays): List of joint sets detected in the first frame.
    - joints2 (list of numpy arrays): List of joint sets detected in the second frame.
    - frame1 (numpy array): First frame.
    - frame2 (numpy array): Second frame.
    - thresh (float): Threshold to consider two joint sets as belonging to the same person.

    Returns:
    - dict: A dictionary where keys are bounding boxes (as tuples) and values are the label of the person.
    """

    # Associate joint sets with bounding boxes in each frame
    associations1 = associate_joints_with_bboxes(joints1, bboxes1)
    associations2 = associate_joints_with_bboxes(joints2, bboxes2)

    # Calculate RGB differences between corresponding joint sets
    rgb_diff_matrix = get_joint_rgb_diff(joints1, joints2, frame1, frame2)
    matches,cameravalue = get_min_indices(rgb_diff_matrix) 
    print("matches")
    print(matches)
    # Initialize a dictionary to store the labels of each bounding box
    bbox_labels = {}

    if len(matches) > 0:  # Check if matches is a list and not empty
      
      if cameravalue == 1:
         i = 0
         for bbox1 in associations1.items():
           if i < len(matches):
             bbox_labels[tuple(bbox1[0])] = get_key_by_value(associations2, joints2[matches[i]])
             i += 1
      elif cameravalue == 2:
         i = 0
         for bbox2 in associations2.items():
            if i < len(matches):
             bbox_labels[tuple(bbox2[0])] = get_key_by_value(associations1, joints1[matches[i]])
             i += 1

    return cameravalue, bbox_labels




