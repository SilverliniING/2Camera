#Libraries
import math
import cv2 as cv
import numpy as np
from ultralytics import YOLO as yolo
import random

#relevent to rgbjoints
def diff(prev, frame):
    #Converts frames to grayscale and computes the Mean Squared Error (MSE) to detect significant changes between consecutive frames.
    #print("Previous frame shape:", prev.shape)  
    #print("Current frame shape:", frame.shape) 

    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # print("Previous frame shape after conversion:", prev.shape) 
    #print("Current frame shape after conversion:", frame.shape)

    # Compute the Mean Squared Error (MSE)
    mse = ((prev - frame) ** 2).mean()
    return mse

#global variables
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

"""
# Function to calculate the average cosine similarity of RGB differences between corresponding joints in two arrays
def get_joint_rgb_diff(joints1, joints2,frame1,frame2,thresh=0.6):
    rgb_diff_matrix = np.zeros((len(joints1), len(joints2)))
    
    # Iterate through joints in joints1
    for m in range(len(joints1)): 
        jointset1 = joints1[m]
        for n in range(len(joints2)): 
            jointset2 = joints2[n]
            count = 0
            sum_cosine_similarity = 0
            
            
             # Iterate through corresponding joint pairs
            for i in range(len(jointset1)): 
                x, y = jointset1[i]
                if x != 0 or y != 0:
                    p, q = jointset2[i]
                    if p != 0 or q != 0:
                        count += 1
                        array1 = frame1[y-1, x-1]
                        array2 = frame2[q-1, p-1]
                    
                        if x==p and y==q:
                          print("ARRAY")
                          print(array1)
                          print("ARRAY")
                          print(array1)

                        # Compute dot product
                        dot_product = np.dot(array1, array2)
                        
                        # Compute magnitudes
                        magnitude1 = np.linalg.norm(array1)
                        magnitude2 = np.linalg.norm(array2)
                        
                        # Compute cosine similarity (handle division by zero)
                        if magnitude1 * magnitude2 != 0:
                            cosine_similarity = dot_product / (magnitude1 * magnitude2)
                            sum_cosine_similarity += abs(cosine_similarity)
            
            # Calculate average cosine similarity
            if count != 0:
                average_cosine_similarity = sum_cosine_similarity / count
                rgb_diff_matrix[m, n] = average_cosine_similarity**(-math.sqrt(count)) 
    
    return rgb_diff_matrix
"""


# Function to calculate the average cosine similarity of RGB differences between corresponding joints in two arrays
def get_joint_rgb_diff(joints1, joints2,frame1,frame2,thresh=0.6):
    rgb_diff_matrix = np.zeros((len(joints1), len(joints2)))
    
    # Iterate through joints in joints1
    for m in range(len(joints1)): 
        jointset1 = joints1[m]
        for n in range(len(joints2)): 
            jointset2 = joints2[n]
            count = 0
            sum= 0
            
            
             # Iterate through corresponding joint pairs
            for i in range(len(jointset1)): 
                x, y = jointset1[i]
                if x != 0 or y != 0:
                    p, q = jointset2[i]
                    if p != 0 or q != 0:
                        count += 1
                        array1 = frame1[y-1, x-1]
                        array2 = frame2[q-1, p-1]
                    
                        if x==p and y==q:
                          print("ARRAY")
                          print(array1)
                          print("ARRAY")
                          print(array1)
                        euc = np.linalg.norm(array1 - array2)
                        sum = sum + euc
            # Calculate average cosine similarity
            if count != 0:
                average_cosine_similarity = sum / count
                rgb_diff_matrix[m, n] = average_cosine_similarity 
    
    return rgb_diff_matrix


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



def associate_joints_with_bboxes(joints, bboxes):

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

def generate_rgb_array(colors, size):
    rgb_array = []
    if size > len(colors):
        rgb_array = colors
        for _ in range((size - len(colors))):
           rgb = [random.randint(0, 255) for _ in range(3)]  # Generate random RGB values
           rgb_array.append(rgb)
    else:
       rgb_array = colors[0:size]
    return rgb_array

def label_same_person(joints1, joints2, frame1, frame2, bboxes1, bboxes2, thresh=0.5):


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



