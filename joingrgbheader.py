#Libraries
import cv2 as cv
import numpy as np
from ultralytics import YOLO as yolo
import random
import math

#helper functions
def get_key_by_value(dictionary, target_value):
    for key, value in dictionary.items():
        if np.array_equal(value, target_value):
            return key
    return None  # Return None if the value is not found in the dictionary
#helper functions end

#change this to make it bounding box specific!!!
def diff(prev, frame):
    prev = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Compute the Mean Squared Error (MSE)
    mse = ((prev - frame) ** 2).mean()
    return mse


def find_closest_element(dictionary,input_tuple ):
    min_distance = float('inf')
    closest_key = None
    closest_value = None
    print("I am input approx")
    print("I am dictionary")
    for key,value in dictionary.items():
        distance = np.linalg.norm(np.array(input_tuple) - np.array(key))
        if distance < min_distance:
            min_distance = distance
            closest_key = key
            closest_value = value
            
    if (closest_key is not None) and (closest_value is not None):
      return closest_key,  closest_value
    else:
       return (0,0,0,0),(0,0,0,0)

#change this to make it bounding box specific!!!
def diff2( element,prevbboxmatchup):
    currentbbox = find_bounding_box_center_width_height(element)
    if prevbboxmatchup != []:
     findclosest, _ =  find_closest_element(prevbboxmatchup,element)
     xcentre1, ycentre1, _, _ = findclosest
     xcentre2, ycentre2, _, _ = currentbbox
     array1 = np.array([xcentre1, ycentre1])
     array2 = np.array([xcentre2, ycentre2])
     euc = np.linalg.norm( array1 - array2 )
     print("EUC")
     print(euc)
     if abs(euc) > 2000 and np.any(findclosest != 0):
        return True
     else:
        return False
    else:
     return True
    
def update_key_by_rgb(rgb_dict, new_key, rgb_value):

    # Find the key corresponding to the given RGB value
    old_key = None
    for key, value in rgb_dict.items():
        if value == rgb_value:
            old_key = key
            break

    if old_key is None:
        print(f"No entry found with RGB value {rgb_value}")
        rgb_dict[new_key] = rgb_value
        return False

    # Update the dictionary
    rgb_dict[new_key] = rgb_dict.pop(old_key)
    print(f"Key {old_key} replaced with {new_key} for RGB value {rgb_value}")
    return True

#global variables for bounding box colors
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

#colors of the boxes
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

# Function to calculate the average of RGB differences between corresponding joints in two arrays
#integrate pruning here!!!
def get_joint_rgb_diff(joints1, joints2,frame1,frame2,thresh=0.6):
    #rgb_diff_matrix = np.zeros((len(joints1), len(joints2)))
    
    rgb_diff_matrix = np.full((len(joints1), len(joints2)), -1, dtype=int)

    # Iterate through joints in joints1
    for m in range(len(joints1)): 
        jointset1 = joints1[m]
        if jointset1 != []:
         for n in range(len(joints2)): 
            jointset2 = joints2[n]
            count = 0
            sum= 0
            if jointset2 != []:
             # Iterate through corresponding joint pairs
             for i in range(len(jointset1)) : 
                x, y = jointset1[i]
                if x != 0 or y != 0:
                    p, q = jointset2[i]
                    if p != 0 or q != 0:
                        count += 1
                        array1 = frame1[y-1, x-1]
                        array2 = frame2[q-1, p-1]
                        euc = np.linalg.norm(array1 - array2)
                        sum = sum + euc
            # Calculate average 
            #THRESHOLDING BASED ON COUNT CAN BE DONE HERE
            if count != 0:
                average = sum / count
                rgb_diff_matrix[m, n] = average
    
    return rgb_diff_matrix


"""
#Function to get match the joints with best match [mapped from frame with min joints to frame with max joints]
def get_min_indices2(array,last_indices):
    
    # Determine the longest dimension
    m, n = array.shape
    longest_dim = max(m, n)
    
    # Determine the other dimension
    other_dim = min(m, n)
    
    # Initialize an array to store the indices of the minimum values
    min_indices = np.full(other_dim, -1, dtype=int)

    # Determine CAMERAVALUE
    if m > n:
        CAMERAVALUE = 1 #this implies Camera2 joints are mapped to Camera 1 (search space is Camera1)
    else:
        CAMERAVALUE = 2 #this implies Camera1 joints are mapped to Camera 2 (search space is Camera2)
    
    # Iterate over the longest dimension (searchspace)
    for i in range(longest_dim) :
     if i not in last_indices:
        print("i am i")
        print(i)
        # Extract the row or column depending on the longest dimension
        if m > n:  # Rows are longer
            row = array[i, :]
        else:  # Columns are longer
            row = array[:, i]
            
        flipped_row = np.where(row == -1, np.inf, row)
        for k in range(len(last_indices)) :
            if last_indices[k]!=-1:
              flipped_row[k] =  np.inf
        # Find the index of the minimum value in the row or column
        min_index = int(np.argmin(flipped_row))
        print("this is min")
        print(row[min_index])
        if  len(last_indices) ==0:
         if row[min_index] != -1  :
          if CAMERAVALUE == 1:
            if array[min_indices[min_index]][min_index] >= row[min_index] or array[min_indices[min_index]][min_index]==-1 : 
              # Update the min_indices array with the index of the minimum value
              min_indices[ min_index ] = i
          if CAMERAVALUE == 2:
            if array[min_index][min_indices[min_index]] >= row[min_index] or array[min_indices[min_index]][min_index]!=-1: 
              # Update the min_indices array with the index of the minimum value
              min_indices[ min_index ] = i
        elif last_indices[min_index] == -1 :
         if row[min_index] != -1  :
          if CAMERAVALUE == 1:
            if array[min_indices[min_index]][min_index] >= row[min_index] or array[min_indices[min_index]][min_index]==-1 : 
              # Update the min_indices array with the index of the minimum value
              min_indices[ min_index ] = i
          if CAMERAVALUE == 2:
            if array[min_index][min_indices[min_index]] >= row[min_index] or array[min_indices[min_index]][min_index]!=-1: 
              # Update the min_indices array with the index of the minimum value
              min_indices[ min_index ] = i    
        else:
             #min_indices[ min_index ] = -1 
             print("hi")
        
    print("MIN INDICE")
    print(min_indices)
    if np.any(min_indices == -1):
           get_min_indices(array,min_indices)

    
    return min_indices, CAMERAVALUE
"""

#Function to get match the joints with best match [mapped from frame with min joints to frame with max joints]
def get_min_indices(array):
    
    # Determine the longest dimension
    m, n = array.shape
    longest_dim = max(m, n)
    
    # Determine the other dimension
    other_dim = min(m, n)
    
    # Initialize an array to store the indices of the minimum values
    min_indices = np.full(other_dim, -1, dtype=int)

    # Determine CAMERAVALUE
    if m > n:
        CAMERAVALUE = 1 #this implies Camera2 joints are mapped to Camera 1 (search space is Camera1)
    else:
        CAMERAVALUE = 2 #this implies Camera1 joints are mapped to Camera 2 (search space is Camera2)
    
    # Iterate over the shorest dimension (over searchspace)
    for i in range(other_dim) :
     
        print("i am i")
        print(i)
        # Extract the row or column depending on the longest dimension
        if m > n:  # Rows are longer
            col = array[:, i]
        else:  # Columns are longer
            col = array[i, :]
            
        flipped_col = np.where(col == -1, np.inf, col)

        # Find the index of the minimum value in the row or column
        min_index = int(np.argmin(flipped_col))
        print("this is min")
        print(col[min_index])
      
        if col[min_index] != -1:
              min_indices[ i ] = min_index
   
        else:
             #min_indices[ min_index ] = -1 
             print("hi")
        
    print("MIN INDICE")
    print(min_indices)


    
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

#not yet used
def find_bounding_box_center_width_height(joints):
    # Extract x and y coordinates from the joints array
    x_coords = joints[:, 0]
    y_coords = joints[:, 1]
    filtered_x_coords = [coord for coord in x_coords if coord != 0]
    filtered_y_coords = [coord for coord in y_coords if coord != 0]  
    
    val = True
    # Determine the minimum and maximum coordinates
    if len(filtered_x_coords) > 0:
     x_min = np.min( filtered_x_coords)
     x_max = np.max( filtered_x_coords)
    if len(filtered_y_coords) > 0:
     y_min = np.min(filtered_y_coords)
     y_max = np.max(filtered_y_coords)
    else:
       val = False
       
    if val:
     # Calculate the center coordinates
     center_x = (x_min + x_max) / 2
     center_y = (y_min + y_max) / 2

     # Calculate width and height
     width = x_max - x_min
     height = y_max - y_min

     # Define the bounding box as (center_x, center_y, width, height)
     bounding_box = (center_x, center_y, width, height)
     return bounding_box
    return None


def label_same_person2(joints1, joints2, frame1, frame2):
#thresholding can be done here
    # Calculate RGB differences between corresponding joint sets
    rgb_diff_matrix = get_joint_rgb_diff(joints1, joints2, frame1, frame2)
    matches,cameravalue = get_min_indices(rgb_diff_matrix) 
    
    print("matches")
    print(matches)

    # Initialize a dictionary to store the labels of each bounding box
    #here is potential to ID!!!
    bbox_labels = {}

    if np.any(matches != -1):  # Check if matches is a list and not empty
      
      if cameravalue == 1:
         for i in range(len(joints2)):
           if matches[i]!=-1:
             bbox_labels[tuple(find_bounding_box_center_width_height(joints2[i]))] = tuple(find_bounding_box_center_width_height(joints1[matches[i]]))
      elif cameravalue == 2:
         for i in range(len(joints1)):
           if matches[i]!=-1:
             bbox_labels[tuple(find_bounding_box_center_width_height(joints1[i]))] = tuple(find_bounding_box_center_width_height(joints2[matches[i]]))

    return cameravalue, bbox_labels


def convert_bbox_to_x1y1x2y2(bbox,val):

     converted_bbox = []
     if val == True:
        x_center, y_center, width, height = bbox
        x1 = x_center - (width / 2)
        y1 = y_center - (height / 2)
        x2 = x_center + (width / 2)
        y2 = y_center + (height / 2)
        converted_bbox = x1, y1, x2, y2
        return tuple(converted_bbox)
     if val == False:
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2
        y_center = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        converted_bbox = x_center, y_center, width, height
        return tuple(converted_bbox)
