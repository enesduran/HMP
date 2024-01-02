import os 
import json
import glob
import joblib
import numpy as np

OP_NUM_JOINTS = 21


def read_bbox(bbox_fn):
    
    # x0, y0, x1, y1, conf
    bbox_list = joblib.load(bbox_fn)
    
    return bbox_list 

def read_keypoints(keypoint_fn):
    '''
    Only reads body keypoint data of first person.
    '''
    with open(keypoint_fn) as keypoint_file:
        data = json.load(keypoint_file)

    if len(data['people']) == 0:
        print('WARNING: Found no keypoints in %s! Returning zeros!' % (keypoint_fn))
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float)

    person_data = data['people'][0]

    rh_keypoints = np.array(person_data['hand_right_keypoints_2d'], dtype=np.float32)    
    rh_keypoints = rh_keypoints.reshape([-1, 3])

    # this means there is no detection, set it all zeros
    if rh_keypoints.shape == (7, 3):
        return np.zeros((OP_NUM_JOINTS, 3), dtype=np.float)
    
    return rh_keypoints

def make_absolute(rel_paths):
    ''' Makes a list of relative paths absolute '''
    return [os.path.join(os.getcwd(), rel_path) for rel_path in rel_paths]

