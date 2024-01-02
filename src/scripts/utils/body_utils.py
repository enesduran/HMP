import os
import sys
import copy
import torch
import socket
import itertools
import numpy as np

device_name = socket.gethostname()


workstation = True if "ps063" else False 
local = True if "DESKTOP-EUAVTI7" else False  

SMPLX_LEFT_HAND_INDEXES = np.arange(25, 40)
SMPLX_RIGHT_HAND_INDEXES = np.arange(40, 55)

MANO_RIGHT_PATH = "./bodymodels/MANO_RIGHT.pkl"
MANO_LEFT_PATH = "./bodymodels/MANO_LEFT.pkl"

SMPLX_BODY_PATH = "./bodymodels/SMPLX_NEUTRAL.npz"


SMPLX_CHAIN = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 15, 15, 15, 20, 25, 
                        26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 
                        50, 21, 52, 53])

OPENPOSE_HAND_CHAIN = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])

# Augmented kinematic chain changes by the design choice of the hand joints. Therefore refer to HAND_TIP_IDS if you want to change the joint order 
AUGMENTED_MANO_CHAIN = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])    
MANO_CHAIN = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])

# Chaining of the right hand joints in the SMPLX model     
RIGHT_HAND_CHAIN = {21: -1, 40: 21, 41: 40, 42: 41, 43: 21, 44: 43, 45: 44, 46: 21, 47: 46, 48: 47, 49: 21, 50: 49, 51: 50, 52: 21, 53: 52, 54: 53}

# Chaining of the left hand joints in the SMPLX model     
LEFT_HAND_CHAIN = {20: -1, 25: 20, 26: 25, 27: 26, 28: 20, 29: 28, 30: 29, 31: 20, 32: 31, 33: 32, 34: 20, 35: 34, 36: 35, 37: 20, 38: 37, 39: 38}

SMPLX_JOINTS = ['Pelvis', 'L_Hip', 'R_Hip', 'Spine1', 'L_Knee', 'R_Knee', 'Spine2', 'L_Ankle', 'R_Ankle', 'Spine3', 'L_Foot', 'R_Foot', 'Neck', 'L_Collar',
                'R_Collar', 'Head', 'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist', 'Jaw', 'L_eye', 'R_eye', 'lindex0', 'lindex1',
                'lindex2', 'lmiddle0', 'lmiddle1', 'lmiddle2', 'lpinky0', 'lpinky1', 'lpinky2', 'lring0', 'lring1', 'lring2', 'lthumb0', 'lthumb1', 'lthumb2',
                'rindex0', 'rindex1', 'rindex2', 'rmiddle0', 'rmiddle1', 'rmiddle2', 'rpinky0', 'rpinky1', 'rpinky2', 'rring0', 'rring1', 'rring2',
                'rthumb0', 'rthumb1', 'rthumb2']

SMPLX_JOINT_NAMES = {0: 'Pelvis', 3: 'Spine1', 6: 'Spine2', 9: 'Spine3', 12: 'Neck', 15: 'Head',
                     1: 'L_Hip', 4: 'L_Knee', 7: 'L_Ankle', 10: 'L_Foot',
                     2: 'R_Hip', 5: 'R_Knee', 8: 'R_Ankle', 11: 'R_Foot',
                     13: 'L_Collar', 16: 'L_Shoulder', 18: 'L_Elbow', 20: 'L_Wrist',
                     14: 'R_Collar', 17: 'R_Shoulder', 19: 'R_Elbow', 21: 'R_Wrist',
                     22: 'Jaw', 23: 'L_eye', 24: 'R_eye',
                     25: 'lindex0', 26: 'lindex1', 27: 'lindex2',
                     28: 'lmiddle0', 29: 'lmiddle1', 30: 'lmiddle2',
                     31: 'lpinky0', 32: 'lpinky1', 33: 'lpinky2',
                     34: 'lring0', 35: 'lring1', 36: 'lring2',
                     37: 'lthumb0', 38: 'lthumb1', 39: 'lthumb2',
                     40: 'rindex0', 41: 'rindex1', 42: 'rindex2',
                     43: 'rmiddle0', 44: 'rmiddle1', 45: 'rmiddle2',
                     46: 'rpinky0', 47: 'rpinky1', 48: 'rpinky2',
                     49: 'rring0', 50: 'rring1', 51: 'rring2',
                     52: 'rthumb0', 53: 'rthumb1', 54: 'rthumb2'}

# Order is in harmony with the MANO joint ordering.
HAND_TIP_IDS = {'thumb':744, 'index':320, 'middle': 443, 'ring': 554, 'pinky': 671}

# Maps the joints to their correspondences for the data augmentation. 
SMPLX_JOINT_MIRROR_DICT = {0: 0, 1: 2, 2: 1, 3: 3, 4: 5, 5: 4, 6: 6, 7: 8, 8: 7, 9: 9, 10: 11, 11: 10, 12: 12, 13: 14, 14: 13, 15: 15,
                            16: 17, 17: 16, 18: 19, 19: 18, 20: 21, 21: 20, 22: 22, 23: 24, 24: 23, 25: 40, 26: 41, 27: 42, 28: 43, 
                            29: 44, 30: 45, 31: 46, 32: 47, 33: 48, 34: 49, 35: 50, 36: 51, 37: 52, 38: 53, 39: 54, 40: 25, 41: 26,
                            42: 27, 43: 28, 44: 29, 45: 30, 46: 31, 47: 32, 48: 33, 49: 34, 50: 35, 51: 36, 52: 37, 53: 38, 54: 39}

SMPLX_JOINT_MIRROR_MAP = sum([[3*i, 3*i+1, 3*i+2] for i in SMPLX_JOINT_MIRROR_DICT.values()], [])

LEFT_HAND_JOINT_DICT = {20: 'L_Wrist', 25: 'lindex0', 26: 'lindex1', 27: 'lindex2', 28: 'lmiddle0', 29: 'lmiddle1', 30: 'lmiddle2',
                     31: 'lpinky0', 32: 'lpinky1', 33: 'lpinky2', 34: 'lring0', 35: 'lring1', 36: 'lring2',
                     37: 'lthumb0', 38: 'lthumb1', 39: 'lthumb2'}

RIGHT_HAND_JOINT_DICT = {21: 'R_Wrist', 40: 'rindex0', 41: 'rindex1', 42: 'rindex2', 43: 'rmiddle0', 44: 'rmiddle1', 45: 'rmiddle2',
                     46: 'rpinky0', 47: 'rpinky1', 48: 'rpinky2', 49: 'rring0', 50: 'rring1', 51: 'rring2',
                     52: 'rthumb0', 53: 'rthumb1', 54: 'rthumb2'}

RIGHT_HAND_MANO_DICT = {'R_Wrist':21, 'rindex0':40, 'rindex1':41, 'rindex2':42, 'rmiddle0':43, 'rmiddle1':44, 'rmiddle2':45,
                        'rpinky0':46, 'rpinky1':47, 'rpinky2':48, 'rring0':49, 'rring1':50, 'rring2':51, 'rthumb0':52, 'rthumb1':53, 'rthumb2':54}

LEFT_WRIST_INDEX = SMPLX_JOINTS.index("L_Wrist")
RIGHT_WRIST_INDEX = SMPLX_JOINTS.index("R_Wrist")

NUM_BODY_JOINTS = len(SMPLX_JOINT_NAMES.keys())
NUM_HAND_JOINTS = len(LEFT_HAND_JOINT_DICT.keys()) 
NUM_HAND_TIP = len(list(HAND_TIP_IDS.values()))

HAND_KEYPOINT_VERTICES_DICT = {
                                "thumb": [709, 713, 240, 89],
                                "index_finger": [58, 167, 300, 297],
                                "middle_finger": [362, 394, 431, 457],
                                "ring_finger": [504, 523, 521, 474],
                                "little_finger" : [599, 609, 659, 642],
                                "palm": [ 113, 774, 40, 61, 157],
                                "knuckle": [144, 290, 270, 202],
                                "outside": [16, 22, 184] 
                              }       

KEYPOINT_VERTICES = list(itertools.chain.from_iterable(HAND_KEYPOINT_VERTICES_DICT.values()))
NUM_KEYPOINT_VERTS = len(KEYPOINT_VERTICES)

RIGHT_WRIST_BASE_LOC = torch.tensor([[0.0957, 0.0064, 0.0062]])
LEFT_WRIST_BASE_LOC = torch.tensor([[-0.0957, 0.0064, 0.0062]])


def return_mean_hand_pose(dataset_name):

    if dataset_name == "amass":
        return np.zeros((1, 45)), np.zeros((1, 45))
    elif dataset_name == "rollout":
        return np.zeros((1, 45)), np.zeros((1, 45))

    elif dataset_name == "interhands":
    
        lh_mean = torch.tensor([[0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
                            -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
                            -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
                            -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
                            -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
                            0.0266,  0.0529,  0.5356,  0.0460, -0.2774]])

        rh_mean = torch.tensor([[0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
                                 0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
                                 0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
                                -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
                                 0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
                                -0.0266, -0.0529,  0.5356, -0.0460,  0.2774]])
        return rh_mean, lh_mean
    
    else:
        print("MISTAKE")
        raise ValueError
        return rh_mean, lh_mean

def smplx_full_chain():
    """
    :return full kinematic tree of the smplx body model: np.array of shape [N, 2], first element is parent the second is children
    """
    tree = []
    end_effectors = np.setdiff1d(np.array(list(SMPLX_JOINT_NAMES.keys())), SMPLX_CHAIN)
    
    for end_eff in end_effectors:
        child = end_eff
        while child != -1:
            parent = SMPLX_CHAIN[child]
            if ([parent, child] not in tree) and (parent!=-1):
                tree.append([parent, child])
            child = parent

    return np.array(tree)

def mano_full_chain(is_rhand):
    """
    :return full kinematic tree of the given hand: np.array of shape [N, 2], first element is parent the second is children
    """
    tree = [] 

    hand_joint_list = np.array(list(RIGHT_HAND_JOINT_DICT.keys())) if is_rhand else np.array(list(LEFT_HAND_JOINT_DICT.keys()))

    wrist_index = RIGHT_WRIST_INDEX if is_rhand else LEFT_WRIST_INDEX
    start_index = hand_joint_list[1]

    hand_joint_array = copy.deepcopy(hand_joint_list) 

    # set the wrist to 0. 
    hand_joint_array[0] = 0
    # make the joints adjacent  
    hand_joint_array[1:] -= (start_index - 1) 

    # set containing end effector indexes
    end_effectors = set(hand_joint_array).difference(MANO_CHAIN)
    

    for end_eff in end_effectors:
        child = end_eff
        while child != -1:
            parent = MANO_CHAIN[child]
            if ([parent, child] not in tree) and (parent!=-1):
                tree.append([parent, child]) 
            child = parent

    return np.array(tree)


def single_chain(children_index):
    """
    :param chain:
    :return: kinematic tree, first element is parent the second is children
    """
    tree = []
    pointer = children_index

    while pointer != -1:
        element = pointer
        tree.append(pointer)
        pointer = SMPLX_CHAIN[element]
    return tree

SMPLX_MARKER_NAMES = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot',
                        'right_foot', 'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                        'left_wrist', 'right_wrist', 'jaw', 'left_eye_smplhf', 'right_eye_smplhf', 'left_index1', 'left_index2', 'left_index3', 'left_middle1',
                        'left_middle2', 'left_middle3', 'left_pinky1', 'left_pinky2', 'left_pinky3', 'left_ring1', 'left_ring2', 'left_ring3', 'left_thumb1',
                        'left_thumb2', 'left_thumb3', 'right_index1', 'right_index2', 'right_index3', 'right_middle1',
                        'right_middle2', 'right_middle3', 'right_pinky1', 'right_pinky2', 'right_pinky3', 'right_ring1', 'right_ring2', 'right_ring3',
                        'right_thumb1', 'right_thumb2', 'right_thumb3', 'nose', 'right_eye', 'left_eye', 'right_ear', 'left_ear', 'left_big_toe', 'left_small_toe', 'left_heel',
                        'right_big_toe', 'right_small_toe', 'right_heel', 'left_thumb', 'left_index', 'left_middle', 'left_ring',
                        'left_pinky', 'right_thumb', 'right_index', 'right_middle', 'right_ring', 'right_pinky', 'right_eye_brow1',
                        'right_eye_brow2', 'right_eye_brow3', 'right_eye_brow4', 'right_eye_brow5', 'left_eye_brow5', 'left_eye_brow4', 'left_eye_brow3', 'left_eye_brow2', 'left_eye_brow1',
                        'nose1', 'nose2', 'nose3', 'nose4', 'right_nose_2', 'right_nose_1', 'nose_middle', 'left_nose_1',
                        'left_nose_2', 'right_eye1', 'right_eye2', 'right_eye3', 'right_eye4', 'right_eye5', 'right_eye6', 'left_eye4', 'left_eye3', 'left_eye2',
                        'left_eye1', 'left_eye6', 'left_eye5', 'right_mouth_1', 'right_mouth_2', 'right_mouth_3', 'mouth_top', 'left_mouth_3', 'left_mouth_2', 'left_mouth_1',
                        'left_mouth_5',  # 59 in OpenPose output
                        'left_mouth_4',  # 58 in OpenPose output
                        'mouth_bottom', 'right_mouth_4', 'right_mouth_5', 'right_lip_1', 'right_lip_2', 'lip_top', 'left_lip_2', 'left_lip_1', 'left_lip_3', 'lip_bottom', 'right_lip_3',
                        # Face contour
                        'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5', 'right_contour_6', 'right_contour_7',
                        'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6', 'left_contour_5', 'left_contour_4',
                        'right_contour_1', 'right_contour_2', 'right_contour_3', 'right_contour_4', 'right_contour_5',
                        'right_contour_6', 'right_contour_7', 'right_contour_8', 'contour_middle', 'left_contour_8', 'left_contour_7', 'left_contour_6',
                        'left_contour_5', 'left_contour_4', 'left_contour_3', 'left_contour_2', 'left_contour_1',]


#
# From https://github.com/vchoutas/smplify-x/blob/master/smplifyx/utils.py
# Please see license for usage restrictions.
#
def to_openpose(model_type='smplx', use_hands=True, use_face=True,
                     use_face_contour=False, openpose_format='coco25'):
    ''' Returns the indices of the permutation that maps SMPL to OpenPose
        Parameters
        ----------
        model_type: str, optional
            The type of SMPL-like model that is used. The default mapping
            returned is for the SMPLX model
        use_hands: bool, optional
            Flag for adding to the returned permutation the mapping for the
            hand keypoints. Defaults to True
        use_face: bool, optional
            Flag for adding to the returned permutation the mapping for the
            face keypoints. Defaults to True
        use_face_contour: bool, optional
            Flag for appending the facial contour keypoints. Defaults to False
        openpose_format: bool, optional
            The output format of OpenPose. For now only COCO-25 and COCO-19 is
            supported. Defaults to 'coco25'
    '''
    # coco25, coco19 are for body keypoints. We are not interested in them
    # interhands pose format is different than openpose for hands 
    if openpose_format.lower() == 'interhands_mano_gt':
        if model_type == "mano":
            return np.array(np.arange(21))
        else: 
            raise ValueError("Only mano model is supported for interhands format")
    
    elif openpose_format.lower() == 'interhands_gt':   
        # PROBABLY WRONG 
        interhands_to_mano = {20: 0, 7:1, 6:2, 5:3, 11:4, 10:5, 9:6, 19:7, 18:8, 17:9, 15:10, 14:11, 13:12, 3:13, 2:14, 1:15} 
        return interhands_to_mano
        
    elif openpose_format.lower() == 'coco25':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8, 1, 4,
                             7, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56, 57, 58, 59,
                                     60, 61, 62], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 63, 22, 23, 24, 64,
                                          25, 26, 27, 65, 31, 32, 33, 66, 28,
                                          29, 30, 67], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 68, 37, 38, 39, 69,
                                          40, 41, 42, 70, 46, 47, 48, 71, 43,
                                          44, 45, 72], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59, 60, 61, 62,
                                     63, 64, 65], dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 66, 25, 26, 27,
                                          67, 28, 29, 30, 68, 34, 35, 36, 69,
                                          31, 32, 33, 70], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 71, 40, 41, 42, 72,
                                          43, 44, 45, 73, 49, 50, 51, 74, 46,
                                          47, 48, 75], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                #  end_idx = 127 + 17 * use_face_contour
                face_mapping = np.arange(76, 127 + 17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        # Mano model
        elif model_type == 'mano':
            
            # from openpose to mano (finger tips are included)
            rhand_mapping = np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], dtype=np.int32)
            
            # from mano to openpose (finger tips are included)
            rhand_mapping_ = np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20], dtype=np.int32)

            return rhand_mapping # np.array(np.arange(21))
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    elif openpose_format == 'coco19':
        if model_type == 'smpl':
            return np.array([24, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5, 8,
                             1, 4, 7, 25, 26, 27, 28],
                            dtype=np.int32)
        elif model_type == 'smplh':
            body_mapping = np.array([52, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 53, 54, 55, 56],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 34, 35, 36, 57, 22, 23, 24, 58,
                                          25, 26, 27, 59, 31, 32, 33, 60, 28,
                                          29, 30, 61], dtype=np.int32)
                rhand_mapping = np.array([21, 49, 50, 51, 62, 37, 38, 39, 63,
                                          40, 41, 42, 64, 46, 47, 48, 65, 43,
                                          44, 45, 66], dtype=np.int32)
                mapping += [lhand_mapping, rhand_mapping]
            return np.concatenate(mapping)
        # SMPLX
        elif model_type == 'smplx':
            body_mapping = np.array([55, 12, 17, 19, 21, 16, 18, 20, 0, 2, 5,
                                     8, 1, 4, 7, 56, 57, 58, 59],
                                    dtype=np.int32)
            mapping = [body_mapping]
            if use_hands:
                lhand_mapping = np.array([20, 37, 38, 39, 60, 25, 26, 27,
                                          61, 28, 29, 30, 62, 34, 35, 36, 63,
                                          31, 32, 33, 64], dtype=np.int32)
                rhand_mapping = np.array([21, 52, 53, 54, 65, 40, 41, 42, 66,
                                          43, 44, 45, 67, 49, 50, 51, 68, 46,
                                          47, 48, 69], dtype=np.int32)

                mapping += [lhand_mapping, rhand_mapping]
            if use_face:
                face_mapping = np.arange(70, 70 + 51 +
                                         17 * use_face_contour,
                                         dtype=np.int32)
                mapping += [face_mapping]

            return np.concatenate(mapping)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))
    else:
        raise ValueError('Unknown joint format: {}'.format(openpose_format))

