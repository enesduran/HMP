import os 
import sys 
import copy
import torch
import numpy as np

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from utils import body_utils

"""Beware that, in the case of reflecting the wrist location, forwarding the output of this method to SMPLX model leads to implausible pose because we only are interested in the hand articulation and wrist(root of hand kinematic chain). """
def reflect_hands_in_body(pose_data, copy_left_to_right, reflect_wrist):
    timestep, joint_num, k = pose_data.shape

    assert k == 3, f"Pose data is not at the required shape. Required: (BatchDim, JointDim, 3) Input: {pose_data.shape}"

    if joint_num != len(body_utils.SMPLX_JOINTS):
        raise ValueError(f"Joint number must be 55. Input joint number: {joint_num}")

    augmented_data = copy.deepcopy(pose_data)

    l_hand_index = body_utils.SMPLX_LEFT_HAND_INDEXES
    r_hand_index = body_utils.SMPLX_RIGHT_HAND_INDEXES

    hand_of_interest = (r_hand_index, l_hand_index) if copy_left_to_right else (l_hand_index, r_hand_index)
    augmented_data[:, hand_of_interest[0], 1:] = copy.deepcopy(-1*pose_data[:, hand_of_interest[1], 1:])
    augmented_data[:, hand_of_interest[0], 0] = copy.deepcopy(pose_data[:, hand_of_interest[1], 0])

    # Reflect the kinematic chain to the wrist
    if reflect_wrist:
        l_wrist_index = body_utils.SMPLX_JOINTS.index("L_Wrist")
        r_wrist_index = body_utils.SMPLX_JOINTS.index("R_Wrist")

        left_hand_chain = body_utils.single_chain(l_wrist_index)
        right_hand_chain = body_utils.single_chain(r_wrist_index)

        hand_of_interest = (right_hand_chain, left_hand_chain) if copy_left_to_right else (left_hand_chain, right_hand_chain)

        chain0 = hand_of_interest[0]
        chain1 = hand_of_interest[1]
        augmented_data[:, chain0, 1:] = - augmented_data[:, chain1, 1:]
        
    return torch.from_numpy(np.resize(augmented_data, (timestep, joint_num * k)))

def reflect_hand(pose_hand):
    ''' 
    Reflect the hand pose only. The aim is to have the mirrored hand articulation. We dont need to know the orientation. 
    '''
    
    if type(pose_hand) == torch.Tensor:
        pose_hand = pose_hand.cpu()

    if len(pose_hand.shape) == 2:
        pose_hand = pose_hand.reshape(pose_hand.shape[0], -1, 3)
    timestep, pose_num, k = pose_hand.shape

    assert k == 3, f"Pose data is not at the required shape. Required: (BatchDim, JointDim, 3) Input: {pose_hand.shape}"

    assert (pose_num == body_utils.NUM_HAND_JOINTS-1) or (pose_num == body_utils.NUM_HAND_JOINTS), "Pose number must be either 15 or 16. Input shape: (T, " + str(pose_num) + ", " + str(k) + ")"

    augmented_data = copy.deepcopy(pose_hand)

    augmented_data[:, :, 1:] = copy.deepcopy(-1*pose_hand[:, :, 1:])
    augmented_data[:, :, 0] = copy.deepcopy(pose_hand[:, :, 0])    
    
    return torch.reshape(torch.from_numpy(augmented_data), (timestep, pose_num * k))


def reflect_body(b_pose, root_transl=None, root_rot=False):
    ''' 
    Reflect the body around yz plane. This is done via reversing the y and z entries of the bodypose.
    Notice that the bodypose is in axis angle representation. 
    
    b_pose = np.array (T, JX3)
    root_transl = np.array (T, 3) : If not None reflect the root also 
    root_rot = bool : If True also reflect the root node which is stored in the body pose (first 3 indices).     
    '''

    T, pose_dim = b_pose.shape
    assert pose_dim == body_utils.NUM_BODY_JOINTS * 3, f"Pose data is not at the required shape. Required: (BatchDim, JointDim * 3) Input: {b_pose.shape}"

    # If translation is unputted then reflect along yz plane. 
    if type(root_transl)==np.ndarray:
        root_transl[:, 0] *= -1
        
    start_ind = 0 if root_rot else 3
    index_set = np.setdiff1d(np.arange(start=start_ind, stop=pose_dim, step=1), np.arange(start=start_ind, stop=pose_dim, step=3))
    
    b_pose_copy = copy.deepcopy(b_pose)
    b_pose_copy[:, index_set] *= -1

    # Now we need to reflect the positons. To do this use directly the mirror mapping dictionary.
    b_pose_copy = b_pose_copy[:, body_utils.SMPLX_JOINT_MIRROR_MAP]
    assert b_pose_copy.shape == (T, pose_dim)

    return b_pose_copy, root_transl


def mirror_mesh(vertices):
    """ vertices shape: (N, 3) """

    if type(vertices) == torch.Tensor:  
        copied_vertices = copy.deepcopy(vertices.cpu().detach())
    else:
        copied_vertices = copy.deepcopy(vertices)

    copied_vertices[:, 0] = - copied_vertices[:, 0]
    return copied_vertices


if __name__ == "__main__":
    t1 = torch.rand(size=(1, 55, 3))
    t1_ = reflect_hands_in_body(t1, copy_left_to_right=True, reflect_wrist=False)

    print(t1[0, 25:, :])
    print(t1_.reshape(1, 55, 3)[0, 25:, :])






