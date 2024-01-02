import torch 
import itertools
import numpy as np

import utils.body_utils as body_utils
from body_model import BodyModel
from utils.torch_utils import copy2cuda, copy2cpu, get_device
from utils.transform_utils import axisangle2matrots, matrot2axisangle

device = get_device()
flat_hand_flag = True
use_finger_tips = True

# size of MANO shape parameter to use
MANO_NUM_BETAS = 10
BODY_NUM_BETAS = 10

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



def get_mesh_sequence(body_path, mano_rh_path, mano_lh_path, num_frames, body_pose, pose_right_hand, pose_left_hand, body_betas, hand_betas, body_root_orient, body_trans):

    assert type(body_trans) == type(body_pose) == type(pose_right_hand) == type(pose_left_hand) == type(hand_betas) \
            == type(body_betas) == type(body_root_orient) == torch.Tensor 
    assert (body_root_orient == body_root_orient[:, :3]).all()
 
    b_out = get_body_model_sequence(body_model_path=body_path, num_frames=num_frames, pose_body=body_pose, betas=body_betas, trans=body_trans)

    # wrist locations
    r_wrist_location = b_out.joints[:, body_utils.RIGHT_WRIST_INDEX, :]
    l_wrist_location = b_out.joints[:, body_utils.LEFT_WRIST_INDEX, :]

    ##############################################
    # set hand translations to zero
    r_wrist_location = torch.zeros_like(r_wrist_location)
    l_wrist_location = torch.zeros_like(l_wrist_location)
    ##############################################

    rh_out, lh_out = get_hand_model_sequence(mano_rh_path=mano_rh_path, mano_lh_path=mano_lh_path, body_pose=body_pose, num_frames=num_frames, 
        pose_right_hand=pose_right_hand, pose_left_hand=pose_left_hand, hand_betas=hand_betas, rh_trans=r_wrist_location, lh_trans=l_wrist_location)

    return b_out, rh_out, lh_out

def get_body_model_sequence(body_model_path, num_frames, pose_body, betas, trans):

    assert type(pose_body) == type(betas) == torch.Tensor

    body_model = BodyModel(model_path=body_model_path, model_type="smplx", device=device, batch_size=num_frames, **{"flat_hand_mean":flat_hand_flag})

    betas = betas.T.repeat(num_frames, 1)
    
    body_input_dict = { "root_orient": pose_body[:, :3],
                        "body_pose": pose_body[:, 3:66],
                        "jaw_pose": pose_body[:, 66:69],
                        "leye_pose": pose_body[:, 69:72],
                        "reye_pose": pose_body[:, 72:75],
                        "left_hand_pose": pose_body[:, 75:120],
                        "right_hand_pose": pose_body[:, 120:],
                        "betas": betas,
                        "transl": trans}
 
    return body_model(body_input_dict)


def get_hand_model_sequence(mano_rh_path, mano_lh_path, body_pose, pose_right_hand, pose_left_hand, num_frames, hand_betas, rh_trans, lh_trans):

    hand_betas = hand_betas.T.repeat(num_frames, 1)
    
    right_hand_model = BodyModel(model_path=mano_rh_path, model_type="mano", device=device, batch_size=num_frames, **{"is_rhand":True, "flat_hand_mean":flat_hand_flag})
    left_hand_model = BodyModel(model_path=mano_lh_path, model_type="mano", device=device, batch_size=num_frames, **{"is_rhand":False, "flat_hand_mean":flat_hand_flag})

    r_orientation_axis, l_orientation_axis = get_wrist_orientations(body_pose=body_pose, num_frames=num_frames)

    # for now assume pseudo right and right are at the same location on space.
    rh_input_dict = {"hand_pose":pose_right_hand, 
                    "betas":hand_betas, 
                    "global_orient":r_orientation_axis, 
                    "no_shift": True,
                    # "transl":rh_trans,
                    "return_finger_tips": use_finger_tips}
    
    lh_input_dict = {"hand_pose":pose_left_hand, 
                    "betas":hand_betas, 
                    "global_orient":l_orientation_axis, 
                    "no_shift": True,
                    # "transl":lh_trans,
                    "return_finger_tips": use_finger_tips}
    
    rh_out = right_hand_model(rh_input_dict)
    lh_out = left_hand_model(lh_input_dict)
    
    return rh_out, lh_out

def get_wrist_orientations(body_pose, num_frames):

    assert type(body_pose) == torch.Tensor
    
    body_pose_reshaped = body_pose.reshape(num_frames, len(body_utils.SMPLX_JOINTS), 3)

    l_orientation = copy2cuda(torch.repeat_interleave(torch.eye(3, 3).unsqueeze(0), num_frames, dim=0))
    r_orientation = copy2cuda(torch.repeat_interleave(torch.eye(3, 3).unsqueeze(0), num_frames, dim=0))

    left_hand_chain = body_utils.single_chain(body_utils.SMPLX_JOINTS.index("L_Wrist"))
    right_hand_chain = body_utils.single_chain(body_utils.SMPLX_JOINTS.index("R_Wrist"))


    # matrix multiplication for all timesteps
    for l_part in left_hand_chain:
        matrix_representation = axisangle2matrots(body_pose_reshaped[:, l_part, :])
        l_orientation = torch.matmul(matrix_representation, l_orientation)

    for r_part in right_hand_chain:
        matrix_representation = axisangle2matrots(body_pose_reshaped[:, r_part, :])
        r_orientation = torch.matmul(matrix_representation, r_orientation)

    r_orientation_axis = matrot2axisangle(r_orientation).to(device)
    l_orientation_axis = matrot2axisangle(l_orientation).to(device)
    
    return r_orientation_axis, l_orientation_axis

def estimate_velocity(data_seq, h):
    """Given some data sequence of T timesteps in the shape (T, ...), estimates
    the velocity for the middle T-2 steps using a second order central difference scheme.
    - h : step size"""
    data_tp1 = data_seq[2:]
    data_tm1 = data_seq[0:-2]
    data_vel_seq = (data_tp1 - data_tm1) / (2 * h)
    return data_vel_seq 


def estimate_angular_velocity(rot_seq, h):
    '''
    Given a sequence of T rotation matrices, estimates angular velocity at T-2 steps.
    Input sequence should be of shape (T, ..., 3, 3)
    '''
    # see https://en.wikipedia.org/wiki/Angular_velocity#Calculation_from_the_orientation_matrix
    dRdt = estimate_velocity(rot_seq, h)
    R = rot_seq[1:-1]
    RT = np.swapaxes(R, -1, -2)
    # compute skew-symmetric angular velocity tensor
    w_mat = torch.matmul(dRdt, RT)
    # pull out angular velocity vector
    # average symmetric entries
    w_x = (-w_mat[..., 1, 2] + w_mat[..., 2, 1]) / 2.0
    w_y = (w_mat[..., 0, 2] - w_mat[..., 2, 0]) / 2.0
    w_z = (-w_mat[..., 0, 1] + w_mat[..., 1, 0]) / 2.0
    w = torch.stack([w_x, w_y, w_z], axis=-1)

    return w


def regress_hand_params():
    return copy2cuda(np.zeros((10, 1)))