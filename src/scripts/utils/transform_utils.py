import cv2
import torch
import numpy as np
from torch.nn import functional as F
from body_model import BodyModel
from utils.torch_utils import copy2cuda, copy2cpu
from utils.body_utils import single_chain, SMPLX_BODY_PATH, SMPLX_JOINTS, MANO_RIGHT_PATH, MANO_LEFT_PATH

# Humor
def convert_to_rotmat(pred_rot, rep='aa'):
    '''
    Converts rotation rep to rotation matrix based on the given type.
    pred_rot : B x T x N
    '''
    B, T, _ = pred_rot.size()
    pred_rot_mat = None
    if rep == 'aa':
        pred_rot_mat = batch_rodrigues(pred_rot.reshape(-1, 3))
    elif rep == '6d':
        pred_rot_mat = rot6d_to_rotmat(pred_rot.reshape(-1, 6))
    elif rep == '9d':
        pred_rot_mat = rot9d_to_rotmat(pred_rot.reshape(-1, 9))
    return pred_rot_mat.reshape((B, T, -1))


#
# Many of these functions taken from https://github.com/mkocabas/VIBE/blob/a859e45a907379aa2fba65a7b620b4a2d65dcf1b/lib/utils/geometry.py
# Please see their license for usage restrictions.
#

# PARE batch_rot2aa 
def matrot2axisangle(Rs):
    """
    Rs is B x 3 x 3
    void cMathUtil::RotMatToAxisAngle(const tMatrix& mat, tVector& out_axis,
                                      double& out_theta)
    {
        double c = 0.5 * (mat(0, 0) + mat(1, 1) + mat(2, 2) - 1);
        c = cMathUtil::Clamp(c, -1.0, 1.0);
        out_theta = std::acos(c);
        if (std::abs(out_theta) < 0.00001)
        {
            out_axis = tVector(0, 0, 1, 0);
        }
        else
        {
            double m21 = mat(2, 1) - mat(1, 2);
            double m02 = mat(0, 2) - mat(2, 0);
            double m10 = mat(1, 0) - mat(0, 1);
            double denom = std::sqrt(m21 * m21 + m02 * m02 + m10 * m10);
            out_axis[0] = m21 / denom;
            out_axis[1] = m02 / denom;
            out_axis[2] = m10 / denom;
            out_axis[3] = 0;
        }
    }
    """
    cos = 0.5 * (torch.stack([torch.trace(x) for x in Rs]) - 1)
    cos = torch.clamp(cos, -1, 1)

    theta = torch.acos(cos)

    m21 = Rs[:, 2, 1] - Rs[:, 1, 2]
    m02 = Rs[:, 0, 2] - Rs[:, 2, 0]
    m10 = Rs[:, 1, 0] - Rs[:, 0, 1]
    denom = torch.sqrt(m21 * m21 + m02 * m02 + m10 * m10)

    axis0 = torch.where(torch.abs(theta) < 0.00001, m21, m21 / denom)
    axis1 = torch.where(torch.abs(theta) < 0.00001, m02, m02 / denom)
    axis2 = torch.where(torch.abs(theta) < 0.00001, m10, m10 / denom)

    return theta.unsqueeze(1) * torch.stack([axis0, axis1, axis2], 1)

# humor 
def axisangle2matrots(axisangle):
    '''
    :param axisangle: N*num_joints*3
    :return: N*num_joints*9
    '''
    tensor_flag = False
    if type(axisangle) == torch.Tensor:
        tensor_flag = True
        tensor_device = axisangle.device
        axisangle = copy2cpu(axisangle)
    
    batch_size = axisangle.shape[0]
    axisangle = axisangle.reshape([batch_size, -1, 3])
    out_matrot = []
    for mIdx in range(axisangle.shape[0]):
        cur_axisangle = []
        for jIdx in range(axisangle.shape[1]):
            a = cv2.Rodrigues(axisangle[mIdx, jIdx:jIdx + 1, :].reshape(1, 3))[0]
            cur_axisangle.append(a)
        out_matrot.append(np.array(cur_axisangle))

    out_matrot = np.vstack(out_matrot)
    
    if tensor_flag:
        out_matrot = copy2cuda(out_matrot)
    
    return out_matrot


def make_rot_homog(rotation_matrix):
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)
    return rotation_matrix


def skew(v):
    '''
    Returns skew symmetric (B x 3 x 3) mat from vector v: B x 3
    '''
    B, D = v.size()
    assert (D == 3)
    skew_mat = torch.zeros((B, 3, 3)).to(v)
    skew_mat[:, 0, 1] = v[:, 2]
    skew_mat[:, 1, 0] = -v[:, 2]
    skew_mat[:, 0, 2] = v[:, 1]
    skew_mat[:, 2, 0] = -v[:, 1]
    skew_mat[:, 1, 2] = v[:, 0]
    skew_mat[:, 2, 1] = -v[:, 0]
    return skew_mat


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def quat2mat(quat):
    """
    This function is borrowed from https://github.com/MandyMo/pytorch_HMR/blob/master/src/util.py#L50
    Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [batch_size, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [batch_size, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                   2], norm_quat[:,
                                                       3]

    batch_size = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).view(batch_size, 3, 3)
    return rotMat


def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.view(-1, 3, 2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)

    # inp = a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1
    # denom = inp.pow(2).sum(dim=1).sqrt().unsqueeze(-1) + 1e-8
    # b2 = inp / denom

    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)


def rot9d_to_rotmat(x):
    '''
    Converts 9D rotation output to valid 3x3 rotation amtrix.
    Based on Levinson et al., An Analysis of SVD for Deep Rotation Estimation.
    Input:
        (B, 9)
    Output:
        (B, 9)
    '''
    B = x.size()[0]
    x = x.reshape((B, 3, 3))
    u, s, v = torch.svd(x)

    v_T = v.transpose(-2, -1)
    s_p = torch.eye(3).to(x).reshape((1, 3, 3)).expand_as(x).clone()
    s_p[:, 2, 2] = torch.det(torch.matmul(u, v_T))
    x_out = torch.matmul(torch.matmul(u, s_p), v_T)

    return x_out.reshape((B, 9))


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to Rodrigues vector
    Args:
        rotation_matrix (Tensor): rotation matrix.
    Returns:
        Tensor: Rodrigues vector transformation.
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`
    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = torch.tensor([0, 0, 1], dtype=torch.float32,
                           device=rotation_matrix.device).reshape(1, 3, 1).expand(rot_mat.shape[0], -1, -1)
        rotation_matrix = torch.cat([rot_mat, hom], dim=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    aa = quaternion_to_angle_axis(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert 3x4 rotation matrix to 4d quaternion vector
    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201
    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.
    Return:
        Tensor: the rotation in quaternion
    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`
    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia
    Convert quaternion vector to angle axis of rotation.
    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h
    Args:
        quaternion (torch.Tensor): tensor with quaternions.
    Return:
        torch.Tensor: tensor with angle axis of rotation.
    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`
    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1: torch.Tensor = quaternion[..., 1]
    q2: torch.Tensor = quaternion[..., 2]
    q3: torch.Tensor = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta)
    cos_theta: torch.Tensor = quaternion[..., 0]
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos: torch.Tensor = two_theta / sin_theta
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def process_pymaf_mano(pickle_obj):
    
    batch_count = len(pickle_obj["smplx_params"])
    frame_count = len(pickle_obj["frame_ids"])

    right_hand_pose, rh_orient, rh_betas = [], [], []
    
    for i in range(batch_count):
        bs = pickle_obj["smplx_params"][i]["body_pose"].shape[0]

        right_hand_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["body_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        rh_betas.append(pickle_obj["smplx_params"][i]["shape"])   
        rh_orient.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["root_orient"].reshape(-1, 3, 3)).reshape(bs, 3))
    
    right_hand_pose = torch.vstack(right_hand_pose)
    rh_betas = torch.vstack(rh_betas)
    rh_orient = torch.vstack(rh_orient)
    
    transl = torch.tensor(pickle_obj["orig_cam_t"], dtype=torch.float32)

    return rh_orient, transl, right_hand_pose, rh_betas 

def process_pymaf_smplx(pickle_obj):
    
    device = "cuda"
    flat_hand_flag = True
    
    batch_count = len(pickle_obj["smplx_params"])
    frame_count = len(pickle_obj["frame_ids"])

    body_model = BodyModel(model_path=SMPLX_BODY_PATH, model_type="smplx", device=device,  
                           batch_size=frame_count, name="body", skeleton_flag=True)
    # right_model = BodyModel(model_path=MANO_RIGHT_PATH, model_type="mano", **{"is_rhand":True, "flat_hand_mean":flat_hand_flag, "use_pca":False}, 
    #                         device=device, batch_size=frame_count, name="right_hand", mesh_color=[1, 0, 0], skeleton_flag=True)
    # left_model = BodyModel(model_path=MANO_LEFT_PATH, model_type="mano", **{"is_rhand":False, "flat_hand_mean":flat_hand_flag, "use_pca":False}, 
    #                        device=device, batch_size=frame_count, name="left_hand", mesh_color=colors["olive"], skeleton_flag=True)
 
    body_pose, jaw_pose, leye_pose, right_hand_pose, left_hand_pose = [], [], [], [], [] 
    reye_pose, expression = [], []
    
    # check if they are the same 
    # pickle_obj["pose"][:8] - pickle_obj["smplx_params"][0]["body_pose"].cpu().numpy()
    
    for i in range(batch_count):
        bs = pickle_obj["smplx_params"][i]["right_hand_pose"].shape[0]

        right_hand_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["right_hand_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        left_hand_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["left_hand_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))

        body_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["body_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        jaw_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["jaw_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        leye_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["leye_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        reye_pose.append(rotation_matrix_to_angle_axis(pickle_obj["smplx_params"][i]["reye_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        expression.append(pickle_obj["smplx_params"][i]["expression"])  	
    
    body_pose = torch.vstack(body_pose)
    right_hand_pose = torch.vstack(right_hand_pose)
    left_hand_pose = torch.vstack(left_hand_pose)
    reye_pose = torch.vstack(reye_pose)
    leye_pose = torch.vstack(leye_pose)
    jaw_pose = torch.vstack(jaw_pose)
    
    betas = pickle_obj["betas"]
    transl = pickle_obj["orig_cam_t"]

    
    body_model_output = body_model.forward({"body_pose":body_pose[:, 1:22].to(device), 
                                            "right_hand_pose":right_hand_pose,
                                            "left_hand_pose":left_hand_pose,
                                            "jaw_pose":jaw_pose,
                                            "betas":betas,
                                            "reye_pose":reye_pose,
                                            "leye_pose":leye_pose, 
                                            "transl":transl,
                                            "root_orient":body_pose[:, 0].to(device)})
    
    l_wrist_index = SMPLX_JOINTS.index("L_Wrist")
    r_wrist_index = SMPLX_JOINTS.index("R_Wrist")

    l_location = body_model_output.joints[:, l_wrist_index, :].detach() 
    r_location = body_model_output.joints[:, r_wrist_index, :].detach() 

    left_hand_chain = single_chain(l_wrist_index)
    right_hand_chain = single_chain(r_wrist_index)

    l_orientation = torch.repeat_interleave(torch.eye(3, 3).unsqueeze(0), frame_count, dim=0).to(device)
    r_orientation = torch.repeat_interleave(torch.eye(3, 3).unsqueeze(0), frame_count, dim=0).to(device)

 
    body_pose_reshaped = body_pose # .reshape(frame_count, len(SMPLX_JOINTS), 3)

    # matrix multiplication for all timesteps. Beware that the root orientation is also included in the pose.
    for l_part in left_hand_chain:
        matrix_representation = batch_rodrigues(body_pose_reshaped[:, l_part, :])
        l_orientation = torch.matmul(matrix_representation, l_orientation)

    for r_part in right_hand_chain:
        matrix_representation = batch_rodrigues(body_pose_reshaped[:, r_part, :])
        r_orientation = torch.matmul(matrix_representation, r_orientation)

    l_orientation_axis = rotation_matrix_to_angle_axis(l_orientation)
    r_orientation_axis = rotation_matrix_to_angle_axis(r_orientation)
 

    return r_location, r_orientation_axis, right_hand_pose, l_location, l_orientation_axis, left_hand_pose