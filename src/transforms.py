import enum
import math
import warnings
from typing import Tuple, List, Optional
import numpy as np

import torch
from torch import Tensor, tensor
import torch.nn.functional as F

from numpy import pi


def rotation_about_x(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def rotation_about_y(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]])


def rotation_about_z(angle: float) -> Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def mat4x4_inverse(mat4x4):
    R = mat4x4[:, :3, :3].clone()
    t = mat4x4[:, :3, 3].clone()
    R = R.transpose(-1, -2)
    t = -R @ t[..., None]
    mat4x4[:, :3, :3] = R
    mat4x4[:, :3, 3] = t[..., 0]
    return mat4x4


def gl_world_to_camera(camtoworld: Tensor) -> Tensor:
    camtoworld = camtoworld @ rotation_about_x(math.pi)[None].to(camtoworld) 
    return mat4x4_inverse(camtoworld)


def gl_camera_to_world(worldtocam: Tensor) -> Tensor:
    worldtocam = mat4x4_inverse(worldtocam)
    return worldtocam @ rotation_about_x(math.pi)[None].to(worldtocam)

def camera_to_world(worldtocam: Tensor) -> Tensor:
    worldtocam = mat4x4_inverse(worldtocam)
    return worldtocam # @ rotation_about_x(math.pi)[None].to(worldtocam)

def estimate_translation_np(S, joints_2d, joints_conf, focal_length=5000., img_size=224.):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (25, 3) 3D joint locations
        joints: (25, 3) 2D joint locations and confidence
    Returns:
        (3,) camera translation vector
    """

    num_joints = S.shape[0]
    # focal length
    f = np.array([focal_length,focal_length])
    # optical center
    center = np.array([img_size/2., img_size/2.])

    # transformations
    Z = np.reshape(np.tile(S[:,2],(2,1)).T,-1)
    XY = np.reshape(S[:,0:2],-1)
    O = np.tile(center,num_joints)
    F = np.tile(f,num_joints)
    weight2 = np.reshape(np.tile(np.sqrt(joints_conf),(2,1)).T,-1)

    # least squares
    Q = np.array([F*np.tile(np.array([1,0]),num_joints), F*np.tile(np.array([0,1]),num_joints), O-np.reshape(joints_2d,-1)]).T
    c = (np.reshape(joints_2d,-1)-O)*Z - F*XY

    # weighted least squares
    W = np.diagflat(weight2)
    Q = np.dot(W,Q)
    c = np.dot(W,c)

    # square matrix
    A = np.dot(Q.T,Q)
    b = np.dot(Q.T,c)

    # solution
    trans = np.linalg.solve(A, b)

    return trans


def estimate_translation(S, joints_2d, focal_length=5000., img_size=224., use_all_joints=False, rotation=None):
    """Find camera translation that brings 3D joints S closest to 2D the corresponding joints_2d.
    Input:
        S: (B, 49, 3) 3D joint locations
        joints: (B, 49, 3) 2D joint locations and confidence
    Returns:
        (B, 3) camera translation vectors
    """

    device = S.device

    if rotation is not None:
        S = torch.einsum('bij,bkj->bki', rotation, S)

    # Use only joints 25:49 (GT joints)
    if use_all_joints:
        S = S.cpu().numpy()
        joints_2d = joints_2d.cpu().numpy()
    else:
        S = S[:, 25:, :].cpu().numpy()
        joints_2d = joints_2d[:, 25:, :].cpu().numpy()

    joints_conf = joints_2d[:, :, -1]
    joints_2d = joints_2d[:, :, :-1]
    trans = np.zeros((S.shape[0], 3), dtype=np.float32)
    # Find the translation for each example in the batch
    for i in range(S.shape[0]):
        S_i = S[i]
        joints_i = joints_2d[i]
        conf_i = joints_conf[i]
        trans[i] = estimate_translation_np(S_i, joints_i, conf_i, focal_length=focal_length, img_size=img_size)
    return torch.from_numpy(trans).to(device)