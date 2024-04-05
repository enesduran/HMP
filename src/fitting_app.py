import os
import cv2
import copy
import glob
import time
import torch
import ffmpeg
import joblib
import random
import shutil
import subprocess
import numpy as np
import torch.nn as nn
from loguru import logger 
from argparse import Namespace
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable

import scenepic_viz
from utils import slerp
import open3d_viz_overlay
from datasets.amass import *
from arguments import Arguments
from argparse import ArgumentParser
from hposer_utils import load_hposer
from body_model.mano import BodyModel
from nemf.generative import Architecture
from nemf.fk import ForwardKinematicsLayer
from scripts.train_gmm import MaxMixturePrior
from keypoints2d.mediapipe_detect import MPJson
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from nemf.losses import GeodesicLoss, pos_smooth_loss, rot_smooth_loss
from keypoints2d.top_down_pose_tracking_demo_with_mmdet import MMPOSEJson
from rotations import (axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_rotation_6d,
                        quaternion_to_matrix, rotation_6d_to_matrix, axis_angle_to_quaternion, quaternion_to_axis_angle, quat_to_aa)
from fitting_utils import (process_gt, gmof, perspective_projection, get_joints2d, run_pymafx, run_metro, process_pymafx_mano, compute_seq_intervals,
                        save_quantitative_evaluation, get_seqname_ho3d_v3, get_seqname_arctic_data, get_seqname_in_the_wild, get_seqname_dexycb,
                        RIGHT_WRIST_BASE_LOC, joints2d_loss, map_openpose_joints_to_mano, map_mano_joints_to_openpose,
                        export_pymafx_json, blend_keypoints)


HAND_JOINT_NUM = 16
mano2openpose = map_mano_joints_to_openpose()
openpose2mano = map_openpose_joints_to_mano()

MANO_RH_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"
IGNORE_KEYS = ['cam_f', 'cam_center', 'img_height', 'img_width', 'img_dir',
                'save_path', 'frame_id', 'config_type', 'rh_verts', 'handedness']
POSSIBLE_KEYP_SOURCES = ["mmpose", "mediapipe", "mediapipe_std", "mediapipe_multiview", 
                                    "pymafx", "pymafx_std", "gt", "metro", "blend",
                                    "blend_mediapipe_std_mmpose",
                                      "blend_std", "blend_smooth"]
 
 
def forward_mano(output):
    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    B, T, J, _, _ = rotmat.size()
    
    # b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.view(-1, J, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(B, -1, J, 3, 3)

    root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
    
    local_rotmat[:, :, 0] = root_orient

    poses = matrix_to_axis_angle(local_rotmat)  # (T, J, 3)
    poses = poses.view(-1, J * 3)

    # no_shift is the flag for not shifting the wrist location to the origin
    mano_out = args.hand_model(input_dict={"betas":output['betas'].view(-1, 10),
                            "global_orient":poses[..., :3].view(-1, 3),
                            "hand_pose":poses[..., 3:].view(-1, 45),
                            "no_shift":True,
                            "return_finger_tips": True,
                            "transl":output['trans'].view(-1, 3)})
 
    return mano_out


def L_pose_prior(output):

    hposer.eval()

    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    B, T, J, _, _ = rotmat.size()
    
    # b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.view(-1, J, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(B, -1, J, 3, 3)

    local_rotmat_casted = local_rotmat[:, :, 1:, ...].view(B*T, -1)
  
    pose_latent_code = hposer.encode(local_rotmat_casted)
    pose_prior_mean_squared = (pose_latent_code.mean ** 2).mean(-1)
    
    loss = pose_prior_mean_squared.mean()
    
    return loss


def L_rot(pred, gt, T, conf=None):
    """
    Args:
        source, target: rotation matrices in the shape B x T x J x 3 x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    if conf is not None:
        criterion_rec = nn.L1Loss(reduction='none') if args.l1_loss else nn.MSELoss(reduction='none')
    else:
        criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()
    
    B, seqlen, J, _, _ = pred.shape
   
    if args.geodesic_loss:
        if conf is not None:
            loss = (conf.squeeze(-1) ** 2) *  criterion_geo(pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3), reduction='none').reshape(B, seqlen, J)
            loss = loss.mean()
        else:
            loss = criterion_geo(pred[:, T].view(-1, 3, 3), gt[:, T].view(-1, 3, 3))
    else:
        if conf is not None:
            loss = (conf.unsqueeze(-1) ** 2) *  criterion_rec(pred[:, T], gt[:, T])
            loss = loss.mean()
        else:
            loss = criterion_rec(pred[:, T], gt[:, T])

    return loss


def L_PCA(pose):
    
    bs = pose.shape[0]
    
    # convert aa to rotmat 
    pose_rotmat_global = axis_angle_to_matrix(pose)
    
    # convert global to local 
    pose_rotmat_local = fk.global_to_local(pose_rotmat_global.reshape(-1, 16, 3, 3))
    
    # convert back to aa 
    pose_aa_local = matrix_to_axis_angle(pose_rotmat_local)
    
    pose_reshaped = pose_aa_local[:, 1:, :].reshape(-1, args.data.clip_length * 15 * 3)
    
    pose_reshaped_centered = pose_reshaped - pca_mean
    
    pose_projected = pose_reshaped_centered @ pca_pc.T  
    
    normalized_pose_loss = abs(pose_projected) / pca_sv 
     
    mp_loss = normalized_pose_loss.mean(0).mean()

    return mp_loss

def L_GMM(pose):
    
    bs = pose.shape[0]
        
    # convert aa to rotmat 
    pose_rotmat_global = axis_angle_to_matrix(pose)
    
    # convert global to local 
    pose_rotmat_local = fk.global_to_local(pose_rotmat_global.reshape(-1, 16, 3, 3))
    
    # convert back to aa 
    pose_aa_local = matrix_to_axis_angle(pose_rotmat_local).reshape(bs, args.data.clip_length, -1, 3)
    
    mp_loss_list = []
    loss_tot = 0
    
    for i in range(bs):
        loss_i = gmm_aa.log_likelihood(pose_aa_local[i, :, 1:].reshape(1, -1).cpu())
        loss_tot += loss_i
        
    mp_loss = loss_tot / bs     
    return mp_loss.to("cuda")

def L_orient(source, target, T, bbox_conf=None):
    """
    Args:
        source: predicted root orientation in the shape B x T x 6.
        target: root orientation in the shape of B x T x 6.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the root orientation.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()

    source = rotation_6d_to_matrix(source)  # (B, T, 3, 3)
    target = rotation_6d_to_matrix(target)  # (B, T, 3, 3)

    if args.geodesic_loss:
        
        if bbox_conf is not None:
            
            loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3), reduction='none')
            bbox_conf_coef = bbox_conf.reshape(-1)
            loss = ((bbox_conf_coef ** 2) * loss).mean()
            
        else:
            loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
            
    else:
        loss = criterion_rec(source[:, T], target[:, T])
    
    return loss


def L_trans(source, target, T, bbox_conf=None):
    """
    Args:
        source: predict global translation in the shape B x T x 3 (the origin is (0, 0, height)).
        target: global translation of the root joint in the shape B x T x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the global translation.
    """
 
    trans = source
    trans_gt = target
     
    # dont make reduction and weight by bbox_conf
    if bbox_conf is not None:
        criterion_pred = nn.L1Loss(reduction='none') if args.l1_loss else nn.MSELoss(reduction='none')
        
        # reshape to (T * N, 3)
        loss = criterion_pred(trans[:, T].reshape(-1, 3), trans_gt[:, T].reshape(-1, 3)).mean(1)
        bbox_conf_coef = bbox_conf.reshape(-1)
        loss = ((bbox_conf_coef **2) * loss).mean()
        
    else:
        criterion_pred = nn.L1Loss() if args.l1_loss else nn.MSELoss()
        loss = criterion_pred(trans[:, T], trans_gt[:, T])

    return loss


def L_pos(source, target, T):
    """
    Args:
        source, target: joint local positions in the shape B x T x J x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the joint local positions.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    loss = criterion_rec(source[:, T], target[:, T])

    return loss

 
def contact_vel_loss(contacts_conf, joints3d):
    '''
    Velocity should be zero at predicted contacts
    '''
    delta_pos = (joints3d[:,1:] - joints3d[:,:-1])**2
    cur_loss = delta_pos.sum(dim=-1) * contacts_conf[:,1:]
    cur_loss = 0.5 * torch.mean(cur_loss)

    return cur_loss

def motion_prior_loss(latent_motion_pred):
    # assume standard normal
    loss = latent_motion_pred**2
    loss = torch.mean(loss)
    
    return loss


def motion_reconstruction(target, output_dir, steps, T=None, offset=0):
 
    z_l, _, _ = model.encode_local()
    z_g, _, _ = model.encode_global()
     
    # optimize pose directly 
    pose_aa = matrix_to_axis_angle(model.input_data["rotmat"].detach().clone()) if motion_prior_type in ['gmm', 'pca'] else None

    z_l, z_g, opt_betas, opt_root_orient, opt_trans, cam_R, cam_t, opt_pose_aa = latent_optimization(target, T=T, z_l=z_l, z_g=z_g, pose=pose_aa)
   
    for step in steps:
        with torch.no_grad():
            B, seqlen, _ = opt_trans.shape
                
            if motion_prior_type in ["pca", "gmm"]:
                output = {"rotmat": axis_angle_to_matrix(opt_pose_aa)}    
            else:
                output = model.decode(z_l, z_g=z_g, length=args.data.clip_length, step=step)
                
            output['betas'] = opt_betas[:, None, None, :].repeat(1, B, seqlen, 1)
            output['trans'] = opt_trans.clone().reshape(B*args.nsubject, seqlen, 3)
            output['root_orient'] = opt_root_orient
            
            rh_mano_out = forward_mano(output) 
            joints3d_pred = rh_mano_out.joints.view(B, seqlen, -1, 3)
            vertices_pred = rh_mano_out.vertices.view(B, seqlen, -1, 3)

            optim_cam_R = rotation_6d_to_matrix(cam_R)
            optim_cam_t = cam_t
                    
            joints2d_pred = get_joints2d(joints3d_pred=joints3d_pred, 
                                    cam_t=optim_cam_t.unsqueeze(0).repeat_interleave(args.nsubject, 0),
                                    cam_R=optim_cam_R.unsqueeze(0).repeat_interleave(args.nsubject, 0),
                                    cam_f=torch.tensor([5000., 5000.]), 
                                    cam_center=target['cam_center'])
            
 
        fps = int(args.data.fps / step)
        criterion_geo = GeodesicLoss()

        rotmat = output['rotmat']  # (B, T, J, 3, 3)
        rotmat_gt = target['rotmat']  # (B, T, J, 3, 3)
        b_size, _, n_joints = rotmat.shape[:3]
        
        local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
        local_rotmat_gt = fk.global_to_local(rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)

        root_orient = rotation_6d_to_matrix(opt_root_orient)  # (B, T, 3, 3)
        root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
        
        global_trans = opt_trans.clone()
        B = global_trans.shape[0] 
    
        if args.data.root_transform:
            local_rotmat[:, :, 0] = root_orient
            local_rotmat_gt[:, :, 0] = root_orient_gt

 
        trans = global_trans 
        trans_gt = target['trans']  # (B, T, 3)
    
        if trans.shape[0] > 1:

            # batch optimization 
            if args.overlap_len == 0:
                # concat_results
                trans = trans.reshape(1, -1, 3)
                trans_gt = trans_gt.reshape(1, -1, 3)
                local_rotmat = local_rotmat.reshape(1, -1, n_joints, 3, 3)
                local_rotmat_gt = local_rotmat_gt.reshape(1, -1, n_joints, 3, 3)
                cam_R = target['cam_R'].reshape(1, -1, 3, 3) 
                cam_t = target['cam_t'].reshape(1, -1, 3) 
                joints2d_pred = joints2d_pred.reshape(1, -1, *joints2d_pred.shape[2:])
                joints3d_pred = joints3d_pred.reshape(1, -1, *joints3d_pred.shape[2:])
                vertices_pred = vertices_pred.reshape(1, -1, *vertices_pred.shape[2:])
                
                for k, v in target.items():
                    if not (k in ['cam_f', 'cam_center', 'img_height', 'img_width', 'img_dir', 'save_path', 'frame_id', 'config_type', 'rh_verts', 'handedness']):
                        
                        target[k] = v.reshape(1, -1, * v.shape[2:])    
            
            else:           
                B, T, J, _, _ = output["rotmat"].shape

                resh = lambda x: x.reshape(1, B, *x.shape[1:])
                
                def concat(x):
                    x = resh(x)
                    ls = [x[:, 0]]
                    for bid in range(1, x.shape[1]):
                        ls.append(x[:, bid, args.overlap_len:])
                    return torch.cat(ls, dim=1)[:, :args.orig_seq_len]
                    
                # concat_cam = lambda x: torch.cat([x[0]] + [y[args.overlap_len:] for y in x[1:]], dim=0).unsqueeze(0)[:, :args.orig_seq_len]
                trans = concat(trans)
                trans_gt = concat(trans_gt)
                local_rotmat = concat(local_rotmat)
                local_rotmat_gt = concat(local_rotmat_gt)
                cam_R = target['cam_R'].reshape(1, -1, 3, 3) 
                cam_t = target['cam_t'].reshape(1, -1, 3) 
                joints2d_pred = concat(joints2d_pred)
                joints3d_pred = concat(joints3d_pred)
                vertices_pred = concat(vertices_pred)
                
                for k, v in target.items():
                    if not (k in IGNORE_KEYS):
                        target[k] = concat(v)
           

        for i in range(local_rotmat.shape[0]): # (1, T. J, 3, 3)
            
            poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
            poses_gt = c2c(matrix_to_axis_angle(local_rotmat_gt[i]))  # (T, J, 3)
            
            poses = poses.reshape((poses.shape[0], -1))  # (T, 48)
            poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, 66)
     
           
 
            limit = poses.shape[0]
            
            pred_save_path = os.path.join(output_dir, f'recon_{offset + i:03d}_{fps}fps.npz')
            gt_save_path = pred_save_path if args.raw_config else os.path.join(output_dir, f'recon_{offset + i:03d}_gt.npz') 

            # save opt results if not _pymafx_raw or _metro_raw 
            if not args.raw_config:
                np.savez(pred_save_path,
                        poses=poses[:limit], 
                        trans=c2c(trans[i][:limit]), 
                        betas=opt_betas.detach().cpu().numpy()[:limit], 
                        gender=args.data.gender, 
                        mocap_framerate=fps,
                        cam_R=c2c(target['cam_R'][i][0]),
                        cam_t=c2c(target["cam_t"][i][0]),
                        cam_f=c2c(target['cam_f'].unsqueeze(0)),
                        cam_center=c2c(target['cam_center'].unsqueeze(0)),
                        joints_2d=c2c(joints2d_pred[i][:limit]),
                        keypoints_2d=c2c(target['joints2d'][i][:limit]),  # detected keypoints with confidence
                        vertices=c2c(vertices_pred[i][:limit]),
                        save_path=pred_save_path,
                        handedness=target["handedness"],
                        img_dir=str(target['img_dir']),
                        frame_id=target['frame_id'],
                        img_height=target['img_height'],
                        img_width=target['img_width'],
                        config_type=target['config_type'],
                        joints_3d=c2c(joints3d_pred)[i][:limit])
             
            if ("_encode_decode" in gt_save_path) or args.raw_config: 
                np.savez(gt_save_path,
                        poses=poses_gt[:limit], 
                        trans=c2c(trans_gt[i][:limit]), 
                        betas=target["betas"][0].detach().cpu().numpy()[:limit], 
                        vertices=target["rh_verts"].detach().cpu().numpy()[:limit], 
                        config_type=target['config_type'] if args.raw_config else None,
                        gender=args.data.gender, 
                        save_path=gt_save_path,
                        mocap_framerate=args.data.fps,
                        frame_id=target['frame_id'],
                        img_height=target['img_height'],
                        img_width=target['img_width'],
                        cam_R=c2c(target['cam_R'][i][0]),
                        cam_t=c2c(target['cam_t'][i][0]),
                        cam_f=c2c(target['cam_f'].unsqueeze(0)),
                        cam_center=c2c(target['cam_center'].unsqueeze(0)),
                        joints2d=c2c(target['joints2d'][i, ...,])[:limit], 
                        joints_2d=c2c(target['joints2d'][i, ...,])[:limit], 
                        keypoints_2d=c2c(target['joints2d'][i, ...,])[:limit], # duplicated, because alignment reads that key value
                        joints_3d=c2c(target['joints3d'][i])[:limit])

def get_gt_path(expname):
    if args.dataname == "DexYCB":
        gt_path = args.vid_path
        subjectname = args.vid_path.split("/")[4]
    elif args.dataname == "HO3D_v3":
        gt_path = os.path.join(os.path.dirname(args.vid_path), "meta")
        subjectname = args.vid_path.split("/")[-2]
    elif args.dataname == "arctic_data":
        subjectname = args.vid_path.split("/")[-3]
        split_path = os.path.join(f"./external/arctic/data/arctic_data/data/splits/p1_{expname}.npy")
        meta_path = os.path.join(f"./external/arctic/data/arctic_data/data/meta/misc.json")
        gt_path = {"gt_path": args.vid_path, "split": split_path, "meta_path": meta_path}
    else:  
        args.dataname = "in_the_wild"
        subjectname = args.vid_path.split("/")[-2]
        gt_path = os.path.join(os.path.dirname(args.vid_path), "meta")

    return gt_path, subjectname 


def slerp_first_stage(stage1_out_obj, num_frames, bbox_confs=None, SLERP_THRESH=0.6, methodname="pymafx", perform_slerp_flag=True):

    try: 
        gt_frame_ids = stage1_out_obj["frame_ids"]
    except:
        gt_frame_ids = stage1_out_obj["frame_id"]
        	
    
    # read output.pkl file and obtain initial pose, root orientation, translation and shape
    if methodname == "pymafx":
        rhand_orient, rhand_trans, rhand_pose, rhand_betas, rh_verts = process_pymafx_mano(stage1_out_obj) 
    else:
        rhand_orient = torch.tensor(stage1_out_obj["orient"])
        rhand_trans = torch.tensor(stage1_out_obj["trans"])
        rhand_pose = torch.tensor(stage1_out_obj["pose"]).reshape(-1, 15, 3)
        rhand_betas = torch.tensor(stage1_out_obj["betas"])
        rh_verts = torch.tensor(stage1_out_obj["vertices"]).reshape(-1, 778, 3)
 
    rhand_orient_padded = torch.zeros((num_frames, 3)).to(rhand_orient)
    rhand_trans_padded = torch.zeros((num_frames, 3)).to(rhand_trans)
    rhand_betas_padded = torch.zeros((num_frames, 10)).to(rhand_betas)
    rhand_pose_padded = torch.zeros((num_frames, 15, 3)).to(rhand_pose)
    rh_verts_padded = torch.zeros((num_frames, 778, 3)).to(rh_verts)

    if rhand_orient is None:
        logger.info("Regressor does not provide any output, means that no hand detected in the video") 
        # rhand_trans_padded = torch.repeat_interleave(torch.tensor([[0, 0, -2.5]]), repeats=num_frames, dim=0).to(rhand_trans)
        stage1_out_obj["pymafx_joints2d"] = np.zeros((num_frames, 21, 3))
        rh_verts = torch.zeros((num_frames, 778, 3))
        return rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts_padded
  
    idx = torch.tensor(np.array(gt_frame_ids)).to("cuda")
    rhand_orient_padded[idx] = rhand_orient
    rhand_trans_padded[idx] = rhand_trans
    rhand_betas_padded[idx] = rhand_betas
    rhand_pose_padded[idx] = rhand_pose
    rh_verts_padded[idx] = rh_verts
     
    rhand_pose_concat = torch.cat((rhand_pose_padded, rhand_orient_padded.unsqueeze(1)), dim=1)
    
    # perform slerp between first and the last detected frames. We cannot hallucinate before the start and after the end of the video.
    if perform_slerp_flag and (bbox_confs is not None):
        mask_bbox = np.where(bbox_confs < SLERP_THRESH)[0]
        pred_frame_ids = np.setdiff1d(gt_frame_ids, mask_bbox, assume_unique=False)
        
        # If there are less than 2 frames 
        if len(pred_frame_ids) < 2:
            pred_frame_ids = gt_frame_ids
         
    else: 
        pred_frame_ids = gt_frame_ids

     
    end_frame = max(pred_frame_ids)
    start_frame = min(pred_frame_ids)
    
    slerp_quat, slerp_trans = slerp(quat=axis_angle_to_quaternion(rhand_pose_concat), 
            trans=rhand_trans_padded, key_times=pred_frame_ids, times=np.arange(start_frame, end_frame), 
            mask=True)
  
    rhand_pose = quaternion_to_axis_angle(torch.tensor(slerp_quat, dtype=torch.float))
 
    # fill the padded values with the slerp values 
    rhand_orient_padded[start_frame:end_frame] = rhand_pose[:, 15, :]
    rhand_pose_padded[start_frame:end_frame] = rhand_pose[:, :15, :] 
    rhand_trans_padded[start_frame:end_frame] = torch.tensor(slerp_trans, dtype=torch.float32)
    
    start_pad = start_frame
    end_pad = num_frames - end_frame
    
     # end infilling
    rhand_pose_padded[end_frame:] = torch.repeat_interleave(rhand_pose[-1, :15, :].unsqueeze(0) , repeats=end_pad, dim=0)
    rhand_orient_padded[end_frame:] = torch.repeat_interleave(rhand_orient[-1].unsqueeze(0) , repeats=end_pad, dim=0)
    
    # start infilling
    rhand_pose_padded[:start_frame] = torch.repeat_interleave(rhand_pose[-1, :15, :].unsqueeze(0) , repeats=start_pad, dim=0)
    rhand_orient_padded[:start_frame] = torch.repeat_interleave(rhand_orient[-1].unsqueeze(0) , repeats=start_pad, dim=0)
      
    return rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts_padded
    # CHECK THIS AGAIN . detect the places where there is a change. make sure that the change is only due to slerp. 
    # np.where(np.sum(np.sum(slerp_quat[:, 1:] - np.array(axis_angle_to_quaternion(rhand_pose_padded[start_frame:end_frame]).cpu()), axis=1), axis=1) > 0.000001)

   
def obtain_keypoints(gt_values, keypoint_blend_weight):

    # define flags for the keypoint sources
    mediapipe_multiview_flag, mmpose_flag, gt_flag, mediapipe_std_flag, metro_flag, mediapipe_flag, pymafx_flag, blend_flag, blend_std_flag = \
            False, False, False, False, False, False, False, False, False
    
    # define flags, paths for the keypoints and their video outputs
    for source in POSSIBLE_KEYP_SOURCES:
        exec(f"{source}_keypoints2d_path = get_keyp_path('{source}')", globals())
        exec(f"{source}_vid_out_path = get_vid_out_path('{source}')", globals())
    
    joints2d_source = eval(args.keypoint_source + "_keypoints2d_path")
    joints2d_vid_out_path = eval(args.keypoint_source + "_vid_out_path")
    
    if args.keypoint_source == "blend":
        blend_flag, pymafx_flag, mediapipe_multiview_flag = True, True, True
    elif args.keypoint_source == "blend_std":
        blend_std_flag, pymafx_flag, mediapipe_std_flag = True, True, True
    elif args.keypoint_source == "mediapipe":
        mediapipe_multiview_flag = True
    elif args.keypoint_source == "mediapipe_std":
        mediapipe_std_flag = True
    elif args.keypoint_source == "pymafx":
        pymafx_flag = True
    elif args.keypoint_source == "metro":
        metro_flag = True
    elif args.keypoint_source == "gt":
        gt_flag = True
    elif args.keypoint_source == "mmpose":
        mmpose_flag = True
    else:
        raise ValueError("Invalid keypoint source")
     
    if pymafx_flag:
        if not os.path.exists(pymafx_keypoints2d_path):     
            export_pymafx_json(fpath=init_out_obj["video_path"], jts_2d=init_out_obj["pymafx_joints2d"])
    
    if metro_flag:
        if not os.path.exists(metro_keypoints2d_path):     
            raise NotImplementedError("Metro is not implemented yet")
            
    if mediapipe_multiview_flag:
        if not os.path.exists(mediapipe_multiview_keypoints2d_path) or (args.N_frames != len(glob.glob(mediapipe_multiview_keypoints2d_path + "/*.json"))):         
            MPJson(gt_dict=gt_values, expand_coef=1.4).detect_mp_multiview(img_dir=args.vid_path, out_dir=mediapipe_muliview_vid_out_path, 
                                    json_out_dir=mediapipe_multiview_keypoints2d_path, video_out=True)

    if mediapipe_flag:
        if not os.path.exists(mediapipe_keypoints2d_path) or (args.N_frames != len(glob.glob(mediapipe_keypoints2d_path + "/*.json"))): 
            MPJson(gt_dict=gt_values, expand_coef=1.4).detect_mp(img_dir=args.vid_path, out_dir=mediapipe_vid_out_path, 
                                    json_out_dir=mediapipe_keypoints2d_path, video_out=True)
        
    # obtain mean and std values for keypoints for different (11 views)
    if mediapipe_std_flag and (not os.path.exists(mediapipe_std_keypoints2d_path) or (args.N_frames != len(glob.glob(mediapipe_std_keypoints2d_path + "/*.json")))):
        MPJson(gt_dict=gt_values, expand_coef=1.4).detect_mp_std(img_dir=args.vid_path, out_dir=mediapipe_std_vid_out_path,
                    json_out_dir=mediapipe_std_keypoints2d_path, video_out=False)
   

    if blend_flag or blend_std_flag:
        # create blend keypoints if not already done. joints2d_source may either be blend or blend_smooth
        if (not os.path.exists(joints2d_source) or (args.N_frames != len(glob.glob(joints2d_source + "/*.json")))):
            smooth_flag = False if "smooth" not in args.keypoint_source else True
            
            assert args.keypoint_source in ["blend_mediapipe_std_mmpose", "blend_std", "blend_smooth", "blend"] 
            
            # mediapipe_std + mmpose  
            if args.keypoint_source == "blend_mediapipe_std_mmpose":
                source1_path = mediapipe_std_keypoints2d_path
                source2_path = mmpose_keypoints2d_path
            # mediapipe_std + pymafx 
            elif args.keypoint_source == "blend_std":
                source1_path = mediapipe_std_keypoints2d_path
                source2_path = pymafx_keypoints2d_path 
            else:
                source1_path = mediapipe_multiview_keypoints2d_path 
                source2_path = pymafx_keypoints2d_path 
                
            blend_keypoints(source_path1=source1_path, source_path2=source2_path, raw_img_path=args.vid_path, gt_frames=gt_values["frame_id"],
                    target_path=joints2d_source, blend_ratio=keypoint_blend_weight, blend_vid_out_path=joints2d_vid_out_path,
                    smooth_flag=smooth_flag, render=True) 
    
    return 
    

def get_keyp_path(keypoint_source, weight=1.0):
    if keypoint_source in ["blend", "blend_std"]:
        keyp_path = os.path.join(os.path.dirname(args.vid_path), f"{keypoint_source}_{weight}_keypoints2d")
    else:
        keyp_path = os.path.join(os.path.dirname(args.vid_path), f"{keypoint_source}_keypoints2d")

    return keyp_path

def get_vid_out_path(keypoint_source, weight=1.0):
    if keypoint_source in ["blend", "blend_std"]:
        vid_out_path = os.path.join(os.path.dirname(args.vid_path), f"rgb_{keypoint_source}_{weight}")
    else:
        vid_out_path = os.path.join(os.path.dirname(args.vid_path), f"rgb_{keypoint_source}")

    return vid_out_path


def multi_stage_opt(config_f, data_name, init_method_name):
    
    logger.info(f"Running reconstruction for {args.vid_path}")
    
    config_type = config_f.split('/')[-1].split('.')[:-1]
    config_type = '.'.join(config_type)
    
    keypoint_blend_weight = 1.0
    abs_video_path = os.path.join(os.getcwd(), args.vid_path)
 
    args.N_frames = len(glob.glob(os.path.join(abs_video_path, "*.jpg")))
    
    if args.N_frames == 0:
        args.N_frames = len(glob.glob(os.path.join(abs_video_path, "*.png")))
    
    args.hand_model = BodyModel(model_type="mano", model_path=MANO_RH_DIR, device='cuda', 
                           **{"flat_hand_mean":True, "use_pca":False, "batch_size":args.N_frames, "is_rhand":True})
    gt_path, subjectname = get_gt_path(data_name)
     
    assert args.keypoint_source in POSSIBLE_KEYP_SOURCES, "Please specify a valid keypoint source"
 
    # either pymafx or metro 
    init_method_jts_vid_out_name = os.path.join(os.getcwd(), os.path.dirname(args.vid_path), f"rgb_{init_method_name}.mp4")
    init_method_out_path = os.path.join(os.path.dirname(args.vid_path), f"{init_method_name}_out")
    init_method_raw_res_path = os.path.join(init_method_out_path, f"{init_method_name}_raw_res.npz") 
    init_method_slerp_res_path = os.path.join(init_method_out_path, f"{init_method_name}_slerp_res.npz") 
    init_method_slerp_res_bbox_path = os.path.join(init_method_out_path, f"{init_method_name}_slerp_res_bbox.npz")
    init_method_vid_out_path = os.path.join(os.path.dirname(args.vid_path), f"rgb_{init_method_name}")
         
    mediapipe_bbox_conf_path = os.path.join(os.path.dirname(args.vid_path), "confidences", "mediapipe_bbox_conf.npz")
      
    gt_values = eval(f"get_{args.dataname.lower()}_gt")(gt_path, render=False)    
    gt_values["exp_setup_name"] = data_name                               # this will be used in quant evaluation 
    seqname = eval(f"get_seqname_{args.dataname.lower()}")(args.vid_path)
    
    gt_frame_id = gt_values["frame_id"]  # empty list if not available    
    
    if len(gt_frame_id) == 0 and args.dataname != "in_the_wild":
        logger.info(f"No ground truth frame detected, skipping pymafx & optimization, {args.vid_path}")
        return 
     
    try:
        width, height, _ = cv2.imread(glob.glob(os.path.join(abs_video_path, "*.png"))[0]).shape
        num_frames = len(glob.glob(os.path.join(abs_video_path, "*.png")))
    except:
        width, height, _ = cv2.imread(glob.glob(os.path.join(abs_video_path, "*.jpg"))[0]).shape
        num_frames = len(glob.glob(os.path.join(abs_video_path, "*.jpg")))
    
    gt_values["width"] = width
    gt_values["height"] = height
    gt_values["source"] = data_name

    # input reflected view in case of left hand. 
    args.vid_path = args.vid_path if gt_values["handedness"] == "right" else os.path.join(os.path.dirname(args.vid_path), "rgb_pseudo_right") 
    cam_center = torch.tensor([[height / 2, width / 2]])   
    

    # put mmpose bbox instead of gt in itw setting. Gt frames are the bbox detected frames for itw setting.
    mmpose_keypoints2d_path = os.path.join(os.path.dirname(args.vid_path), "mmpose_keypoints2d")
    
    # check if mmpose keypoints are already there and full. If not, run mmpose.
    cond_mmpose = (not os.path.exists(mmpose_keypoints2d_path)) or  \
                    (args.N_frames != len(glob.glob(mmpose_keypoints2d_path + "/*.json")))
       
    # Notice that for any case independent of the dataset we need bounding boxes through mmpose.
    if cond_mmpose:        
        vid_p = os.path.join(os.path.dirname(args.vid_path), "rgb_raw.mp4")
        
        if "handedness" in  gt_values.keys():
            if gt_values["handedness"] == 'left':
                vid_p = os.path.join(os.path.dirname(args.vid_path), "rgb_pseudo_raw.mp4")    
         
        MMPOSEJson().main(video_path=vid_p, out_folder_path=get_vid_out_path('mmpose'), 
                            json_folder=mmpose_keypoints2d_path, 
                            gt_dict=gt_values)

    if args.dataname == "in_the_wild":
        bbox_pickles = sorted(glob.glob(mmpose_keypoints2d_path + "/*.pkl"))
        gt_values["bbox"] = []
        
        for i, mmpose_bbox_path in enumerate(bbox_pickles):
            bbox_temp = joblib.load(mmpose_bbox_path)
        
            if not np.equal(bbox_temp, np.array([0, 0, gt_values["width"], gt_values["height"], 0.0])).all():
                gt_values["bbox"].append(bbox_temp)
                gt_values["frame_id"].append(i)
            else:
                pass 

     
    global init_out_obj; init_out_obj = dict()
  
    # take initialization from either PyMaF-X or MeTro
    init_out_obj = eval(f"run_{init_method_name}")(args.vid_path, init_method_out_path, 
                                    joints2d_image_path=init_method_vid_out_path, 
                                    jts_vid=init_method_jts_vid_out_name,
                                    gt_bbox=gt_values)
    init_out_obj["video_path"] = args.vid_path
  
    # dont change ordering, we first need to have keypoints. 
    obtain_keypoints(gt_values, keypoint_blend_weight)  


    joints2d_source = eval(args.keypoint_source + "_keypoints2d_path")
    joints2d_vid_out_path = eval(args.keypoint_source + "_vid_out_path")
     
    # invoke the initial estimate method     
    if not os.path.exists(mediapipe_bbox_conf_path):
        MPJson(gt_dict=gt_values, expand_coef=1.4).mp_bbox_conf(img_dir=args.vid_path, out_dir=mediapipe_bbox_conf_path)
 
    mediapipe_bbox_conf = torch.tensor(np.load(mediapipe_bbox_conf_path)['arr_0']) 

    # If the config setting is not raw, then we perform slerp. We need raw settings for comparison. 
    args.raw_config = config_type in ["_pymafx_raw", "_metro_raw"]

    # have both raw and slerp results for comparison
    if (not os.path.exists(init_method_raw_res_path)):
        rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts = \
                slerp_first_stage(init_out_obj, num_frames, bbox_confs=None, methodname=init_method_name, perform_slerp_flag=False)
        joblib.dump([rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts], init_method_raw_res_path)
    
    # Slerped version 
    if (not os.path.exists(init_method_slerp_res_path)):
        rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts = \
                slerp_first_stage(init_out_obj, num_frames, bbox_confs=None, methodname=init_method_name, perform_slerp_flag=True)
        joblib.dump([rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts], init_method_slerp_res_path)

    # Decide which one to use based on the config type
    if args.raw_config:
        print(f"Loading non-slerped version of the init method {init_method_name}.")
        init_method_res_path = init_method_raw_res_path
    else:
        print(f"Loading slerped version of the init method {init_method_name}.")
        init_method_res_path = init_method_slerp_res_path
    
    rhand_orient_padded, rhand_betas_padded, rhand_trans_padded, rhand_pose_padded, rh_verts = \
                        np.load(init_method_res_path, allow_pickle=True)

       
    init_dict = {"path": os.path.join(init_method_out_path, "stage2_results.npz"), 
                "keyp2d_path": joints2d_source,
                "betas": rhand_betas_padded.cpu().detach().numpy(),
                "trans": rhand_trans_padded.cpu().detach().numpy(),
                "root_orient": rhand_orient_padded.cpu().detach().numpy(),
                "poses": rhand_pose_padded.cpu().detach().numpy().reshape(-1, 45),
                "cam_f":init_out_obj["cam_f"],
                "cam_center":cam_center.cpu().detach().numpy(),
                "img_dir":abs_video_path,
                "img_width":width,
                "img_height":height,
                "gt_bbox":np.array(gt_values["bbox"], dtype=object)}

    # save the results, expand the length by padding so that it matches the number of frames in the video	
    np.savez(os.path.join(init_method_out_path, "stage2_results.npz"), **init_dict)
    
    args.pkl_output_dir = os.path.join(args.save_path, "pkls")
    os.makedirs(args.pkl_output_dir, exist_ok=True)
    
    # run stage3 optimization
    os.makedirs(args.save_path, exist_ok=True)
     
    data = get_stage2_res(init_dict, joints2d_path=joints2d_source)
    
    # map from openpose to mano if not gt. GT s are already in that format. 
    if not args.keypoint_source == "gt": 
        data['joints2d'] = data['joints2d'][:, openpose2mano]

    data['save_path'] =  os.path.join(args.save_path, 'pymaf_output.npz') 
    data['config_type'] = config_type
    data['mediapipe_bbox_conf'] = mediapipe_bbox_conf
    args.orig_seq_len = data['trans'].shape[0]
     
    shutil.copy2(config_f, args.save_path)
    
    for k, v in data.items():
        if k in IGNORE_KEYS:
            continue        
        else:
            if v.shape[0] > 128:
                
                # in case of batch optimization, we need to split the data into chunks of 128 frames
                if args.overlap_len > 0:
                    # compute start and end indices
                    seq_intervals = compute_seq_intervals(v.shape[0], 128, args.overlap_len)
                    data_split = []
                    for seq_s, seq_e in seq_intervals:
                        data_split.append(v[seq_s:seq_e])
                else:
                    data_split = list(torch.split(v, 128))
                    
                if data_split[-1].shape[0] == 128:
                    data[k] = torch.stack(data_split, dim=0)
                else:
                    pad_repeat = 128 - data_split[-1].shape[0]
                    last_el = data_split[-1]
                    last_el = torch.cat([last_el, last_el[-1:].repeat_interleave(pad_repeat, 0)])
                    data_split[-1] = last_el
                    data[k] = torch.stack(data_split, dim=0)
            else:
                pad_repeat = 128 - v.shape[0]
                data[k] = torch.cat([v, v[-1:].repeat_interleave(pad_repeat, 0)]).unsqueeze(0) # BxTxJxD

    args.data.clip_length = data['pos'].shape[1]
    model.set_input(data)
    
    target = dict()
    target['pos'] = data['pos'].to(model.device)
    target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
    target['trans'] = data['trans'].to(model.device)  
    target['root_orient'] = data['root_orient'].to(model.device)
    target['cam_R'] = data['cam_R'].to(model.device) 
    target['cam_t'] = data['cam_t'].to(model.device)
    target['cam_f'] = data['cam_f'].to(model.device).squeeze(0)
    target['cam_center'] = data['cam_center'].to(model.device).squeeze(0)
    target['joints2d'] = data['joints2d'].to(model.device)
    target['joints3d'] = data['joints3d'].to(model.device)
    target['betas'] = data['betas'].to(model.device)
    target['img_width'] = data['img_width']
    target['img_height'] = data['img_height']
    target['img_dir'] = data['img_dir']
    target['frame_id'] = gt_frame_id
    target['save_path'] = data['save_path']
    target['rh_verts'] = rh_verts   # for debug purposes
    target['config_type'] = data['config_type']
    target['handedness'] = gt_values['handedness']
    target['mediapipe_bbox_conf'] = data['mediapipe_bbox_conf']
    
    motion_reconstruction(target, args.save_path, steps=[1.0], offset=0)
        
    # mediapipe mv has 4 views, ineligible for visualization (will change it later)
    if args.keypoint_source == "mediapipe_multiview":
        vis_vid_name = mediapipe_vid_out_path
    else:
        vis_vid_name = eval(f"{args.keypoint_source}_vid_out_path") 
    
    # call evaluation code 
    if gt_values["poses"]:
        pred_path = os.path.join(args.save_path, f'recon_000_30fps.npz')        

        # read pymafx values from _slerp_encode_decode. Pymafx there is raw pymafx with slerped.
        pymafx_path = os.path.join(args.save_path, f'recon_000_gt.npz') if config_type == "_slerp_encode_decode" else None
            
        logger.info("Running quantitative evaluation")
        run_quantitative_evaluation(pred_npz_path=pred_path, 
                                    gt_dict=gt_values,
                                    pymafx_npz_path=pymafx_path,
                                    viz_flag=False, 
                                    misc={})
 
    # assign that according to the initialization method    
    init_phase_npz_path = os.path.join(args.save_path, f'recon_000_30fps.npz')
    
    open3d_viz_flag, scenepic_viz_flag = True, False 
    pymafx_pred_path = args.save_path.replace(config_type, "_pymafx_raw")
    gt_npz_filepath = f"{pymafx_pred_path}/recon_000_30fps.npz"
    
    # vis_vid_name = vid_path  
    vis_vid_name = joints2d_vid_out_path


      
    # visualize the results
    if open3d_viz_flag:
        print("Visualizing optimization results") 
        open3d_viz_overlay.vis_opt_results(pred_file_path=f"{args.save_path}/recon_000_30fps.npz", 
                                 gt_file_path=gt_npz_filepath, 
                                  img_dir=vis_vid_name,
                                  flip_flag = gt_values["handedness"] == "left")
    if scenepic_viz_flag: 
        scenepic_viz.vis_opt_results(pred_file_path=f"{args.save_path}/recon_000_30fps.npz", 
                                 gt_file_path=gt_npz_filepath, 
                                 img_dir=vis_vid_name)

    return 



def run_quantitative_evaluation(pred_npz_path, gt_dict, pymafx_npz_path, viz_flag, misc={}):

    from eval import alignment

    out_dir = "/".join(pred_npz_path.split("/")[:-1])
    evaluator_object = alignment.Evaluator(out_dir)
 
    # load predictions 
    pred_dict = dict(np.load(pred_npz_path))
    
    # change keynames in gt_dict for quant evaluation
    gt_dict["joints_3d"] = torch.tensor(np.array(gt_dict["joints_3d"]))
    gt_dict["joints_2d"] = torch.tensor(np.array(gt_dict["joints_2d"]))
    

    pred_dict, gt_dict = process_gt(pred_dict, gt_dict, args.dataname)
    
    if pymafx_npz_path is not None:
        pymafx_dict = dict(np.load(pymafx_npz_path, allow_pickle=True))
        pymafx_dict, gt_dict = process_gt(pymafx_dict, gt_dict, args.dataname)
    else:
        pymafx_dict = None
        
    
    # MPJPE
    no_align = alignment.PointError(alignment_object=alignment.NoAlignment(), return_aligned=True)
    procrustes_align = alignment.PointError(alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    root_align = alignment.PointError(alignment_object=alignment.RootAlignment(), return_aligned=True)
    
    # ACCELERATION
    no_align_accel = alignment.AccelError(alignment_object=alignment.NoAlignment(), return_aligned=True)
    root_align_accel = alignment.AccelError(alignment_object=alignment.RootAlignment(), return_aligned=True)
    procrustes_align_accel = alignment.AccelError(alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    
    # F-SCORE
    f_thresholds = np.array([5/1000, 15/1000])
    root_align_f = alignment.FScores(thresholds=f_thresholds, alignment_object=alignment.RootAlignment(), return_aligned=True)
    procrustes_align_f = alignment.FScores(thresholds=f_thresholds, alignment_object=alignment.ProcrustesAlignment(), return_aligned=True)
    
    # scale alignment and procrustes alignment are the same. Only input shapes are different. SO, there is no need to use. 
    align_dict_3d = {"no_align": no_align, "root_align": root_align, "procrustes_align": procrustes_align}  
    align_dict_fscore = {"root_align": root_align_f, "procrustes_align": procrustes_align_f}  
    align_dict_2d = {"no_align": no_align}

    align_dict_accel_score = {"no_align": no_align_accel, "root_align": root_align_accel, "procrustes_align": procrustes_align_accel}
 
    metrics = {"mpjpe_3d": align_dict_3d, "mpjpe_2d": align_dict_2d, "acc_err": align_dict_accel_score, "f_score": align_dict_fscore}

    save_quantitative_evaluation(evaluator_object, pred_dict, pymafx_dict=pymafx_dict, gt_dict=gt_dict, metrics=metrics, viz_flag=viz_flag, misc=misc)
    
    return  


def latent_optimization(target, T=None, z_l=None, z_g=None, pose=None):

    if T is None:
        T = torch.arange(args.data.clip_length)
 
    cam_R = target['cam_R'].clone()
    cam_t = target['cam_t'].clone()
  
    optim_trans = target['trans'].clone()
    optim_root_orient = target["root_orient"].clone()
    
    mp_bbox_conf = target["mediapipe_bbox_conf"]

    B, seqlen, _ = optim_trans.shape
    optim_trans.requires_grad = True 
    optim_root_orient.requires_grad = True
    
    if not pose is None:
        optim_pose = pose.clone()
        optim_pose.requires_grad = False
    else:
        optim_pose = pose

    z_global = torch.zeros_like(z_g).to(z_g)
    z_global.requires_grad = False
    
    init_z_l = z_l.clone().detach()
    init_z_l.requires_grad = False
  
    # torch.autograd.set_detect_anomaly(True)
    init_cam_orient = torch.eye(3).to(cam_R)
    init_cam_scale = torch.ones(1).to(cam_t) * 2.5
    
    init_cam_orient = matrix_to_rotation_6d(init_cam_orient) 
        
    full_cam_R = matrix_to_rotation_6d(cam_R.clone())
    full_cam_t = torch.zeros_like(cam_t)
    
    # take pymafx mean as starting point
    mean_betas = target["betas"].mean(dim=1).mean(dim=0)

    
    if args.opt_betas:
        betas = Variable(mean_betas.clone().unsqueeze(0).repeat_interleave(args.nsubject, 0), requires_grad=True)
        logger.info(f'Optimizing betas: {betas}')
    else:
        betas = mean_betas.clone().unsqueeze(0).repeat_interleave(args.nsubject, 0)
        betas.requires_grad = False
    
    stg_configs = [args.stg1]

    if hasattr(args, 'stg2'):
        stg_configs.append(args.stg2)
    
    if hasattr(args, 'stg3'):
        stg_configs.append(args.stg3)
    
    if hasattr(args, 'stg4'):
        stg_configs.append(args.stg4)
        
    if hasattr(args, 'stg5'):
        stg_configs.append(args.stg5)

    stg_int_results = (init_cam_orient, init_cam_scale, optim_trans, optim_root_orient, z_l, z_global, betas, 
                       target, B, seqlen, mean_betas, T, full_cam_R, full_cam_t)
    
    joblib.dump(stg_int_results, f'{args.pkl_output_dir}/stg_0.pkl')
    logger.info(f'Saved intermediate results to {args.pkl_output_dir}/stg_0.pkl')
            
    is_nan_loss = False
    iter = 0
    
    # iterate over different optimization steps 
    while iter < len(stg_configs):
        stg_conf = stg_configs[iter]
        logger.info(f'Stage {iter+1}: Learning rate: {stg_conf.lr}')
        stg_id = iter
        
        # break is better than continue here
        if stg_conf.niters == 0 and iter!=0:
            break    
        # this corresponds to encode-decode stage, we need to calculate joints2d, joints3d etc. 
        elif stg_conf.niters == 0 and iter == 0:
            logger.info('Encode-Decode case')
            # cannot plot loss this case 
            args.plot_loss = False
            break 
        
        if is_nan_loss:
            # will give error here
            prev_stg_results = joblib.load(f'{args.pkl_output_dir}/stg_{stg_id}.pkl')       
            _, _, optim_trans, optim_root_orient, z_l, z_global, betas, target, B, seqlen, mean_betas, T, \
            full_cam_R, full_cam_t = prev_stg_results
                
 
        stg_results = optim_step(stg_conf, stg_id, init_cam_scale, z_l, z_global, betas, target,
                                 B, seqlen, optim_trans, optim_root_orient, init_z_l, mean_betas,
                                 T, full_cam_R, full_cam_t, bbox_conf=mp_bbox_conf.to("cuda"), pose=optim_pose)
        
        if isinstance(stg_results, int):
            is_nan_loss = True
            logger.error(f'[Stage {stg_id+1}] NaN loss detected, restarting stage {stg_id+1}')
            logger.warning(f'Decreasing learning rate by 0.5 for the current stage')
            stg_configs[stg_id].lr *= 0.5
          
        else:
            z_l, z_global, optim_cam_R, optim_cam_t, optim_trans, optim_root_orient, betas, optim_pose = stg_results
            
             
            full_cam_R = matrix_to_rotation_6d(optim_cam_R.detach())
            full_cam_t = optim_cam_t.detach()
            
            stg_int_results = (init_cam_orient, init_cam_scale, optim_trans, optim_root_orient,
                               z_l, z_global, betas, target, B, seqlen,
                               mean_betas, T, full_cam_R, full_cam_t)
            
            joblib.dump(stg_int_results, f'{args.pkl_output_dir}/stg_{stg_id+1}.pkl')
            logger.info(f'Saved intermediate results to {args.pkl_output_dir}/stg_{stg_id+1}.pkl')
            
            iter += 1

    if args.plot_loss:
        plot_list = []
        
        for num in range(iter):  
            loss_i = joblib.load(f'{args.pkl_output_dir}/stage_{num}_loss.pkl')
            plt.figure()
            
            for k, v in loss_i.items():
                if not v == []:
                    plt.plot(v, label=k)
            plt.legend()
            plt.title(f'Stage {num}')
            plt.savefig(f'{args.pkl_output_dir}/stage_{num}_loss.jpg')
        
            # concatenate all the losses
            plot_list.append(cv2.imread(f'{args.pkl_output_dir}/stage_0_loss.jpg'))
        
        plt_concat = np.concatenate(plot_list, axis=0)
        cv2.imwrite(f'{args.pkl_output_dir}/all_stages_loss.jpg', plt_concat)

    return z_l, z_global, betas, optim_root_orient, optim_trans, full_cam_R, full_cam_t, optim_pose


def optim_step(stg_conf, stg_id, init_cam_scale, z_l, z_g, betas, target, B,
               seqlen, trans, root_orient, init_z_l, mean_betas, T, full_cam_R, full_cam_t, bbox_conf=None, pose=None):
    
    logger.info(f'Running optimization stage {stg_id+1} ...')

    opt_params = []
    for param in stg_conf.opt_params:
        if param == 'root_orient':
            opt_params.append(root_orient)
        elif param == 'trans':
            opt_params.append(trans)
        elif param == 'z_l':
            opt_params.append(z_l)
        elif param == 'pose':
            opt_params.append(pose)
        elif param == 'betas':
            if betas is None:
                logger.error('Cannot optimize betas if args.opt_betas is False')
            opt_params.append(betas)
        else:
            raise ValueError(f'Unknown parameter {param}')
    
    for param in opt_params:
        param.requires_grad = True
        
    optimizer = torch.optim.Adam(opt_params, lr=stg_conf.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)
    
    def mask_data(data, mask):
        ml = len(mask.shape)
        dl = len(data.shape)
        for _ in range(dl-ml):
            mask = mask[..., None]
        return data * mask
      
    loss_dict_by_step = {"rot": [], "reproj": [], "rot_sm": [], "orient_sm": [], 'betas_prior': [], "j3d_sm": [], "pose_prior": [],
                    "trans_sm": [], "mot_prior": [], "init_z_prior": [], "orient": [], "trans": [], "loss": []} 
    
    # optimize the z_l and root_orient, pos, trans, 2d kp objectives
    start_time = time.time()
    for i in range(stg_conf.niters):
        
        optimizer.zero_grad()
        
        if motion_prior_type == "hmp":
            output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

            for k, v in output.items():
                if torch.isnan(v).any():
                    logger.warning(f'{k} in output is NaN, skipping this stage...')
                    return 0
      
        # instead of latent code, work with pose, it is in global coordinates  
        else:
            output = {"rotmat": axis_angle_to_matrix(pose)}       
        
        
        output['betas'] = betas[:, None, None, :].repeat(1, B, seqlen, 1)
      
        # For batch optimization 
        global_trans = trans.clone().reshape(args.nsubject, B, seqlen, 3) 
        global_trans = global_trans.reshape(B*args.nsubject, seqlen, 3)
        
        output['trans'] = global_trans
        output['root_orient'] = root_orient
        
        rh_mano_out = forward_mano(output) 
        joints3d = rh_mano_out.joints.view(B, seqlen, -1, 3)
        vertices3d = rh_mano_out.vertices.view(B, seqlen, -1, 3)
 
        optim_cam_R = rotation_6d_to_matrix(full_cam_R)
        optim_cam_t = full_cam_t
        
        joints2d_pred = get_joints2d(joints3d_pred=joints3d, 
                                cam_t=optim_cam_t.unsqueeze(0).repeat_interleave(args.nsubject, 0),
                                cam_R=optim_cam_R.unsqueeze(0).repeat_interleave(args.nsubject, 0),
                                cam_f=torch.tensor([5000., 5000.]), 
                                cam_center=target['cam_center'])
        
        output['joints2d'] = joints2d_pred
        output['joints3d'] = joints3d
        
        _bbox_conf_ = None
        # _bbox_conf_[_bbox_conf_<0.6] = 0.0
        
        local_rotmat = fk.global_to_local(output['rotmat'].view(-1, HAND_JOINT_NUM, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(B*args.nsubject, -1, HAND_JOINT_NUM, 3, 3) # (B x T, J, 3, 3)
        
        local_rotmat_gt = target['rotmat'] 
        loss_dict = {}

        if stg_conf.lambda_rot > 0:    
            mano_joint_conf = torch.zeros_like(target['rotmat'][..., :1, 0])
            
            for si in range(16):
                op_conf = [target['joints2d'][:, :, si, 2]]
                max_conf = torch.stack(op_conf, dim=0).max(0).values
                mano_joint_conf[:, :, si] = max_conf.unsqueeze(-1)
            
            # use bbox conf if that is the case 
            if _bbox_conf_ is not None:
                bbox_coef = torch.repeat_interleave(_bbox_conf_[..., None], dim=2, repeats=15) 
                
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=bbox_coef)
            else:
                rot_loss = L_rot(local_rotmat[:, :, 1:], 
                                local_rotmat_gt[:, :, 1:], 
                                T, conf=mano_joint_conf[:, :, 1:])
            
            loss_dict['rot'] = stg_conf.lambda_rot * rot_loss
   
        if stg_conf.lambda_reproj > 0:
            reproj_loss = joints2d_loss(joints2d_obs=target['joints2d'], joints2d_pred=joints2d_pred, bbox_conf=_bbox_conf_) 
            loss_dict['reproj'] = stg_conf.lambda_reproj * reproj_loss
            
        if stg_conf.lambda_orient > 0:
            orient_loss = L_orient(output['root_orient'], target['root_orient'], T, bbox_conf=_bbox_conf_)
            loss_dict['orient'] = stg_conf.lambda_orient * orient_loss     
        
        if stg_conf.lambda_trans > 0:
            trans_loss = L_trans(output['trans'], target['trans'], T, bbox_conf=_bbox_conf_)
            loss_dict['trans'] = stg_conf.lambda_trans * trans_loss

        if stg_conf.lambda_rot_smooth > 0:
            rot_smooth_l = rot_smooth_loss(local_rotmat)
            loss_dict['rot_sm'] = stg_conf.lambda_rot_smooth * rot_smooth_l   

        if stg_conf.lambda_orient_smooth > 0: 
            matrot_root_orient = rotation_6d_to_matrix(root_orient)            
            orient_smooth_l = rot_smooth_loss(matrot_root_orient)
            loss_dict['orient_sm'] = stg_conf.lambda_orient_smooth * orient_smooth_l
                
        # Smoothness objectives
        if stg_conf.lambda_j3d_smooth > 0:
            joints3d = output['joints3d']

            j3d_smooth_l = pos_smooth_loss(joints3d)
            loss_dict['j3d_sm'] = stg_conf.lambda_j3d_smooth * j3d_smooth_l   
        
        if stg_conf.lambda_trans_smooth > 0:
            # tr = mask_data(output['trans'], mask)
            tr = output['trans']
            tr = tr.reshape(args.nsubject, B, seqlen, 3)
            trans_smooth_l = 0
            for sid in range(args.nsubject):
                trans_smooth_l += pos_smooth_loss(tr[sid])
            loss_dict['trans_sm'] = stg_conf.lambda_trans_smooth * trans_smooth_l
        
        if stg_conf.lambda_motion_prior > 0:
            
            if motion_prior_type == "pca":
                mp_local_loss = L_PCA(pose)
            elif motion_prior_type == "gmm":
                mp_local_loss = L_GMM(pose)                 
            else:    
                mp_local_loss = motion_prior_loss(z_l)
            loss_dict['mot_prior'] = stg_conf.lambda_motion_prior * mp_local_loss
            
            
        if stg_conf.lambda_init_z_prior > 0:
            zl_init_prior_l = F.mse_loss(z_l, init_z_l)
            loss_dict['init_z_prior'] = stg_conf.lambda_init_z_prior * (zl_init_prior_l)

        if stg_conf.lambda_pose_prior > 0 and opt.use_hposer:
            loss_dict['pose_prior'] = stg_conf.lambda_pose_prior * L_pose_prior(output)
        
        if hasattr(stg_conf, 'lambda_batch_cs'):
            if stg_conf.lambda_batch_cs > 0:
                if args.overlap_len == 0:
                    logger.warning('Batch consistency won\'t be effective since overlap_len is 0')
                if B > 1:
                    # joints3d = mask_data(output['joints3d'], mask)
                    joints3d = joints3d.reshape(args.nsubject, B, seqlen, -1, 3)
                    batch_cs_l = 0
                    for sid in range(args.nsubject):
                        batch_cs_l += L_pos(joints3d[sid, :-1, -args.overlap_len:], joints3d[sid, 1:, :args.overlap_len], T)
                    loss_dict['batch_cs'] = stg_conf.lambda_batch_cs * batch_cs_l
                else:
                    if i < 5:
                        logger.warning('Batch consistency won\'t be effective since batch size is 1')
                
        if hasattr(stg_conf, 'betas_prior'):
            if stg_conf.betas_prior > 0:
                if betas is None:
                    logger.error('Cannot compute betas prior since args.opt_betas is False')
                betas_prior_l = torch.pow(betas - mean_betas, 2).mean()
                loss_dict['betas_prior'] = stg_conf.betas_prior * betas_prior_l
            
      
        
        loss = sum(loss_dict.values())
        loss_dict['loss'] = loss
        
        # copy loss values to loss_dict_by_step
        for k, v in loss_dict.items():
            loss_dict_by_step[k].append(v.detach().item()) 

        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 5.0)
            optimizer.step()
        else:
            logger.warning('Loss is NaN, skipping this stage')
            return 0

        scheduler.step()
        loss_log_str = f'Stage {stg_id+1} [{i:03d}/{stg_conf.niters}]'
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
            loss_log_str += f'{k}: {v.item():.3f}\t'
        logger.info(loss_log_str)
        
    end_time = time.time()
    
    # save the loss dict. 
    joblib.dump(loss_dict_by_step, open(os.path.join(args.pkl_output_dir, f'stage_{stg_id}_loss.pkl'), 'wb'))
    
    print(f'Stage {stg_id+1} finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    
    if not betas is None:
        logger.info(f'mean_betas: {mean_betas.detach().cpu().numpy()}')
        logger.info(f'betas: {betas.detach().cpu().numpy()}')
    
    return z_l, z_g, optim_cam_R, optim_cam_t, trans, root_orient, betas, pose

def run_opt(opt):

    global model, fk, ngpu, hposer, motion_prior_type, pca_aa, gmm_aa 
    init_method = args.init_method if hasattr(args, 'init_method') else "pymafx"
    assert init_method in ["metro", "pymafx"]
    
    if hasattr(args, 'motion_prior_type'):
        motion_prior_type = args.motion_prior_type 
        
        if motion_prior_type == "pca":
            pca_aa = joblib.load("data/pca.pkl") 
            pca_mean = torch.tensor(pca_aa.mean_).to("cuda")
            pca_cov = torch.tensor(pca_aa.get_covariance()).to("cuda")
            
            pca_sv = torch.tensor(pca_aa.singular_values_).to("cuda")
            pca_pc = torch.tensor(pca_aa.components_).to("cuda")
        else:
            gmm_aa = MaxMixturePrior()
    else:
        motion_prior_type = "hmp"


    assert motion_prior_type in ["hmp", "pca", "gmm"]

    # load hposer 
    if opt.use_hposer:
        hposer, _ = load_hposer()
        hposer.to("cuda")
        hposer.eval()
    else:
        hposer = None

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    ngpu = 1
    
    model = Architecture(args, ngpu)
    model.load(optimal=True)
    model.eval()

    fk = ForwardKinematicsLayer(args)
     
    multi_stage_opt(os.path.join('./configs', opt.config), opt.dataname, init_method)
    
def main(exp_name, vid_path, config, save_path=None, misc={}):

    global args
    args = Arguments('./configs', filename=config)  
    args.save_path = save_path  
    args.plot_loss = True 
    args.vid_path = vid_path
    
    opt = Namespace(config=config, exp_name=exp_name, vid_path=vid_path)

    run_opt(opt)


if __name__ == '__main__':
    
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--vid-path', type=str, required=True, help='path to the video frames')
    parser.add_argument('--config', type=str, default='in_the_wild_sample_config.yaml', help='name of the configutation file')
    parser.add_argument('--init-method', type=str, default="pymafx", help='Which method to use for initialization. pymafx or metro.')

    opt = parser.parse_args()
    args = Arguments('./configs', filename=opt.config)
    
    # python src/fitting_app.py --vid-path ./data/rgb_data/in_the_wild/cand_20/rgb --config in_the_wild_sample_config.yaml
    if "HO3D_v3" in opt.vid_path:    
        opt.dataname = "HO3D_v3"
    elif "DexYCB" in opt.vid_path:
        opt.dataname = "DexYCB"
    else:
        opt.dataname = "in_the_wild"
         
    args.dataname = opt.dataname
    cfg_name = opt.config.split(".")[0]
    opt.use_hposer = False
    
    args.save_path = opt.save_path = os.path.join("./optim", cfg_name, opt.dataname, opt.vid_path.split("/")[-2])   
    args.plot_loss = True
    
    if os.path.splitext(opt.vid_path)[-1] == ".mp4":

        # vid_frame_num = cv2.VideoCapture(opt.vid_path).get(cv2.CAP_PROP_FRAME_COUNT)

        rgb_frames_path = os.path.join(os.path.dirname(opt.vid_path), "rgb")
        
        # create a folder and save the frames there
        os.makedirs(rgb_frames_path, exist_ok=True)

        # vid_frame_num = int(ffmpeg.probe(opt.vid_path, v='error')['streams'][0]['nb_frames'])
        subprocess.run(f"ffmpeg -y -i {opt.vid_path} {rgb_frames_path}/%06d.jpg -r 30 ", shell=True)
 
        # do this if it has not been already converted. 
        # if len(glob.glob(f"{rgb_frames_path}/*.jpg")) != vid_frame_num: 
            # convert to jpg rgb frames with 30 fps
            # pass 

        # refer to raw images folder
        opt.vid_path = rgb_frames_path

    args.vid_path = opt.vid_path
    run_opt(opt)