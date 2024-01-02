import glob
import itertools
import os
import pickle
import random
import subprocess
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from umap import UMAP

from arguments import Arguments
from datasets.amass import AMASS
from nemf.fk import ForwardKinematicsLayer
from nemf.generative import Architecture
from nemf.losses import GeodesicLoss
from rotations import axis_angle_to_matrix, matrix_to_axis_angle, matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_matrix, rotation_6d_to_matrix
from soft_dtw_cuda import SoftDTW
from utils import build_canonical_frame, compute_orient_angle, estimate_angular_velocity, estimate_linear_velocity, export_ply_trajectory, slerp


def L_rot(source, target, T):
    """
    Args:
        source, target: rotation matrices in the shape B x T x J x 3 x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the rotation matrices.
    """
    criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
    criterion_geo = GeodesicLoss()

    if args.geodesic_loss:
        loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
    else:
        loss = criterion_rec(source[:, T], target[:, T])

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


def L_orient(source, target, T):
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
        loss = criterion_geo(source[:, T].view(-1, 3, 3), target[:, T].view(-1, 3, 3))
    else:
        loss = criterion_rec(source[:, T], target[:, T])

    return loss


def L_trans(source, target, T, up=True):
    """
    Args:
        source: predict global translation in the shape B x T x 3 (the origin is (0, 0, height)).
        target: global translation of the root joint in the shape B x T x 3.
        T: temporal masks.

    Returns:
        reconstruction loss evaluated on the global translation.
    """
    criterion_pred = nn.L1Loss() if args.l1_loss else nn.MSELoss()

    origin = target[:, 0]
    trans = source
    trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
    trans_gt = target

    if not up:
        trans = trans[..., model.v_axis]
        trans_gt = trans_gt[..., model.v_axis]

    loss = criterion_pred(trans[:, T], trans_gt[:, T])

    return loss


def latent_optimization(target, T=None, z_l=None, z_g=None):
    """
        Slove the latent optimization problem to minimize the reconstruction loss.
    """
    if T is None:
        T = torch.arange(args.data.clip_length)
    if z_l is None:
        z_l = Variable(torch.randn(target['rotmat'].shape[0], args.function.local_z).to(model.device), requires_grad=True)
    else:
        z_l = Variable(z_l, requires_grad=True)
        pass
    if z_g is None:
        if args.data.root_transform:  # optimize both z_l and z_g
            z_g = Variable(torch.randn(target['rotmat'].shape[0], args.function.global_z).to(model.device), requires_grad=True)
            optimizer = torch.optim.Adam([z_l, z_g], lr=args.learning_rate)
        else:  # optimize z_l ONLY
            optimizer = torch.optim.Adam([z_l], lr=args.learning_rate)
    else:
        z_g = Variable(z_g, requires_grad=True)
        optimizer = torch.optim.Adam([z_l, z_g], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)
    
    start_time = time.time()
    # for i in range(10):
    for i in range(args.iterations):
        optimizer.zero_grad()
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)
        
        # evaluate the objective function
        rot_loss = L_rot(output['rotmat'], target['rotmat'], T)
        pos_loss = L_pos(output['pos'], target['pos'], T)
        orient_loss = L_orient(output['root_orient'], target['root_orient'], T)
        # trans_loss = L_trans(output['trans'], target['trans'], T, up=True)
        trans_loss = L_trans(target['trans'], target['trans'], T, up=True)

        loss = args.lambda_rot * rot_loss + args.lambda_pos * pos_loss # + args.lambda_orient * orient_loss # + args.lambda_trans * trans_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        print('[{:03d}/{}] rot_loss: {:.4f}\t pos_loss: {:.4f}\t orient_loss: {:.4f}\t trans_loss: {:.4f}\t loss: {:.4f}'.format(
            i, args.iterations, rot_loss.item(), pos_loss.item(), orient_loss.item(), trans_loss.item(), loss.item()))

    end_time = time.time()
    print(f'Optimization finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    return z_l, z_g


def interpolation(idx1, sample1, source1, idx2, sample2, source2, output_dir):
    model.set_input(sample1)
    z_l1, _, _ = model.encode_local()
    z_g1, _, _ = model.encode_global()
    z_l1, z_g1 = latent_optimization(source1, T=None, z_l=z_l1, z_g=z_g1)

    model.set_input(sample2)
    z_l2, _, _ = model.encode_local()
    z_g2, _, _ = model.encode_global()
    z_l2, z_g2 = latent_optimization(source2, T=None, z_l=z_l2, z_g=z_g2)

    z_l = [torch.lerp(z_l1, z_l2, i / 11) for i in range(1, 11)]
    z_l = torch.cat(z_l, dim=0)
    if args.data.root_transform:
        z_g = [torch.lerp(z_g1, z_g2, i / 11) for i in range(1, 11)]
        z_g = torch.cat(z_g, dim=0)
    else:
        z_g = None

    with torch.no_grad():
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

    rotmat = output['rotmat']  # (B, T, J, 3, 3)
    b_size, _, n_joints = rotmat.shape[:3]
    local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
    if args.data.root_transform:
        root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
        local_rotmat[:, :, 0] = root_orient

    origin1 = source1['trans'][:, 0]
    origin2 = source2['trans'][:, 0]
    origin = [torch.lerp(origin1, origin2, i / 11) for i in range(1, 11)]
    origin = torch.cat(origin, dim=0)
    trans = output['trans']
    trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)

    for i in range(z_l.shape[0]):
        poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
        poses = poses.reshape((poses.shape[0], -1))  # (T, J x 3)
        poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')

        np.savez(os.path.join(output_dir, f'{idx1}-{idx2}_lerp_{i}.npz'),
                 poses=poses, trans=c2c(trans[i]), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=args.data.fps)


def run_interpolation():
    index = list(range(len(data_loader)))
    random.shuffle(index)
    subset1 = index[:len(index) // 2]
    subset2 = index[len(index) // 2:]

    output_dir = os.path.join(opt.save_path, 'latent_interpolation', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    for pair in zip(subset1, subset2):
        print(pair)
        sample1 = next(itertools.islice(data_loader, pair[0], None))
        sample2 = next(itertools.islice(data_loader, pair[1], None))

        source1 = dict()
        source1['pos'] = sample1['pos'].to(model.device)
        source1['rotmat'] = rotation_6d_to_matrix(sample1['global_xform'].to(model.device))
        source1['trans'] = sample1['trans'].to(model.device)
        source1['root_orient'] = sample1['root_orient'].to(model.device)

        source2 = dict()
        source2['pos'] = sample2['pos'].to(model.device)
        source2['rotmat'] = rotation_6d_to_matrix(sample2['global_xform'].to(model.device))
        source2['trans'] = sample2['trans'].to(model.device)
        source2['root_orient'] = sample2['root_orient'].to(model.device)

        interpolation(pair[0], sample1, source1, pair[1], sample2, source2, output_dir)


def motion_reconstruction(target, output_dir, steps, T=None, offset=0):
    if args.initialization:
        z_l, _, _ = model.encode_local()
        if args.data.root_transform:
            z_g, _, _ = model.encode_global()
        else:
            z_g = None
  
        z_l, z_g = latent_optimization(target, T=T, z_l=z_l, z_g=z_g)
    else:
        z_l, z_g = latent_optimization(target, T=T)
    for step in steps:
        with torch.no_grad():
            # output gets none in the last version of pytorch 
            output = model.decode(z_l, z_g, length=args.data.clip_length, step=step)
        fps = int(args.data.fps / step)
        criterion_geo = GeodesicLoss()

        rotmat = output['rotmat']  # (B, T, J, 3, 3)
        rotmat_gt = target['rotmat']  # (B, T, J, 3, 3)

        b_size, _, n_joints = rotmat.shape[:3]

        local_rotmat = fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
        local_rotmat_gt = fk.global_to_local(rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)

        root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
        root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
        if args.data.root_transform:
            local_rotmat[:, :, 0] = root_orient
            local_rotmat_gt[:, :, 0] = root_orient_gt

        pos = output['pos']  # (B, T, J, 3)
        pos_gt = target['pos']  # (B, T, J, 3)

        origin = target['trans'][:, 0]
        # trans = output['trans']
        trans =  target['trans'][:, 0]
        
        trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
        trans_gt = target['trans']  # (B, T, 3)
        

        for i in range(rotmat.shape[0]):
            poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
            poses_gt = c2c(matrix_to_axis_angle(local_rotmat_gt[i]))  # (T, J, 3)

            poses = poses.reshape((poses.shape[0], -1))  # (T, 66)
            poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')
            poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, 66)
            poses_gt = np.pad(poses_gt, [(0, 0), (0, 93)], mode='constant')

            # np.savez(os.path.join(output_dir, f'recon_{offset + i:03d}_{fps}fps.npz'),
            #          poses=poses, trans=c2c(trans[i]), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=fps)
            # if step == 1:
            #     np.savez(os.path.join(output_dir, f'recon_{offset + i:03d}_gt.npz'),
            #              poses=poses_gt, trans=c2c(trans_gt[i]), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=args.data.fps)

        if step == 1:
            orientation_error = criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))
            rotation_error = criterion_geo(local_rotmat[:, :, 1:].reshape(-1, 3, 3), local_rotmat_gt[:, :, 1:].reshape(-1, 3, 3))
            position_error = torch.linalg.norm((pos - pos_gt), dim=-1).mean()
            translation_error = torch.linalg.norm((trans - trans_gt), dim=-1).mean()

    return {
        'rotation': c2c(rotation_error),
        'position': c2c(position_error),
        'orientation': c2c(orientation_error),
        'translation': c2c(translation_error)
    }


def run_motion_reconstruction():
    offset = 0
    output_dir = os.path.join(opt.save_path, 'motion_reconstruction', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)
    errors = dict()
    
    ee = 0 
    
    for data in data_loader:
        
        ee += 1
        # if ee > 10:
        #     break 
         
        model.set_input(data)
        target = dict()
        target['pos'] = data['pos'].to(model.device)
        target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
        target['trans'] = data['trans'].to(model.device)
        target['root_orient'] = data['root_orient'].to(model.device)

        error = motion_reconstruction(target, output_dir, steps=[1.0], offset=offset)

        if not errors:
            errors = {
                'rotation_error': [error['rotation'] * 180.0 / np.pi],
                'position_error': [error['position'] * 100.0],
                'orientation_error': [error['orientation'] * 180.0 / np.pi],
                'translation_error': [error['translation'] * 100.0]
            }
        else:
            errors['rotation_error'].append(error['rotation'] * 180.0 / np.pi)
            errors['position_error'].append(error['position'] * 100.0)
            errors['orientation_error'].append(error['orientation'] * 180.0 / np.pi)
            errors['translation_error'].append(error['translation'] * 100.0)

        offset += ngpu * args.batch_size
        
    errors["mre"] = np.mean(errors["rotation_error"])
    errors["moe"] = np.mean(errors["orientation_error"])
    errors["mpe"] = np.mean(errors["position_error"])

    df = pd.DataFrame.from_dict(errors)
    df.to_csv(os.path.join(output_dir, 'motion_recon.csv'), index=False)


def slerp_initialization(data, T, mask):
    global_rotmat = rotation_6d_to_matrix(data['global_xform'])  # (B, T, J, 3, 3)
    b_size, _, n_joints = global_rotmat.shape[:3]
    local_rotmat = fk.global_to_local(global_rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
    local_quat = matrix_to_quaternion(local_rotmat)  # (B x T, J, 4)
    local_quat = local_quat.view(b_size, -1, n_joints, 4)  # (B, T, J, 4)

    root_orient_rotmat = rotation_6d_to_matrix(data['root_orient'])  # (B, T, 3, 3)
    root_orient = matrix_to_quaternion(root_orient_rotmat).unsqueeze(2)  # (B, T, 1, 4)

    slerp_quat = []
    slerp_orient = []
    lerp_trans = []
    for i in range(local_quat.shape[0]):
        quat, trans = slerp(local_quat[i], data['trans'][i], T, range(args.data.clip_length), mask)
        orient, _ = slerp(root_orient[i], data['trans'][i], T, range(args.data.clip_length), mask)
        slerp_quat.append(quat)
        slerp_orient.append(orient)
        lerp_trans.append(trans)

    slerp_quat = np.stack(slerp_quat, axis=0)
    slerp_orient = np.stack(slerp_orient, axis=0)
    lerp_trans = np.stack(lerp_trans, axis=0)
    slerp_quat = torch.from_numpy(slerp_quat).to(model.device)  # (B, T, J, 4)
    slerp_orient = torch.from_numpy(slerp_orient).squeeze(2).to(model.device)  # (B, T, 4)
    lerp_trans = torch.from_numpy(lerp_trans).to(model.device)  # (B, T, 3)

    slerp_rot = quaternion_to_matrix(slerp_quat)  # (B, T, J, 3, 3)
    slerp_ori = matrix_to_rotation_6d(quaternion_to_matrix(slerp_orient))  # (B, T, 6)

    slerp_pos, slerp_xform = fk(slerp_rot.view(-1, n_joints, 3, 3))
    slerp_pos = slerp_pos.view(b_size, -1, n_joints, 3)  # (B, T, J, 3)
    slerp_xform = slerp_xform[:, :, :3, :3].view(b_size, -1, n_joints, 3, 3)  # (B, T, J, 3, 3)
    slerp_xform = matrix_to_rotation_6d(slerp_xform)  # (B, T, J, 6)

    slerp_angular = estimate_angular_velocity(slerp_rot.clone(), dt=1.0 / args.data.fps)
    slerp_velocity = estimate_linear_velocity(slerp_pos, dt=1.0 / args.data.fps)

    lerp_root_vel = estimate_linear_velocity(lerp_trans, dt=1.0 / args.data.fps)

    return {
        'pos': slerp_pos,
        'velocity': slerp_velocity,
        'global_xform': slerp_xform,
        'angular': slerp_angular,
        'root_orient': slerp_ori,
        'root_vel': lerp_root_vel,
        'trans': lerp_trans
    }


def run_motion_inbetweening():
    output_dir = os.path.join(opt.save_path, 'motion_inbetweening', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    clip_inbetweens = [10, 20, 30, 45, 60, 90]
    keyframe_inbetweens = [5, 10, 15, 20]

    calc_slerp = True

    for interval in clip_inbetweens:
        sub_folder_dir = os.path.join(output_dir, f'clip_{interval}')
        os.makedirs(sub_folder_dir, exist_ok=True)
        errors = dict()
        slerp_errors = dict()
        
        
        end_frame = (args.data.clip_length - interval) // 2
        T = list(range(end_frame)) + list(range(end_frame + interval, args.data.clip_length))
        print(T)

        offset = 0
        ee1 = 0
        for data in data_loader:
            
            ee1 += 1
            # if ee1 > 10:
            #     break
            
            initial = slerp_initialization(data, T, mask=True)
            
            target = dict()
            target['pos'] = data['pos'].to(model.device)
            target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
            target['trans'] = data['trans'].to(model.device)
            target['root_orient'] = data['root_orient'].to(model.device)
            
            ###################################################################
            initial['root_orient'] = target['root_orient'].to(model.device)
            ###################################################################
            
            
            if calc_slerp:
                criterion_geo = GeodesicLoss()
                
                root_orient = rotation_6d_to_matrix(initial['root_orient']).to(torch.float)  # (B, T, 3, 3)
                root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
                
                b_size, _, n_joints = target['rotmat'].shape[:3]
                
                initial['rotmat'] = rotation_6d_to_matrix(initial['global_xform'].to(model.device)).to(torch.float)
                             
                local_rotmat = fk.global_to_local(initial["rotmat"].view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
                local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
                local_rotmat_gt = fk.global_to_local(target['rotmat'].view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
                local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)
                

                orientation_error = criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))
                rotation_error = criterion_geo(local_rotmat[:, :, 1:].reshape(-1, 3, 3), local_rotmat_gt[:, :, 1:].reshape(-1, 3, 3))
                position_error = torch.linalg.norm((initial['pos'] - target['pos']), dim=-1).mean()
                translation_error = torch.linalg.norm((initial['trans'] - target['trans']), dim=-1).mean()
                
                if not slerp_errors:
                    slerp_errors = {'rotation_error': [c2c(rotation_error) * 180.0 / np.pi],
                                    'position_error': [c2c(position_error) * 100.0],
                                    'orientation_error': [c2c(orientation_error) * 180.0 / np.pi],
                                    'translation_error': [c2c(translation_error) * 100.0]}
                else:
                    slerp_errors['rotation_error'].append(c2c(rotation_error) * 180.0 / np.pi)
                    slerp_errors['position_error'].append(c2c(position_error) * 100.0)
                    slerp_errors['orientation_error'].append(c2c(orientation_error) * 180.0 / np.pi)
                    slerp_errors['translation_error'].append(c2c(translation_error) * 100.0)

            
            
            model.set_input(initial)
            error = motion_reconstruction(target, sub_folder_dir, steps=[1], T=T, offset=offset)

            if not errors:
                errors = {
                    'rotation_error': [error['rotation'] * 180.0 / np.pi],
                    'position_error': [error['position'] * 100.0],
                    'orientation_error': [error['orientation'] * 180.0 / np.pi],
                    'translation_error': [error['translation'] * 100.0]
                }
            else:
                errors['rotation_error'].append(error['rotation'] * 180.0 / np.pi)
                errors['position_error'].append(error['position'] * 100.0)
                errors['orientation_error'].append(error['orientation'] * 180.0 / np.pi)
                errors['translation_error'].append(error['translation'] * 100.0)

            offset += ngpu * args.batch_size

         
        errors["slerp_rotation_error"] = slerp_errors["rotation_error"]
        errors["slerp_position_error"] = slerp_errors["position_error"]
        errors["slerp_orientation_error"] = slerp_errors["orientation_error"]
        errors["slerp_translations_error"] = slerp_errors["translation_error"]
        
        errors["slerp_mre"] = np.mean(slerp_errors["rotation_error"])
        errors["slerp_moe"] = np.mean(slerp_errors["orientation_error"])
        errors["slerp_mpe"] = np.mean(slerp_errors["position_error"])
        errors["mre"] = np.mean(errors["rotation_error"])
        errors["moe"] = np.mean(errors["orientation_error"])
        errors["mpe"] = np.mean(errors["position_error"])
        
        df = pd.DataFrame.from_dict(errors)
        df.to_csv(os.path.join(sub_folder_dir, 'motion_recon.csv'), index=False)

    for interval in keyframe_inbetweens:
        sub_folder_dir = os.path.join(output_dir, f'keyframe_{interval}')
        os.makedirs(sub_folder_dir, exist_ok=True)
        errors = dict()
        slerp_errors_ = dict()

        T = list(range(0, args.data.clip_length, interval))
        T.append(args.data.clip_length - 1)
        print(T)

        offset = 0
        ee2 = 0
        
        for data in data_loader:
            
            ee2 +=1 
            # if ee2 > 10:
            #     break
            target = dict()
            initial = slerp_initialization(data, T, mask=True)
            target['pos'] = data['pos'].to(model.device)
            target['rotmat'] = rotation_6d_to_matrix(data['global_xform'].to(model.device))
            target['trans'] = data['trans'].to(model.device)
            target['root_orient'] = data['root_orient'].to(model.device)
            
            
            ###################################################################
            initial['root_orient'] = target['root_orient'].to(model.device)
            ###################################################################
            model.set_input(initial)
            
            if calc_slerp:
                criterion_geo = GeodesicLoss()
                
                root_orient = rotation_6d_to_matrix(initial['root_orient']).to(torch.float)  # (B, T, 3, 3)
                root_orient_gt = rotation_6d_to_matrix(target['root_orient'])  # (B, T, 3, 3)
                
                b_size, _, n_joints = target['rotmat'].shape[:3]
                
                initial['rotmat'] = rotation_6d_to_matrix(initial['global_xform'].to(model.device)).to(torch.float)
                             
                local_rotmat = fk.global_to_local(initial["rotmat"].view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
                local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)
                local_rotmat_gt = fk.global_to_local(target['rotmat'].view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
                local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)
                

                orientation_error = criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))
                rotation_error = criterion_geo(local_rotmat[:, :, 1:].reshape(-1, 3, 3), local_rotmat_gt[:, :, 1:].reshape(-1, 3, 3))
                position_error = torch.linalg.norm((initial['pos'] - target['pos']), dim=-1).mean()
                translation_error = torch.linalg.norm((initial['trans'] - target['trans']), dim=-1).mean()
                
                if not slerp_errors_:
                    slerp_errors_ = {'rotation_error': [c2c(rotation_error) * 180.0 / np.pi],
                                    'position_error': [c2c(position_error) * 100.0],
                                    'orientation_error': [c2c(orientation_error) * 180.0 / np.pi],
                                    'translation_error': [c2c(translation_error) * 100.0]}
                else:
                    slerp_errors_['rotation_error'].append(c2c(rotation_error) * 180.0 / np.pi)
                    slerp_errors_['position_error'].append(c2c(position_error) * 100.0)
                    slerp_errors_['orientation_error'].append(c2c(orientation_error) * 180.0 / np.pi)
                    slerp_errors_['translation_error'].append(c2c(translation_error) * 100.0)


            error = motion_reconstruction(target, sub_folder_dir, steps=[1], T=T, offset=offset)

            if not errors:
                errors = {
                    'rotation_error': [error['rotation'] * 180.0 / np.pi],
                    'position_error': [error['position'] * 100.0],
                    'orientation_error': [error['orientation'] * 180.0 / np.pi],
                    'translation_error': [error['translation'] * 100.0]
                }
            else:
                errors['rotation_error'].append(error['rotation'] * 180.0 / np.pi)
                errors['position_error'].append(error['position'] * 100.0)
                errors['orientation_error'].append(error['orientation'] * 180.0 / np.pi)
                errors['translation_error'].append(error['translation'] * 100.0)

            offset += ngpu * args.batch_size

        errors["slerp_rotation_error"] = slerp_errors_["rotation_error"]
        errors["slerp_position_error"] = slerp_errors_["position_error"]
        errors["slerp_orientation_error"] = slerp_errors_["orientation_error"]
        errors["slerp_translations_error"] = slerp_errors_["translation_error"]
        
        errors["slerp_mre"] = np.mean(slerp_errors_["rotation_error"])
        errors["slerp_moe"] = np.mean(slerp_errors_["orientation_error"])
        errors["slerp_mpe"] = np.mean(slerp_errors_["position_error"])
        errors["mre"] = np.mean(errors["rotation_error"])
        errors["moe"] = np.mean(errors["orientation_error"])
        errors["mpe"] = np.mean(errors["position_error"])

        df = pd.DataFrame.from_dict(errors)
        df.to_csv(os.path.join(sub_folder_dir, 'motion_recon.csv'), index=False)


def load_aist_data(path):
    with open(path, 'rb') as f:
        bdata = pickle.load(f)
        trans = bdata['smpl_trans'][::2]  # downsample to 30 fps
        poses = bdata['smpl_poses'][::2]
        scaling = bdata['smpl_scaling']

    # process the original data
    trans = trans / scaling
    trans[:, 1] -= 1
    poses = poses.reshape(-1, 72)

    trans = torch.from_numpy(np.asarray(trans, np.float32)).cuda()  # global translation (T, 3)
    T = trans.shape[0]
    poses = torch.from_numpy(np.asarray(poses[:, :72], np.float32)).cuda()
    poses = poses.view(T, 24, 3)  # axis-angle (T, J, 3)
    rotmat = axis_angle_to_matrix(poses)  # (T, J, 3, 3)

    # change from y-up to z-up
    change_of_basis = torch.FloatTensor([[[1, 0, 0],
                                        [0, 0, -1],
                                        [0, 1, 0]]]).cuda()
    trans = torch.matmul(change_of_basis, trans.unsqueeze(-1)).squeeze(-1)
    rotmat[:, 0] = torch.matmul(change_of_basis, rotmat[:, 0])

    # convert to local space
    root_orient = rotmat[:, 0].clone()  # (T, 3, 3)
    identity = torch.eye(3).cuda()
    identity = identity.view(1, 3, 3).repeat(T, 1, 1)
    rotmat[:, 0] = identity
    pos, global_xform = fk(rotmat)
    pos = pos.contiguous()  # local joint positions (T, J, 3)
    global_xform = global_xform[:, :, :3, :3]  # global transformation matrix for each joint (T, J, 3, 3)

    pos = pos.unsqueeze(0)
    global_xform = global_xform.unsqueeze(0)
    root_orient = root_orient.unsqueeze(0)
    trans = trans.unsqueeze(0)
    rotmat = rotmat.unsqueeze(0)

    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / args.data.fps)  # angular velocity of all the joints (1, T, 24, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (1, T, J, 3)
    root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (1, T, 3)

    return {
        'pos': pos,
        'velocity': velocity,
        'global_xform': matrix_to_rotation_6d(global_xform),
        'angular': angular,
        'root_orient': matrix_to_rotation_6d(root_orient),
        'root_vel': root_vel,
        'trans': trans,
        'rotmat': rotmat
    }


def process_video(path, output_dir, start_frame, end_frame):
    filename = os.path.splitext(os.path.split(path)[1])[0]

    # downsample videos to 30 fps
    command = ['ffmpeg',
               '-i', path,
               '-filter:v',
               f'fps=fps={args.data.fps}',
               f'{output_dir}/{filename}.mp4']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    # trim videos
    command = ['ffmpeg',
               '-i', f'{output_dir}/{filename}.mp4',
               '-ss', str(start_frame / args.data.fps),
               '-to', str(end_frame / args.data.fps),
               '-c:v', 'libx264',
               f'{output_dir}/{filename}_trim.mp4']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)


def run_aist_dance_inbetweening():
    aist_dataset = './data/aist'
    video_files = glob.glob(os.path.join(aist_dataset, 'video', '*.mp4'))
    random.shuffle(video_files)
    subset1 = video_files[:len(video_files) // 2]
    subset2 = video_files[len(video_files) // 2:]

    output_dir = os.path.join(opt.save_path, 'aist_inbetweening', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    interval_length = [30]
    for (video_file1, video_file2) in list(zip(subset1, subset2)):
        filename1 = os.path.splitext(os.path.split(video_file1)[1])[0]
        filename2 = os.path.splitext(os.path.split(video_file2)[1])[0]
        print(filename1, filename2)
        aist_file1 = glob.glob(os.path.join(aist_dataset, 'motions', f"{filename1.replace('c01', 'cAll')}.pkl"))
        aist_file2 = glob.glob(os.path.join(aist_dataset, 'motions', f"{filename2.replace('c01', 'cAll')}.pkl"))
        print(aist_file1, aist_file2)
        sample1 = load_aist_data(aist_file1[0])
        sample2 = load_aist_data(aist_file2[0])

        # |<- 128 (clip 1) ->| |<- 22 (clip 1) ->| |<- t (inbetween) ->| |<- 106 - t (clip 2) ->| |<- 128 (clip 2) ->|
        data = [None] * 3
        for t in interval_length:
            sub_output_dir = os.path.join(output_dir, f'{filename1}_{filename2}_{t}')
            os.makedirs(sub_output_dir, exist_ok=True)

            process_video(video_file1, sub_output_dir, start_frame=0, end_frame=150)
            process_video(video_file2, sub_output_dir, start_frame=0, end_frame=106 - t + 128)

            data[0] = dict()  # clip 1
            data[1] = dict()  # inbetween
            data[2] = dict()  # clip 2

            for key in sample1.keys():
                data[0][key] = sample1[key][:, :128]
                data[1][key] = torch.cat((sample1[key][:, 128:150], sample2[key][:, :106 - t]), dim=1)
                data[2][key] = sample2[key][:, 106 - t:106 - t + 128]

            z_l_total = []
            z_g_total = []
            for i in range(3):
                target = dict()
                if i == 1:
                    T = list(range(22)) + list(range(22 + t, 128))
                    initial = slerp_initialization(data[i], T, mask=False)
                    model.set_input(initial)
                    target['pos'] = initial['pos'].float().to(model.device)
                    target['rotmat'] = rotation_6d_to_matrix(initial['global_xform'].float().to(model.device))
                    target['trans'] = initial['trans'].float().to(model.device)
                    target['root_orient'] = initial['root_orient'].float().to(model.device)
                else:
                    T = None
                    model.set_input(data[i])
                    target['pos'] = data[i]['pos'].to(model.device)
                    target['rotmat'] = rotation_6d_to_matrix(data[i]['global_xform'].to(model.device))
                    target['trans'] = data[i]['trans'].to(model.device)
                    target['root_orient'] = data[i]['root_orient'].to(model.device)

                z_l, _, _ = model.encode_local()
                if args.data.root_transform:
                    z_g, _, _ = model.encode_global()
                else:
                    z_g = None
                z_l, z_g = latent_optimization(target, T, z_l=z_l, z_g=z_g)

                z_l_total.append(z_l)
                z_g_total.append(z_g)

            z_l_total = torch.cat(z_l_total, dim=0)
            z_g_total = torch.cat(z_g_total, dim=0)

            print(f'z_l: {z_l_total.shape}')
            print(f'z_g: {z_g_total.shape}')

            with torch.no_grad():
                output = model.decode(z_l_total, z_g_total, length=args.data.clip_length, step=1)

            rotmat = output['rotmat'].view(-1, 24, 3, 3)  # (3 x 128, 24, 3, 3)
            local_rotmat = fk.global_to_local(rotmat)  # (3 x 128, 24, 3, 3)
            root_orient = rotation_6d_to_matrix(output['root_orient'].view(-1, 6))  # (3 x 128, 3, 3)
            if args.data.root_transform:
                local_rotmat[:, 0] = root_orient

            origin = torch.cat((data[0]['trans'][:, 0], data[1]['trans'][:, 0], data[2]['trans'][:, 0]), dim=0)  # (3, 3)
            trans = output['trans']  # (3, 128, 3)
            trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
            trans = trans.view(-1, 3)  # (3 x 128, 3)

            poses = c2c(matrix_to_axis_angle(local_rotmat))  # (3 x 128, 24, 3)
            poses = poses.reshape((poses.shape[0], -1))  # (3 x 128, 24 x 3)
            poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')

            np.savez(os.path.join(sub_output_dir, f'{filename1}_{filename2}_{t}.npz'),
                     poses=poses, trans=c2c(trans), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=args.data.fps)


def motion_renavigating(reference, traj, z_l=None):
    """
        Slove the latent optimization problem to minimize the reconstruction and similarity loss.
    """
    if z_l is None:
        z_l = Variable(torch.randn(reference['rotmat'].shape[0], args.local_prior.z_dim).to(model.device), requires_grad=True)
    else:
        z_l = Variable(z_l, requires_grad=True)
    if args.data.root_transform:  # optimize both z_l and z_g
        z_g = Variable(torch.randn(reference['rotmat'].shape[0], args.global_prior.z_dim).to(model.device), requires_grad=True)
        optimizer = torch.optim.Adam([z_l, z_g], lr=args.learning_rate)
    else:  # optimize z_l ONLY
        z_g = None
        optimizer = torch.optim.Adam([z_l], lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.scheduler.step_size, args.scheduler.gamma, verbose=False)
    L_DTW = SoftDTW(use_cuda=False, gamma=0.1)
    T = torch.arange(args.data.clip_length)

    root_orient_ref = rotation_6d_to_matrix(reference['root_orient'])  # (B, T, 3, 3)
    forward_ref = root_orient_ref[:, :, :, 2].clone()
    canonical_frame_ref = build_canonical_frame(forward_ref, up_axis=args.data.up)  # (B, T, 3, 3)
    root_rotation_ref = torch.matmul(torch.linalg.inv(canonical_frame_ref), root_orient_ref)  # (B, T, 3, 3), local -> world -> canonical
    root_rotation_ref = root_rotation_ref.unsqueeze(2)  # (B, T, 1, 3, 3)
    pos_ref = torch.matmul(root_rotation_ref, reference['pos'].unsqueeze(-1)).squeeze(-1)  # (B, T, J, 3)
    pos_ref[..., 'xyz'.index(args.data.up)] += reference['trans'][..., 'xyz'.index(args.data.up)].unsqueeze(-1)  # add root height to joint positions
    angle_ref = compute_orient_angle(root_orient_ref, reference['trans'].clone())  # (B, T)
    angle_ref = angle_ref.unsqueeze(-1)  # (B, T, 1)

    start_time = time.time()
    for i in range(args.iterations):
        optimizer.zero_grad()
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

        # evaluate the objective function
        trans_loss = L_trans(output['trans'], traj['trans'], T, up=False)

        root_orient = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
        forward = root_orient[:, :, :, 2].clone()
        canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
        root_rotation = torch.matmul(torch.linalg.inv(canonical_frame), root_orient)  # (B, T, 3, 3)
        root_rotation = root_rotation.unsqueeze(2)  # (B, T, 1, 3, 3)
        pos = torch.matmul(root_rotation, output['pos'].unsqueeze(-1)).squeeze(-1)  # (B, T, J, 3)
        pos[..., 'xyz'.index(args.data.up)] += output['trans'][..., 'xyz'.index(args.data.up)].unsqueeze(-1)

        angle = compute_orient_angle(root_orient, traj['trans'].clone())  # (B, T)
        angle = angle.unsqueeze(-1)  # (B, T, 1)

        if args.dtw_loss:
            # normalized dtw loss, check: https://github.com/mblondel/soft-dtw/issues/10
            x = pos.reshape(1, T.shape[0], -1)
            y = pos_ref.reshape(1, T.shape[0], -1)
            sim_loss = L_DTW(x, y) - 0.5 * (L_DTW(x, x) + L_DTW(y, y))
            angle_loss = L_DTW(angle, angle_ref) - 0.5 * (L_DTW(angle, angle) + L_DTW(angle_ref, angle_ref))
            loss = args.lambda_trans * trans_loss + args.lambda_dtw * sim_loss.mean() + args.lambda_angle * angle_loss.mean()
        else:
            sim_loss = L_pos(pos, pos_ref, T)
            angle_loss = L_pos(angle, angle_ref, T)
            loss = args.lambda_trans * trans_loss + args.lambda_pos * sim_loss + args.lambda_angle * angle_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        print('[{:03d}/{}] trans_loss: {:.4f}\t sim_loss: {:.4f}\t angle_loss: {:.4f}\t loss: {:.4f}'.format(i,
              args.iterations, trans_loss.item(), sim_loss.item(), angle_loss.item(), loss.item()))

    end_time = time.time()
    print(f'Optimization finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')

    return z_l, z_g


def analytical_trajectory(traj_type, num_points):
    traj = torch.zeros(1, num_points, 3)

    if traj_type == 'line':
        traj[..., 0] = torch.linspace(start=0, end=4, steps=num_points)
    elif traj_type == 'circle':
        theta = torch.linspace(start=0, end=2.0 * np.pi, steps=num_points)
        r1 = 0.5
        r2 = 0.5
        traj[..., 0] = r1 * theta.cos()
        traj[..., 1] = r2 * theta.sin()
    elif traj_type == 'sinusoidal':
        theta = torch.linspace(start=0, end=np.pi, steps=num_points)
        traj[..., 0] = theta
        traj[..., 1] = (2.0 * theta).sin() * 0.5
    elif traj_type == 'two':
        scale_x = 0.05
        scale_y = 0.07
        theta = torch.linspace(start=0, end=np.pi, steps=100)
        traj[:, :theta.shape[0], 0] = 16 * (torch.sin(theta) ** 3) * scale_x
        traj[:, :theta.shape[0], 1] = (13 * torch.cos(theta) - 5 * torch.cos(2 * theta) - 2 * torch.cos(3 * theta) - torch.cos(4 * theta)) * scale_y
        traj[:, theta.shape[0]:, 0] = torch.linspace(start=0, end=16 * scale_x, steps=28)
        traj[:, theta.shape[0]:, 1] = -17 * scale_y
    else:
        raise RuntimeError(f'unsupported trajectory type: {traj_type}')

    return traj.to(model.device)


def evaluate_diversity(data):
        
    index = list(range(data['pos'].shape[0]))
    random.shuffle(index)
    index1 = index[:data['pos'].shape[0] // 2]
    index2 = index[data['pos'].shape[0] // 2:]

    with torch.no_grad():
        model.set_input(data)
        f_l, _, _ = model.encode_local()
        f_g, _, _ = model.encode_global()
        feature = torch.cat((f_l, f_g), dim=-1)

    diversity = 0
    for first_idx, second_idx in zip(index1, index2):
        diversity += torch.dist(feature[first_idx], feature[second_idx])
    diversity /= len(index1)

    print(f'Diversity: {diversity.item():.4f}')

    return diversity.item()


def synthetic_trajectory():
    
    N_motions = 100
    
    z_l = torch.normal(mean=0, std=1, size=(N_motions, args.local_prior.z_dim)).to(model.device)
    z_g = torch.zeros((N_motions, args.global_prior.z_dim)).to(model.device)
    
    with torch.no_grad():
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

    rotmat = output["rotmat"]  # (N, T, J, 3, 3)
    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)
    pos, global_xform = fk(rotmat.view(-1, 16, 3, 3))
    pos = pos.contiguous().view(N_motions, args.data.clip_length, 16, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(N_motions, args.data.clip_length, 16, 4, 4)
   
    global_xform = global_xform[:, :, :, :3, :3]  # global transformation matrix for each joint (N, T, J, 3, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)


    # root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (N, T, 3)

 
    root_rotation = rotmat[:, :, 0]  # (N, T, 3, 3)
    global_pos = torch.matmul(root_rotation.unsqueeze(2), pos.unsqueeze(-1)).squeeze(-1)  # (N, T, J, 3)
    # global_pos = global_pos + trans.unsqueeze(2)


    output = {
        'pos': pos,
        'velocity': velocity,
        'global_xform': matrix_to_rotation_6d(global_xform),
        'angular': angular,
        'root_orient': matrix_to_rotation_6d(root_rotation),
        # 'root_vel': root_vel,
        # 'global_pos': global_pos,
        'rotmat': rotmat,
        # 'trans': trans,
        }

    div = evaluate_diversity(output)
    print(div)
    
    return 


def run_motion_renavigating():
    sample = next(itertools.islice(data_loader, 1521, None))

    reference = dict()
    reference['pos'] = sample['pos'].to(model.device)
    reference['rotmat'] = rotation_6d_to_matrix(sample['global_xform'].to(model.device))
    reference['trans'] = sample['trans'].to(model.device)
    reference['root_orient'] = sample['root_orient'].to(model.device)

    traj = dict()
    traj['trans'] = analytical_trajectory('line', 128)
    # traj['trans'] = analytical_trajectory('circle', 128)
    # traj['trans'] = analytical_trajectory('sinusoidal', 128)

    output_dir = os.path.join(opt.save_path, 'motion_renavi', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    if args.initialization:
        model.set_input(sample)
        z_l, _, _ = model.encode_local()
    else:
        z_l = None

    z_l, z_g = motion_renavigating(reference, traj, z_l)

    with torch.no_grad():
        output = model.decode(z_l, z_g, length=args.data.clip_length, step=1)

    rotmat = output['rotmat'][0]  # (T, J, 3, 3)
    rotmat_ref = reference['rotmat'][0]  # (T, J, 3, 3)
    local_rotmat = fk.global_to_local(rotmat)
    local_rotmat_ref = fk.global_to_local(rotmat_ref)
    if args.data.root_transform:
        root_orient = rotation_6d_to_matrix(output['root_orient'][0])  # (T, 3, 3)
        root_orient_ref = rotation_6d_to_matrix(reference['root_orient'][0])  # (T, 3, 3)
        local_rotmat[:, 0] = root_orient
        local_rotmat_ref[:, 0] = root_orient_ref

    origin = traj['trans'][:, 0]
    trans = output['trans']
    trans[..., model.v_axis] = trans[..., model.v_axis] + origin[..., model.v_axis].unsqueeze(1)
    trans = trans[0]  # (T, 3)
    trans_ref = reference['trans'][0]  # (T, 3)
    trans_gt = traj['trans'][0]  # (T, 3)

    poses = c2c(matrix_to_axis_angle(local_rotmat))  # (T, J, 3)
    poses = poses.reshape((poses.shape[0], -1))  # (T, J x 3)
    poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')

    poses_ref = c2c(matrix_to_axis_angle(local_rotmat_ref))  # (T, J, 3)
    poses_ref = poses_ref.reshape((poses_ref.shape[0], -1))  # (T, J x 3)
    poses_ref = np.pad(poses_ref, [(0, 0), (0, 93)], mode='constant')

    np.savez(os.path.join(output_dir, f'renavi.npz'),
             poses=poses, trans=c2c(trans), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=args.data.fps)

    np.savez(os.path.join(output_dir, f'reference.npz'),
             poses=poses_ref, trans=c2c(trans_ref), betas=np.zeros(10), gender=args.data.gender, mocap_framerate=args.data.fps)

    trans[..., model.up_axis] = 0
    trans_gt[..., model.up_axis] = 0

    export_ply_trajectory(points=trans, color=(0, 255, 255), ply_fname=os.path.join(output_dir, 'renavi_traj.ply'))
    export_ply_trajectory(points=trans_gt, color=(0, 255, 0), ply_fname=os.path.join(output_dir, 'gt_traj.ply'))


def time_translation():
    plt.rcParams['text.usetex'] = True

    data_dir = '/home/chhe/proj_nemf/sample'
    output_dir = os.path.join(opt.save_path, 'time_translation', datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(output_dir, exist_ok=True)

    z_ls = []
    z_gs = []
    seqs = []
    for subject in ['subject_02', 'subject_03', 'subject_04']:
        pose_file = glob.glob(os.path.join(data_dir, subject, '*.pkl'))[0]
        print(pose_file)

        all_motion = load_aist_data(pose_file)
        print(all_motion['rotmat'].shape)

        offset = 10
        seq_num = int((all_motion['rotmat'].shape[1] - args.data.clip_length) // offset) + 1
        seqs.append(seq_num)
        for i in range(seq_num):
            data = dict()
            for key in all_motion.keys():
                if key != 'scaling':
                    data[key] = all_motion[key][:, i * offset:i * offset + args.data.clip_length]
            print(f'sequence: [{i * offset}, {i * offset + args.data.clip_length})')

            model.set_input(data)
            z_l, _, _ = model.encode_local()
            z_g, _, _ = model.encode_global()

            z_ls.append(z_l)
            z_gs.append(z_g)

    z_ls = torch.cat(z_ls, dim=0)
    z_gs = torch.cat(z_gs, dim=0)

    print(f'z_ls: {z_ls.shape}')
    print(f'z_gs: {z_gs.shape}')
    seq = [sum(seqs[:i]) for i in range(len(seqs))]
    print(seq)

    z_l_data = UMAP(n_components=3, random_state=42).fit_transform(c2c(z_ls))
    z_g_data = UMAP(n_components=3, random_state=42).fit_transform(c2c(z_gs))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(z_l_data[seq[0]:seq[1], 0], z_l_data[seq[0]:seq[1], 1], z_l_data[seq[0]:seq[1], 2], linestyle='-', marker='o', c='red')
    ax.plot(z_l_data[seq[1]:seq[2], 0], z_l_data[seq[1]:seq[2], 1], z_l_data[seq[1]:seq[2], 2], linestyle='-', marker='o', c='green')
    ax.plot(z_l_data[seq[2]:, 0], z_l_data[seq[2]:, 1], z_l_data[seq[2]:, 2], linestyle='-', marker='o', c='blue')
    # ax.view_init(0, 120)
    plt.title(r'Latent Distribution of $z_l$')
    plt.draw()
    plt.savefig(os.path.join(output_dir, 'z_l_trans.pdf'))
    plt.cla()
    plt.close()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(z_g_data[seq[0]:seq[1], 0], z_g_data[seq[0]:seq[1], 1], z_g_data[seq[0]:seq[1], 2], linestyle='-', marker='o', c='red')
    ax.plot(z_g_data[seq[1]:seq[2], 0], z_g_data[seq[1]:seq[2], 1], z_g_data[seq[1]:seq[2], 2], linestyle='-', marker='o', c='green')
    ax.plot(z_g_data[seq[2]:, 0], z_g_data[seq[2]:, 1], z_g_data[seq[2]:, 2], linestyle='-', marker='o', c='blue')
    # ax.view_init(0, 30)
    plt.title(r'Latent Distribution of $z_g$')
    plt.draw()
    plt.savefig(os.path.join(output_dir, 'z_g_trans.pdf'))
    plt.cla()
    plt.close()


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='application.yaml', help='name of the configutation file')
    parser.add_argument('--task', type=str, help='name of application tasks', required=True)
    parser.add_argument('--save_path', type=str, help='path to save the generated results', default='./optim')

    opt = parser.parse_args()
    args = Arguments('./configs', filename=opt.config)

    torch.set_default_dtype(torch.float32)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ngpu = 1
    if args.multi_gpu is True:
        ngpu = torch.cuda.device_count()
        if ngpu == 1:
            args.multi_gpu = False
    print(f'Number of GPUs: {ngpu}')

    if opt.task != 'motion_renavigating':
        dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'test'))
        data_loader = DataLoader(dataset, batch_size=ngpu * args.batch_size, num_workers=args.num_workers, shuffle=False,
                                 pin_memory=True, drop_last=True if opt.task == 'latent_interpolation' else False)
    else:
        # dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'valid'))
        dataset = AMASS(dataset_dir=os.path.join(args.d, 'test'))
        data_loader = DataLoader(dataset, batch_size=ngpu * args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
       
    model = Architecture(args, ngpu)
    model.load(optimal=True)
    model.eval()
    fk = ForwardKinematicsLayer(args)

    if opt.task == 'motion_reconstruction':
        print('Running motion reconstruction ...')
        run_motion_reconstruction()
    elif opt.task == 'latent_interpolation':
        print('Running latent interpolation ...')
        run_interpolation()
    elif opt.task == 'motion_inbetweening':
        print('Running motion inbetweening ...')
        run_motion_inbetweening()
    elif opt.task == 'motion_renavigating':
        print('Running motion renavigating ...')
        run_motion_renavigating()
    elif opt.task == 'aist_inbetweening':
        print('Running motion inbetweening for AIST++ data ...')
        run_aist_dance_inbetweening()
    elif opt.task == 'time_translation':
        print('Visualizing the latent space ...')
        time_translation()
        
    elif opt.task == 'synthetic_trajectory':
        synthetic_trajectory()
    else:
        raise NotImplementedError(f'unknown application: {opt.task}')
