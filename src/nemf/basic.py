import os

import holden.BVH as BVH
import numpy as np
import torch
import torch.nn as nn
from holden.Animation import Animation
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from rotations import matrix_to_quaternion, quaternion_to_axis_angle, rotation_6d_to_matrix
from utils import align_joints, compute_trajectory

from .base_model import BaseModel
from .fk import ForwardKinematicsLayer
from .losses import GeodesicLoss
from .neural_motion import NeuralMotionField


class Architecture(BaseModel):
    def __init__(self, args, ngpu):
        super(Architecture, self).__init__(args)

        self.args = args

        if args.amass_data:
            self.fk = ForwardKinematicsLayer(args)
        else:
            self.fk = ForwardKinematicsLayer(parents=args.animal.parents, positions=args.animal.offsets)

        self.field = NeuralMotionField(args.nemf).to(self.device)
        if args.multi_gpu is True:
            self.field = nn.DataParallel(self.field, device_ids=range(ngpu))

        self.models = [self.field]

        ordermap = {
            'x': 0,
            'y': 1,
            'z': 2,
        }
        self.up_axis = ordermap[self.args.data.up]
        self.v_axis = [x for x in ordermap.values() if x != self.up_axis]

        print(self.up_axis)
        print(self.v_axis)

        self.input_data = dict()
        self.recon_data = dict()

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.field.parameters(), args.learning_rate)
            self.optimizers = [self.optimizer]
            self.criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()

        self.criterion_geo = GeodesicLoss().to(self.device)
        self.fps = args.data.fps
        if args.bvh_viz:
            self.viz_dir = os.path.join(args.save_dir, 'results', 'bvh')
        else:
            self.viz_dir = os.path.join(args.save_dir, 'results', 'smpl')

    def set_input(self, input):
        self.input_data = {k: v.float().to(self.device) for k, v in input.items() if k in ['pos', 'global_xform', 'root_orient', 'root_vel', 'trans']}
        self.input_data['rotmat'] = rotation_6d_to_matrix(self.input_data['global_xform'])

    def forward(self, step=1):
        b_size, t_length, n_joints = self.input_data['pos'].shape[:3]

        t = torch.arange(start=0, end=t_length, step=step).unsqueeze(0)  # (1, T)
        t = t / t_length * 2 - 1  # map t to [-1, 1]
        t = t.expand(b_size, -1).unsqueeze(-1).to(self.device)  # (B, T, 1)
        f, _ = self.field(t)  # (B, T, D), rot6d + root orient + root vel + root height
        # import ipdb; ipdb.set_trace()
        self.recon_data = dict()
        rot6d_recon = f[:, :, :n_joints * 6]  # (B, T, J x 6)
        rot6d_recon = rot6d_recon.view(b_size, -1, n_joints, 6)  # (B, T, J, 6)
        rotmat_recon = rotation_6d_to_matrix(rot6d_recon)  # (B, T, J, 3, 3)
        if self.args.data.root_transform:
            identity = torch.zeros(b_size, t.shape[1], 3, 3).to(self.device)
            identity[:, :, 0, 0] = 1
            identity[:, :, 1, 1] = 1
            identity[:, :, 2, 2] = 1
            rotmat_recon[:, :, 0] = identity
        # rotmat_recon[:, :, -2] = rotmat_recon[:, :, -4]
        # rotmat_recon[:, :, -1] = rotmat_recon[:, :, -3]
        local_rotmat = self.fk.global_to_local(rotmat_recon.view(-1, n_joints, 3, 3))  # (B x T, J, 3. 3)
        pos_recon, _ = self.fk(local_rotmat)  # (B x T, J, 3)
        pos_recon = pos_recon.contiguous().view(b_size, -1, n_joints, 3)  # (B, T, J, 3)
        self.recon_data['rotmat'] = rotmat_recon
        self.recon_data['pos'] = pos_recon

        if self.args.data.root_transform:
            root_orient = f[:, :, n_joints * 6:n_joints * 6 + 6]  # (B, T, 6)
            self.recon_data['root_orient'] = root_orient
        else:
            self.recon_data['root_orient'] = rot6d_recon[:, :, 0]  # (B, T, 6)

        root_vel = f[:, :, -4:-1]  # (B, T, 3)
        root_height = f[:, :, -1]  # (B, T)

        self.recon_data['root_vel'] = root_vel
        self.recon_data['root_height'] = root_height

        dt = 1.0 / self.args.data.fps * step
        origin = torch.zeros(b_size, 3).to(self.device)
        trans = compute_trajectory(root_vel, root_height, origin, dt, up_axis=self.args.data.up)
        self.recon_data['trans'] = trans

    def backward(self, validation=False):
        loss = 0

        root_orient = rotation_6d_to_matrix(self.recon_data['root_orient'])
        root_orient_gt = rotation_6d_to_matrix(self.input_data['root_orient'])
        if self.args.geodesic_loss:
            orient_recon_loss = self.criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))
        else:
            orient_recon_loss = self.criterion_rec(root_orient, root_orient_gt)
        self.loss_recorder.add_scalar('orient_recon_loss', orient_recon_loss, validation=validation)
        loss += self.args.lambda_orient * orient_recon_loss

        if self.args.geodesic_loss:
            rotmat_recon_loss = self.criterion_geo(self.recon_data['rotmat'].view(-1, 3, 3), self.input_data['rotmat'].view(-1, 3, 3))
        else:
            rotmat_recon_loss = self.criterion_rec(self.recon_data['rotmat'], self.input_data['rotmat'])
        self.loss_recorder.add_scalar('rotmat_recon_loss', rotmat_recon_loss, validation=validation)
        loss += self.args.lambda_rotmat * rotmat_recon_loss

        pos_recon_loss = self.criterion_rec(self.recon_data['pos'], self.input_data['pos'])
        self.loss_recorder.add_scalar('pos_recon_loss', pos_recon_loss, validation=validation)
        loss += self.args.lambda_pos * pos_recon_loss

        v_loss = self.criterion_rec(self.recon_data['root_vel'], self.input_data['root_vel'])
        up_loss = self.criterion_rec(self.recon_data['root_height'], self.input_data['trans'][..., 'xyz'.index(self.args.data.up)])
        self.loss_recorder.add_scalar('v_loss', v_loss, validation=validation)
        self.loss_recorder.add_scalar('up_loss', up_loss, validation=validation)
        loss += self.args.lambda_v * v_loss + self.args.lambda_up * up_loss

        origin = self.input_data['trans'][:, 0]
        trans = self.recon_data['trans']
        trans[..., self.v_axis] = trans[..., self.v_axis] + origin[..., self.v_axis].unsqueeze(1)
        trans_loss = self.criterion_rec(trans, self.input_data['trans'])
        self.loss_recorder.add_scalar('trans_loss', trans_loss, validation=validation)
        loss += self.args.lambda_trans * trans_loss

        self.loss_recorder.add_scalar('total_loss', loss, validation=validation)

        if not validation:
            loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward(validation=False)
        self.optimizer.step()

    def report_errors(self):
        root_orient = rotation_6d_to_matrix(self.recon_data['root_orient'])  # (B, T, 3, 3)
        root_orient_gt = rotation_6d_to_matrix(self.input_data['root_orient'])  # (B, T, 3, 3)
        orientation_error = self.criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))

        local_rotmat = self.fk.global_to_local(self.recon_data['rotmat'].view(-1, self.args.smpl.joint_num, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = self.fk.global_to_local(self.input_data['rotmat'].view(-1, self.args.smpl.joint_num, 3, 3))  # (B x T, J, 3, 3)
        rotation_error = self.criterion_geo(local_rotmat[:, 1:].reshape(-1, 3, 3), local_rotmat_gt[:, 1:].reshape(-1, 3, 3))

        pos = self.recon_data['pos']  # (B, T, J, 3)
        pos_gt = self.input_data['pos']  # (B, T, J, 3)
        position_error = torch.linalg.norm((pos - pos_gt), dim=-1).mean()

        origin = self.input_data['trans'][:, 0]
        trans = self.recon_data['trans']
        trans[..., self.v_axis] = trans[..., self.v_axis] + origin[..., self.v_axis].unsqueeze(1)
        trans_gt = self.input_data['trans']  # (B, T, 3)
        translation_error = torch.linalg.norm((trans - trans_gt), dim=-1).mean()

        return {
            'rotation': c2c(rotation_error),
            'position': c2c(position_error),
            'orientation': c2c(orientation_error),
            'translation': c2c(translation_error)
        }

    def verbose(self):
        res = {}
        for loss in self.loss_recorder.losses.values():
            res[loss.name] = {'train': loss.current()[0]}

        return res

    def save(self, optimal=False):
        if optimal:
            path = os.path.join(self.args.save_dir, 'results', 'model')
        else:
            path = os.path.join(self.model_save_dir, f'{self.epoch_cnt:04d}')

        os.makedirs(path, exist_ok=True)
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        torch.save(nemf.state_dict(), os.path.join(path, 'nemf.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(path, 'optimizer.pth'))
        if self.args.scheduler.name:
            torch.save(self.schedulers[0].state_dict(), os.path.join(path, 'scheduler.pth'))
        self.loss_recorder.save(path)

        print(f'Save at {path} succeeded')

    def load(self, epoch=None, optimal=False):
        if optimal:
            path = os.path.join(self.args.save_dir, 'results', 'model')
        else:
            if epoch is None:
                all = [int(q) for q in os.listdir(self.model_save_dir)]
                if len(all) == 0:
                    raise RuntimeError(f'Empty loading path {self.model_save_dir}')
                epoch = sorted(all)[-1]
            path = os.path.join(self.model_save_dir, f'{epoch:04d}')

        print(f'Loading from {path}')
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        nemf.load_state_dict(torch.load(os.path.join(path, 'nemf.pth'), map_location=self.device))
        if self.is_train:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
            self.loss_recorder.load(path)
            if self.args.scheduler.name:
                self.schedulers[0].load_state_dict(torch.load(os.path.join(path, 'scheduler.pth')))
        self.epoch_cnt = epoch if not optimal else 0

        print('Load succeeded')

    def super_sampling(self, step=1):
        with torch.no_grad():
            self.forward(step)
            self.fps = self.args.data.fps / step
            self.compute_test_result()

    def compute_test_result(self):
        os.makedirs(self.viz_dir, exist_ok=True)

        rotmat = self.recon_data['rotmat'][0]  # (T, J, 3, 3)
        rotmat_gt = self.input_data['rotmat'][0]  # (T, J, 3, 3)
        local_rotmat = self.fk.global_to_local(rotmat)  # (T, J, 3, 3)
        local_rotmat_gt = self.fk.global_to_local(rotmat_gt)  # (T, J, 3, 3)

        if self.args.data.root_transform:
            root_orient = rotation_6d_to_matrix(self.recon_data['root_orient'][0])  # (T, 3, 3)
            root_orient_gt = rotation_6d_to_matrix(self.input_data['root_orient'][0])  # (T, 3, 3)
            local_rotmat[:, 0] = root_orient
            local_rotmat_gt[:, 0] = root_orient_gt
        rotation = matrix_to_quaternion(local_rotmat)  # (T, J, 4)
        rotation_gt = matrix_to_quaternion(local_rotmat_gt)  # (T, J, 4)

        origin = self.input_data['trans'][0, 0].clone()  # [3]
        position = self.recon_data['trans'][0].clone()  # (T, 3)
        position[:, self.v_axis] = position[:, self.v_axis] + origin[self.v_axis].unsqueeze(0)
        position_gt = self.input_data['trans'][0]  # (T, 3)

        if self.args.bvh_viz:
            if self.args.amass_data:
                rotation = align_joints(rotation)
                rotation_gt = align_joints(rotation_gt)

            position = position.unsqueeze(1)  # (T, 1, 3)
            position_gt = position_gt.unsqueeze(1)  # (T, 1, 3)

            # export data to BVH files
            if self.args.amass_data:
                offsets = self.args.smpl.offsets[self.args.data.gender]
                parents = self.args.smpl.parents
                names = self.args.smpl.joint_names
            else:
                offsets = self.args.animal.offsets
                parents = self.args.animal.parents
                names = self.args.animal.joint_names

            anim = Animation(Quaternions(c2c(rotation)), c2c(position), None, offsets=offsets, parents=parents)
            BVH.save(os.path.join(self.viz_dir, f'test_{int(self.fps)}fps.bvh'), anim, names=names, frametime=1 / self.fps)

            anim.rotations = Quaternions(c2c(rotation_gt))
            anim.positions = c2c(position_gt)
            BVH.save(os.path.join(self.viz_dir, 'test_gt.bvh'), anim, names=names, frametime=1 / self.args.data.fps)
        else:
            poses = c2c(quaternion_to_axis_angle(rotation))  # (T, J, 3)
            poses_gt = c2c(quaternion_to_axis_angle(rotation_gt))  # (T, J, 3)

            poses = poses.reshape((poses.shape[0], -1))  # (T, J x 3)
            poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')
            poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, J x 3)
            poses_gt = np.pad(poses_gt, [(0, 0), (0, 93)], mode='constant')

            np.savez(os.path.join(self.viz_dir, f'test_{int(self.fps)}fps.npz'),
                     poses=poses, trans=c2c(position), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.fps)
            np.savez(os.path.join(self.viz_dir, 'test_gt.npz'),
                     poses=poses_gt, trans=c2c(position_gt), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.args.data.fps)
