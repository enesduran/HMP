import os

import holden.BVH as BVH
import numpy as np
import torch
import torch.nn as nn
from arguments import Arguments
from holden.Animation import Animation
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from rotations import matrix_to_quaternion, matrix_to_rotation_6d, quaternion_to_axis_angle, rotation_6d_to_matrix
from utils import align_joints, compute_trajectory, estimate_angular_velocity, estimate_linear_velocity, normalize

from .base_model import BaseModel
from .fk import ForwardKinematicsLayer
from .global_motion import GlobalMotionPredictor
from .losses import GeodesicLoss
from .neural_motion import NeuralMotionField
from .prior import GlobalEncoder, LocalEncoder
from .skeleton import build_edge_topology


class Architecture(BaseModel):
    def __init__(self, args, ngpu, batch_num=0):
        super(Architecture, self).__init__(args)

        self.batch_num = batch_num
        self.args = args

        self.fk = ForwardKinematicsLayer(args)

        smpl_fname = os.path.join(args.smpl.smpl_body_model, "MANO_RIGHT.pkl")
        smpl_data = np.load(smpl_fname, allow_pickle=True, encoding='latin1')
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)

        self.local_encoder = LocalEncoder(args.local_prior, edges).to(self.device)
        self.global_encoder = GlobalEncoder(args.global_prior).to(self.device)
        self.field = NeuralMotionField(args.nemf).to(self.device)
        if args.multi_gpu is True:
            self.local_encoder = nn.DataParallel(self.local_encoder, device_ids=range(ngpu))
            self.global_encoder = nn.DataParallel(self.global_encoder, device_ids=range(ngpu))
            self.field = nn.DataParallel(self.field, device_ids=range(ngpu))

        self.models = [self.local_encoder, self.field]
        if args.data.root_transform:
            self.models.append(self.global_encoder)

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
        self.mean = torch.load(os.path.join(args.dataset_dir, f'mean-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        print(f'mean: {list(self.mean.keys())}')
        self.std = torch.load(os.path.join(args.dataset_dir, f'std-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        print(f'std: {list(self.std.keys())}')

        if args.pretrained_gmp:
            assert args.output_trans is False, 'already output global translation in this model'
            gmp_args = Arguments('./configs', filename=args.pretrained_gmp)
            self.gmp = GlobalMotionPredictor(gmp_args, ngpu)
            self.gmp.load(optimal=True)
            self.gmp.eval()

        self.criterion_geo = GeodesicLoss().to(self.device)
        if self.is_train:
            parameters = list(self.local_encoder.parameters()) + list(self.field.parameters())
            if args.data.root_transform:
                parameters += list(self.global_encoder.parameters())

            if args.adam_optimizer:
                self.optimizer = torch.optim.Adam(parameters, args.learning_rate, weight_decay=args.weight_decay)
            else:
                self.optimizer = torch.optim.Adamax(parameters, args.learning_rate, weight_decay=args.weight_decay)
            self.optimizers = [self.optimizer]

            self.criterion_rec = nn.L1Loss() if args.l1_loss else nn.MSELoss()
            self.criterion_bce = nn.BCEWithLogitsLoss()

            self.iteration = 0
        else:
            self.fps = args.data.fps
            self.test_index = 0
            if args.bvh_viz:
                self.viz_dir = os.path.join(args.save_dir, 'results', 'bvh')
            else:
                self.viz_dir = os.path.join(args.save_dir, 'results', 'smpl')

    def set_input(self, input):
        self.input_data = {k: v.float().to(self.device) for k, v in input.items() if k in ['pos', 'velocity', 'global_xform', 'angular',
                                                                                           'root_orient', 'root_vel', 'trans', 'contacts']}
        self.input_data['rotmat'] = rotation_6d_to_matrix(self.input_data['global_xform'])

    def reparameterize(self, mu, logvar):
        eps = torch.randn_like(mu)
        std = torch.exp(0.5 * logvar)

        return mu + eps * std

    def encode_local(self):
        b_size, t_length = self.input_data['pos'].shape[:2]


        if 'pos' in self.args.data.normalize:
            pos = normalize(self.input_data['pos'], mean=self.mean['pos'], std=self.std['pos'])
        else:
            pos = self.input_data['pos']
        if 'velocity' in self.args.data.normalize:
            velocity = normalize(self.input_data['velocity'], mean=self.mean['velocity'], std=self.std['velocity'])
        else:
            velocity = self.input_data['velocity']
        if 'global_xform' in self.args.data.normalize:
            global_xform = normalize(self.input_data['global_xform'], mean=self.mean['global_xform'], std=self.std['global_xform'])
        else:
            global_xform = self.input_data['global_xform']
        if 'angular' in self.args.data.normalize:
            angular = normalize(self.input_data['angular'], mean=self.mean['angular'], std=self.std['angular'])
        else:
            angular = self.input_data['angular']

        x = torch.cat((pos, velocity, global_xform, angular), dim=-1)
        x = x.view(b_size, t_length, -1)  # (B, T, J x D)
        x = x.permute(0, 2, 1)  # (B, J x D, T)

        mu, logvar = self.local_encoder(x)  # (B, D)
        if self.args.lambda_kl != 0:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar

    def encode_global(self):
        
        if 'root_orient' in self.args.data.normalize:
            root_orient = normalize(self.input_data['root_orient'], mean=self.mean['root_orient'], std=self.std['root_orient'])
        else:
            root_orient = self.input_data['root_orient']
        if self.args.global_prior.in_channels == 9:
            if 'root_vel' in self.args.data.normalize:
                root_vel = normalize(self.input_data['root_vel'], mean=self.mean['root_vel'], std=self.std['root_vel'])
            else:
                root_vel = self.input_data['root_vel']
            x = torch.cat((root_orient, root_vel), dim=-1)  # (B, T, D)
        elif self.args.global_prior.in_channels == 6:
            x = root_orient
        else:
            raise RuntimeError('unsupported input dimensions for global prior')
        x = x.permute(0, 2, 1)  # (B, D, T)

        mu, logvar = self.global_encoder(x)  # (B, D)
        if self.args.lambda_kl != 0:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu

        return z, mu, logvar

    def forward(self, step=1):
        z_l, mu_l, logvar_l = self.encode_local()  # (B, D)
        if self.args.data.root_transform:
            z_g, mu_g, logvar_g = self.encode_global()  # (B, D)
        else:
            z_g = None

        self.recon_data.clear()
        self.recon_data = self.decode(z_l, z_g, length=self.args.data.clip_length, step=step)

        self.recon_data['mu_l'] = mu_l
        self.recon_data['logvar_l'] = logvar_l
        if self.args.data.root_transform:
            self.recon_data['mu_g'] = mu_g
            self.recon_data['logvar_g'] = logvar_g

    def decode(self, z_l, z_g, length, step=1):
        b_size = z_l.shape[0]
        n_joints = self.args.smpl.joint_num
        n_joints = 16

        # generate f(t; z)
        t = torch.arange(start=0, end=length, step=step).unsqueeze(0)  # (1, T)
        t = t / self.args.data.clip_length * 2 - 1  # map t to [-1, 1]
        t = t.expand(b_size, -1).unsqueeze(-1).to(self.device)  # (B, T, 1)
        local_motion, global_motion = self.field(t, z_l, z_g)  # (B, T, N)

        rot6d_recon = local_motion[:, :, :n_joints * 6].contiguous()  # (B, T, J x 6)
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

        output = dict()
        output['rotmat'] = rotmat_recon
        output['pos'] = pos_recon

        if self.args.data.root_transform:
            root_orient = global_motion[:, :, :6]  # (B, T, 6)
            output['root_orient'] = root_orient
        else:
            output['root_orient'] = rot6d_recon[:, :, 0]  # (B, T, 6)

        dt = 1.0 / self.args.data.fps * step
        if self.args.output_trans:
            root_vel = global_motion[:, :, 6:9]  # (B, T, 3)
            root_height = global_motion[:, :, 9]  # (B, T)

            output['root_vel'] = root_vel
            output['root_height'] = root_height

            contacts = global_motion[:, :, 10:]  # (B, T, 8)
            output['contacts'] = contacts

            origin = torch.zeros(b_size, 3).to(self.device)
            trans = compute_trajectory(root_vel, root_height, origin, dt, up_axis=self.args.data.up)
            output['trans'] = trans
        elif self.args.pretrained_gmp:
            local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)  # (B, T, J, 3, 3)
            if self.args.data.root_transform:
                root_orient_rotmat = rotation_6d_to_matrix(output['root_orient'])  # (B, T, 3, 3)
                local_rotmat[:, :, 0] = root_orient_rotmat
                pos = torch.matmul(root_orient_rotmat.unsqueeze(2), pos_recon.unsqueeze(-1)).squeeze(-1)
            else:
                pos = pos_recon

            gmp_data = dict()
            gmp_data['rot6d'] = matrix_to_rotation_6d(local_rotmat)  # (B, T, J, 6)
            gmp_data['angular'] = estimate_angular_velocity(local_rotmat.clone(), dt=dt)  # (B, T, J, 3)

            gmp_data['pos'] = pos
            gmp_data['velocity'] = estimate_linear_velocity(pos, dt=dt)

            pred_data = self.gmp.predict(gmp_data, dt=dt)

            output |= pred_data  # merge two dicts

        return output

    def kl_scheduler(self):
        """
        Cyclical Annealing Schedule
        """
        if (self.epoch_cnt % self.args.annealing_cycles == 0) and (self.iteration % self.batch_num == 0):
            print('KL annealing restart')
            return 0.01

        return min(1, 0.01 + self.iteration / (self.args.annealing_warmup * self.batch_num))

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

        if self.args.output_trans:
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

            contacts_loss = self.criterion_bce(self.recon_data['contacts'], self.input_data['contacts'])
            self.loss_recorder.add_scalar('contacts_loss', contacts_loss, validation=validation)
            loss += self.args.lambda_contacts * contacts_loss

        if self.args.lambda_kl != 0:
            mu_l = self.recon_data['mu_l']
            logvar_l = self.recon_data['logvar_l']
            local_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar_l - mu_l.pow(2) - logvar_l.exp(), dim=-1))
            self.loss_recorder.add_scalar('local_kl_loss', local_kl_loss, validation=validation)

            mu_g = self.recon_data['mu_g']
            logvar_g = self.recon_data['logvar_g']
            global_kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar_g - mu_g.pow(2) - logvar_g.exp(), dim=-1))
            self.loss_recorder.add_scalar('global_kl_loss', global_kl_loss, validation=validation)
            if not validation:
                loss += self.args.lambda_kl * self.kl_scheduler() * (local_kl_loss + global_kl_loss)

        self.loss_recorder.add_scalar('total_loss', loss, validation=validation)

        if not validation:
            loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward(validation=False)
        self.optimizer.step()

        self.iteration += 1

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

        if 'trans' in self.recon_data.keys():
            origin = self.input_data['trans'][:, 0]
            trans = self.recon_data['trans']
            trans[..., self.v_axis] = trans[..., self.v_axis] + origin[..., self.v_axis].unsqueeze(1)
            trans_gt = self.input_data['trans']  # (B, T, 3)
            translation_error = torch.linalg.norm((trans - trans_gt), dim=-1).mean()
        else:
            translation_error = torch.zeros(1).mean()

        return {
            'rotation': c2c(rotation_error),
            'position': c2c(position_error),
            'orientation': c2c(orientation_error),
            'translation': c2c(translation_error)
        }

    def verbose(self):
        res = {}
        for loss in self.loss_recorder.losses.values():
            res[loss.name] = {'train': loss.current()[0], 'val': loss.current()[1]}

        return res

    def validate(self):
        with torch.no_grad():
            self.forward()
            self.backward(validation=True)

    def save(self, optimal=False):
        if optimal:
            path = os.path.join(self.args.save_dir, 'results', 'model')
        else:
            path = os.path.join(self.model_save_dir, f'{self.epoch_cnt:04d}')

        os.makedirs(path, exist_ok=True)
        local_encoder = self.local_encoder.module if isinstance(self.local_encoder, nn.DataParallel) else self.local_encoder
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        torch.save(local_encoder.state_dict(), os.path.join(path, 'local_encoder.pth'))
        torch.save(nemf.state_dict(), os.path.join(path, 'nemf.pth'))
        if self.args.data.root_transform:
            global_encoder = self.global_encoder.module if isinstance(self.global_encoder, nn.DataParallel) else self.global_encoder
            torch.save(global_encoder.state_dict(), os.path.join(path, 'global_encoder.pth'))
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
        local_encoder = self.local_encoder.module if isinstance(self.local_encoder, nn.DataParallel) else self.local_encoder
        nemf = self.field.module if isinstance(self.field, nn.DataParallel) else self.field
        local_encoder.load_state_dict(torch.load(os.path.join(path, 'local_encoder.pth'), map_location=self.device))
        nemf.load_state_dict(torch.load(os.path.join(path, 'nemf.pth'), map_location=self.device))
        if self.args.data.root_transform:
            global_encoder = self.global_encoder.module if isinstance(self.global_encoder, nn.DataParallel) else self.global_encoder
            global_encoder.load_state_dict(torch.load(os.path.join(path, 'global_encoder.pth'), map_location=self.device))
        if self.is_train:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
            self.loss_recorder.load(path)
            self.iteration = len(self.loss_recorder.losses['total_loss'].loss_step)
            print(f'trained iterations: {self.iteration}')
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

        b_size = self.input_data['pos'].shape[0]
        n_joints = self.input_data['pos'].shape[2]

        rotmat = self.recon_data['rotmat']  # (B, T, J, 3, 3)
        local_rotmat = self.fk.global_to_local(rotmat.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat = local_rotmat.view(b_size, -1, n_joints, 3, 3)  # (B, T, J, 3, 3)
        rotmat_gt = self.input_data['rotmat']  # (B, T, J, 3, 3)
        local_rotmat_gt = self.fk.global_to_local(rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)  # (B, T, J, 3, 3)

        if self.args.data.root_transform:
            root_orient = rotation_6d_to_matrix(self.recon_data['root_orient'])  # (B, T, 3, 3)
            root_orient_gt = rotation_6d_to_matrix(self.input_data['root_orient'])  # (B, T, 3, 3)
            local_rotmat[:, :, 0] = root_orient
            local_rotmat_gt[:, :, 0] = root_orient_gt
        rotations = matrix_to_quaternion(local_rotmat)  # (B, T, J, 4)
        rotations_gt = matrix_to_quaternion(local_rotmat_gt)  # (B, T, J, 4)

        if 'trans' in self.recon_data.keys():
            origin = self.input_data['trans'][:, 0]
            positions = self.recon_data['trans']
            positions[..., self.v_axis] = positions[..., self.v_axis] + origin[..., self.v_axis].unsqueeze(1)
            positions_gt = self.input_data['trans']  # (B, T, 3)
        else:
            positions = torch.zeros_like(self.input_data['trans'])  # (B, T, 3)
            positions_gt = torch.zeros_like(self.input_data['trans'])  # (B, T, 3)

        for i in range(b_size):
            if self.args.bvh_viz:
                rotation = align_joints(rotations[i])
                position = positions[i].unsqueeze(1)  # (T, 1, 3)
                anim = Animation(Quaternions(c2c(rotation)), c2c(position), None, offsets=self.args.smpl.offsets[self.args.data.gender], parents=self.args.smpl.parents)
                BVH.save(os.path.join(self.viz_dir, f'test_{self.test_index + i:03d}_{int(self.fps)}fps.bvh'), anim, names=self.args.smpl.joint_names, frametime=1 / self.fps)

                rotation_gt = align_joints(rotations_gt[i])
                position_gt = positions_gt[i].unsqueeze(1)  # (T, 1, 3)
                anim.rotations = Quaternions(c2c(rotation_gt))
                anim.positions = c2c(position_gt)
                BVH.save(os.path.join(self.viz_dir, f'test_{self.test_index + i:03d}_gt.bvh'), anim, names=self.args.smpl.joint_names, frametime=1 / self.args.data.fps)
            else:
                poses = c2c(quaternion_to_axis_angle(rotations[i]))  # (T, J, 3)
                poses_gt = c2c(quaternion_to_axis_angle(rotations_gt[i]))  # (T, J, 3)

                poses = poses.reshape((poses.shape[0], -1))  # (T, J x 3)
                poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')
                poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, J x 3)
                poses_gt = np.pad(poses_gt, [(0, 0), (0, 93)], mode='constant')

                np.savez(os.path.join(self.viz_dir, f'test_{self.test_index + i:03d}_{int(self.fps)}fps.npz'),
                         poses=poses, trans=c2c(positions[i]), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.fps)
                np.savez(os.path.join(self.viz_dir, f'test_{self.test_index + i:03d}_gt.npz'),
                         poses=poses_gt, trans=c2c(positions_gt[i]), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.args.data.fps)

        self.test_index += b_size
