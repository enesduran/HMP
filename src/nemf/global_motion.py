import os

import numpy as np
import torch
import torch.nn as nn
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from rotations import matrix_to_axis_angle, rotation_6d_to_matrix
from utils import compute_trajectory, export_ply_trajectory, normalize

from .base_model import BaseModel
from .fk import ForwardKinematicsLayer
from .residual_blocks import ResidualBlock, SkeletonResidual, residual_ratio
from .skeleton import SkeletonConv, SkeletonPool, build_edge_topology, find_neighbor


class Predictor(nn.Module):
    def __init__(self, args, topology):
        super(Predictor, self).__init__()
        self.topologies = [topology]
        self.channel_base = [args.channel_base]
        self.channel_list = []
        self.edge_num = [len(topology)]
        self.pooling_list = []
        self.layers = nn.ModuleList()
        self.args = args
        # self.convs = []

        kernel_size = args.kernel_size
        padding = (kernel_size - 1) // 2
        bias = True

        for _ in range(args.num_layers):
            self.channel_base.append(self.channel_base[-1] * 2)

        for i in range(args.num_layers):
            seq = []
            neighbour_list = find_neighbor(self.topologies[i], args.skeleton_dist)
            in_channels = self.channel_base[i] * self.edge_num[i]
            out_channels = self.channel_base[i + 1] * self.edge_num[i]
            if i == 0:
                self.channel_list.append(in_channels)
            self.channel_list.append(out_channels)
            last_pool = True if i == args.num_layers - 1 else False

            # (T, J, D) => (T, J', D)
            pool = SkeletonPool(edges=self.topologies[i], pooling_mode=args.skeleton_pool,
                                channels_per_edge=out_channels // len(neighbour_list), last_pool=last_pool)

            if args.use_residual_blocks:
                # (T, J, D) => (T, J', 2D)
                seq.append(SkeletonResidual(self.topologies[i], neighbour_list, joint_num=self.edge_num[i], in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=1, padding=padding, padding_mode=args.padding_mode, bias=bias,
                                            extra_conv=args.extra_conv, pooling_mode=args.skeleton_pool, activation=args.activation, last_pool=last_pool))
            else:
                for _ in range(args.extra_conv):
                    # (T, J, D) => (T, J, D)
                    seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=in_channels,
                                            joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                            padding=padding, padding_mode=args.padding_mode, bias=bias))
                    seq.append(nn.PReLU())
                # (T, J, D) => (T, J, 2D)
                seq.append(SkeletonConv(neighbour_list, in_channels=in_channels, out_channels=out_channels,
                                        joint_num=self.edge_num[i], kernel_size=kernel_size, stride=1,
                                        padding=padding, padding_mode=args.padding_mode, bias=bias, add_offset=False,
                                        in_offset_channel=3 * self.channel_base[i] // self.channel_base[0]))
                # self.convs.append(seq[-1])

                seq.append(pool)
                seq.append(nn.PReLU())
            self.layers.append(nn.Sequential(*seq))

            self.topologies.append(pool.new_edges)
            self.pooling_list.append(pool.pooling_list)
            self.edge_num.append(len(self.topologies[-1]))

        in_channels = self.channel_base[-1] * len(self.pooling_list[-1])
        out_channels = args.out_channels  # root orient (6) + root vel (3) + root height (1) + contacts (8)

        self.global_motion = nn.Sequential(
            ResidualBlock(in_channels=in_channels, out_channels=512, kernel_size=kernel_size, stride=1, padding=padding, residual_ratio=residual_ratio(1), activation=args.activation),
            ResidualBlock(in_channels=512, out_channels=256, kernel_size=kernel_size, stride=1, padding=padding, residual_ratio=residual_ratio(2), activation=args.activation),
            ResidualBlock(in_channels=256, out_channels=128, kernel_size=kernel_size, stride=1, padding=padding, residual_ratio=residual_ratio(3), activation=args.activation),
            ResidualBlock(in_channels=128, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=padding,
                          residual_ratio=residual_ratio(4), activation=args.activation, last_layer=True)
        )

    def forward(self, input):
        feature = input
        for layer in self.layers:
            feature = layer(feature)

        return self.global_motion(feature)


class GlobalMotionPredictor(BaseModel):
    def __init__(self, args, ngpu):
        super(GlobalMotionPredictor, self).__init__(args)

        self.args = args
        # smpl_fname = os.path.join(args.smpl.smpl_body_model, args.data.gender, 'model.npz')
        # smpl_data = np.load(smpl_fname, encoding='latin1')
        smpl_fname = os.path.join(args.smpl.smpl_body_model, "MANO_RIGHT.pkl")
        smpl_data = np.load(smpl_fname, encoding='latin1', allow_pickle=True)
        parents = smpl_data['kintree_table'][0].astype(np.int32)
        edges = build_edge_topology(parents)

        self.fk = ForwardKinematicsLayer(args)

        self.predictor = Predictor(args.global_motion, edges).to(self.device)
        if args.multi_gpu is True:
            self.predictor = nn.DataParallel(self.predictor, device_ids=range(ngpu))
        self.models = [self.predictor]

        ordermap = {
            'x': 0,
            'y': 1,
            'z': 2,
        }
        self.up_axis = ordermap[self.args.data.up]
        self.v_axis = [x for x in ordermap.values() if x != self.up_axis]

        print(self.up_axis)
        print(self.v_axis)

        self.input_data = {}
        self.pred_data = {}

        self.mean = torch.load(os.path.join(args.dataset_dir, f'mean-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        print(f'mean: {list(self.mean.keys())}')
        self.std = torch.load(os.path.join(args.dataset_dir, f'std-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), map_location=self.device)
        print(f'std: {list(self.std.keys())}')

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.predictor.parameters(), args.learning_rate, weight_decay=args.weight_decay)
            self.optimizers = [self.optimizer]
            self.criterion_pred = nn.L1Loss() if args.l1_loss else nn.MSELoss()
            self.criterion_bce = nn.BCEWithLogitsLoss()
        else:
            self.test_index = 0
            self.smpl_dir = os.path.join(args.save_dir, 'results', 'smpl')

    def set_input(self, input):
        self.input_data = {k: v.to(self.device) for k, v in input.items() if k in ['pos', 'velocity', 'rot6d', 'angular', 'global_xform',
                                                                                   'root_vel', 'trans', 'contacts']}

    def forward(self):
        self.pred_data.clear()
        self.input_data['origin'] = self.input_data['trans'][:, 0]
        self.pred_data = self.predict(data=self.input_data, dt=1.0 / self.args.data.fps)

    def predict(self, data, dt):
        b_size, t_length = data['pos'].shape[:2]

        if 'pos' in self.args.data.normalize:
            pos = normalize(data['pos'], mean=self.mean['pos'], std=self.std['pos'])
        else:
            pos = data['pos']
        if 'velocity' in self.args.data.normalize:
            velocity = normalize(data['velocity'], mean=self.mean['velocity'], std=self.std['velocity'])
        else:
            velocity = data['velocity']
        if 'rot6d' in self.args.data.normalize:
            rot6d = normalize(data['rot6d'], mean=self.mean['rot6d'], std=self.std['rot6d'])
        else:
            rot6d = data['rot6d']
        if 'angular' in self.args.data.normalize:
            angular = normalize(data['angular'], mean=self.mean['angular'], std=self.std['angular'])
        else:
            angular = data['angular']

        x = torch.cat((pos, velocity, rot6d, angular), dim=-1)
        x = x.view(b_size, t_length, -1)  # (B, T, J x D)
        x = x.permute(0, 2, 1)  # (B, J x D, T)

        global_motion = self.predictor(x)
        global_motion = global_motion.permute(0, 2, 1)  # (B, T, D)

        output = dict()
        output['root_vel'] = global_motion[:, :, :3]
        output['root_height'] = global_motion[:, :, 3]
        output['contacts'] = global_motion[:, :, 4:]
        if 'origin' in data.keys():
            origin = data['origin']
        else:
            origin = torch.zeros(b_size, 3).to(self.device)
        output['trans'] = compute_trajectory(output['root_vel'], output['root_height'], origin, dt, up_axis=self.args.data.up)

        return output

    def backward(self, validation=False):
        loss = 0

        v_loss = self.criterion_pred(self.pred_data['root_vel'], self.input_data['root_vel'])
        up_loss = self.criterion_pred(self.pred_data['root_height'], self.input_data['trans'][..., self.up_axis])
        self.loss_recorder.add_scalar('v_loss', v_loss, validation=validation)
        self.loss_recorder.add_scalar('up_loss', up_loss, validation=validation)
        loss += self.args.lambda_v * v_loss + self.args.lambda_up * up_loss

        trans_loss = self.criterion_pred(self.pred_data['trans'], self.input_data['trans'])
        self.loss_recorder.add_scalar('trans_loss', trans_loss, validation=validation)
        loss += self.args.lambda_trans * trans_loss

        contacts_loss = self.criterion_bce(self.pred_data['contacts'], self.input_data['contacts'])
        self.loss_recorder.add_scalar('contacts_loss', contacts_loss, validation=validation)
        loss += self.args.lambda_contacts * contacts_loss

        self.loss_recorder.add_scalar('total_loss', loss, validation=validation)

        if not validation:
            loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.forward()
        self.backward(validation=False)
        self.optimizer.step()

    def report_errors(self):
        trans = self.pred_data['trans']  # (B, T, 3)
        trans_gt = self.input_data['trans']  # (B, T, 3)
        translation_error = torch.linalg.norm((trans - trans_gt), dim=-1).mean()

        return {
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
        predictor = self.predictor.module if isinstance(self.predictor, nn.DataParallel) else self.predictor
        torch.save(predictor.state_dict(), os.path.join(path, 'predictor.pth'))
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
        predictor = self.predictor.module if isinstance(self.predictor, nn.DataParallel) else self.predictor
        predictor.load_state_dict(torch.load(os.path.join(path, 'predictor.pth'), map_location=self.device))
        if self.is_train:
            self.optimizer.load_state_dict(torch.load(os.path.join(path, 'optimizer.pth')))
            self.loss_recorder.load(path)
            if self.args.scheduler.name:
                self.schedulers[0].load_state_dict(torch.load(os.path.join(path, 'scheduler.pth')))
        self.epoch_cnt = epoch if not optimal else 0

        print('Load succeeded')

    def compute_test_result(self):
        os.makedirs(self.smpl_dir, exist_ok=True)

        b_size = self.input_data['trans'].shape[0]
        n_joints = self.input_data['pos'].shape[2]

        trans = self.pred_data['trans']
        trans_gt = self.input_data['trans']

        rotmat_gt = rotation_6d_to_matrix(self.input_data['global_xform'])  # (B, T, J, 3, 3)
        local_rotmat_gt = self.fk.global_to_local(rotmat_gt.view(-1, n_joints, 3, 3))  # (B x T, J, 3, 3)
        local_rotmat_gt = local_rotmat_gt.view(b_size, -1, n_joints, 3, 3)  # (B, T, J, 3, 3)
        local_rotmat = local_rotmat_gt.clone()

        for i in range(b_size):
            export_ply_trajectory(points=trans[i], color=(255, 0, 0), ply_fname=os.path.join(self.smpl_dir, f'test_{self.test_index + i:03d}.ply'))
            export_ply_trajectory(points=trans_gt[i], color=(0, 255, 0), ply_fname=os.path.join(self.smpl_dir, f'test_{self.test_index + i:03d}_gt.ply'))

            poses = c2c(matrix_to_axis_angle(local_rotmat[i]))  # (T, J, 3)
            poses_gt = c2c(matrix_to_axis_angle(local_rotmat_gt[i]))  # (T, J, 3)

            poses = poses.reshape((poses.shape[0], -1))  # (T, J x 3)
            poses = np.pad(poses, [(0, 0), (0, 93)], mode='constant')
            poses_gt = poses_gt.reshape((poses_gt.shape[0], -1))  # (T, J x 3)
            poses_gt = np.pad(poses_gt, [(0, 0), (0, 93)], mode='constant')

            np.savez(os.path.join(self.smpl_dir, f'test_{self.test_index + i:03d}.npz'),
                     poses=poses, trans=c2c(trans[i]), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.args.data.fps)
            np.savez(os.path.join(self.smpl_dir, f'test_{self.test_index + i:03d}_gt.npz'),
                     poses=poses_gt, trans=c2c(trans_gt[i]), betas=np.zeros(10), gender=self.args.data.gender, mocap_framerate=self.args.data.fps)

        self.test_index += b_size
