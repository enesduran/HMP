import os
import glob
import json
import torch
import joblib
import random
import numpy as np
import pandas as pd
from scipy import linalg
from arguments import Arguments
from nemf.losses import GeodesicLoss
from nemf.generative import Architecture
from nemf.fk import ForwardKinematicsLayer
from scripts.train_gmm import MaxMixturePrior
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from utils import FOOT_IDX, estimate_angular_velocity, estimate_linear_velocity
from rotations import axis_angle_to_matrix, matrix_to_rotation_6d, rotation_6d_to_matrix


def padding_prior_data(data):
    N, T, J = data['pos'].shape[:3]
    # pad inputs to match the training sequence length
    zero_padding = torch.zeros(N, args.data.clip_length - T, J, 3).to(prior.device)
    identity_padding = torch.zeros(N, args.data.clip_length - T, 3, 3).to(prior.device)
    identity_padding[:, :, 0, 0] = 1
    identity_padding[:, :, 1, 1] = 1
    identity_padding[:, :, 2, 2] = 1
    rotation_6d_padding = torch.zeros(N, args.data.clip_length - T, 6).to(prior.device)
    rotation_6d_padding[..., 0] = 1
    rotation_6d_padding[..., 4] = 1

    pos = torch.cat((data['pos'], zero_padding), dim=1)
    velocity = torch.cat((data['velocity'], zero_padding), dim=1)
    angular = torch.cat((data['angular'], zero_padding), dim=1)
    root_orient = torch.cat((data['root_orient'], rotation_6d_padding), dim=1)
    root_vel = torch.cat((data['root_vel'], zero_padding[:, :, 0]), dim=1)
    rotmat_padding = rotation_6d_padding.unsqueeze(2).expand(-1, -1, J, -1)
    global_xform = torch.cat((data['global_xform'], rotmat_padding), dim=1)

    data['pos'] = pos
    data['velocity'] = velocity
    data['global_xform'] = global_xform
    data['angular'] = angular
    data['root_orient'] = root_orient
    data['root_vel'] = root_vel

    return data


def load_data(files, padding=False, max_samples=400):
    poses = []
    trans = []
    visibility = []
    assert len(files) != 0, 'files not found'

    max_samples = min(max_samples, len(files))
    
    # import ipdb; ipdb.set_trace()
  
    for f in files[:max_samples]:
        bdata = dict(np.load(f, allow_pickle=True))
        if 'r_wrist_orient' in bdata.keys() and 'pose_right_hand' in bdata.keys():
            
            coef = len(bdata['r_wrist_orient']) // args.data.clip_length
            
            for k_ in range(coef):
                root_orient = bdata['r_wrist_orient'][k_*args.data.clip_length:(k_+1)*args.data.clip_length]
                pose_body = bdata['pose_right_hand'][k_*args.data.clip_length:(k_+1)*args.data.clip_length]
                trans.append(bdata['r_wrist_location'][k_*args.data.clip_length:(k_+1)*args.data.clip_length])
            
                poses.append(np.concatenate((root_orient, pose_body), axis=-1))
            
        elif 'pose_right_hand' in bdata.keys():
            pass 
            # poses.append(bdata['pose_right_hand'][:, :72])
        else:
            raise RuntimeError(f'missing pose parameters in the file: {f}')
        
        
        if 'visibility' in bdata.keys():
            visibility.append(bdata['visibility'])

        # if 'poses' in bdata.keys():
        #     poses.append(bdata['poses'][:, :72])
        # elif 'root_orient' in bdata.keys() and 'pose_body' in bdata.keys():
        #     root_orient = bdata['root_orient']
        #     pose_body = bdata['pose_body']
        #     poses.append(np.concatenate((root_orient, pose_body), axis=-1))
        # else:
        #     raise RuntimeError(f'missing pose parameters in the file: {f}')
        # trans.append(bdata['trans'])
        # if 'visibility' in bdata.keys():
        #     visibility.append(bdata['visibility'])

    trans = torch.from_numpy(np.asarray(trans, np.float32)).cuda()  # global translation (N, T, 3)
    N, T = trans.shape[:2]
    poses = torch.from_numpy(np.asarray(poses, np.float32)).cuda()
    poses = poses.view(N, T, -1, 3)  # axis-angle (N, T, J, 3)

    
    # poses = poses.view(N, T, 24, 3)  # axis-angle (N, T, J, 3)

    root_orient = poses[:, :, 0].clone()
    root_rotation = axis_angle_to_matrix(root_orient)  # (N, T, 3, 3)
    poses[:, :, 0] = 0

    rotmat = axis_angle_to_matrix(poses)  # (N, T, J, 3, 3)
    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)
    pos, global_xform = fk(rotmat.view(-1, 16, 3, 3))
    pos = pos.contiguous().view(N, T, 16, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(N, T, 16, 4, 4)
    global_xform = global_xform[:, :, :, :3, :3]  # global transformation matrix for each joint (N, T, J, 3, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)

    root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (N, T, 3)

    global_pos = torch.matmul(root_rotation.unsqueeze(2), pos.unsqueeze(-1)).squeeze(-1)  # (N, T, J, 3)
    global_pos = global_pos + trans.unsqueeze(2)

    data = {
        'pos': pos,
        'velocity': velocity,
        'global_xform': matrix_to_rotation_6d(global_xform),
        'angular': angular,
        'root_orient': matrix_to_rotation_6d(root_rotation),
        'root_vel': root_vel,
        'global_pos': global_pos,
        'rotmat': rotmat,
        'trans': trans
    }

    if len(visibility) != 0:
        visibility = torch.from_numpy(np.asarray(visibility, bool)).cuda()
        padding = torch.zeros(visibility.shape[:2] + torch.Size([2]), dtype=torch.bool).cuda()
        visibility = torch.cat((visibility, padding), dim=-1)  # (T, 24)
        print(f'visibility: {visibility.shape}')
        data['visibility'] = visibility

    if T < args.data.clip_length:
        padding = True

    if padding:
        if T == args.data.clip_length:
            data['pos'] = data['pos'].view(2 * N, T // 2, 24, 3)
            data['velocity'] = data['velocity'].view(2 * N, T // 2, 24, 3)
            data['global_xform'] = data['global_xform'].view(2 * N, T // 2, 24, 6)
            data['angular'] = data['angular'].view(2 * N, T // 2, 24, 3)
            data['root_orient'] = data['root_orient'].view(2 * N, T // 2, 6)
            data['root_vel'] = data['root_vel'].view(2 * N, T // 2, 3)
            data['global_pos'] = data['global_pos'].view(2 * N, T // 2, 24, 3)
            data['rotmat'] = data['rotmat'].view(2 * N, T // 2, 24, 3, 3)
            data['trans'] = data['trans'].view(2 * N, T // 2, 3)

        padding_data = padding_prior_data(data)
        for key in padding_data.keys():
            data[key] = padding_data[key]

    return data


def calculate_feature_statistics(feature):
    feature = feature.cpu().numpy()
    mu = np.mean(feature, axis=0)
    sigma = np.cov(feature, rowvar=False)

    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def evaluate_fid(data, gt_data, num_motions=None, mask=None):
    if num_motions is None:
        index = list(range(data['pos'].shape[0]))
        index_gt = list(range(gt_data['pos'].shape[0]))
    else:
        # randomly pick [num_motions] motions from the data
        assert num_motions <= data['pos'].shape[0]
        assert num_motions <= gt_data['pos'].shape[0]

        index = list(range(data['pos'].shape[0]))
        index = random.sample(index, num_motions)
        index_gt = list(range(gt_data['pos'].shape[0]))
        index_gt = random.sample(index_gt, num_motions)

    data_selected = {}
    gt_data_selected = {}
    for key in data.keys():
        data_selected[key] = data[key][index].clone()
        gt_data_selected[key] = gt_data[key][index_gt].clone()

    if mask is not None:
        # replace with ground truth
        for key in data.keys():
            data_selected[key][:, mask] = gt_data[key][:, mask].clone()

    with torch.no_grad():
        prior.set_input(data_selected)
        f_l, _, _ = prior.encode_local()
        f_g, _, _ = prior.encode_global()
        feature = torch.cat((f_l, f_g), dim=-1)

        prior.set_input(gt_data_selected)
        f_l_gt, _, _ = prior.encode_local()
        f_g_gt, _, _ = prior.encode_global()
        feature_gt = torch.cat((f_l_gt, f_g_gt), dim=-1)

    statistics = calculate_feature_statistics(feature.view(feature.shape[0], -1))
    statistics_gt = calculate_feature_statistics(feature_gt.view(feature_gt.shape[0], -1))

    fid = calculate_frechet_distance(statistics[0], statistics[1], statistics_gt[0], statistics_gt[1])
    print(f'FID: {fid:.4f}')

    return fid


def evaluate_diversity(data):
    index = list(range(data['pos'].shape[0]))
    random.shuffle(index)
    index1 = index[:data['pos'].shape[0] // 2]
    index2 = index[data['pos'].shape[0] // 2:]

    with torch.no_grad():
        prior.set_input(data)
        f_l, _, _ = prior.encode_local()
        # f_g, _, _ = prior.encode_global()
        # feature = torch.cat((f_l, f_g), dim=-1)
        
        feature = f_l.clone()
        

    diversity = 0
    for first_idx, second_idx in zip(index1, index2):
        diversity += torch.dist(feature[first_idx], feature[second_idx])
    diversity /= len(index1)

    print(f'Diversity: {diversity.item():.4f}')

    return diversity.item()


def evaluate_recon_error(data, gt_data):
    criterion_geo = GeodesicLoss()

    root_orient = rotation_6d_to_matrix(data['root_orient'])
    root_orient_gt = rotation_6d_to_matrix(gt_data['root_orient'])
    orientation_error = criterion_geo(root_orient.view(-1, 3, 3), root_orient_gt.view(-1, 3, 3))

    rotation_error = criterion_geo(data['rotmat'][:, :, 1:].reshape(-1, 3, 3), gt_data['rotmat'][:, :, 1:].reshape(-1, 3, 3))
    position_error = torch.linalg.norm((data['pos'] - gt_data['pos']), dim=-1).mean()
    translation_error = torch.linalg.norm((data['trans'] - gt_data['trans']), dim=-1).mean()
    final_displacement_error = torch.linalg.norm(data['global_pos'][:, -1] - gt_data['global_pos'][:, -1], dim=-1).mean()

    return {
        'rotation': c2c(rotation_error) * 180.0 / np.pi,
        'position': c2c(position_error) * 100.0,
        'orientation': c2c(orientation_error) * 180.0 / np.pi,
        'translation': c2c(translation_error) * 100.0,
        'final_displacement': c2c(final_displacement_error) * 100.0
    }


def synthetic_trajectory(N_motions):

    z_l = torch.normal(mean=0, std=1, size=(N_motions, args.local_prior.z_dim)).to(prior.device)
    z_g = torch.normal(mean=0, std=1, size=(N_motions, args.global_prior.z_dim)).to(prior.device)
    # z_g = torch.zeros((N_motions, args.global_prior.z_dim)).to(prior.device)
 
    
    with torch.no_grad():
        output = prior.decode(z_l, z_g, length=args.data.clip_length, step=1)

    rotmat = output["rotmat"]  # (N, T, J, 3, 3)
    angular = estimate_angular_velocity(rotmat.clone(), dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)
    pos, global_xform = fk(rotmat.view(-1, 16, 3, 3))
    pos = pos.contiguous().view(N_motions, args.data.clip_length, 16, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(N_motions, args.data.clip_length, 16, 4, 4)
   
    global_xform = global_xform[:, :, :, :3, :3]  # global transformation matrix for each joint (N, T, J, 3, 3)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)
 
    root_rotation = rotmat[:, :, 0]  # (N, T, 3, 3)
    global_pos = torch.matmul(root_rotation.unsqueeze(2), pos.unsqueeze(-1)).squeeze(-1)  # (N, T, J, 3)

    output_ = {'pos': pos,
                'velocity': velocity,
                'global_xform': matrix_to_rotation_6d(global_xform),
                'angular': angular,
                'root_orient': matrix_to_rotation_6d(root_rotation),
                'rotmat': rotmat}

    return output_


def ablation_study(path, subtask):
    
    num_motions = 100
    print(os.path.join(path, 'TCDHands', '*.npz'))
    
    test_data = load_data(sorted(glob.glob(os.path.join('./outputs/generative_copy/TCDHands', '*.npz'))))    # test data TCDHands
    val_data = load_data(sorted(glob.glob(os.path.join('./outputs/generative_copy/GRAB', '*.npz'))))    # test data TCDHands
    train_data = load_data(sorted(glob.glob(os.path.join('./outputs/generative_copy/SAMP', '*.npz'))))    # test data TCDHands
    synthetic_data = synthetic_trajectory(N_motions=num_motions)

    pca_aa = joblib.load("data/pca.pkl") 
    pca_mean = torch.tensor(pca_aa.mean_).to("cuda")
    pca_cov = torch.tensor(pca_aa.get_covariance()).to("cuda")
    pca_sv = torch.tensor(pca_aa.singular_values_).to("cuda")
    pca_pc = torch.tensor(pca_aa.components_).to("cuda")
    
    random_samples_low_dim = np.random.randn(num_motions, pca_aa.n_components_)
    pca_data = pca_aa.inverse_transform(random_samples_low_dim)

    gmm_aa = MaxMixturePrior()
    gmm_data = gmm_aa.gmm.sample(num_motions)
    
    identity = torch.eye(3)
    identity = identity.view(1, 1, 3, 3).repeat(num_motions, args.data.clip_length, 1, 1)
    
    # convert to tensor 
    gmm_data = torch.tensor(gmm_data[0].reshape(num_motions, args.data.clip_length, -1, 3))
    gmm_data = torch.cat((identity.unsqueeze(2), axis_angle_to_matrix(gmm_data)), dim=2)
    gmm_rot6d = matrix_to_rotation_6d(gmm_data).reshape(num_motions * args.data.clip_length, -1, 6)
    
    # convert to tensor 
    pca_data = torch.tensor(pca_data.reshape(num_motions, args.data.clip_length, -1, 3))
    pca_data = torch.cat((identity.unsqueeze(2), axis_angle_to_matrix(pca_data)), dim=2)
    pca_rot6d = matrix_to_rotation_6d(pca_data).reshape(num_motions * args.data.clip_length, -1, 6)
    
    gmm_pos, gmm_xform = fk(gmm_rot6d.to(fk.device))
    pca_pos, pca_xform = fk(pca_rot6d.to(fk.device))

    gmm_xform = matrix_to_rotation_6d(gmm_xform[:, :, :3, :3]) 
    pca_xform = matrix_to_rotation_6d(pca_xform[:, :, :3, :3]) 
    
    gmm_velocity = estimate_linear_velocity(gmm_pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)
    pca_velocity = estimate_linear_velocity(pca_pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)
    
    gmm_angular = estimate_angular_velocity(gmm_data.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)
    pca_angular = estimate_angular_velocity(pca_data.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)
    
    gmm_dict = {'pos': gmm_pos.reshape(num_motions, args.data.clip_length, -1, 3),
                'global_xform': gmm_xform.reshape(num_motions, args.data.clip_length, -1, 6),
                'velocity': gmm_velocity.reshape(num_motions, args.data.clip_length, -1, 3),
                'angular': gmm_angular}
    
    pca_dict = {'pos': pca_pos.reshape(num_motions, args.data.clip_length, -1, 3),
                'global_xform': pca_xform.reshape(num_motions, args.data.clip_length, -1, 6),
                'velocity': pca_velocity.reshape(num_motions, args.data.clip_length, -1, 3),
                'angular': pca_angular}
     
    metrics = dict()
    
    def f(data, num_motions, key):
        div = evaluate_diversity(data)
        if key in metrics.keys():
            pass
        else:
            metrics[key] = {'DIV': [div]}

    print('Motion Synthesis:')
    div_test = f(test_data, num_motions, 'TEST_DATA (TDCHANDS)')
    div_val = f(val_data, num_motions, 'VAL_DATA (SAMP)')
    div_train = f(train_data, num_motions, 'TRAIN_DATA (GRAB)')
    div_synt = f(synthetic_data, num_motions, 'SYNTHETIC')
    div_gmm = f(gmm_dict, num_motions, 'GMM')
    div_pca = f(pca_dict, num_motions, 'PCA')

    df = pd.DataFrame.from_dict(metrics)
    df.to_csv(os.path.join(path, f'diversity.csv'), index=False)



def comparison(path, subtask):
    gt_data = load_data(sorted(glob.glob(os.path.join(path, 'gt', '*.npz'))))

    file_path = os.path.join(path, subtask)

    metrics = dict()
    if subtask == 'recon':
        humor_data = load_data(sorted(glob.glob(os.path.join(file_path, 'humor', '*.npz'))))
        hm_vae_data = load_data(sorted(glob.glob(os.path.join(file_path, 'hm-vae', '*.npz'))))
        nemf_data = load_data(sorted(glob.glob(os.path.join(file_path, 'nemf', '*.npz'))))

        metrics['HUMOR'] = evaluate_recon_error(humor_data, gt_data)
        metrics['HMVAE'] = evaluate_recon_error(hm_vae_data, gt_data)
        metrics['NEMF'] = evaluate_recon_error(nemf_data, gt_data)
        print('Motion Reconstruction:')
        print(json.dumps(metrics, sort_keys=True, indent=4))

        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(file_path, f'{subtask}.csv'))
    elif subtask == 'synth':
        humor_data = load_data(sorted(glob.glob(os.path.join(file_path, 'humor', '.*.npz'))), padding=True)
        hm_vae_data = load_data(sorted(glob.glob(os.path.join(file_path, 'hm-vae', '*.npz'))), padding=True)
        nemf_data = load_data(sorted(glob.glob(os.path.join(file_path, 'nemf', '*.npz'))), padding=True)

        num_motions = 100

        def f(data, num_motions, key):
            fid = evaluate_fid(data, gt_data, num_motions)
            div = evaluate_diversity(data)
            index = list(range(data['global_pos'].shape[0]))
            index = random.sample(index, num_motions)
            s = foot_skating(data['global_pos'][index])
            if key in metrics.keys():
                metrics[key]['FID'].append(fid)
                metrics[key]['DIV'].append(div)
                metrics[key]['FOOT'].append(s)
            else:
                metrics[key] = {'FID': [fid], 'DIV': [div], 'FOOT': [s]}

        print('Motion Synthesis:')
        for replication in range(20):
            print(f'==================== Replication {replication} ====================')
            f(humor_data, num_motions, 'HUMOR')
            f(hm_vae_data, num_motions, 'HMVAE')
            f(nemf_data, num_motions, 'NEMF')

        for key in metrics.keys():
            df = pd.DataFrame.from_dict(metrics[key])
            df.to_csv(os.path.join(file_path, f'{subtask}_{key}.csv'), index=False)
    else:
        raise RuntimeError(f'unsupported evaluation task for comparison: {subtask}')


def inbetween_benchmarks(path, keyframes=False):
    trans_lengths = [5, 10, 15, 20] if keyframes else [10, 20, 30]

    gt_data = load_data(sorted(glob.glob(os.path.join(path, 'gt', '*.npz'))))
    s = foot_skating(gt_data['global_pos'])
    print(f'average foot skating of the real motions: {s:.4f}')

    def f(data, key, mask):
        fid = evaluate_fid(data, gt_data, mask=mask)
        s = foot_skating(data['global_pos'], mask=mask)
        if key in statistics.keys():
            statistics[key]['FID'].append(fid)
            statistics[key]['FOOT'].append(s)
        else:
            statistics[key] = {'FID': [fid], 'FOOT': [s]}

    file_path = os.path.join(path, 'keyframe') if keyframes else os.path.join(path, 'clip')
    statistics = {}
    for t in trans_lengths:
        print(f'==================== Length {t} ====================')
        if keyframes:
            mask = None  # don't mask data for keyframe inbetweening
        else:
            end_frame = (args.data.clip_length - t) // 2
            mask = list(range(end_frame)) + list(range(end_frame + t, args.data.clip_length))
        print(f'mask: {mask}')

        slerp_data = load_data(sorted(glob.glob(os.path.join(file_path, 'slerp', f'{t}_frames', '*.npz'))))
        f(slerp_data, 'SLERP', mask)

        if not keyframes:
            inertia_data = load_data(sorted(glob.glob(os.path.join(file_path, 'inertialization', f'{t}_frames', '*.npz'))))
            f(inertia_data, 'INERTIALIZATION', mask)

            rmi_data = load_data(sorted(glob.glob(os.path.join(file_path, 'rmi', f'{t}_frames', '*.npz'))))
            f(rmi_data, 'RMI', mask)

        hm_vae_data = load_data(sorted(glob.glob(os.path.join(file_path, 'hm-vae', f'{t}_frames', '*.npz'))))
        f(hm_vae_data, 'HMVAE', mask)

        nemf_data = load_data(sorted(glob.glob(os.path.join(file_path, 'nemf', f'{t}_frames', '*.npz'))))
        f(nemf_data, 'NEMF', mask)

    for key in statistics.keys():
        df = pd.DataFrame.from_dict(statistics[key])
        df.to_csv(os.path.join(file_path, f'inbetween_{key}.csv'))


def compute_joint_accel(joint_seq, fps):
    ''' Magnitude of joint accelerations for joint_seq : B x T x J x 3 '''
    joint_accel = joint_seq[:, :-2] - (2 * joint_seq[:, 1:-1]) + joint_seq[:, 2:]
    DATA_h = 1.0 / fps
    joint_accel = joint_accel / ((DATA_h**2))
    joint_accel_mag = torch.norm(joint_accel, dim=-1)

    return joint_accel, joint_accel_mag


def evaluate_super_sampling(path):
    fps_30_data = load_data(sorted(glob.glob(os.path.join(path, '30fps/*.npz'))))
    fps_60_data = load_data(sorted(glob.glob(os.path.join(path, '60fps/*.npz'))))
    fps_120_data = load_data(sorted(glob.glob(os.path.join(path, '120fps/*.npz'))))
    fps_240_data = load_data(sorted(glob.glob(os.path.join(path, '240fps/*.npz'))))

    smoothness = dict()
    joint_vel = estimate_linear_velocity(fps_30_data['global_pos'], dt=1.0 / 30.0)
    joint_vel_mag = torch.norm(joint_vel, dim=-1).mean()
    smoothness['30FPS'] = [c2c(joint_vel_mag * 100.0)]
    joint_vel = estimate_linear_velocity(fps_60_data['global_pos'], dt=1.0 / 60.0)
    joint_vel_mag = torch.norm(joint_vel, dim=-1).mean()
    smoothness['60FPS'] = [c2c(joint_vel_mag * 100.0)]
    joint_vel = estimate_linear_velocity(fps_120_data['global_pos'], dt=1.0 / 120.0)
    joint_vel_mag = torch.norm(joint_vel, dim=-1).mean()
    smoothness['120FPS'] = [c2c(joint_vel_mag * 100.0)]
    joint_vel = estimate_linear_velocity(fps_240_data['global_pos'], dt=1.0 / 240.0)
    joint_vel_mag = torch.norm(joint_vel, dim=-1).mean()
    smoothness['240FPS'] = [c2c(joint_vel_mag * 100.0)]
  
    df = pd.DataFrame.from_dict(smoothness)
    df.to_csv(os.path.join(path, f'smoothness.csv'))


def evaluate_smoothness_test(path):
    smoothness = dict()

    for fps in [30, 60]:
        smoothness[f'{fps}FPS'] = dict()
        for L in range(1, 23, 2):
            data = load_data([os.path.join(path, f'L{L}', f'test_{fps}fps.npz')])
            # _, joint_accel_mag = compute_joint_accel(data['global_pos'], fps=fps)
            # smoothness[f'{fps}FPS'][L] = c2c(joint_accel_mag.mean() * 100.0)
            joint_vel = estimate_linear_velocity(data['global_pos'], dt=1.0 / fps)
            joint_vel_mag = torch.norm(joint_vel, dim=-1).mean()
            smoothness[f'{fps}FPS'][L] = c2c(joint_vel_mag.mean() * 100.0)

    df = pd.DataFrame.from_dict(smoothness)
    df.to_csv(os.path.join(path, f'smoothness.csv'))

    import matplotlib.pyplot as plt
    plt.rcParams['text.usetex'] = True

    smoothness_30fps = np.array(list(smoothness['30FPS'].values()))
    smoothness_60fps = np.array(list(smoothness['60FPS'].values()))

    plt.plot(np.arange(1, 23, 2), smoothness_30fps, marker='o', label='30fps')
    plt.plot(np.arange(1, 23, 2), smoothness_60fps, marker='o', label='60fps')
    plt.legend()
    plt.grid(linestyle='--')
    plt.xticks(np.arange(1, 23, 2))
    plt.xlabel(r'$L$')
    # plt.yscale('log')
    plt.ylabel('Mean Per-Joint Velocity')
    plt.savefig(os.path.join(path, 'smoothness.pdf'))


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--config', type=str, default='evaluation.yaml', help='name of the configutation file')
    parser.add_argument('--task', type=str, help='name of application tasks', required=True)
    parser.add_argument('--load_path', type=str, help='path to load results for evaluation', required=True)

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

    prior = Architecture(args, ngpu)
    prior.load(optimal=True)
    prior.eval()
    fk = ForwardKinematicsLayer(args)

    if opt.task == 'ablation_study':
        print('Evaluating ablation study ...')
        ablation_study(opt.load_path, subtask='synth')
    elif opt.task == 'comparison':
        print('Evaluating comparison with other methods ...')
        comparison(opt.load_path, subtask='recon')
        comparison(opt.load_path, subtask='synth')
    elif opt.task == 'inbetween_benchmark':
        print('Evaluating motion in-betweening ...')
        inbetween_benchmarks(opt.load_path, keyframes=False)
        inbetween_benchmarks(opt.load_path, keyframes=True)
    elif opt.task == 'super_sampling':
        print('Evaluating super sampling ...')
        evaluate_super_sampling(opt.load_path)
    elif opt.task == 'smoothness_test':
        print('Evaluating smoothness testing ...')
        evaluate_smoothness_test(opt.load_path)
    else:
        raise NotImplementedError(f'unknown evaluation: {opt.task}')
