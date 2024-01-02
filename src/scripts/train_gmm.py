import os
import sys
import torch
import random

import joblib 
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from torch.utils.data.dataloader import DataLoader


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from arguments import Arguments
from nemf.losses import GeodesicLoss
from src.datasets.amass import AMASS
from nemf.fk import ForwardKinematicsLayer
from rotations import  matrix_to_rotation_6d, matrix_to_axis_angle
   
class MaxMixturePrior(nn.Module):

    def __init__(self, num_gaussians=1, dtype=torch.float32, epsilon=1e-8,
                 use_merged=False,
                 **kwargs):
        super(MaxMixturePrior, self).__init__()

        if dtype == torch.float32:
            np_dtype = np.float32
        elif dtype == torch.float64:
            np_dtype = np.float64
        else:
            print('Unknown float type {}, exiting!'.format(dtype))
            sys.exit(-1)

        self.num_gaussians = num_gaussians
        self.epsilon = epsilon
        self.use_merged = use_merged

        full_gmm_fn = "./data/gmm.pkl"
        if not os.path.exists(full_gmm_fn):
            print('The path to the mixture prior "{}"'.format(full_gmm_fn) +
                  ' does not exist, exiting!')
            sys.exit(-1)

        with open(full_gmm_fn, 'rb') as f:
            gmm = joblib.load(f)

        if type(gmm) == dict:
            means = gmm['means'].astype(np_dtype)
            covs = gmm['covariances_'].astype(np_dtype)
            weights = gmm['weights'].astype(np_dtype)
        elif 'sklearn.mixture._gaussian_mixture.GaussianMixture' in str(type(gmm)):
            means = gmm.means_.astype(np_dtype)
            covs = gmm.covariances_.astype(np_dtype)
            weights = gmm.weights_.astype(np_dtype)
        else:
            print('Unknown type for the prior: {}, exiting!'.format(type(gmm)))
            sys.exit(-1)

        self.register_buffer('means', torch.tensor(means, dtype=dtype))

        self.register_buffer('covs', torch.tensor(covs, dtype=dtype))

        precisions = [np.linalg.inv(cov) for cov in covs]
        precisions = np.stack(precisions).astype(np_dtype)

        self.gmm = gmm
        
        self.register_buffer('precisions',
                             torch.tensor(precisions, dtype=dtype))

        # The constant term:
        sqrdets = np.array([(np.sqrt(np.linalg.det(c) +  epsilon)) for c in gmm.covariances_])
        const = (2 * np.pi)**(69 / 2.)


        nll_weights = np.asarray(gmm.weights_ / (const *
                                                   (sqrdets / sqrdets.min())))
        nll_weights = torch.tensor(nll_weights, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('nll_weights', nll_weights)

        weights = torch.tensor(gmm.weights_, dtype=dtype).unsqueeze(dim=0)
        self.register_buffer('weights', weights)

        self.register_buffer('pi_term', torch.log(torch.tensor(2 * np.pi, dtype=dtype)))

        cov_dets = [np.log(np.linalg.det(cov.astype(np_dtype)) + epsilon)
                    for cov in covs]
        
        self.register_buffer('cov_dets',
                             torch.tensor(cov_dets, dtype=dtype))

        # The dimensionality of the random variable
        self.random_var_dim = self.means.shape[1]

    def get_mean(self):
        ''' Returns the mean of the mixture '''
        mean_pose = torch.matmul(self.weights, self.means)
        return mean_pose

    def merged_log_likelihood(self, pose):
        diff_from_mean = pose.unsqueeze(dim=1) - self.means
        
        prec_diff_prod = torch.einsum('mij,bmj->bmi',
                                      [self.precisions, diff_from_mean])
        diff_prec_quadratic = (prec_diff_prod * diff_from_mean).sum(dim=-1)

        curr_loglikelihood = 0.5 * diff_prec_quadratic - \
            torch.log(self.nll_weights)
        #  curr_loglikelihood = 0.5 * (self.cov_dets.unsqueeze(dim=0) +
        #  self.random_var_dim * self.pi_term +
        #  diff_prec_quadratic
        #  ) - torch.log(self.weights)
        min_likelihood, _ = torch.min(curr_loglikelihood, dim=1)
        
        
        return min_likelihood

    def log_likelihood(self, pose, *args, **kwargs):
        ''' Create graph operation for negative log-likelihood calculation
        '''
        likelihoods = []

        for idx in range(self.num_gaussians):

            mean = self.means[idx]
            prec = self.precisions[idx]
            cov = self.covs[idx]
            diff_from_mean = pose - mean

            
            curr_loglikelihood = torch.einsum('bj,ji->bi',
                                              [diff_from_mean, prec])
            curr_loglikelihood = torch.einsum('bi,bi->b',
                                              [curr_loglikelihood,
                                               diff_from_mean])
            cov_term = torch.log(torch.det(cov) + self.epsilon)
            curr_loglikelihood += 0.5 * (cov_term +
                                         self.random_var_dim *
                                         self.pi_term)
            likelihoods.append(curr_loglikelihood)

        log_likelihoods = torch.stack(likelihoods, dim=1)
        min_idx = torch.argmin(log_likelihoods, dim=1)
        weight_component = self.nll_weights[:, min_idx]
        weight_component = -torch.log(weight_component)


        return weight_component + log_likelihoods[:, min_idx]

    def forward(self, pose):
        if self.use_merged:
            return self.merged_log_likelihood(pose)
        else:
            return self.log_likelihood(pose)    
    
    

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    
    args = Arguments('./configs', filename="pca_fit.yaml")

    fk = ForwardKinematicsLayer(args)
    args.batch_size = 1
    ngpu = 1
    geoloss = GeodesicLoss()

    train_dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'train'))
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=1, shuffle=True, pin_memory=True, drop_last=True)

    poses_aa, geoloss_list = [], []

    for item in tqdm(train_data_loader):
        
        pose_locat_rotmat = fk.global_to_local(item['rotmat'].view(-1, 16, 3, 3)) 
        pose_locat_6d = matrix_to_axis_angle(pose_locat_rotmat.view(-1, 16, 3, 3))  
        poses_aa.append(pose_locat_6d[:, 1:, ...].reshape(args.batch_size, -1))
        
        geoloss_list_item = []
        
        for t in range(127):
            geoloss_t = geoloss(pose_locat_rotmat[:-1][t], pose_locat_rotmat[1:][t])
            geoloss_list_item.append(geoloss_t)
        
        geoloss_list.append(geoloss_list_item)
        
        
    num_samples = 100
            
    poses_aa = torch.cat(poses_aa, dim=0)
    poses_aa_mean = torch.mean(poses_aa, 0)
    poses_aa_centered = poses_aa
    
    gmm_geoloss = GMM(n_components=1)
    gmm_aa = GMM(n_components=1)
    pca_geoloss = PCA(n_components=256)
    pca_aa = PCA(n_components=256)

    gmm_aa.fit(poses_aa)
    gmm_geoloss.fit(geoloss_list)
    pca_aa.fit(poses_aa_centered)

    joblib.dump(gmm_aa, "./data/gmm.pkl")
    joblib.dump(pca_aa, "./data/pca.pkl")



