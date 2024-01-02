import os 
import sys
import cv2
import glob
import torch
import shutil
import tempfile
import numpy as np
import open3d as o3d
import scenepic as sp
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
from loguru import logger
import torch.nn.functional as F
from fitting_utils import AUGMENTED_MANO_CHAIN
from typing import Dict, Union, Optional, Tuple
import open3d.visualization.rendering as rendering
from .typing import Tensor, Array, IntList, StructureList

dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dirname, ".."))

from body_model.mano import BodyModel 
from open3d_viz_overlay import images_to_video
from fitting_utils import vis_mano_skeleton_matplotlib
from datasets.amass import get_ho3d_v3_gt, get_dexycb_gt

def get_mano_skeleton():
    return np.array([[0, 1], 
                     [1, 2], 
                     [2, 3],
                     [0, 4],
                     [4, 5], 
                     [5, 6], 
                     [0, 7],
                     [7, 8], 
                     [8, 9], 
                     [0, 10], 
                     [10, 11], 
                     [11, 12],
                     [0, 13],
                     [13, 14], 
                     [14, 15],
                     [15, 16],
                     [3, 17], 
                     [6, 18],
                     [12, 19],
                     [9, 20]])

    
# for hand 
def get_openpose_skeleton():
    return np.array([[0, 1], 
                     [1, 2], 
                     [2, 3], 
                     [3, 4], 
                     [0, 5], 
                     [5, 6], 
                     [6, 7], 
                     [7, 8], 
                     [0, 9],
                     [9, 10], 
                     [10, 11], 
                     [11, 12], 
                     [0, 13], 
                     [13, 14],
                     [14, 15],
                     [15, 16],
                     [0, 17],
                     [17, 18],
                     [18, 19], 
                     [19, 20]])


MANO_RIGHT_MODEL_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"

def np2o3d_pcl(x: np.ndarray) -> o3d.geometry.PointCloud:
    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(x.reshape(-1, 3))
    return pcl

def build_alignment(name: str, **kwargs):
    if name == 'procrustes':
        return ProcrustesAlignment()
    elif name == 'root':
        return RootAlignment(**kwargs)
    elif name == 'scale':
        return ScaleAlignment()
    elif name == 'no' or name == 'none':
        return NoAlignment()
    else:
        raise ValueError(f'Unknown alignment type: {name}')


def compute_error_accel(joints_gt, joints_pred, vis=None):
    """
    Computes acceleration error: (for joints)
        1/(n-2) \sum_{i=1}^{n-1} X_{i-1} - 2X_i + X_{i+1}
    Note that for each frame that is not visible, three entries in the
    acceleration error should be zero'd out.
    Args:
        joints_gt (Nx21x3).
        joints_pred (Nx21x3).
        vis (N).
    Returns:
        error_accel (N-2).
    """
    joints_pred = joints_pred.reshape(-1, 21, 3)
    joints_gt = joints_gt.reshape(-1, 21, 3)

    # (N-2)x21x3
    accel_gt = joints_gt[:-2] - 2 * joints_gt[1:-1] + joints_gt[2:]
    accel_pred = joints_pred[:-2] - 2 * joints_pred[1:-1] + joints_pred[2:]

    normed = np.linalg.norm(accel_pred - accel_gt, axis=2)

    if vis is None:
        new_vis = np.ones(len(normed), dtype=bool)
    else:
        invis = np.logical_not(vis)
        invis1 = np.roll(invis, -1)
        invis2 = np.roll(invis, -2)
        new_invis = np.logical_or(invis, np.logical_or(invis1, invis2))[:-2]
        new_vis = np.logical_not(new_invis)

    return np.mean(normed[new_vis], axis=1)

def point_error(input_points: Union[Array, Tensor],
                target_points: Union[Array, Tensor]) -> Array:
    ''' Calculate point error
    Parameters
    ----------
        input_points: numpy.array, BxPx3
            The estimated points
        target_points: numpy.array, BxPx3
            The ground truth points
    Returns
    -------
        numpy.array, BxJ
            The point error for each element in the batch
    '''
    if torch.is_tensor(input_points):
        input_points = input_points.detach().cpu().numpy()
    if torch.is_tensor(target_points):
        target_points = target_points.detach().cpu().numpy()

    return np.sqrt(np.power(input_points - target_points, 2).sum(axis=-1))


def weighted_point_error(input_points: Union[Array, Tensor],
                target_points: Union[Array, Tensor]) -> Array:
    ''' Calculate point error
    Parameters
    ----------
        input_points: numpy.array, BxPx3 The estimated points
        target_points: numpy.array, BxPx2 The ground truth points
    Returns
    -------
        numpy.array, BxJ
            The point error for each element in the batch
    '''   
    
    input_points_ = input_points[:, :, :-1]
    conf = input_points[:, :, -1][..., None]
     
    if torch.is_tensor(input_points_):
        input_points_ = input_points_.detach().cpu().numpy()
        conf = conf.detach().cpu().numpy()
    if torch.is_tensor(target_points):
        target_points = target_points.detach().cpu().numpy()

    err = np.power(input_points_ - target_points, 2)
    weighted_err = np.sqrt((err * conf).sum(axis=-1))
    
    return weighted_err




def vertex_to_vertex_error(input_vertices, target_vertices):
    return np.sqrt(np.power(input_vertices - target_vertices, 2).sum(axis=-1))


class NoAlignment(object):
    def __init__(self):
        super(NoAlignment, self).__init__()

    def __repr__(self):
        return 'NoAlignment'

    @property
    def name(self):
        return 'none'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        return S1, S2


class ProcrustesAlignment(object):
    def __init__(self):
        super(ProcrustesAlignment, self).__init__()

    def __repr__(self):
        return 'ProcrustesAlignment'

    @property
    def name(self):
        return 'procrustes'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrustes problem.
        '''
        if len(S1.shape) < 2:
            S1 = S1.reshape(1, *S1.shape)
            S2 = S2.reshape(1, *S2.shape)

        transposed = False
        if S1.shape[1] != 3 and S1.shape[1] != 3:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.transpose(S2, [0, 2, 1])
            transposed = True

        assert(S2.shape[1] == S1.shape[1])
        batch_size = len(S1)

        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        # 1. Remove mean.
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2, axis=(1, 2))

        # 3. The outer product of X1 and X2.
        K = X1 @ np.transpose(X2, [0, 2, 1])

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = np.transpose(Vh, [0, 2, 1])
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.tile(np.eye(3)[np.newaxis], [batch_size, 1, 1])
        Z[:, -1, -1] *= np.sign(np.linalg.det(U @ Vh))
        # Construct R.
        R = V @ (Z @ np.transpose(U, [0, 2, 1]))

        # 5. Recover scale.
        scale = np.einsum('bii->b', R @ K) / var1

        # 6. Recover translation.
        t = mu2.squeeze(-1) - scale[:, np.newaxis] * np.einsum(
            'bmn,bn->bm', R, mu1.squeeze(-1))

        # 7. Error:
        S1_hat = scale.reshape(-1, 1, 1) * (R @ S1) + t.reshape(
            batch_size, -1, 1)

        if transposed:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.ascontiguousarray(np.transpose(S2, [0, 2, 1]))
            S1_hat = np.ascontiguousarray(np.transpose(S1_hat, [0, 2, 1]))

        return S1_hat, S2


class ScaleAlignment(object):
    def __init__(self):
        super(ScaleAlignment, self).__init__()

    def __repr__(self):
        return 'ScaleAlignment'

    @property
    def name(self):
        return 'scale'

    def __call__(self, S1: Array, S2: Array) -> Tuple[Array, Array]:
        '''
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        '''
        transposed = False
        if len(S1.shape) < 2:
            S1 = S1.reshape(1, *S1.shape)
            S2 = S2.reshape(1, *S2.shape)

        batch_size = len(S1)
        if S1.shape[1] != 3 and S1.shape[1] != 3:
            S1 = np.transpose(S1, [0, 2, 1])
            S2 = np.transpose(S2, [0, 2, 1])
            transposed = True

        assert(S2.shape[1] == S1.shape[1])

        mu1 = S1.mean(axis=-1, keepdims=True)
        mu2 = S2.mean(axis=-1, keepdims=True)

        # 1. Remove mean.
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2, axis=(1, 2))
        var2 = np.sum(X2 ** 2, axis=(1, 2))

        # 5. Recover scale.
        scale = np.sqrt(var2 / var1)

        # 6. Recover translation.
        t = mu2 - scale * mu1

        # 7. Error:
        S1_hat = scale.reshape(-1, 1, 1) * S1 + t.reshape(batch_size, -1, 1)

        if transposed:
            S1_hat = np.transpose(S1_hat, [0, 2, 1])

        return S1_hat, S2


class RootAlignment(object):
    def __init__(self, root: Optional[IntList] = None, **kwargs) -> None:
        super(RootAlignment, self).__init__()
        if root is None:
            root = [0]
        self.root = root

    def set_root(self, new_root):
        self.root = new_root

    @property
    def name(self):
        return 'root'

    def __repr__(self):
        return f'RootAlignment: root = {self.root}'

    def align_by_root(self, joints: Array) -> Array:
        root_joint = joints[:, self.root, :].mean(axis=1, keepdims=True)
        return joints - root_joint

    def __call__(self, est: Array, gt: Array) -> Tuple[Array, Array]:
        est_out = self.align_by_root(est)    
        est_out += gt[:, self.root, :] 
        
        # gt_out = self.align_by_root(gt)
        return est_out, gt


def point_fscore(pred: torch.Tensor, gt: torch.Tensor, thresh: float) -> Dict[str, float]:
    if torch.is_tensor(pred):
        pred = pred.detach().cpu().numpy()
    if torch.is_tensor(gt):
        gt = gt.detach().cpu().numpy()
        
    f_scores_dict = {}

    for threshold in thresh:
        
        f_score_list = []
        precision_list = []
        recall_list = []
        
        for t in range(len(pred)):
            pred_pcl = np2o3d_pcl(pred[t])
            gt_pcl = np2o3d_pcl(gt[t])
  
            gt_to_pred = np.asarray(gt_pcl.compute_point_cloud_distance(pred_pcl))
            pred_to_gt = np.asarray(pred_pcl.compute_point_cloud_distance(gt_pcl))
 

            recall = (pred_to_gt < threshold).reshape(-1, 778).sum(1) / 778
            precision = (gt_to_pred < threshold).reshape(-1, 778).sum(1) / 778
 
            valid_fscore_flag = (recall + precision > 0.0)[0]
            fscore = 0
            
            if valid_fscore_flag:
                fscore = ((2 * recall * precision) / (recall + precision))
                
            f_score_list.append(fscore)
        
        f_scores_dict[f'f@{threshold}'] = {'fscore': np.array(f_score_list),
                                            'precision': np.array(precision_list),
                                            'recall': np.array(recall_list)}
    
    return f_scores_dict

class AccelError(object):
    def __init__(self,
                 alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment],
                 name: str = '',
                 return_aligned=False) -> None:
        
        super(AccelError, self).__init__()
        self._alignment = alignment_object
        self._return_aligned = return_aligned
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'AccelError: Alignment = {self._alignment}'

    def set_root(self, new_root):
        
        if hasattr(self._alignment, 'set_root'):
            self._alignment.set_root(new_root)

    def set_alignment(self, alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment]) -> None:
        self._alignment = alignment_object

    def __call__(self, est_points, gt_points, valid_timesteps=None):
        aligned_est_points, aligned_gt_points = self._alignment(est_points, gt_points)

        err = compute_error_accel(joints_gt=aligned_gt_points, joints_pred=aligned_est_points , vis=valid_timesteps)
        
        if self._return_aligned:
            return err, aligned_est_points, aligned_gt_points
        else:
            return err     


class PointError(object):
    def __init__(self,
                 alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment],
                 name: str = '',
                 return_aligned=False) -> None:
        
        super(PointError, self).__init__()
        self._alignment = alignment_object
        self._return_aligned = return_aligned
        self._name = name

    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'PointError: Alignment = {self._alignment}'

    def set_root(self, new_root):
        
        if hasattr(self._alignment, 'set_root'):
            self._alignment.set_root(new_root)

    def set_alignment(self, alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment]) -> None:
        self._alignment = alignment_object

    def __call__(self, est_points, gt_points, weighted_error=False):
        aligned_est_points, aligned_gt_points = self._alignment(est_points, gt_points)
        # vis_mano_skeleton_matplotlib({"joints_3d": aligned_est_points}, {"joints_3d": aligned_gt_points})
        
        if weighted_error:
            err = weighted_point_error(aligned_est_points, aligned_gt_points)
        else:
            err = point_error(aligned_est_points, aligned_gt_points)
            
        if self._return_aligned:
            return err, aligned_est_points, aligned_gt_points
        else:
            return err
        

class FScores(object):
    def __init__(self, thresholds: Array,
                 alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment],
                 name: str = '',
                 return_aligned=False) -> None:
        super(FScores, self).__init__()
        self._alignment = alignment_object
        self._return_aligned = return_aligned
        self._name = name
        self.thresholds = thresholds
        
    @property
    def name(self):
        return self._name

    def __repr__(self):
        return f'PointError: Alignment = {self._alignment}'

    def set_root(self, new_root):
        
        if hasattr(self._alignment, 'set_root'):
            self._alignment.set_root(new_root)
            
    def set_alignment(self, alignment_object: Union[ProcrustesAlignment, RootAlignment, NoAlignment]) -> None:
        self._alignment = alignment_object
    
    def __call__(self, est_points, gt_points):
        
        aligned_est_points, aligned_gt_points = self._alignment(est_points, gt_points)
        
        f_score_dict = point_fscore(pred=aligned_est_points, gt=aligned_gt_points, thresh=self.thresholds) 
            
        if self._return_aligned:
            return f_score_dict, aligned_est_points, aligned_gt_points
        else:
            return f_score_dict
        

class Evaluator(object):
    def __init__(self, output_folder, rank=0, distributed=False):
        super(Evaluator, self).__init__()
     
     
        self.mano_right = BodyModel(model_type="mano", model_path=MANO_RIGHT_MODEL_DIR, device='cuda', 
                        **{"flat_hand_mean":True, "use_pca":False, "is_rhand":True}).model
        
        self.rh_faces = self.mano_right.faces

        self.output_folder = output_folder
        self.np_filepath = None

        self.summary_folder = osp.join(self.output_folder, "quantitative_summary")
        os.makedirs(self.summary_folder, exist_ok=True)
    
        self.results_folder = osp.join(self.output_folder)
        os.makedirs(self.results_folder, exist_ok=True)
        
        self.f_score_thresholds = [5/1000, 15/1000]    

    @torch.no_grad()
    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def _compute_mpjpe(self,
                       model_output,
                       targets: StructureList,
                       metric_align_dicts: Dict,
                       mpjpe_root_joints_names: Optional[Array] = None) -> Dict[str, Array]:
        """ Takes the ground truth joints and predicted joints to compute mean per joint position error. """

        # keypoint annotations.
        gt_joints_3d_indices = np.array(targets["frame_id"])
        gt_joints3d = np.array(targets["joints_3d"]) 
        output = {}
    
        # Get the number of valid instances
        num_instances = len(gt_joints_3d_indices)
        if num_instances < 1:
            return output

        # Get the data from the output of the model. 
        est_joints_np = model_output.get("joints_3d")
             
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [0]
                # root_indices = [target_names.index(name) for name in mpjpe_root_joints_names]
                alignment.set_root(root_indices)
            
            metric_value, aligned_pred_joints3d, aligned_gt_joints3d = alignment(est_joints_np[gt_joints_3d_indices], gt_joints3d)
            
            
            # save them aligned joints. eval() method does not work
            setattr(self, f'{self.source}_{alignment_name}_joints3d_pred', aligned_pred_joints3d)
            setattr(self, f'{self.source}_{alignment_name}_joints3d_gt', aligned_gt_joints3d)  
            self.aligned_data[f'{self.source}_{alignment_name}_joints3d_pred'] = aligned_pred_joints3d
            self.aligned_data[f'{self.source}_{alignment_name}_joints3d_gt'] = aligned_gt_joints3d          
            
            name = f'{alignment_name}_mpjpe'
            # convert to mm
            output[name] = metric_value * 1000
        return output

    @staticmethod
    def _compute_mpjpe14(model_output,
                        targets: StructureList,
                        metric_align_dicts: Dict,
                        J14_regressor: Array,
                        **extra_args) -> Dict[str, Array]:
        output = {}
        gt_joints_3d_indices = np.array([ii for ii, t in enumerate(targets)
             if t.has_field('joints14')], dtype=np.long)
        if len(gt_joints_3d_indices) < 1:
            return output
        # Stack all 3D joint tensors
        gt_joints3d = np.stack(
            [t.get_field('joints14').joints.detach().cpu().numpy()
             for t in targets if t.has_field('joints14')])

        # Get the data from the output of the model
        est_vertices = model_output.get('vertices', None)
        est_vertices_np = est_vertices.detach().cpu().numpy()
        est_joints_np = np.einsum(
            'jv,bvn->bjn', J14_regressor, est_vertices_np)
        for alignment_name, alignment in metric_align_dicts.items():
            metric_value = alignment(est_joints_np[gt_joints_3d_indices], gt_joints3d)
            name = f'{alignment_name}_mpjpe14'
            output[name] = metric_value
        return output

    
    def _compute_v2v(self, 
                    model_output,
                    targets: StructureList,
                    metric_align_dicts: Dict,
                    **extra_args) -> Dict[str, Array]:
        
        
        ''' Computes the Vertex-to-Vertex error for the current input'''
        output = {}
        # Ground truth vertices     
        gt_verts_indices = np.array(targets["frame_id"])
        gt_vertices = np.array(targets["vertices_3d"])

        if len(gt_verts_indices) < 1:
            return output
 
        # Get the data from the output of the model
        est_verts_np = model_output.get("vertices", None)
 
        for alignment_name, alignment in metric_align_dicts.items():
            
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [117]                          # vertex id close to wrist
                alignment.set_root(root_indices)	
             
              # pymafx does not contain vertices for interim timesteps 
            if est_verts_np.shape[0] != gt_verts_indices.shape[0]:
                cast_verts_np = est_verts_np[gt_verts_indices]
            else:    
                cast_verts_np = est_verts_np
                
            metric_value, aligned_pred_vertices, aligned_gt_vertices = alignment(cast_verts_np, gt_vertices)
        
            name = f'{alignment_name}_v2v'
            
            setattr(self, f'{self.source}_{alignment_name}_vertices_pred', aligned_pred_vertices)
            setattr(self, f'{self.source}_{alignment_name}_vertices_gt', aligned_gt_vertices)
            self.aligned_data[f'{self.source}_{alignment_name}_vertices_pred'] = aligned_pred_vertices
            self.aligned_data[f'{self.source}_{alignment_name}_vertices_gt'] = aligned_gt_vertices
                        
            # convert to mm
            output[name] = metric_value * 1000
        return output


    def _compute_2d_mpjpe(self, model_output,         # pred
                          targets: StructureList,     # gt
                          metric_align_dicts: Dict,
                          mpjpe_root_joints_names: Optional[Array] = None): 
        
        # keypoint annotations

        gt_joints_2d_indices = np.array(targets["frame_id"])
        gt_joints2d = np.array(targets["joints_2d"])
             
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_joints_2d_indices)
        if num_instances < 1:
            return output
       
        est_joints_np = model_output.get("keypoints_2d", None)
        
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [0]
                alignment.set_root(root_indices)	
                
            # weigthed mean of the 2d keypoint error. Last column of est_joints_np is the confidence
            metric_value, _, _ = alignment(est_joints_np[gt_joints_2d_indices], gt_joints2d[:, :, :2], weighted_error=True)
            
            name = f'{alignment_name}_mpjpe_2d'
            output[name] = metric_value 

        return output


    def _compute_f_score(self, model_output,          # pred
                          targets: StructureList,     # gt
                          metric_align_dicts: Dict,
                          mpjpe_root_joints_names: Optional[Array] = None): 
        
        # keypoint annotations.
    
        gt_verts_indices = np.array(targets["frame_id"])
        gt_vertices = np.array(targets["vertices_3d"])
        
        
          
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_vertices)
        if num_instances < 1:
            return output
        
        # Get the data from the output of the model. No need to map the estimated joints 
        # to the order used by the ground truth joints. They are already aligned.
        est_verts_np = model_output.get("vertices", None)
   
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [117]    # looked at from meshlab, this is for the vertices of the rigth hand, close to wrist.
                alignment.set_root(root_indices)	
                

            # pymafx does not contain vertices for interim timesteps 
            if est_verts_np.shape[0] != gt_verts_indices.shape[0]:
                cast_verts_np = est_verts_np[gt_verts_indices]
            else:    
                cast_verts_np = est_verts_np

            # weigthed mean of the 2d keypoint error. Last column of est_joints_np is the confidence
            metric_value, _, _ = alignment(cast_verts_np, gt_vertices)
            
            # metric value is in the form of 'f@0.005': {'fscore': x, 'precision': y, 'recall': z}, 
            # 'f@0.015': {'fscore': x_, 'precision': y_, 'recall': z_}}}

            name_f5 = f'{alignment_name}_f@5_score'
            name_f15 = f'{alignment_name}_f@15_score'
    
            output[name_f5] = np.array(metric_value['f@0.005']['fscore'])
            output[name_f15] = np.array(metric_value['f@0.015']['fscore'])

        return output
    
    def _compute_acc_err(self, model_output,            # pred
                            targets: StructureList,     # gt
                            metric_align_dicts: Dict,
                            mpjpe_root_joints_names: Optional[Array] = None):
        
        gt_joints_3d_indices = np.array(targets["frame_id"])
        gt_joints3d = np.array(targets["joints_3d"]) 
        output = {}
    
    
        visibility_arr = np.zeros(targets["num_frames"])
        visibility_arr[gt_joints_3d_indices] = 1
        
        gt_joints_3d_pad = np.zeros((targets["num_frames"], 21, 3))
        gt_joints_3d_pad[gt_joints_3d_indices] = gt_joints3d
        
        
        output = {}
        # Get the number of valid instances
        num_instances = len(gt_joints_3d_indices)
        if num_instances < 1:
            return output
        
        
        # Get the data from the output of the model.
        est_joints_np = model_output.get("joints_3d")
        
        # cut the last parts padded if it is the case 
        if est_joints_np.shape[0] > gt_joints_3d_pad.shape[0]:
            est_joints_np = est_joints_np[:gt_joints_3d_pad.shape[0]]
             
        for alignment_name, alignment in metric_align_dicts.items():
            # Update the root joint for the current dataset
            if hasattr(alignment, 'set_root'):
                root_indices = [0]
                # root_indices = [target_names.index(name) for name in mpjpe_root_joints_names]
                alignment.set_root(root_indices)
            
            metric_value, _, _ = alignment(est_joints_np, gt_joints_3d_pad, visibility_arr)
      
            name = f'{alignment_name}_acc_err'
            # convert to mm
            output[name] = metric_value[:, None] * 1000
    
        return output


    def compute_metric(self, model_output, 
                             targets: StructureList,
                             metrics: Dict,
                             mpjpe_root_joints_names: Optional[Array] = None, 
                             **extra_args,):
        
        self.aligned_data, output_metric_values = {}, {}
        self.source = model_output["source"]
        
        for metric_name, metric in metrics.items():
            if metric_name == 'mpjpe_3d':
                curr_vals = self._compute_mpjpe(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'v2v':
                curr_vals = self._compute_v2v(
                    model_output, targets, metric, **extra_args)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'mpjpe_2d':
                curr_vals = self._compute_2d_mpjpe(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'f_score':
                curr_vals = self._compute_f_score(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
            elif metric_name == 'acc_err':                
                curr_vals = self._compute_acc_err(
                    model_output, targets, metric,
                    mpjpe_root_joints_names=mpjpe_root_joints_names)
                for key, val in curr_vals.items():
                    output_metric_values[key] = val
                
            else:
                raise ValueError(f'Unsupported metric: {metric_name}')
        
        self.aligned_data["frame_id"] = targets["frame_id"]
        self.np_filepath = os.path.join(os.path.dirname(str(model_output["save_path"])), 'aligned_data.npz')
       
        self.aligned_data["img_width"] = model_output["img_width"]
        self.aligned_data["img_height"] = model_output["img_height"]
        
        if type(model_output["cam_f"]) == torch.Tensor:
            self.aligned_data["cam_f"] = model_output["cam_f"].cpu().numpy()
            self.aligned_data["cam_R"] = model_output["cam_R"].cpu().numpy()
            self.aligned_data["cam_t"] = model_output["cam_t"].cpu().numpy()
        else:    
            self.aligned_data["cam_f"] = model_output["cam_f"]
            self.aligned_data["cam_R"] = model_output["cam_R"]
            self.aligned_data["cam_t"] = model_output["cam_t"]

        # Save both initial stage or hmp results to a dict. If the file exists, overwrite the values. 
        if os.path.isfile(self.np_filepath):
            npz_dict = dict(np.load(self.np_filepath, allow_pickle=True))
            for key in self.aligned_data.keys():
                npz_dict[key] = self.aligned_data[key]  
            np.savez(self.np_filepath, **npz_dict)
        else:
            # save all aligned values to a dict.   
            np.savez(self.np_filepath, **self.aligned_data)    
            
        output_metric_values["valid_frames"] = len(targets["frame_id"])

        return output_metric_values
    
    # this method is for visualization of the aligned joints and vertices.
    def create_canvas_scenepic(self, img_width, img_height, cam_intrinsics, save_dir, alignment_type, valid_indices=None, img_dir=None):
    
        scene = sp.Scene()
        canvas = scene.create_canvas_3d(width=img_width, height=img_height)
        
        assert alignment_type in ['no', 'procrustes', 'root', 'all']
        rh_joints = getattr(self, f'{self.source}_{alignment_type}_align_joints3d_pred')
        rh_joints_gt = getattr(self, f'{self.source}_{alignment_type}_align_joints3d_gt')
        
        # load corresponding joints and vertices 
        try:
            rh_verts = getattr(self, f'{self.source}_{alignment_type}_align_vertices_pred')
            rh_verts_gt = getattr(self, f'{self.source}_{alignment_type}_align_vertices_gt')
        except:
            rh_verts = None
            rh_verts_gt = None
        
        B = rh_joints.shape[0]

        cam_R = np.repeat(np.identity(3)[None, ...], B, axis=0).reshape((-1,3,3))
        cam_t = np.zeros((B, 3))

        coord_ax = scene.create_mesh(layer_id="coord")
        coord_ax.add_coordinate_axes()

        label = scene.create_label(text=alignment_type, color=sp.Colors.White, size_in_pixels=80, offset_distance=0.0, camera_space=True)
        
        in_joint_pos = None
        
    
        gt_cam_lines = []
        for j in range(B):      
            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3,:3] = cam_R[j] 
            cam_extrinsics[:3,3] = cam_t[j] 

      
        # compatible with DexYCB
        sorted_images = sorted(glob.glob(f'{img_dir}/*.jpg'))
        if len(sorted_images)== 0:
            sorted_images = sorted(glob.glob(f'{img_dir}/*.png'))

        for i in range(B):
            frame = canvas.create_frame()
            
            rh_mesh = scene.create_mesh(shared_color=(0.7, 0.7, 0.7), layer_id="rh_mesh")
            rh_gt_mesh = scene.create_mesh(shared_color=(0.7, 0.2, 0.2), layer_id="rh_gt_mesh")
            
            if rh_verts is not None:
                rh_mesh.add_mesh_without_normals(rh_verts[i], self.rh_faces)
                frame.add_mesh(rh_mesh)
                
            if rh_verts_gt is not None:
                rh_gt_mesh.add_mesh_without_normals(rh_verts_gt[i], self.rh_faces)
                frame.add_mesh(rh_gt_mesh)     
                
            
            if rh_joints is not None:
                joints_mesh = scene.create_mesh(shared_color = sp.Color(1.0, 0.0, 0.0), layer_id="joints")
                joint_colors = np.zeros_like(rh_joints[i])
                joint_colors[:, 0] = 1.0
                joints_mesh.add_sphere(transform = sp.Transforms.Scale(0.005))
                joints_mesh.enable_instancing(positions=rh_joints[i], colors = joint_colors)
            
            if rh_joints_gt is not None:
                joints_mesh_gt = scene.create_mesh(shared_color = sp.Color(0.0, 0.0, 1.0), layer_id="joints_gt")
                joint_colors_gt = np.zeros_like(rh_joints_gt[i])
                joint_colors_gt[:, 2] = 1.0
                joints_mesh_gt.add_sphere(transform = sp.Transforms.Scale(0.005))
                joints_mesh_gt.enable_instancing(positions=rh_joints_gt[i], colors = joint_colors_gt)
                            
            for j, pa in enumerate(AUGMENTED_MANO_CHAIN):
                if pa >= 0:
                    bone_mesh = scene.create_mesh(shared_color=(1.0, 0.0, 0.0), layer_id="bones")
                    bone_mesh.add_lines(rh_joints[i][j][None], rh_joints[i][pa][None])
                    frame.add_mesh(bone_mesh)
                    
                    bone_mesh_gt = scene.create_mesh(shared_color=(0.0, 0.0, 1.0), layer_id="bones_gt")
                    bone_mesh_gt.add_lines(rh_joints_gt[i][j][None], rh_joints_gt[i][pa][None])
                    frame.add_mesh(bone_mesh_gt)   
                    
            if in_joint_pos is not None:
                in_joints_mesh = scene.create_mesh(shared_color = sp.Color(64./255., 224./255., 208./255.), layer_id="in_joints")
                in_joints_mesh.add_sphere(transform = sp.Transforms.Scale(0.005))
                in_joints_mesh.enable_instancing(positions=in_joint_pos[i])
                
                frame.add_mesh(in_joints_mesh)
                
                for j, pa in enumerate(AUGMENTED_MANO_CHAIN):
                    if pa >= 0:
                        in_bone_mesh = scene.create_mesh(shared_color=(0.5, 0.0, 0.5), layer_id="in_joints")
                        in_bone_mesh.add_lines(in_joint_pos[i][j][None], in_joint_pos[i][pa][None])
                        frame.add_mesh(in_bone_mesh)
        
            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3,:3] = cam_R[i] # smpl_seq["cam_rot"][i]
            cam_extrinsics[:3, 3] = cam_t[i] # smpl_seq["cam_trans"][i]
            cam_extrinsics[1:3] *= -1
            


            try:
                cam_dataset = self.load_camera_from_ext_int(cam_extrinsics, cam_intrinsics)
            except:
                raise ValueError("Could not load camera")
            
            if img_dir is not None:
                
                val_ind = valid_indices[i]
                
                img_length = len(sorted_images)
                image_index = val_ind if val_ind < img_length else img_length-1
                
                cam_image = scene.create_image(sorted_images[image_index])
                full_img = False   
                
                try:
                    if full_img:
                        cam_image.load(sorted_images[image_index])
                    else:
                        cvimg = cv2.imread(sorted_images[image_index])
                        cvimg = cv2.resize(cvimg, (cvimg.shape[0]//2, cvimg.shape[1]//2))
                        tmimg = f'/tmp/{next(tempfile._get_candidate_names())}.jpg'
                        cv2.imwrite(tmimg, cvimg)
                        
                        cam_image.load(tmimg)
                        os.remove(tmimg)
                
                    image_mesh = scene.create_mesh(f"image_{i:02d}",
                                                layer_id="images",
                                                shared_color=sp.Colors.Gray,
                                                double_sided=True,
                                                texture_id=cam_image.image_id)
                    
                    image_mesh.add_camera_image(cam_dataset, 20.0)
                    frame.add_mesh(image_mesh)
                except:
                    print(f"Could not load image {i+1:06d}.png")
            
            frame.camera = cam_dataset
            
            if rh_joints_gt is not None:
                frame.add_mesh(joints_mesh_gt)
            if rh_joints is not None:
                frame.add_mesh(joints_mesh)
                
            # frame.add_mesh(coord_ax)
            frame.add_label(label=label, position=[-1.0, 1.0, -5.0])
        
            for m_id in range(len(gt_cam_lines)):
                frame.add_mesh(gt_cam_lines[m_id])
        
        scene.link_canvas_events(*[canvas])
        scene.framerate = 30
        scene.save_as_html(save_dir, title=save_dir)
        # canvas.set_layer_settings(layer_settings)
        
        return canvas
    
    # renew it every 2^16 time
    def create_renderer_object(self, img_width, img_height, cam_intrinsics, cam_extrinsics):
        
        if hasattr(self, 'render'):
            del self.render
            # print('deleting renderer object')
        
        self.render = rendering.OffscreenRenderer(width=self.window_size[0], height=self.window_size[1])
            
        self.render.scene.view.set_ambient_occlusion(True)
        self.render.scene.view.set_antialiasing(True)
        self.render.scene.view.set_shadowing(True)
        self.render.scene.view.set_post_processing(False)
        self.render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
        self.render.scene.scene.enable_sun_light(True)
        self.render.scene.set_background((1., 1., 1., 1.))     # invoking second time causes segmentation fault
        
        o3d_pinhole_cam = o3d.camera.PinholeCameraIntrinsic(int(img_width.item()), int(img_height.item()), cam_intrinsics)
        self.render.setup_camera(o3d_pinhole_cam, cam_extrinsics.astype(np.float64))
        
        return 
            
    def create_canvas_open3d(self, img_width, img_height, cam_intrinsics, video_path, alignment_type, 
                             valid_indices=None, 
                             img_dir=None,
                             fps=30,
                             frame_dir=None,
                             cam_eye=(0.0, 0.0, 1.5),
                             cam_look_at=(0.0, -1.0, 0.0),
                             cam_up=(0.0, 0.0, 1.0),
                             hr=2.0):
        
        mano_verts_list, mano_joints_list = [], []
        
        assert alignment_type in ['no', 'procrustes', 'root', 'all']
        rh_joints = getattr(self, f'{self.source}_{alignment_type}_align_joints3d_pred')
        rh_joints_gt = getattr(self, f'{self.source}_{alignment_type}_align_joints3d_gt')
        
        mano_joints_list.append(rh_joints)
        mano_joints_list.append(rh_joints_gt)
        
        # load corresponding joints and vertices 
        try:
            rh_verts = getattr(self, f'{self.source}_{alignment_type}_align_vertices_pred')
            rh_verts_gt = getattr(self, f'{self.source}_{alignment_type}_align_vertices_gt')
            
            mano_verts_list.append(rh_verts)
            mano_verts_list.append(rh_verts_gt)
            
        except:
            rh_verts = None
            rh_verts_gt = None
        
        imgfnames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        mesh_colors = {0: np.array([0, 0, 255]),           # gt 
                        1: np.array([0, 255, 0])}          # regressed
        mesh_colors = [v / 255. for k, v in mesh_colors.items()]
        alpha_val = 0.9
                
        B = rh_joints.shape[0]
        cam_extrinsics = np.eye(4)
        cam_extrinsics[:3,:3] = np.identity(3)
        cam_extrinsics[:3,3] = np.zeros((1, 3))
        
        self.window_size = (int(img_width.item()), int(img_height.item()))
        
        
        self.create_renderer_object(img_width, img_height, cam_intrinsics, cam_extrinsics)
          
        if osp.exists(frame_dir):
            shutil.rmtree(frame_dir)
        os.makedirs(frame_dir)

        mat_gt = rendering.MaterialRecord()
        mat_gt.shader = "defaultUnlit"
        mat_gt.base_color = [*mesh_colors[0].tolist(), 1.0] 
        
        bone_mat_gt = rendering.MaterialRecord()
        bone_mat_gt.shader = "unlitLine"
        bone_mat_gt.base_color = [*mesh_colors[0].tolist(), 1.0] 
        
        mat_opt = rendering.MaterialRecord()
        mat_opt.shader = "defaultUnlit"
        mat_opt.base_color = [*mesh_colors[1].tolist(), 1.0] 
        
        bone_mat_opt = rendering.MaterialRecord()
        bone_mat_opt.shader = "unlitLine"
        bone_mat_opt.base_color = [*mesh_colors[1].tolist(), 1.0]

        for j, val_ind in enumerate(tqdm(valid_indices)): 
            
            for i, pa in enumerate(AUGMENTED_MANO_CHAIN):
                # indicate joints through spheres
                joint_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                joint_sphere.compute_vertex_normals()
                # joint_sphere.paint_uniform_color([1.0, 0.3, 0.3])
                
                # regressed joints
                jts_transformation = np.identity(4)        
                jts_transformation[:3, 3] = rh_joints[j][i]
                joint_sphere.transform(jts_transformation)
                
                joint_sphere_gt = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                joint_sphere_gt.compute_vertex_normals()
                # joint_sphere_gt.paint_uniform_color([0.3, 0.3, 1.0])
                
                # gt joints (select by j)
                jts_transformation_gt = np.identity(4)        
                jts_transformation_gt[:3, 3] = rh_joints_gt[j][i]
                joint_sphere_gt.transform(jts_transformation_gt)

                
                # add joints for timestep j
                if self.render.scene.has_geometry(f'joint_{i}'):
                    self.render.scene.remove_geometry(f'joint_{i}')
                self.render.scene.add_geometry(f'joint_{i}', geometry=joint_sphere, material=mat_opt)
                if self.render.scene.has_geometry(f'joint_gt_{i}'):
                    self.render.scene.remove_geometry(f'joint_gt_{i}')
                self.render.scene.add_geometry(f'joint_gt_{i}', geometry=joint_sphere_gt, material=mat_gt)
                
                if pa == -1:
                    continue
                
            # indicate bones through lineset
            lineset_opt = o3d.geometry.LineSet()
            lineset_opt.points = o3d.utility.Vector3dVector(rh_joints[j])
            lineset_opt.lines = o3d.utility.Vector2iVector(get_mano_skeleton())
            
            if self.render.scene.has_geometry('bone_opt'):
                self.render.scene.remove_geometry('bone_opt')
            self.render.scene.add_geometry(geometry=lineset_opt, name='bone_opt', material=bone_mat_opt)
            
            lineset_gt = o3d.geometry.LineSet()
            lineset_gt.points = o3d.utility.Vector3dVector(rh_joints_gt[j])
            lineset_gt.lines = o3d.utility.Vector2iVector(get_mano_skeleton())
            
            if self.render.scene.has_geometry('bone_gt'):
                self.render.scene.remove_geometry('bone_gt')
            self.render.scene.add_geometry(geometry=lineset_gt, name='bone_gt', material=bone_mat_gt)
            
            # j in gt corresponds to val_ind in images             
            img_file_j = np.asarray(o3d.io.read_image(imgfnames[val_ind]))
            
            img_path = f'{frame_dir}/{j:06d}.jpg'
            
            # renew it every 1000 timestep at max. add_geometry() causes memory leak if it is called 2^16 time. 
            if j % 1000 == 0:
                self.create_renderer_object(img_width, img_height, cam_intrinsics, cam_extrinsics)

            alignment_img_rgb = self.render.render_to_image()
 
            valid_mask = (np.sum(alignment_img_rgb, axis=-1) < 765)[:, :, np.newaxis]
            
            # blend the two images through masking alpha blending
            img_overlay = alignment_img_rgb * valid_mask * alpha_val + img_file_j * (1 - alpha_val) * valid_mask + img_file_j * (1 - valid_mask)
            
            img_overlay = o3d.geometry.Image((img_overlay).astype(np.uint8))
            o3d.io.write_image(img_path, img_overlay)
    
        images_to_video(frame_dir, video_path, fps=fps, crf=25, verbose=False)    
        return 
    
    def load_camera_from_ext_int(self, extrinsics, intrinsics):
    
        img_width = intrinsics[0, 2] * 2
        img_height = intrinsics[1, 2] * 2
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        
        vfov = 2. * np.arctan(img_height / (2. * fy))
        
        world_to_camera = extrinsics  
        aspect_ratio = img_width / img_height
        projection = sp.Transforms.gl_projection(np.degrees(vfov), aspect_ratio, 0.01, 100)

        return sp.Camera(world_to_camera, projection)