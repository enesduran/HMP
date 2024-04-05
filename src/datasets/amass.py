# -*- coding: utf-8 -*-
# Copyright (C) 2024 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
# If you use this code in a research publication please consider citing the following:
#
# HMP: Hand Motion Priors for Pose and Shape Estimation from Video (https://hmp.is.tue.mpg.de/)

import os
import cv2
import sys
import glob
import yaml
import copy
import json
import torch
import pickle
import numpy as np
from tqdm import tqdm
import os.path as osp	    
from tqdm import tqdm 
from datetime import datetime
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(file_path, '..'))

import holden.BVH as BVH
from arguments import Arguments
from body_model.mano import BodyModel
from holden.Animation import Animation
from utils import ho3d_cam_extrinsics
from nemf.fk import ForwardKinematicsLayer
from holden.Quaternions import Quaternions
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import log2file, makepath
from kornia_transform import quaternion_to_angle_axis, rotation_about_x, rotation_about_z, slam_to_opencv
from utils import align_joints, build_canonical_frame, estimate_angular_velocity, estimate_linear_velocity, normalize
from rotations import axis_angle_to_matrix, axis_angle_to_quaternion, matrix_to_rotation_6d, rotation_6d_to_matrix, batch_rodrigues
from fitting_utils import map_mano_joints_to_openpose, map_openpose_joints_to_mano, map_mano_joints_to_openpose, convert_pred_to_full_img_cam, RIGHT_WRIST_BASE_LOC

J = 16
openpose_skeleton = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
augmented_mano_skeleton = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]

openpose2mano = map_openpose_joints_to_mano()
mano2openpose = map_mano_joints_to_openpose()

MANO_RH_DIR = "./data/body_models/mano/MANO_RIGHT.pkl" 

def expand_bbox(_bbox_, image_width, image_height, EXPAND_COEF=1.2):
    ''' _bbox_ is a np array of shape N, 5 (x1, y1, x2, y2, score)
    '''
    # expand the bbox by a factor of coef
    center = ((_bbox_[:, 0]+_bbox_[:, 2])/2, (_bbox_[:, 1]+_bbox_[:, 3])/2)
    
    assert (center[0] < image_width).all() & (center[0] >= 0).all(), "Problem with image width"
    assert (center[1] < image_height).all() and (center[1] >= 0).all(), "Problem with image height"
    
    width = abs(_bbox_[:, 2] - _bbox_[:, 0])
    height = abs(_bbox_[:, 3] - _bbox_[:, 1])

    new_width = width * EXPAND_COEF
    new_height = height * EXPAND_COEF
    
 
    x1 = np.maximum(np.zeros_like(center[0]), center[0] - new_width/2).astype(np.int32)[:, None]
    x2 = np.minimum(image_width, center[0] + new_width/2).astype(np.int32)[:, None]
    y1 = np.maximum(np.zeros_like(center[1]), center[1] - new_height/2).astype(np.int32)[:, None]
    y2 = np.minimum(image_height, center[1] + new_height/2).astype(np.int32)[:, None]
    
    return np.concatenate([x1, y1, x2, y2], axis=1)

def get_mano_gt(trans_, pose_, betas):
    
    
    hand_model = BodyModel(model_type="mano", model_path=MANO_RH_DIR, device='cuda', 
                           **{"flat_hand_mean":True, "use_pca":False, "batch_size":pose_.shape[0], "is_rhand":True}).to()
    mano_motion = hand_model(input_dict={"betas":betas,
                            "global_orient":pose_[:, 0,  :3].view(-1, 3),
                            "hand_pose":pose_[:, 1:, :].view(-1, 45),
                            "no_shift":True,
                            "return_finger_tips": True,
                            "transl":trans_.view(-1, 3)})
    return mano_motion.joints, mano_motion.vertices  


def reflect_gt(gt_dict): 
    
    # reflect img & obtain right hand image 
    img_width = gt_dict["img_width"]
    img_height = gt_dict["img_height"]
    
    trans_reflection_array = np.array([-1, 1, 1])
    pos_reflection_array = np.array([-1, 1, 1] * 21)
    pose_reflection_array = np.array([1, -1, -1] * 16)

    # create path for the pseudo right hand image
    pseudo_right_path = os.path.join(os.path.dirname(gt_dict["path"]), "rgb_pseudo_right") 
    os.makedirs(pseudo_right_path, exist_ok=True)
    
    reflected_bbox, reflected_joints2d, reflected_pose, \
            reflected_joints3d, reflected_trans = [], [], [], [], []
    
    # reflect gt values for a left hand.
    for i, rgb_im_path in enumerate(sorted(glob.glob(os.path.join(gt_dict["path"], "*jpg")))): 
        left_img = cv2.imread(rgb_im_path)
        pse_right_img = cv2.flip(left_img, 1)        
        
        if i in gt_dict["frame_id"]:    
            # find corresponding id 
            real_i = gt_dict["frame_id"].index(i)
            
            # find corresponding bbox, joints2d, and joints3d 
            bbox_real_i = gt_dict["bbox"][real_i]
            joints2d_real_i = gt_dict["joints_2d"][real_i]
            joints3d_real_i = gt_dict["joints_3d"][real_i]
            trans_real_i = gt_dict["trans"][real_i]
            pose_real_i = gt_dict["poses"][real_i]
            
            joints2d_real_i[:, 0] = img_width - joints2d_real_i[:, 0]
            reflected_joints2d.append(joints2d_real_i)
            
            joints3d_real_i = joints3d_real_i * pos_reflection_array.reshape(-1, 3)
            reflected_joints3d.append(joints3d_real_i)
            
            trans_real_i = trans_real_i * trans_reflection_array
            reflected_trans.append(trans_real_i)
            
            pose_real_i = pose_real_i * pose_reflection_array
            reflected_pose.append(pose_real_i)
                        
            if bbox_real_i is None:
                reflected_bbox.append(None)
                
            else:
                bbox_real_i[0] = img_width - bbox_real_i[0]
                bbox_real_i[2] = img_width - bbox_real_i[2]
                
                # arrange again 
                if bbox_real_i[0] > bbox_real_i[2]:
                    bbox_real_i[0], bbox_real_i[2] = bbox_real_i[2], bbox_real_i[0] 
                
                reflected_bbox.append(bbox_real_i)
                        
        # write image 
        cv2.imwrite(os.path.join(pseudo_right_path, os.path.basename(rgb_im_path)), pse_right_img)  
        
    gt_dict["bbox"] = reflected_bbox  
    gt_dict["joints_2d"] = reflected_joints2d  
    gt_dict["joints_3d"] = reflected_joints3d  
    gt_dict["path"] = pseudo_right_path
    gt_dict["trans"] = reflected_trans
    gt_dict["poses"] = reflected_pose
    
    pseudo_vid_path = os.path.join(os.path.dirname(pseudo_right_path), "rgb_pseudo_raw.mp4")
    os.system(f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{pseudo_right_path}/*.jpg' -vcodec libx264 -pix_fmt yuv420p {pseudo_vid_path}")
    
     
def render_gt(rgb_img_list, gt_dict, rgb_images_path, circle_rad=2, line_width=2):
    
    for i, img_rgb in enumerate(tqdm(rgb_img_list)):
    
            _img = Image.open(img_rgb)
            draw = ImageDraw.Draw(_img)
                
            # check if there is a corresponding gt for this frame
            if i in gt_dict["frame_id"]:
                
                real_i = gt_dict["frame_id"].index(i)
                joints2d = gt_dict["joints_2d"][real_i]
                
                # draw bbox
                bb = gt_dict["bbox"][real_i]
                if bb is not None:
                    draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], outline=(0, 255, 0), width=line_width)  

                for k in range(joints2d.shape[0]):                  
                    kps_parent = joints2d[augmented_mano_skeleton[k]]
                    kps_child = joints2d[k]
                    
                    if (kps_parent == [-1, -1]).all() or (kps_child == [-1, -1]).all():
                        continue

                    if augmented_mano_skeleton[k] != -1:
                        draw.line([(kps_child[0], kps_child[1]), (kps_parent[0], kps_parent[1])], fill=(0, 0, 200), width=line_width)
                    
                    draw.ellipse((joints2d[k][0]-circle_rad, joints2d[k][1]-circle_rad, joints2d[k][0]+circle_rad, joints2d[k][1]+circle_rad), fill=(200, 0, 0))
            
            os.makedirs(rgb_images_path + "_gt", exist_ok=True)
            _img.save(rgb_images_path + f"_gt/{i:04d}.png")
    return 


def form_bbox(jts2d, bbox_coef, img_dim):
    return 

def read_camera_intrinsics(dataname, datapath=None):

    assert dataname in ["HO3D", "DexYCB", "other"]
    if dataname == "HO3D":
        return torch.tensor([[615, 615]])
    elif dataname == "DexYCB":
        return torch.tensor([[615, 615]])
    else:
        return torch.tensor([[1060, 1060]])

def get_dexycb_gt(rgb_images_path, render=True):

    # drop suffix
    if "_mmpose_vid" in rgb_images_path or "_rgb_mediapipe" in rgb_images_path:
        rgb_images_path = os.path.join(os.path.dirname(rgb_images_path), os.path.basename(rgb_images_path).split("_")[0])
        
    # initialize calibration path
    dexycb_calibration_path = "./data/rgb_data/DexYCB/calibration"
    meta_yaml_path = os.path.join(os.path.dirname(os.path.dirname(rgb_images_path)), "meta.yml") 
    gt_label_paths = sorted(glob.glob(os.path.join(os.path.dirname(rgb_images_path), "labels", "labels_*.npz")))

    cam_id = os.path.basename(os.path.dirname(rgb_images_path))
    
    img_height, img_width, _ = cv2.imread(glob.glob(os.path.join(rgb_images_path, "*.jpg"))[0]).shape
    
    # read meta yaml file
    meta_dict = yaml.load(open(meta_yaml_path, "r"), Loader=yaml.FullLoader)
    cam_serial = meta_dict["serials"]
    pcnn_init = meta_dict["pcnn_init"]
    num_frames = meta_dict["num_frames"]
    handedness = meta_dict["mano_sides"][0]
    assert handedness in ["left", "right"], "invalid handedness"
  
    assert len(gt_label_paths) == num_frames, f"Something is wrong with the number of frames {len(gt_label_paths)}, {num_frames}"
        
    cam_extrinsics_path = os.path.join(dexycb_calibration_path, "extrinsics_" + meta_dict["extrinsics"], "extrinsics.yml")
    cam_extrinsics_dict = yaml.load(open(cam_extrinsics_path, "r"), Loader=yaml.FullLoader)
    
    cam_intrinsics_path = os.path.join(dexycb_calibration_path, "intrinsics", cam_id + "_640x480.yml")
    cam_intrinsics_dict = yaml.load(open(cam_intrinsics_path, "r"), Loader=yaml.FullLoader)
    
    # color camera intrinsics
    K_matrix = np.zeros((3, 3))
    K_matrix[0, 0] = cam_intrinsics_dict["color"]["fx"]
    K_matrix[0, 2] = cam_intrinsics_dict["color"]["ppx"]
    K_matrix[1, 1] = cam_intrinsics_dict["color"]["fy"]
    K_matrix[1, 2] = cam_intrinsics_dict["color"]["ppy"]
    K_matrix[2, 2] = 1
    
    R_t = np.array(cam_extrinsics_dict["extrinsics"][cam_id]).reshape(3, 4)
     
    gt_dict = {"joints_3d": [],
                "joints_2d": [],
                "betas": [],
                "poses": [],
                "bbox": [],
                "frame_id": [],
                "trans": [], 
                "num_frames": num_frames, 
                "img_width": img_width,
                "img_height": img_height,
                "path": rgb_images_path,
                "dataset_name": "DexYCB",
                "cam_intrinsics" : K_matrix, 
                "handedness": handedness}
        
    beta_yml_path = os.path.join(dexycb_calibration_path, "mano_" + meta_dict["mano_calib"][0], "mano.yml")    
    beta_yaml_file = yaml.load(open(beta_yml_path, "r"), Loader=yaml.FullLoader)

    mano_betas = np.array(beta_yaml_file["betas"])
    at_least_one_gt = False

    # we may exclude meaningless gt frames 
    for i, gt_path in enumerate(tqdm(gt_label_paths)):
        
        gt_file_i = np.load(gt_path)
        bbox_i = None 
    
        seg_mask = gt_file_i["seg"]
        hand_mask = np.argwhere(seg_mask == 255)
 
        # if joints are equal to -1. Then dont include them because they are not visible.  
        if (not np.all(gt_file_i["joint_3d"] == -1)) and len(hand_mask) > 0:
    
            y_min, y_max = hand_mask[:, 0].min(), hand_mask[:, 0].max()
            x_min, x_max = hand_mask[:, 1].min(), hand_mask[:, 1].max()
            
            delta_x = x_max - x_min
            delta_y = y_max - y_min
            
            # this means an invalid bounding box.
            if delta_x < 1 or delta_y < 1:
                continue
            
            gt_dict["frame_id"].append(i)
            gt_dict["betas"].append(mano_betas)
            at_least_one_gt = True
            
            # format is [x, y, x2, y2]
            bbox_i = np.array([x_min, y_min, x_max, y_max, 1.0])           
            
            assert gt_file_i["joint_2d"].shape == (1, 21, 2), "Something is wrong with the number of joints"
            assert gt_file_i["joint_3d"].shape == (1, 21, 3), "Something is wrong with the number of joints"
            
            # check if the projection holds
            proj_joints = (K_matrix @ gt_file_i["joint_3d"][0][openpose2mano].T).T
            proj_joints = proj_joints[:, :2] / proj_joints[:, 2:]

            # there are some joints equal to -1. We should exclude them before having this assertion statement.
            # assert np.allclose(proj_joints, gt_file_i["joint_2d"][0][openpose2mano], r_tol=1e-4, atol=1e-4), print(((proj_joints - gt_file_i["joint_2d"][0][openpose2mano])**2).mean()) 
            gt_dict["joints_3d"].append(gt_file_i["joint_3d"][0][openpose2mano])
            gt_dict["joints_2d"].append(gt_file_i["joint_2d"][0][openpose2mano])
            gt_dict["bbox"].append(bbox_i)
            
            # first 48 entry is the pose, Last 3 is for translation 
            gt_dict["poses"].append(gt_file_i["pose_m"][0, 0:48])
            gt_dict["trans"].append(gt_file_i[ "pose_m"][0, 48:])

    if handedness == "left":
        reflect_gt(gt_dict)    

    if at_least_one_gt:
        # get gt vertices        
        jts3d, vert3d = get_mano_gt(trans_=torch.tensor(np.array(gt_dict["trans"]), dtype=torch.float).to("cuda"), 
                    pose_=torch.tensor(np.array(gt_dict["poses"]), dtype=torch.float).reshape(-1, 16, 3).to("cuda"), 
                    betas=np.array(gt_dict["betas"]))

        gt_dict["vertices_3d"] = vert3d.cpu().numpy()

    rgb_img_list = sorted(glob.glob(os.path.join(gt_dict["path"], "color_*.jpg")))

    
    # plot gt joints. if no gt is available for this frame, then implicitly skip it 
    if render and at_least_one_gt:       
        render_gt(rgb_img_list, gt_dict, rgb_images_path, circle_rad=2, line_width=2)        
    
    return gt_dict

def get_in_the_wild_gt(gt_path, render=False):
    return {"frame_id": [],
            "poses": None,
            "handedness": "right"}
    
    
def get_arctic_data_gt(gt_path, render=False):
    
    # what is the split 
    gt_dict_split = np.load(gt_path["split"], allow_pickle=True).item()

    subject_id, seqname, view_id = gt_path["gt_path"].split("/")[-4:-1]
    view_id = int(view_id)
    
    keyval = f"{subject_id}/{seqname}"
    read_dict = gt_dict_split["data_dict"][keyval]
    
    gt_dict = {}
    
    # if valid 
    gt_dict["frame_id"] = np.nonzero(read_dict["cam_coord"]["right_valid"][:, view_id])[0].tolist()
    
    gt_dict["joints_2d"] = read_dict["2d"]["joints.right"][gt_dict["frame_id"], view_id]
    gt_dict["joints_3d"] = read_dict["cam_coord"]["joints.right"][gt_dict["frame_id"], view_id]
    gt_dict["num_frames"] = read_dict["cam_coord"]["right_valid"][:, view_id].shape[0]
    gt_dict["handedness"] = 'right'
   

    min_x, max_x = gt_dict["joints_2d"][:, :, 0].min(1), gt_dict["joints_2d"][:, :, 0].max(1)
    min_y, max_y = gt_dict["joints_2d"][:, :, 1].min(1), gt_dict["joints_2d"][:, :, 1].max(1)
 
    # concatenate to form bbox as well as confidence values. 
    gt_dict["bbox"] = np.concatenate([min_x[..., None], 
                            min_y[..., None], 
                            max_x[..., None], 
                            max_y[..., None]], axis=1).astype(np.int32)
                            
                            
 
    # read metadata
    with open(gt_path["meta_path"], "r") as f:
        meta_dict = json.load(f)

    
    gt_dict["cam_intrinsics"] = np.array(meta_dict[subject_id]["intris_mat"][view_id])
    gt_dict["cam_extrinsics"] = np.array(meta_dict[subject_id]["world2cam"][view_id])
    gt_dict["image_width"] = meta_dict[subject_id]["image_size"][view_id][0]
    gt_dict["image_height"] = meta_dict[subject_id]["image_size"][view_id][1]
    gt_dict["vertices"] = 0 

    
    # expand bbox
    gt_dict["bbox"] = expand_bbox(gt_dict["bbox"], 
                                image_width=gt_dict["image_width"], 
                                image_height=gt_dict["image_height"], 
                                EXPAND_COEF=1.2)
    gt_dict["bbox"] = np.concatenate([gt_dict["bbox"],
                                    np.ones_like(min_x)[..., None]], axis=1)
    
    return gt_dict
    
def get_ho3d_v3_gt(gt_path, render=False):
    
    valid_timesteps_train = "./data/rgb_data/HO3D_v3/train.txt" 
    valid_timesteps_eval = "./data/rgb_data/HO3D_v3/evaluation.txt" 

    valid_timesteps_path = valid_timesteps_train if "train" in gt_path else valid_timesteps_eval
    valid_timesteps = open(valid_timesteps_path).read()

    seqname = gt_path.split("/")[-2]
    segmentation_path = None
    
    rgb_img_list = sorted(glob.glob(gt_path.replace("meta", "rgb")+ "/*.jpg"))
    rgb_images_path = gt_path.replace("meta", "rgb")
    img_height, img_width, _ = cv2.imread(rgb_img_list[0]).shape
    
    if "train" in gt_path:
        gt_path.replace("train", "calibration").replace("/meta", "")
        split_type = "train"
        segmentation_path = f"./data/rgb_data/HO3D_v3/HO3D_v3_segmentations_rendered/{split_type}/{seqname}/seg"
        segmentation_imgs = sorted(glob.glob(segmentation_path + "/*.png"))

    elif "evaluation" in gt_path:
        gt_path.replace("evaluation", "calibration").replace("/meta", "")
        split_type = "evaluation"
    else:
        split_type = None
        raise ValueError("Something is wrong with the gt path")
    
    pkl_files = sorted(glob.glob(gt_path + "/*.pkl"))
    bs = len(pkl_files)
    
    gt_dict = {"joints_3d": [],
                "joints_2d": [],
                "betas": [],
                "poses":  [],
                "bbox": [],
                "frame_id": [],
                "num_frames": bs, 
                "trans": [],
                "handedness": "right"}
 
    # HO3D has the same joint ordering with MANO. No need to reorder.
    coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
   
    print(f"\nREADING GT: {gt_path}")
    
    cam_intrinsic_seq = None
   
    for i, pkl_file in enumerate(tqdm(pkl_files)):
        
        pkl_data = pickle.load(open(pkl_file, "rb"))
        cam_intrinsic = pkl_data["camMat"]

        timestep_id = seqname + f"/{i:04d}" 
        if timestep_id not in valid_timesteps:
            continue
        else:
            gt_dict["frame_id"].append(i)

        # there are some Nans in camMat as well 
        if (cam_intrinsic_seq is None) and (cam_intrinsic is not None):
            cam_intrinsic_seq = cam_intrinsic

        # evaluation set provides bounding box for each frame. 
        if "handBoundingBox" in pkl_data.keys():
            # Make sure that split type is evaluation
            assert split_type == "evaluation"
            joints3d = (pkl_data['handJoints3D'][None, :] @ coord_change_mat)  
            gt_dict["trans"].append(joints3d) 
            gt_dict["joints_3d"].append(joints3d) 
            joints2d = joints3d @ cam_intrinsic.T
            joints2d = joints2d / joints2d[:, 2:]
            gt_dict["joints_2d"].append(joints2d[:, :2])
            gt_dict["bbox"].append(pkl_data["handBoundingBox"])
            continue
         
        try:
            # Make sure that split type is train. Joint annotations are not fully available for evaluation set.
            assert split_type == "train"

            joints3d = (pkl_data['handJoints3D'] @ coord_change_mat)  
            gt_dict["joints_3d"].append(joints3d) 
            
            joints2d = joints3d @ cam_intrinsic.T
            joints2d = joints2d / joints2d[:, 2:]
            gt_dict["joints_2d"].append(joints2d[:, :2])

            gt_dict["betas"].append(pkl_data['handBeta']) 
            gt_dict["poses"].append(pkl_data['handPose'])
            gt_dict["trans"].append(pkl_data['handTrans'])

            # get segmentation mask
            segmentation_img_i = cv2.imread(segmentation_imgs[i])


            img_seg_height, img_seg_width, _ = segmentation_img_i.shape

            coef = img_height / img_seg_height
            bbox_i = None
            hand_mask = np.argwhere(segmentation_img_i[:, :, 0] == 255)

            if len(hand_mask) > 0:
                
                y_min, y_max = coef * hand_mask[:, 0].min(), coef * hand_mask[:, 0].max()
                x_min, x_max = coef * hand_mask[:, 1].min(), coef * hand_mask[:, 1].max()

                bbox_i = np.array([x_min, y_min, x_max, y_max, 1.0])  

            gt_dict["bbox"].append(bbox_i)

        except:
            import ipdb; ipdb.set_trace()
            print(f"Exception in data reading!!!!!!. Step {i}")
       
    if split_type == "train":         
        # get gt vertices        
        jts3d, vert3d = get_mano_gt(trans_=torch.tensor(np.array(gt_dict["trans"]), dtype=torch.float).to("cuda"), 
                    pose_=torch.tensor(np.array(gt_dict["poses"]), dtype=torch.float).reshape(-1, 16, 3).to("cuda"), 
                    betas=np.array(gt_dict["betas"]))
        gt_dict["vertices_3d"] = vert3d.cpu().numpy()    
    
    # diff = jts3d.cpu().numpy() @ coord_change_mat - np.array(gt_dict["joints_3d"]) 
    
    gt_dict["path"] = gt_path
    gt_dict["dataset_name"] = "HO3D_v3"
    gt_dict["cam_intrinsics"] = cam_intrinsic_seq
    
    if render:
        render_gt(rgb_img_list, gt_dict, rgb_images_path, circle_rad=2, line_width=2)
        
    gt_kyp_pathname = os.path.join(os.path.dirname(gt_path), "gt_keypoints2d")
    os.makedirs(gt_kyp_pathname, exist_ok=True)
    
    # write gt keypoints if it does not exists
    if not len(glob.glob(os.path.join(gt_kyp_pathname, "*_keypoints.json"))) == bs and split_type == "train":
        write_gt_kyp(gt_kyp_pathname, gt_dict["joints_2d"], valid_frames=gt_dict["frame_id"])
    
    gt_dict["bbox"] = np.array(gt_dict["bbox"])
 
    return gt_dict

def get_stage2_res(npz_init_dict, joints2d_path=None):
    from datasets.data_utils import read_keypoints, read_bbox
    args = Arguments('./configs', filename='amass.yaml')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    fk = ForwardKinematicsLayer(args)
    to_th = lambda x: torch.from_numpy(x).to(device)
    
    cdata = npz_init_dict

    gender = 'neutral' 
    if gender.startswith('b'):
        gender = str(cdata['gender'], encoding='utf-8')

    N = len(cdata['poses'])
    print(f'Sequence has {N} frames')
    
    vname = npz_init_dict["path"].split("/")[-2]
 
    # no matter what the keypoint source is, we need to read the bbox from mmpose
    mmpose_bbox_path = os.path.join(os.path.dirname(joints2d_path), "mmpose_keypoints2d")

    keyp_paths = sorted(glob.glob(osp.join(joints2d_path, '*_keypoints.json')))
    bbox_paths = sorted(glob.glob(osp.join(mmpose_bbox_path, '*.pkl')))
    
    
    keyp_frames = [read_keypoints(f) for f in keyp_paths]
    bbox_frames = [read_bbox(f_) for f_ in bbox_paths]
    
 
    joint2d_data = to_th(np.stack(keyp_frames, axis=0)) # T x J x 3 (x,y,conf)
    bbox_data = to_th(np.stack(bbox_frames, axis=0)) # T x 5 (x0, y0, x1, y1, conf)

    cam_f = to_th(cdata['cam_f'])
    cam_center = to_th(cdata['cam_center'])
    
    # ONLY FOR RIGHT HAND
    root_orient_aa = cdata['root_orient']
    pose_body = cdata['poses']
    data_poses = np.concatenate((root_orient_aa, pose_body), axis=1)
 
    pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
    pose = pose.view(-1, 15 + 1, 3)  # axis-angle (T, J, 3)
    trans = torch.from_numpy(np.asarray(cdata['trans'], np.float32)).to(device)  # global translation (T, 3)

    # Compute necessary data for model training.
    rotmat = axis_angle_to_matrix(pose)  # rotation matrix (T, J, 3, 3)
    root_orient = rotmat[:, 0].clone()
    root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (T, 6)     

    # defined in amass.yaml. Set as True
    if args.unified_orientation:
        identity = torch.eye(3).cuda()
        identity = identity.view(1, 3, 3).repeat(rotmat.shape[0], 1, 1)
        rotmat[:, 0] = identity
    rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (T, J, 6)

    rot_seq = rotmat.clone()
    angular = estimate_angular_velocity(rot_seq.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # angular velocity of all the joints (T, J, 3)

    pos, global_xform = fk(rot6d)  # local joint positions (T, J, 3), global transformation matrix for each joint (T, J, 4, 4)
    pos = pos.contiguous()
    global_xform = global_xform.contiguous()
    velocity = estimate_linear_velocity(pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of all the joints (T, J, 3)
    
    # defined in amass.yaml. Set as True
    if args.unified_orientation:
        root_rotation = rotation_6d_to_matrix(root_orient)  # (T, 3, 3)
        root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)
        global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        height = global_pos + trans.unsqueeze(1)
    else:
        height = pos + trans.unsqueeze(1)
    height = height[..., 'xyz'.index(args.data.up)]  # (T, J)
    
    root_vel = estimate_linear_velocity(trans.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of the root joint (T, 3)

    global_xform = global_xform[:, :, :3, :3]  # (T, J, 3, 3)
    global_xform = matrix_to_rotation_6d(global_xform)  # (T, J, 6)
   

    betas = to_th(cdata["betas"])
    joints3d, vertices3d = get_mano_gt(trans_=trans, pose_=pose, betas=cdata["betas"])
    
    
    cam_t = torch.zeros_like(trans)
    cam_R = torch.zeros(*trans.shape, 3)
    cam_R[..., 0, 0] = 1.
    cam_R[..., 1, 1] = 1.
    cam_R[..., 2, 2] = 1.
    
    data = {'rotmat': rotmat,
            'pos': pos,
            'trans': trans,
            'root_vel': root_vel,
            'height': height,
            'rot6d': rot6d,
            'angular': angular,
            'betas': betas,
            'global_xform': global_xform,
            'velocity': velocity,
            'root_orient': root_orient,
            'joints2d': joint2d_data,
            'joints3d': joints3d,
            'vertices': vertices3d,
            'img_width': cdata["img_width"],
            'img_height': cdata["img_height"],
            'img_dir': cdata["img_dir"],
            'cam_t': cam_t,
            "cam_R": cam_R,
            'cam_f': cam_f,
            'cam_center': cam_center}
    
    return data


def write_gt_kyp(out_keypoints_path, right_hand_tmp, valid_frames):
    
    rgb_image_list = sorted(glob.glob(os.path.join(os.path.dirname(out_keypoints_path), "rgb", "*.jpg")))
        
    for im_k, im_path_k in enumerate(rgb_image_list):
        json_pathname = os.path.basename(im_path_k).split(".")[0] + "_keypoints.json"
        json_pathname = os.path.join(out_keypoints_path, json_pathname)

        dic = {}
        dic['version'] = '1.5'
        dic["people"] = []
        person_dic = {}

        person_dic["person_id"] = [-1]
        person_dic["pose_keypoints_2d"] = []
        person_dic["face_keypoints_2d"] = []
        person_dic["hand_left_keypoints_2d"] = []   
        
        if im_k in valid_frames: 
            
            real_k = valid_frames.index(im_k)
            
            j2d_with_conf = np.concatenate((right_hand_tmp[real_k], np.ones((21, 1))), axis=1)
            person_dic["hand_right_keypoints_2d"] = list(j2d_with_conf.reshape(-1))
        else:
            person_dic["hand_right_keypoints_2d"] = [0.0] * 21
        
        
        dic["people"].append(person_dic)

        with open(json_pathname, 'w') as fp:
            json.dump(dic, fp)
            
    return 

 
def dump_amass2pytroch(datasets, amass_dir, out_posepath, logger=None):
    makepath(out_posepath, isfile=True)

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(out_posepath.replace(f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt', '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % out_posepath)

    data_poses = []
    data_trans = []
    data_contacts = []

    clip_frames = args.data.clip_length
    for ds_name in datasets:
        npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/**.npz'))
        logger('processing data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname)
            except:
                logger('Could not read %s! skipping..' % npz_fname)
                continue
            N = len(cdata['pose_right_hand'])
            # gender = str(cdata['gender'])
            gender = "neutral"
            if gender.startswith('b'):
                gender = str(cdata['gender'], encoding='utf-8')

            # Only process data with the specific gender.
            if gender != args.data.gender:
                continue

            # root_orient = cdata['root_orient']
            root_orient = cdata['r_wrist_orient']    # matrix rotation
            pose_body = cdata['pose_right_hand']
    
            # poses = np.concatenate((root_orient, pose_body, np.zeros((N, 6))), axis=1)
            poses = np.concatenate((root_orient, pose_body), axis=1)
            # Chop the data into evenly splitted sequence clips.
            nclips = [np.arange(i, i + clip_frames) for i in range(0, N, clip_frames)]
            if N % clip_frames != 0:
                nclips.pop()
            for clip in nclips:
                data_poses.append(poses[clip])
                data_trans.append(cdata['right_hand_joints'][:, 0, :][clip])
                # data_contacts.append(cdata['contacts'][clip])
       
    assert len(data_poses) != 0

    # Choose the device to run the body model on.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")  # when GPU memory is limited
    pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
    pose = pose.view(-1, clip_frames, J, 3)  # axis-angle (N, T, J, 3)
    trans = torch.from_numpy(np.asarray(data_trans, np.float32)).to(device)  # global translation (N, T, 3)

    # Compute necessary data for model training.
    rotmat = axis_angle_to_matrix(pose)  # rotation matrix (N, T, J, 3, 3)
    root_orient = rotmat[:, :, 0].clone()
    root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (N, T, 6)
    if args.unified_orientation:
        identity = torch.zeros_like(rotmat[:, :, 0])  # (N, T, 3, 3)
        identity[:, :, 0, 0] = 1
        identity[:, :, 1, 1] = 1
        identity[:, :, 2, 2] = 1
        rotmat[:, :, 0] = identity

    rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (N, T, J, 6)
    rot_seq = rotmat.clone()
    angular = estimate_angular_velocity(rot_seq, dt=1.0 / args.data.fps)  # angular velocity of all the joints (N, T, J, 3)

    fk = ForwardKinematicsLayer(args, device=device)
    pos, global_xform = fk(rot6d.view(-1, J, 6))
    pos = pos.contiguous().view(-1, clip_frames, J, 3)  # local joint positions (N, T, J, 3)
    global_xform = global_xform.view(-1, clip_frames, J, 4, 4)  # global transformation matrix for each joint (N, T, J, 4, 4)
    velocity = estimate_linear_velocity(pos, dt=1.0 / args.data.fps)  # linear velocity of all the joints (N, T, J, 3)
    # contacts = torch.from_numpy(np.asarray(data_contacts, np.float32)).to(device)
    # contacts = contacts[:, :, CONTACTS_IDX]  # contacts information (N, T, 8)

    if args.unified_orientation:
        root_rotation = rotation_6d_to_matrix(root_orient)  # (N, T, 3, 3)
        root_rotation = root_rotation.unsqueeze(2).repeat(1, 1, J, 1, 1)  # (N, T, J, 3, 3)
        global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        height = global_pos + trans.unsqueeze(2)
    else:
        height = pos + trans.unsqueeze(2)
    height = height[:, :, :, 'xyz'.index(args.data.up)]  # (N, T, J)
    root_vel = estimate_linear_velocity(trans, dt=1.0 / args.data.fps)  # linear velocity of the root joint (N, T, 3)

    _, rotmat_mean, rotmat_std = normalize(rotmat)
    _, pos_mean, pos_std = normalize(pos)
    _, trans_mean, trans_std = normalize(trans)
    _, root_vel_mean, root_vel_std = normalize(root_vel)
    _, height_mean, height_std = normalize(height)
    # _, contacts_mean, contacts_std = normalize(contacts)

    mean = {
        'rotmat': rotmat_mean.detach().cpu(),
        'pos': pos_mean.detach().cpu(),
        'trans': trans_mean.detach().cpu(),
        'root_vel': root_vel_mean.detach().cpu(),
        'height': height_mean.detach().cpu(),
       
    }
    std = {
        'rotmat': rotmat_std.detach().cpu(),
        'pos': pos_std.detach().cpu(),
        'trans': trans_std.detach().cpu(),
        'root_vel': root_vel_std.detach().cpu(),
        'height': height_std.detach().cpu(),
    }

    torch.save(rotmat.detach().cpu(), out_posepath.replace('pose', 'rotmat'))  # (N, T, J, 3, 3)
    torch.save(pos.detach().cpu(), out_posepath.replace('pose', 'pos'))  # (N, T, J, 3)
    torch.save(trans.detach().cpu(), out_posepath.replace('pose', 'trans'))  # (N, T, 3)
    torch.save(root_vel.detach().cpu(), out_posepath.replace('pose', 'root_vel'))  # (N, T, 3)
    torch.save(height.detach().cpu(), out_posepath.replace('pose', 'height'))  # (N, T, J)
    # torch.save(contacts.detach().cpu(), out_posepath.replace('pose', 'contacts'))  # (N, T, J)

    if args.canonical:
        forward = rotmat[:, :, 0, :, 2].clone()
        canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
        root_rotation = canonical_frame.transpose(-2, -1)  # (N, T, 3, 3)
        root_rotation = root_rotation.unsqueeze(2).repeat(1, 1, args.smpl.joint_num, 1, 1)  # (N, T, J, 3, 3)

        theta = torch.atan2(forward[..., 1], forward[..., 0])
        dt = 1.0 / args.data.fps
        forward_ang = (theta[:, 1:] - theta[:, :-1]) / dt
        forward_ang = torch.cat((forward_ang, forward_ang[..., -1:]), dim=-1)  # (N, T)

        local_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
        local_vel = torch.matmul(root_rotation, velocity.unsqueeze(-1)).squeeze(-1)
        local_rot = torch.matmul(root_rotation, global_xform[:, :, :, :3, :3])
        local_rot = matrix_to_rotation_6d(local_rot)
        local_ang = torch.matmul(root_rotation, angular.unsqueeze(-1)).squeeze(-1)

        _, forward_mean, forward_std = normalize(forward)
        _, forward_ang_mean, forward_ang_std = normalize(forward_ang)
        _, local_pos_mean, local_pos_std = normalize(local_pos)
        _, local_vel_mean, local_vel_std = normalize(local_vel)
        _, local_rot_mean, local_rot_std = normalize(local_rot)
        _, local_ang_mean, local_ang_std = normalize(local_ang)

        mean['forward'] = forward_mean.detach().cpu()
        mean['forward_ang'] = forward_ang_mean.detach().cpu()
        mean['local_pos'] = local_pos_mean.detach().cpu()
        mean['local_vel'] = local_vel_mean.detach().cpu()
        mean['local_rot'] = local_rot_mean.detach().cpu()
        mean['local_ang'] = local_ang_mean.detach().cpu()

        std['forward'] = forward_std.detach().cpu()
        std['forward_ang'] = forward_ang_std.detach().cpu()
        std['local_pos'] = local_pos_std.detach().cpu()
        std['local_vel'] = local_vel_std.detach().cpu()
        std['local_rot'] = local_rot_std.detach().cpu()
        std['local_ang'] = local_ang_std.detach().cpu()

        torch.save(forward.detach().cpu(), out_posepath.replace('pose', 'forward'))  # (N, T, 3)
        torch.save(local_pos.detach().cpu(), out_posepath.replace('pose', 'local_pos'))  # (N, T, J, 3)
        torch.save(local_vel.detach().cpu(), out_posepath.replace('pose', 'local_vel'))  # (N, T, J, 3)
        torch.save(local_rot.detach().cpu(), out_posepath.replace('pose', 'local_rot'))  # (N, T, J, 6)
        torch.save(local_ang.detach().cpu(), out_posepath.replace('pose', 'local_ang'))  # (N, T, J, 3)
    else:
        global_xform = global_xform[:, :, :, :3, :3]  # (N, T, J, 3, 3)
        global_xform = matrix_to_rotation_6d(global_xform)  # (N, T, J, 6)

        _, rot6d_mean, rot6d_std = normalize(rot6d)
        _, angular_mean, angular_std = normalize(angular)
        _, global_xform_mean, global_xform_std = normalize(global_xform)
        _, velocity_mean, velocity_std = normalize(velocity)
        _, orientation_mean, orientation_std = normalize(root_orient)

        mean['rot6d'] = rot6d_mean.detach().cpu()
        mean['angular'] = angular_mean.detach().cpu()
        mean['global_xform'] = global_xform_mean.detach().cpu()
        mean['velocity'] = velocity_mean.detach().cpu()
        mean['root_orient'] = orientation_mean.detach().cpu()

        std['rot6d'] = rot6d_std.detach().cpu()
        std['angular'] = angular_std.detach().cpu()
        std['global_xform'] = global_xform_std.detach().cpu()
        std['velocity'] = velocity_std.detach().cpu()
        std['root_orient'] = orientation_std.detach().cpu()
 
        torch.save(rot6d.detach().cpu(), out_posepath.replace('pose', 'rot6d'))  # (N, T, J, 6)
        torch.save(angular.detach().cpu(), out_posepath.replace('pose', 'angular'))  # (N, T, J, 3)
        torch.save(global_xform.detach().cpu(), out_posepath.replace('pose', 'global_xform'))  # (N, T, J, 6)
        torch.save(velocity.detach().cpu(), out_posepath.replace('pose', 'velocity'))  # (N, T, J, 3)
        torch.save(root_orient.detach().cpu(), out_posepath.replace('pose', 'root_orient'))  # (N, T, 3, 3)

    if args.normalize:
        torch.save(mean, out_posepath.replace('pose', 'mean'))
        torch.save(std, out_posepath.replace('pose', 'std'))

    return len(data_poses)


def dump_amass2single(datasets, amass_dir, split_name, logger=None, bvh_viz=False):
    # Choose the device to run the body model on.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fk = ForwardKinematicsLayer(args)
    
    clip_frames_threshold = 512

    index = 0
    for ds_name in datasets:
        npz_fnames = sorted(glob.glob(os.path.join(amass_dir, ds_name, '*/*.npz')))
        logger('processing data points from %s.' % (ds_name))
        for npz_fname in tqdm(npz_fnames):
            try:
                cdata = np.load(npz_fname)
            except:
                logger('Could not read %s! skipping..' % npz_fname)
                continue

            # gender = str(cdata['gender'])
            gender = "neutral"
            if gender.startswith('b'):
                gender = str(cdata['gender'], encoding='utf-8')

            # Only process data with the specific gender.
            if gender != args.data.gender:
                continue

            N = len(cdata['pose_body'])
            logger(f'Sequence {index} has {N} frames')
            
            if N < clip_frames_threshold:
                continue

            root_orient = cdata['r_wrist_orient']
            pose_body = cdata['pose_right_hand']
            # data_poses = np.concatenate((root_orient, pose_body, np.zeros((N, 6))), axis=1)
            data_poses = np.concatenate((root_orient, pose_body), axis=1)
            pose = torch.from_numpy(np.asarray(data_poses, np.float32)).to(device)
            pose = pose.view(-1, J, 3)  # axis-angle (T, J, 3)
            
            trans = torch.from_numpy(np.asarray(cdata['right_hand_joints'][:, 0, :], np.float32)).to(device)  # global translation (T, 3)

            # Compute necessary data for model training.
            rotmat = axis_angle_to_matrix(pose)  # rotation matrix (T, J, 3, 3)
            root_orient = rotmat[:, 0].clone()
            root_orient = matrix_to_rotation_6d(root_orient)  # root orientation (T, 6)
            if args.unified_orientation:
                identity = torch.eye(3).cuda()
                identity = identity.view(1, 3, 3).repeat(rotmat.shape[0], 1, 1)
                rotmat[:, 0] = identity
            rot6d = matrix_to_rotation_6d(rotmat)  # 6D rotation representation (T, J, 6)

            rot_seq = rotmat.clone()
            angular = estimate_angular_velocity(rot_seq.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # angular velocity of all the joints (T, J, 3)

            pos, global_xform = fk(rot6d)  # local joint positions (T, J, 3), global transformation matrix for each joint (T, J, 4, 4)
            pos = pos.contiguous()
            global_xform = global_xform.contiguous()
            velocity = estimate_linear_velocity(pos.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of all the joints (T, J, 3)

            # contacts = torch.from_numpy(np.asarray(cdata['contacts'], np.float32)).to(device)
            # contacts = contacts[:, CONTACTS_IDX]  # contacts information (T, 8)

            if args.unified_orientation:
                root_rotation = rotation_6d_to_matrix(root_orient)  # (T, 3, 3)
                root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)
                global_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
                height = global_pos + trans.unsqueeze(1)
            else:
                height = pos + trans.unsqueeze(1)
            height = height[..., 'xyz'.index(args.data.up)]  # (T, J)
            root_vel = estimate_linear_velocity(trans.unsqueeze(0), dt=1.0 / args.data.fps).squeeze(0)  # linear velocity of the root joint (T, 3)

            out_posepath = os.path.join(work_dir, split_name, f'trans_{index}.pt')
            torch.save(trans.detach().cpu(), out_posepath)  # (T, 3)
            torch.save(root_vel.detach().cpu(), out_posepath.replace('trans', 'root_vel'))  # (T, 3)
            torch.save(pos.detach().cpu(), out_posepath.replace('trans', 'pos'))  # (T, J, 3)
            torch.save(rotmat.detach().cpu(), out_posepath.replace('trans', 'rotmat'))  # (T, J, 3, 3)
            torch.save(height.detach().cpu(), out_posepath.replace('trans', 'height'))  # (T, J)
            # torch.save(contacts.detach().cpu(), out_posepath.replace('trans', 'contacts'))  # (T, J)

            if args.canonical:
                forward = rotmat[:, 0, :, 2].clone()
                canonical_frame = build_canonical_frame(forward, up_axis=args.data.up)
                root_rotation = canonical_frame.transpose(-2, -1)  # (T, 3, 3)
                root_rotation = root_rotation.unsqueeze(1).repeat(1, args.smpl.joint_num, 1, 1)  # (T, J, 3, 3)

                if args.data.up == 'x':
                    theta = torch.atan2(forward[..., 2], forward[..., 1])
                elif args.data.up == 'y':
                    theta = torch.atan2(forward[..., 0], forward[..., 2])
                else:
                    theta = torch.atan2(forward[..., 1], forward[..., 0])
                dt = 1.0 / args.data.fps
                forward_ang = (theta[1:] - theta[:-1]) / dt
                forward_ang = torch.cat((forward_ang, forward_ang[-1:]), dim=-1)

                local_pos = torch.matmul(root_rotation, pos.unsqueeze(-1)).squeeze(-1)
                local_vel = torch.matmul(root_rotation, velocity.unsqueeze(-1)).squeeze(-1)
                local_rot = torch.matmul(root_rotation, global_xform[:, :, :3, :3])
                local_rot = matrix_to_rotation_6d(local_rot)
                local_ang = torch.matmul(root_rotation, angular.unsqueeze(-1)).squeeze(-1)

                torch.save(forward.detach().cpu(), out_posepath.replace('trans', 'forward'))
                torch.save(forward_ang.detach().cpu(), out_posepath.replace('trans', 'forward_ang'))
                torch.save(local_pos.detach().cpu(), out_posepath.replace('trans', 'local_pos'))
                torch.save(local_vel.detach().cpu(), out_posepath.replace('trans', 'local_vel'))
                torch.save(local_rot.detach().cpu(), out_posepath.replace('trans', 'local_rot'))
                torch.save(local_ang.detach().cpu(), out_posepath.replace('trans', 'local_ang'))
            else:


                global_xform = global_xform[:, :, :3, :3]  # (T, J, 3, 3)
                global_xform = matrix_to_rotation_6d(global_xform)  # (T, J, 6)

                torch.save(rot6d.detach().cpu(), out_posepath.replace('trans', 'rot6d'))  # (N, T, J, 6)
                torch.save(angular.detach().cpu(), out_posepath.replace('trans', 'angular'))  # (N, T, J, 3)
                torch.save(global_xform.detach().cpu(), out_posepath.replace('trans', 'global_xform'))  # (N, T, J, 6)
                torch.save(velocity.detach().cpu(), out_posepath.replace('trans', 'velocity'))  # (N, T, J, 3)
                torch.save(root_orient.detach().cpu(), out_posepath.replace('trans', 'root_orient'))  # (N, T, 3, 3)

            if bvh_viz:
                bvh_fname = os.path.join(work_dir, 'bvh', split_name, f'pose_{index}.bvh')
                makepath(bvh_fname, isfile=True)
                rotation = axis_angle_to_quaternion(pose)
                rotation = align_joints(rotation)
                position = trans.unsqueeze(1)
                # Export the motion data to .bvh file with SMPL skeleton.
                anim = Animation(Quaternions(c2c(rotation)), c2c(position), None, offsets=args.smpl.offsets["right"], parents=args.smpl.parents)
                BVH.save(bvh_fname, anim, names=args.smpl.joint_names, frametime=1 / args.data.fps)

            index += 1

    return index


def prepare_amass(amass_splits, amass_dir, work_dir, logger=None):

    if logger is None:
        starttime = datetime.now().replace(microsecond=0)
        log_name = datetime.strftime(starttime, '%Y%m%d_%H%M')
        logger = log2file(os.path.join(work_dir, '%s.log' % (log_name)))
        logger('Creating pytorch dataset at %s' % work_dir)

    logger('Fetch data from AMASS npz files, augment the data and dump every data field as final pytorch pt files')

    for split_name, datasets in amass_splits.items():
        outpath = makepath(os.path.join(work_dir, split_name, f'pose-{args.data.gender}-{args.data.clip_length}-{args.data.fps}fps.pt'), isfile=True)
        if os.path.exists(outpath):
            continue
        if args.single_motion:
            ndatapoints = dump_amass2single(datasets, amass_dir, split_name, logger=logger, bvh_viz=True)
        else:  
            ndatapoints = dump_amass2pytroch(datasets, amass_dir, outpath, logger=logger)
        logger('%s has %d data points!' % (split_name, ndatapoints))


def dump_amass2bvh(amass_splits, amass_dir, out_bvhpath, logger=None):
    from utils import export_bvh_animation

    if logger is None:
        log_name = os.path.join(out_bvhpath, 'bvh_viz.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        logger = log2file(log_name)
        logger('Creating bvh visualization at %s' % out_bvhpath)

    for split_name, datasets in amass_splits.items():
        outpath = makepath(os.path.join(out_bvhpath, split_name))
        if os.path.exists(outpath) and len(os.listdir(outpath)) != 0:
            continue

        for ds_name in datasets:
            npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses_*.npz'))
            logger('visualizing data points at %s.' % (ds_name))
            for npz_fname in tqdm(npz_fnames):
                bvh_fname = os.path.split(npz_fname)[1].replace('.npz', '.bvh')
                try:
                    export_bvh_animation(npz_fname, os.path.join(outpath, ds_name + '_' + bvh_fname))
                except AssertionError:
                    logger('Could not export %s because its gender is not supported! skipping..' % npz_fname)
                    continue


def collect_amass_stats(amass_splits, amass_dir, logger=None):
    import matplotlib.pyplot as plt

    if logger is None:
        log_name = os.path.join('./data/amass', 'amass_stats.log')
        if os.path.exists(log_name):
            os.remove(log_name)
        logger = log2file(log_name)

    logger('Collecting stats for AMASS datasets:')

    gender_stats = {}
    fps_stats = {}
    durations = []
    for split_name, datasets in amass_splits.items():
        for ds_name in datasets:
            logger(f'\t{ds_name} dataset')
            npz_fnames = glob.glob(os.path.join(amass_dir, ds_name, '*/*_poses_*.npz'))
            for npz_fname in tqdm(npz_fnames):
                try:
                    cdata = np.load(npz_fname)
                except:
                    logger('Could not read %s! skipping..' % npz_fname)
                    continue

                gender = str(cdata['gender'])
                if gender.startswith('b'):
                    gender = str(cdata['gender'], encoding='utf-8')
                fps = 30
                duration = len(cdata['pose_body']) / fps

                durations.append(duration)
                if gender in gender_stats.keys():
                    gender_stats[gender].append(duration)
                else:
                    gender_stats[gender] = [duration]
                if fps in fps_stats.keys():
                    fps_stats[fps] += 1
                else:
                    fps_stats[fps] = 1

    logger('\n')
    logger('Total motion sequences: {:,}'.format(len(durations)))
    logger('\tTotal Duration: {:,.2f}s'.format(sum(durations)))
    logger('\tMin Duration: {:,.2f}s'.format(min(durations)))
    logger('\tMax Duration: {:,.2f}s'.format(max(durations)))
    logger('\tAverage Duration: {:,.2f}s'.format(sum(durations) / len(durations)))
    logger('\tSequences longer than 5s: {:,} ({:.2f}%)'.format(sum(i > 5 for i in durations), sum(i > 5 for i in durations) / len(durations) * 100))
    logger('\tSequences longer than 10s: {:,} ({:.2f}%)'.format(sum(i > 10 for i in durations), sum(i > 10 for i in durations) / len(durations) * 100))
    logger('\n')
    logger('Gender:')
    for key, value in gender_stats.items():
        logger('\t{}: {:,} (Duration: {:,.2f}s)'.format(key, len(value), sum(value)))
    logger('\n')
    logger('FPS:')
    for key, value in fps_stats.items():
        logger('\t{}: {:,}'.format(key, value))

    # Plot histograms for duration distributions.
    fig, axes = plt.subplots(3, sharex=True)
    fig.tight_layout(pad=3.0)
    fig.suptitle('AMASS Data Duration Distribution')
    axes[0].hist(durations, density=False, bins=100)
    axes[0].set_title('total')
    for key, value in gender_stats.items():
        if key == 'female':
            f = axes[1]
        else:
            f = axes[2]
        f.hist(value, density=False, bins=100)
        f.set_title(f'{key}')
    # Add a big axis, hide frame.
    fig.add_subplot(111, frameon=False)
    # Hide tick and tick label of the big axis.
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Duration (second)")
    plt.ylabel("Counts", labelpad=10)
    fig.savefig('./data/amass/duration_dist.pdf')


class AMASS(Dataset):
    """AMASS: a pytorch loader for unified human motion capture dataset. http://amass.is.tue.mpg.de/"""

    def __init__(self, dataset_dir):
        self.ds = {}
        for data_fname in glob.glob(os.path.join(dataset_dir, '*.pt')):
            k = os.path.basename(data_fname).split('-')[0]
            self.ds[k] = torch.load(data_fname)

    def __len__(self):
        return len(self.ds['trans'])

    def __getitem__(self, idx):
        data = {k: self.ds[k][idx] for k in self.ds.keys() if k not in ['mean', 'std']} 
        return data


if __name__ == '__main__':
    msg = ''' Using standard AMASS dataset preparation pipeline: 
    0) Download all npz files from https://amass.is.tue.mpg.de/ 
    1) Convert npz files to pytorch readable pt files. '''

    amass_dir = 'data/amass_processed_hist/'
    args = Arguments('./configs', filename='amass.yaml')
    
    work_dir = makepath(args.dataset_dir)

    log_name = os.path.join(work_dir, 'amass.log')
    if os.path.exists(log_name):
        os.remove(log_name)
    logger = log2file(log_name)
    logger('AMASS Data Preparation Began.')
    logger(msg)

    if args.single_motion:
        amass_splits = {
            'train': ['GRAB']
        }
    else:
        amass_splits = {
            'valid': ['SAMP'],
            'test': ['TCDHands'],
            'train': ['GRAB', 'ARCTIC']
        }
     
    # collect_amass_stats(amass_splits, amass_dir)
    prepare_amass(amass_splits, amass_dir, work_dir, logger=logger)
