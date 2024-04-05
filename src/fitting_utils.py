import os 
import cv2 
import glob 
import json
import torch
import joblib
import importlib
import numpy as np
from tqdm import tqdm
import scenepic as sp
from smplx import MANO
from loguru import logger 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from rotations import axis_angle_to_matrix, matrix_to_axis_angle

from nemf.losses import pos_smooth_loss

AUGMENTED_MANO_CHAIN = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])   
openpose_skeleton = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]

RIGHT_WRIST_BASE_LOC = torch.tensor([[0.0957, 0.0064, 0.0062]])
LEFT_WRIST_BASE_LOC = torch.tensor([[-0.0957, 0.0064, 0.0062]])
MANO_RH_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"


def convert_pred_to_full_img_cam(pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length, crop_res=224):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]

    r = bbox_height / crop_res
    
    
    tz = 2 * focal_length / (r * crop_res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    kk = 1.0
 
    tx_final = (tx + cx) / kk
    ty_final = (ty + cy) / kk
    tz_final = tz / kk

    cam_t = torch.stack([tx_final, ty_final, tz_final], dim=-1)

    return cam_t

# reverse 
def map_openpose_joints_to_mano():
    return np.array([0, 5, 6, 7, 9, 10, 11, 17, 18, 19, 13, 14, 15, 1, 2, 3, 4, 8, 12, 16, 20], dtype=np.int32) 

def map_mano_joints_to_openpose(): 
    return np.array([0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20], dtype=np.int32)

def forward_mano_pymafx(output):
    root_orient = output['root_orient'].to("cuda")  # (T, 3)
    poses = output['poses'].to("cuda")  # (T, J, 3)
    
    T, J, _ = poses.size()

    hand_model = MANO(model_path=MANO_RH_DIR, flat_hand_mean=True, use_pca=False, batch_size=poses.shape[0]).to('cuda')

    mano_out = hand_model(betas=output['betas'].view(-1, 10).to("cuda"),
                            global_orient=root_orient.to("cuda"),
                            hand_pose=poses.view(-1, 45).to("cuda"),
                            return_tips=True,
                            transl=output['trans'].view(-1, 3).to("cuda"))  
        
    # mano_out.vertices -= RIGHT_WRIST_BASE_LOC.to(mano_out.vertices)
    # mano_out.joints -= RIGHT_WRIST_BASE_LOC.to(mano_out.joints)
    
    return mano_out


def joints2d_loss(joints2d_obs, joints2d_pred, bbox_conf=None):
    '''
    Cam extrinsics are assumed the same for entire sequence
    - cam_t : (B, 3)
    - cam_R : (B, 3, 3)
    '''
    B, T, _, _ = joints2d_obs.size()
    
    # compared to observations
    
    if bbox_conf is None:
        joints2d_obs_conf = joints2d_obs[:,:,:,2:3]
    else:    
        bbox_conf = torch.repeat_interleave(bbox_conf.unsqueeze(-1), 21, dim=-1)    
        joints2d_obs_conf = torch.repeat_interleave(bbox_conf.unsqueeze(-1), 2, dim=-1).detach()
       
    # weight errors by detection confidence
    robust_sqr_dist = gmof(joints2d_pred[:, :, :, :] - joints2d_obs[:,:,:,:2], sigma=40)
    reproj_err = (joints2d_obs_conf**2) * robust_sqr_dist
    loss = torch.mean(reproj_err)
    return loss

def get_seqname_ho3d_v3(path):
    return path.split("/")[-2]
    
def get_seqname_dexycb(path):
    # return SUBJECTNAME/SEQNAME/SUBSEQNAME'
    return "/".join(path.split("/")[-4:-1])

def get_seqname_arctic_data(path):
    # return SUBJECTNAME/SEQNAME/VIEVNUM'
    return "/".join(path.split("/")[-4:-1])

def get_seqname_in_the_wild(path):
    return path.split("/")[-2].split(".")[0]

def compute_seq_intervals(seq_len, split_len=128, overlap_len=16):
    # for seq_len = 256 [0, 128], [112, 240], [224, 256]
    # for seq_len = 320 [0, 128], [112, 240], [224, 320]
    # for seq_len = 565 [0, 128], [112, 240], [224, 352], [336, 464], [448, 565]
    
    intervals = []
    for i in range(0, seq_len, split_len - overlap_len):
        start = i
        end = min(i + split_len, seq_len)
        
        intervals.append((start, end))
        if (end - start) < split_len:
            break
    return intervals

# optimize on pymafx. Optimize translation only. 
def optimize_on_pymafx(pymafx_dict, pymafx_keypoints2d_path):
    
    os.makedirs(pymafx_keypoints2d_path, exist_ok=True)
   
    seqlen = 128 
    
    # pad values 
    for k, v in pymafx_dict.items():
        if k in ['cam_f', 'cam_center']:
            continue        
        else:
            data_split = list(torch.split(v, 128))
            pad_repeat = 128 - data_split[-1].shape[0]
            last_el = data_split[-1]
            last_el = torch.cat([last_el, last_el[-1:].repeat_interleave(pad_repeat, 0)])
            data_split[-1] = last_el
            pymafx_dict[k] = torch.stack(data_split, dim=0)
            
    rh_opt_trans = torch.tensor(pymafx_dict["trans"].clone(), dtype=torch.float32)
    B = rh_opt_trans.shape[0]
    
    cam_R = torch.zeros((seqlen * B, 3, 3), dtype=torch.float32).to(rh_opt_trans)
    cam_R[:, 0, 0] = 1.0
    cam_R[:, 1, 1] = 1.0
    cam_R[:, 2, 2] = 1.0
    cam_t = torch.zeros((seqlen * B, 3), dtype=torch.float32).to(rh_opt_trans)
    
    rh_opt_trans.requires_grad = True
    cam_R.requires_grad = False
    cam_t.requires_grad = False

    # optimize based on 2d keypoints 
    opt_params = [rh_opt_trans]
    optimizer = torch.optim.Adam(opt_params, lr=0.3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.95, verbose=False)
    
    K_iter = 3000
    
    # project to image and get 2d keypoints 
    # for i in tqdm(range(500)):
    for i in range(K_iter):
        
        loss_dict = {}
        optimizer.zero_grad()
                 
        # values change in every iteration 
        output = {"poses": pymafx_dict["poses"].view(-1, 15, 3),
                  "root_orient": pymafx_dict["root_orient"].view(-1, 3),
                  "betas": pymafx_dict["betas"].view(-1, 10),
                  "trans": rh_opt_trans}

        # rotmat ==> (B, T, J, 3, 3), root_orient ==> (B, T, 3, 3), trans ==> (B, T, 3)
        rh_mano_out = forward_mano_pymafx(output) 
        joints3d = rh_mano_out.joints

        joints2d_pred = get_joints2d(joints3d_pred=joints3d.reshape(B, seqlen, -1, 3), 
                                cam_t=cam_t.unsqueeze(0).repeat_interleave(1, 0),
                                cam_R=cam_R.unsqueeze(0).repeat_interleave(1, 0),
                                cam_f=torch.tensor([5000., 5000.]), 
                                cam_center=pymafx_dict['cam_center'][0])
        
        reproj_loss = joints2d_loss(joints2d_obs=pymafx_dict['keypoints2d'].reshape(B, seqlen, -1, 3), joints2d_pred=joints2d_pred)
      
        loss_dict['reproj'] = 0.01 * reproj_loss
        loss_dict['trans_sm'] = 400 * pos_smooth_loss(rh_opt_trans)
        loss_dict['j3d_sm'] = 100 * pos_smooth_loss(joints3d.reshape(B, seqlen, -1, 3)) 
        
        loss = sum(loss_dict.values())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_log_str = f'[{i:03d}/{K_iter}]'
        for k, v in loss_dict.items():
            loss_dict[k] = v.item()
            loss_log_str += f'{k}: {v.item():.3f}\t'
        logger.info(loss_log_str)
    
    pymafx_dict["trans"] = rh_opt_trans.detach()
    
    return pymafx_dict

# process ground truth and predictions. Discard frames with no or invalid ground truth    
def process_gt(pred_dict, gt_dict, dataset_name):
    
    # DEXYCB
    if dataset_name == "DexYCB":
        
        jts3d = - torch.ones((1, 21, 3))
        jts2d = - torch.ones((1, 21, 2))
        
        invalid_indices3d = torch.where(torch.all(torch.all(torch.eq(gt_dict["joints_3d"], jts3d), dim=1), dim=1))[0]
        invalid_indices2d = torch.where(torch.all(torch.all(torch.eq(gt_dict["joints_2d"], jts2d), dim=1), dim=1))[0]
        
        assert torch.all(torch.eq(invalid_indices3d, invalid_indices2d)), "Inconsistency in ground truth data"
        
        valid_indices = np.setdiff1d(np.arange(gt_dict["joints_3d"].shape[0]), invalid_indices3d)
        
        # discard those frames from frame_id
        gt_dict["frame_id"] = list(set(gt_dict["frame_id"]).difference(set(np.array(invalid_indices3d))))
   
        # find indices of frames with value -1. This indicates the ground truth is not available        
        gt_dict["joints_3d"] = gt_dict["joints_3d"][valid_indices]
        gt_dict["vertices_3d"] = gt_dict["vertices_3d"][valid_indices]
        gt_dict["joints_2d"] = gt_dict["joints_2d"][valid_indices]
        gt_dict["poses"] = np.array(gt_dict["poses"])[valid_indices]
        
        # cast to seq length 
        pred_dict["joints_3d"] = pred_dict["joints_3d"][:gt_dict['num_frames']]
        pred_dict["joints_2d"] = pred_dict["joints_2d"][:gt_dict['num_frames']]
    

    return pred_dict, gt_dict


def gmof(res, sigma):
    """
    Geman-McClure error function
    - residual
    - sigma scaling factor
    """
    x_squared = res ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def get_joints2d(joints3d_pred, cam_t, cam_R, cam_f, cam_center):
    B, T, _, _ = joints3d_pred.size()

    # project points to 2D
    joints2d_pred = perspective_projection(joints3d_pred.reshape((B*T, -1, 3)),
                        cam_R.reshape((B*T, 3, 3)),
                        cam_t.reshape((B*T, 3)),
                        cam_f.unsqueeze(0).unsqueeze(0).repeat(B, T, 1).reshape((B*T, 2)),
                        cam_center.unsqueeze(0).unsqueeze(0).repeat(B, T, 1).reshape((B*T, 2))).reshape(B, T, -1, 2)
    return joints2d_pred


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    Adapted from https://github.com/mkocabas/VIBE/blob/master/lib/models/spin.py
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs, 2): Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length[:,0]
    K[:,1,1] = focal_length[:,1]
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)
    
    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]

def run_metro(video_p, out_path, joints2d_image_path=None, jts_vid=None, gt_bbox=None):

    import open3d_viz_overlay   

    pickle_path = os. path.join(out_path, "output.pkl")
    cur_path = os.getcwd()
    
    if not os.path.exists(pickle_path): 
 
        os.makedirs(out_path, exist_ok=True)
   
        rel_vid_p = os.path.join("../..", video_p)
        rel_out_p = os.path.join("../..", out_path)

        command_list = [
                        "/home/eduran2/miniconda3/envs/nemf/bin/python", 
                        "./metro/tools/end2end_inference_handmesh.py",
                        "--resume_checkpoint", 
                        "./models/metro_hand_state_dict.bin",
                        "--image_file_or_path",
                        rel_vid_p
                        ]
    
        command_str = " ".join(command_list)   
        os.chdir("./external/MeshTransformer")
        os.system(command_str)
                
        os.chdir(cur_path)
        # visualize with our open3d code
        # cropped image dir 
        # open3d_viz_overlay.vis_opt_results(pred_file_path=os.path.join(out_path, "output.pkl"), 
        #                             gt_file_path=None, 
        #                             img_dir=os.path.join(os.path.dirname(video_p), "metro_out/rgb_crop")
        

    open3d_viz_overlay.vis_opt_results(pred_file_path=os.path.join(out_path, "output.pkl"), 
                                gt_file_path=None, 
                                img_dir=video_p)    
    
    # read the output pickle file, and get the hand pose and shape parameters if not exist
    if joblib.load(pickle_path).get("orient") is None:

        import external.v2a_code.scripts.fit_mano_to_metro  
        external.v2a_code.scripts.fit_mano_to_metro.main(video_p)
    
    out_obj = joblib.load(pickle_path)
   
    return out_obj 


def run_pymafx(video_p, pymaf_out_path, joints2d_image_path=None, jts_vid=None, gt_bbox=None):

    pickle_path = os. path.join(pymaf_out_path, "output.pkl")
    
    # run PyMAF-X if not already done
    if not os.path.exists(pickle_path): 
        os.makedirs(pymaf_out_path, exist_ok=True)
 
        # first save the gt bbox so that pymafx read it 
        if gt_bbox is not None:
            np.savez(os.path.join(pymaf_out_path, "gt_bbox.npz"), **gt_bbox)
        
        cur_path = os.getcwd()
        os.chdir("./external/PyMAF-X")
        
        rel_vid_p = os.path.join("../..", video_p)
        rel_out_p = os.path.join("../..", pymaf_out_path)
         
        command_list = ["python", "-m", "apps.demo_mano", "--image_folder", rel_vid_p, "--detection_threshold", "0.3",
        "--misc", "MODEL.PyMAF.OPT_HEAD", "False", "TRAIN.BHF_MODE", "hand_only", "MODEL.MESH_MODEL", "mano",
        "--output_folder", rel_out_p, "--pretrained_model", "./data/pretrained_model/PyMAF-X_model_checkpoint_v1.1.pt", 
         "--out_vid_path", jts_vid]

        if joints2d_image_path is not None:
            joints2d_image_path = os.path.join("../..", joints2d_image_path) 
            command_list.extend(["--joints2d_image_folder", joints2d_image_path])

        # uncomment this line if you want to have the direct open3d rendering of the PyMAF-X output
        # commant_list.extend([", --render"])

        command_str = " ".join(command_list)
       
        try: 
            os.system(command_str)
        except:
            exit("PyMAF-X failed")
            
        
        os.chdir(cur_path)

    out_obj = joblib.load(pickle_path)
    out_obj["cam_f"] = np.array([5000, 5000])
    return out_obj 

def process_pymafx_mano(pickle_obj):

    if not pickle_obj["smplx_params"]:
        return None, None, None, None, None 

    
    batch_count = len(pickle_obj["smplx_params"])
    frame_count = len(pickle_obj["frame_ids"])

    right_hand_pose, rh_orient, rh_betas = [], [], []
    
    for i in range(batch_count):
        bs = pickle_obj["smplx_params"][i]["body_pose"].shape[0]

        right_hand_pose.append(matrix_to_axis_angle(pickle_obj["smplx_params"][i]["body_pose"].reshape(-1, 3, 3)).reshape(bs, -1, 3))
        rh_betas.append(pickle_obj["smplx_params"][i]["shape"])   
        rh_orient.append(matrix_to_axis_angle(pickle_obj["smplx_params"][i]["root_orient"].reshape(-1, 3, 3)).reshape(bs, 3))
    
    right_hand_pose = torch.vstack(right_hand_pose)
    rh_betas = torch.vstack(rh_betas)
    rh_orient = torch.vstack(rh_orient)
    
    transl = torch.tensor(pickle_obj["orig_cam_t"], dtype=torch.float32)

    export_pymafx_json(fpath=pickle_obj["video_path"], jts_2d=pickle_obj["pymafx_joints2d"])
    rh_verts = pickle_obj["verts"]
 
    return rh_orient, transl, right_hand_pose, rh_betas, rh_verts


def export_pymafx_json(fpath, jts_2d):
    
    if ("HO3D_v3" in fpath) or ("DexYCB" in fpath) or ("in_the_wild" in fpath) or ("arctic_data"):
        out_keypoints_path = os.path.join(os.path.dirname(fpath), "pymafx_keypoints2d")    
    else:
        raise NotImplementedError("PYMAFX export not implemented for this dataset")

    # slerp(time smoothing) for 3d case is done in fitting_app  
    os.makedirs(out_keypoints_path, exist_ok=True)
        
    right_hand_tmp = jts_2d  # confidences come from mmpose detection confidence values 
    rgb_image_list = sorted([os.path.join(fpath, x) for x in os.listdir(fpath) if x.endswith('.png') or x.endswith('.jpg')])
        
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
        person_dic["hand_right_keypoints_2d"] = list(right_hand_tmp[im_k].reshape(-1))
        
        dic["people"].append(person_dic)

        with open(json_pathname, 'w') as fp:
            json.dump(dic, fp)
        
    return 

def blend_keypoints(source_path1, source_path2, target_path, gt_frames=None, raw_img_path=None,
                    blend_ratio=0.5, blend_vid_out_path=None, smooth_flag=True,
                    render=False, circle_rad=4, line_width=2):
    """ source_path1: path to mediapipe keypoints1
        source_path2: path to pymafx keypoints2
    """

    # green for pymafx, red for mediapipe
    COLORS = {"pymafx": (0, 255, 0), "together": (0, 0, 255)}
    
    from datasets.data_utils import read_keypoints

    smoother = None    
    
    if blend_vid_out_path is not None and render:
        os.makedirs(blend_vid_out_path, exist_ok=True)
      
    if raw_img_path is not None:
        raw_img_list = sorted([os.path.join(raw_img_path, x) for x in os.listdir(raw_img_path) if x.endswith('.png') or x.endswith('.jpg')])
    
    os.makedirs(target_path, exist_ok=True)
    
    assert os.path.exists(source_path1), "Source path 1 does not exist"
    assert os.path.exists(source_path2), "Source path 1 does not exist"
    
    kypts1 = sorted(glob.glob(os.path.join(source_path1, "*.json")))
    kypts2 = sorted(glob.glob(os.path.join(source_path2, "*.json")))
         
    assert len(kypts1) == len(kypts2), "Number of keypoints do not match"
 
    
    kypts1 = [read_keypoints(f) for f in kypts1]
    kypts2 = [read_keypoints(f_) for f_ in kypts2]
    
    blend_res_list, smooth_list, valid_ind, keyp_source_list = [], [], [], []
    track_id = 0
    
    for i, (kypt1, kypt2) in enumerate(tqdm((zip(kypts1, kypts2)))):
        
        # check if the mediapipe confidence are all zero ( 0,0,0 sometimes is violated (0, 480, 0) input comes )
        if np.array_equiv(kypt1[:, -1], np.zeros_like(kypt1[:, -1])):
            blend_rat = 0.0  
            keyp_source_list.append("pymafx")
        else:
            blend_rat = blend_ratio
            keyp_source_list.append("together")
            
        blend_res_i = np.array(kypt1) * blend_rat + np.array(kypt2) * (1 - blend_rat)
 
        # to see if the gt bbox is discontinued
        if not np.array_equiv(kypt2, np.zeros_like(kypt2)):            
            # print(kypt2.mean(), kypt1.mean(), blend_res_i.mean())
            
            if smoother:
                blend_res_i[:, :2] = smoother.smooth([{"keypoints": blend_res_i[:, :2],
                            "track_id": track_id}])[0]["keypoints"]        
            valid_ind.append(i)     
        else:
            track_id += 1
    
        blend_res_list.append(blend_res_i)
        target_json_pathname = os.path.join(target_path, f"{i:04d}_keypoints.json")
        
        if sum(blend_res_i[:, -1]) == 0.0:
            assert i not in gt_frames, "GT frame is not valid"
    
        dic = {}
        dic['version'] = '1.5'
        dic["people"] = []
        person_dic = {}

        # openpose format
        person_dic["person_id"] = [-1]
        person_dic["pose_keypoints_2d"] = []
        person_dic["face_keypoints_2d"] = []
        person_dic["hand_left_keypoints_2d"] = []        
        person_dic["hand_right_keypoints_2d"] = blend_res_i.reshape(-1, 1).tolist()
        
        # if blend_res_i[:, -1].mean() == 0.0:
        #     import ipdb; ipdb.set_trace()
        
        dic["people"].append(person_dic)
        
        with open(target_json_pathname, 'w') as fp:
            json.dump(dic, fp)
            
        if render:  
            img_rgb = raw_img_list[i]
            color = COLORS[keyp_source_list[i]]
            
            _img = Image.open(img_rgb)
            draw = ImageDraw.Draw(_img)
             
            # check if there is a corresponding gt for this frame
            joints2d = blend_res_i[:, :2]

            for k in range(21):                  
                kps_parent = joints2d[openpose_skeleton[k]]
                kps_child = joints2d[k]
                

                if (kps_parent == [0.0, 0.0]).all() or (kps_child == [0.0, 0.0]).all():
                    continue

                if openpose_skeleton[k] != -1:
                    draw.line([(kps_child[0], kps_child[1]), (kps_parent[0], kps_parent[1])], fill=color, width=line_width)
                
                if k in [3, 4]:
                    jts_color = (255, 0, 0)
                elif k in [7, 8]:
                    jts_color = (139,69,19)
                elif k in [19, 20]:
                    jts_color = (0, 0, 0)
                else:
                    jts_color = (100, 100, 100)
                    
                
                draw.ellipse((joints2d[k][0]-circle_rad, joints2d[k][1]-circle_rad, joints2d[k][0]+circle_rad, joints2d[k][1]+circle_rad), fill=jts_color)
            
            os.makedirs(blend_vid_out_path, exist_ok=True)
            _img.save(blend_vid_out_path + f"/{i:04d}.jpg")
    
       
    if not blend_vid_out_path is None and render:
        os.system(f"/usr/bin/ffmpeg -y -framerate 30 -i {blend_vid_out_path}/%04d.jpg -vcodec libx264 -pix_fmt yuv420p {blend_vid_out_path}.mp4")  
    
    return


def save_quantitative_evaluation(evaluator_object, pred_dict, pymafx_dict, gt_dict, metrics, viz_flag=True, misc={}):
    
    # form camera intrinsics
    img_H, img_W = pred_dict["img_height"], pred_dict["img_width"]
    pred_dict["source"] = "hmp"

    hmp_quant_metric_vals = evaluator_object.compute_metric(model_output=pred_dict, 
                                                            targets=gt_dict, 
                                                            metrics=metrics)
    
    if viz_flag:
        for alignment_type in ['root', 'procrustes']:
            html_filepath = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_hmp.html")
            vid_filepath = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_hmp.mp4")
            frame_dir = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_hmp_open3d_img")
            
            evaluator_object.create_canvas_open3d(img_width=img_H, img_height=img_W, cam_intrinsics=gt_dict["cam_intrinsics"], 
                                        img_dir=pred_dict["img_dir"].item(), video_path=vid_filepath, alignment_type=alignment_type, 
                                        valid_indices=gt_dict["frame_id"], frame_dir=frame_dir)

    
    # optim/CONFIGNAME/performance_sequence.pkl
    cfg_name = pred_dict["config_type"].item()
    pymafx_quant_metric_vals = None
    
    if cfg_name in ["_slerp_encode_decode", "_encode_decode"]:
        pymafx_dict["source"] = "pymafx"
        pymafx_quant_metric_vals = evaluator_object.compute_metric(model_output=pymafx_dict, 
                                                                   targets=gt_dict, 
                                                                   metrics=metrics)
    
    # if viz_flag:
    #     for alignment_type in ['root', 'procrustes']:
    #         html_filepath = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_pymafx.html")
    #         vid_filepath = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_pymafx.mp4")
    #         frame_dir = os.path.join(os.path.dirname(pred_dict["save_path"].item()), f"{alignment_type}_aligned_pymafx_open3d_img")
        
    #         # evaluator_object.create_canvas_scenepic(img_width=img_H , img_height=img_W, cam_intrinsics=gt_dict["cam_intrinsics"], 
    #         #                            img_dir=pred_dict["img_dir"], save_dir=html_filepath, alignment_type=alignment_type, 
    #         #                            valid_indices=gt_dict["frame_id"])
    #         evaluator_object.create_canvas_open3d(img_width=img_H, img_height=img_W, cam_intrinsics=gt_dict["cam_intrinsics"], 
    #                                     img_dir=pred_dict["img_dir"].item(), video_path=vid_filepath, alignment_type=alignment_type, 
    #                                     valid_indices=gt_dict["frame_id"], frame_dir=frame_dir)
     
    dname = gt_dict["dataset_name"]
    exp_setup_name = gt_dict["exp_setup_name"]


    # performance_sequence = f"./optim/{cfg_name}/{dname}/performance_sequence_{exp_setup_name}.pkl"
    # performance_per_frame = f"./optim/{cfg_name}/{dname}/performance_per_frame_{exp_setup_name}.pkl"

    name = gt_dict["path"].replace("/", "_").replace(".", "")

    os.makedirs(f"./optim/{cfg_name}/{dname}/performance_sequence", exist_ok=True)
    os.makedirs(f"./optim/{cfg_name}/{dname}/performance_per_frame", exist_ok=True)

    performance_sequence = f"./optim/{cfg_name}/{dname}/performance_sequence/{name}.pkl"
    performance_per_frame = f"./optim/{cfg_name}/{dname}/performance_per_frame/{name}.pkl"
 

    if not os.path.isfile(performance_sequence):
        print("Creating new performance file")
        joblib.dump({cfg_name: {}, "pymafx": {}}, performance_sequence)
        
    if not os.path.isfile(performance_per_frame):
        print("Creating new performance file")
        joblib.dump({cfg_name: {}, "pymafx": {}}, performance_per_frame)
        
    print("===================HMP====================="	)
    for metric_name, metric_val in hmp_quant_metric_vals.items():
        if metric_name != "valid_frames":    
            # take mean over joints. 
            hmp_quant_metric_vals[metric_name] = metric_val.mean(axis=1)   
            print(metric_name, "(mm): ", f"{metric_val.mean():.3f}")
            
    if not pymafx_quant_metric_vals is None:
        print("==================PYMAFX==================="	)
        for metric_name, pymafx_metric_val in pymafx_quant_metric_vals.items():    
            if metric_name != "valid_frames":    
                pymafx_quant_metric_vals[metric_name] = pymafx_metric_val.mean(axis=1)  
                print(metric_name, "(mm): ", f"{pymafx_metric_val.mean():.3f}")
        
    print("==========================================="	)  
   
    with open(performance_per_frame, "rb") as f:
        perframe_pkl_file = joblib.load(f)

        if cfg_name not in perframe_pkl_file.keys(): 
            perframe_pkl_file[cfg_name] = {}

        # save per frame values
        perframe_pkl_file[cfg_name][gt_dict["path"]] = hmp_quant_metric_vals
        perframe_pkl_file["pymafx"][gt_dict["path"]] = pymafx_quant_metric_vals
        
        joblib.dump(perframe_pkl_file, performance_per_frame)

    with open(performance_sequence, "rb") as f:
        seq_pkl_file = joblib.load(f)

        if cfg_name not in seq_pkl_file.keys(): 
            seq_pkl_file[cfg_name] = {}
        
        # take mean over time.  
        seq_pkl_file[cfg_name][gt_dict["path"]] = {k:v.mean(axis=0) if k!= "valid_frames" else v for k, v in hmp_quant_metric_vals.items()} 
        if pymafx_quant_metric_vals is not None:
            seq_pkl_file["pymafx"][gt_dict["path"]] = {k:v.mean(axis=0) if k!= "valid_frames" else v for k, v in pymafx_quant_metric_vals.items()}
        joblib.dump(seq_pkl_file, performance_sequence)
    
    return 

def vis_mano_skeleton_matplotlib(gt_dict, pred_dict, idx=None):
  
    if idx is None:
        idx = np.random.randint(0, len(gt_dict["joints_3d"]))
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(pred_dict["joints_3d"][idx, :, 0], pred_dict["joints_3d"][idx, :, 1], pred_dict["joints_3d"][idx, :, 2], color='red')
    ax.scatter(gt_dict["joints_3d"][idx, :, 0], gt_dict["joints_3d"][idx, :, 1], gt_dict["joints_3d"][idx, :, 2], color='green')

    for i in range(21):
        if AUGMENTED_MANO_CHAIN[i] == -1:
            continue
        
        parent = pred_dict["joints_3d"][idx][AUGMENTED_MANO_CHAIN[i]]
        child = pred_dict["joints_3d"][idx][i]
        ax.plot([child[0], parent[0]], [child[1], parent[1]], zs=[child[2], parent[2]], color='red')
                
        parent_i = gt_dict["joints_3d"][idx][AUGMENTED_MANO_CHAIN[i]]
        child_i = gt_dict["joints_3d"][idx][i]
        
        ax.plot([child_i[0], parent_i[0]], [child_i[1], parent_i[1]], [child_i[2], parent_i[2]], color='green')
        

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.savefig("test.png")
    
    return 