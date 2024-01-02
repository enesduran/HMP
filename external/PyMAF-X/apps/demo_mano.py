# -*- coding: utf-8 -*-
# This script is borrowed and extended from https://github.com/mkocabas/VIBE/blob/master/demo.py and https://github.com/nkolot/SPIN/blob/master/demo.py
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'

import cv2
import time
import glob
import torch
import joblib
import argparse
import subprocess
import numpy as np
import pickle as pkle
import os.path as osp
from tqdm import tqdm
from PIL import Image, ImageDraw
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.transforms import Normalize

from core import path_config
from models import hmr, pymaf_net
from core.cfgs import cfg, parse_args
from models.smpl import get_partial_smpl
from datasets.inference import Inference
from utils.geometry import convert_to_full_img_cam, rotation_matrix_to_angle_axis
from utils.demo_utils import (convert_crop_cam_to_orig_img, video_to_images, images_to_video)
 


def obtain_conf(std_t, CONF_THRESHOLD=4):
    # std_t is of shape (N, 21)    
    conf_ = np.zeros_like(std_t)
    
    nonzero_idx = np.where(std_t < CONF_THRESHOLD)
    conf_[nonzero_idx] = (CONF_THRESHOLD - std_t[nonzero_idx]) / CONF_THRESHOLD
    
    return conf_ 


def transform_2d_pts(pts, tr):
    pts = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=1)
    pts = np.matmul(tr, pts.T).T
    return pts
 
def prepare_rendering_results(person_data, nframes):
    frame_results = [{} for _ in range(nframes)]
    for idx, frame_id in enumerate(person_data['frame_ids']):
        person_id = person_data['person_ids'][idx],
        frame_results[frame_id][person_id] = {
            'verts': person_data['verts'][idx],
            'smplx_verts': person_data['smplx_verts'][idx] if 'smplx_verts' in person_data else None,
            'cam': person_data['orig_cam'][idx],
            'cam_t': person_data['orig_cam_t'][idx] if 'orig_cam_t' in person_data else None}

    # naive depth ordering based on the scale of the weak perspective camera
    for frame_id, frame_data in enumerate(frame_results):
        # sort based on y-scale of the cam in original image coords
        sort_idx = np.argsort([v['cam'][1] for k,v in frame_data.items()])
        frame_results[frame_id] = OrderedDict(
            {list(frame_data.keys())[i]:frame_data[list(frame_data.keys())[i]] for i in sort_idx}
        )

    return frame_results

# in-the-wild setting
def read_mmpose_bbox(index):   
 
    if "DexYCB" in args.image_folder or "HO3D" in args.image_folder:
        bbox_folder_path = os.path.join(os.path.dirname(args.image_folder), "mmpose_keypoints2d")
    else:
        # fill the same way as in fitting_app.py
        bbox_folder_path = args.image_folder.replace("raw_images", "keypoints2d") 
    
    bbox_file = sorted(glob.glob(os.path.join(bbox_folder_path, "*.pkl")))[index]

    return joblib.load(bbox_file)


def read_gt_bbox():
   
    bb_path = os.path.join(os.path.dirname(args.image_folder), "pymafx_out", "gt_bbox.npz")
    
    try:
        gt_dict = np.load(bb_path, allow_pickle=True)  
    except:
        gt_dict = None
        print("No gt bbox found for PYMAFX, using mmpose bbox instead") 
 
    return gt_dict



def run_demo(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.image_folder is None:
        video_file = args.vid_file

        if not os.path.isfile(video_file):
            exit(f'Input video \"{video_file}\" does not exist!')
        
        output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))

        image_folder, num_frames, _ = video_to_images(video_file, return_info=True)
    else:
        image_folder = args.image_folder

        jpg_files = glob.glob(os.path.join(image_folder, '*.jpg'))
        
        if len(jpg_files) == 0:    
            jpg_files = glob.glob(os.path.join(image_folder, '*.png'))
        
        num_frames =  len(jpg_files)
        output_path = args.output_folder

    os.makedirs(output_path, exist_ok=True)

    print(f'Input video number of frames {num_frames}')

    total_time = time.time()

    args.device = device
    args.pin_memory = True if torch.cuda.is_available() else False

    gt_bbox_all = read_gt_bbox()

    # hand detection 
    pp_det_file_path = os.path.join(output_path, 'pp_det_results.pkl')

   
    if args.vid_file is not None:
        pass
    elif args.image_folder is not None:

        if "DexYCB" in args.image_folder:
            image_file_names = sorted([osp.join(image_folder, x) 
                for x in os.listdir(image_folder) if x.endswith('.jpg')])
        else:
            image_file_names = sorted([osp.join(image_folder, x) 
                for x in os.listdir(image_folder) if x.endswith('.png') or x.endswith('.jpg')])

    tracking_results = {}
    
    if os.path.isfile(pp_det_file_path):
        print(f'Loading person detection results from {pp_det_file_path}')
        tracking_results = joblib.load(pp_det_file_path)
    else:        
        print('Reading bounding boxes from mmpose')

        for i in tqdm(range(num_frames)):

            det_wb_kps = np.ones(shape=(133, 3))
            det_wb_kps[:, 2] = 1e-4
            det_face_kps = det_wb_kps[23:91]

            # if gt bbox is not available, use it. If no, use mmpose bbox 
            if i not in gt_bbox_all["frame_id"]:
                continue
            else:
                frame_i = list(gt_bbox_all["frame_id"]).index(i)
                gt_bbox_t = gt_bbox_all["bbox"][frame_i]
                handedness_i = gt_bbox_all["handedness"] if gt_bbox_all is not None else "right"
                
                bbox_delta_x, bbox_delta_y = 0, 0
                
                if gt_bbox_t is not None:
                    bbox_delta_x = abs(gt_bbox_t[2] - gt_bbox_t[0])
                    bbox_delta_y = abs(gt_bbox_t[3] - gt_bbox_t[1])
                    
                    # check entries in bbox_i, height or width may be 0. There are some cases 
                    # e.g. './data/rgb_data/DexYCB/20200820-subject-03/20200820_142158/932122060861/rgb_gt'
                    if bbox_delta_x > 0 and bbox_delta_y > 0:
                        bbox_i = np.concatenate((gt_bbox_t, [1]))
                    else:
                        continue

                # there are some frames with no bbox but joints2d and joints3d is provided, use mmpose bbox instead
                else:          
                    bbox_i = read_mmpose_bbox(index=i)
                    if bbox_i[-1] == 0:
                        continue 
                
            tracking_results[i] = {
                            'frames': [i],
                            'handedness': handedness_i,
                            'joints2d': [det_wb_kps[:17]],
                            'joints2d_lhand': [det_wb_kps[91:112]],
                            'joints2d_rhand': [det_wb_kps[112:133]],
                            'joints2d_face': [np.concatenate([det_face_kps[17:], det_face_kps[:17]])],
                            'vis_face': [np.mean(det_face_kps[17:, -1])],
                            'vis_lhand': [np.mean(det_wb_kps[91:112, -1])],
                            'vis_rhand': [np.mean(det_wb_kps[112:133, -1])],
                            'bbox': list(bbox_i)}
            
        pkle.dump(tracking_results, open(pp_det_file_path, 'wb'))

    bbox_scale = 1.0
    
    # ========= Define model ========= #
    model = pymaf_net(path_config.SMPL_MEAN_PARAMS, is_train=False).to(device)

    # ========= Load pretrained weights ========= #
    checkpoint_paths = {'body': args.pretrained_body, 'hand': args.pretrained_hand, 'face': args.pretrained_face}
    if args.pretrained_model is not None:
        print(f'Loading pretrained weights from \"{args.pretrained_model}\"')
        checkpoint = torch.load(args.pretrained_model)

        # remove the state_dict overrode by hand and face sub-models
        for part in ['hand', 'face']:
            if checkpoint_paths[part] is not None:
                key_start_list = model.part_module_names[part].keys()
                for key in list(checkpoint['model'].keys()):
                    for key_start in key_start_list:
                        if key.startswith(key_start):
                            checkpoint['model'].pop(key)

        # load the state_dict. strict=false for hands. 
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f'loaded checkpoint: {args.pretrained_model}')

    if not all([args.pretrained_body is None, args.pretrained_hand is None, args.pretrained_face is None]):
        for part in ['body', 'hand', 'face']:
            checkpoint_path = checkpoint_paths[part]
            if checkpoint_path is not None:
                print(f'Loading checkpoint for the {part} part.')
                checkpoint = torch.load(checkpoint_path)['model']
                checkpoint_filtered = {}
                key_start_list = model.part_module_names[part].keys()
                for key in list(checkpoint.keys()):
                    for key_start in key_start_list:
                        if key.startswith(key_start):
                            checkpoint_filtered[key] = checkpoint[key]
                model.load_state_dict(checkpoint_filtered, strict=False)
                print(f'Loaded checkpoint for the {part} part.')

    model.eval()

    smpl2limb_vert_faces = get_partial_smpl(args.render_model)
 
    # ========= Run pred on each person ========= #
    if args.recon_result_file:
        pred_results = joblib.load(args.recon_result_file)
        print('Loaded results from ' + args.recon_result_file)
    else:
        if args.pre_load_imgs:
            image_file_names = [
                osp.join(image_folder, x)
                for x in os.listdir(image_folder)
                if x.endswith('.png') or x.endswith('.jpg')
            ]
            image_file_names = sorted(image_file_names)
            image_file_names = np.array(image_file_names)
            pre_load_imgs = []
            for file_name in image_file_names:
                pre_load_imgs.append(cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB))
            pre_load_imgs = np.array(pre_load_imgs)
          
        else:
            image_file_names = None
        
       
        joints2d = []
        frames= []
        
        print(f'Running reconstruction on each tracklet...')
        pred_time = time.time()
        pred_results = {}
        frames= []
        if args.tracking_method == 'pose':
            wb_kps = {'joints2d_lhand': [],
                      'joints2d_rhand': [],
                      'joints2d_face': [],
                      'vis_face': [],
                      'vis_lhand': [],
                      'vis_rhand': [],
                      'bbox': [], 'handedness': []}
                    
        person_id_list = list(tracking_results.keys())
        for person_id in person_id_list:
            if args.tracking_method == 'bbox':
                raise NotImplementedError
            elif args.tracking_method == 'pose':
               
                joints2d.extend(tracking_results[person_id]['joints2d'])
                wb_kps['joints2d_lhand'].extend(tracking_results[person_id]['joints2d_lhand'])
                wb_kps['joints2d_rhand'].extend(tracking_results[person_id]['joints2d_rhand'])
                wb_kps['joints2d_face'].extend(tracking_results[person_id]['joints2d_face'])
                wb_kps['vis_lhand'].extend(tracking_results[person_id]['vis_lhand'])
                wb_kps['vis_rhand'].extend(tracking_results[person_id]['vis_rhand'])
                wb_kps['vis_face'].extend(tracking_results[person_id]['vis_face'])

                # wb_kps['handedness'].extend(tracking_results[person_id]['handedness'])
                wb_kps['bbox'].extend(tracking_results[person_id]['bbox'])

            frames.extend(tracking_results[person_id]['frames'])    
            
        
        joints2d_image_folder = args.joints2d_image_folder
        os.makedirs(joints2d_image_folder, exist_ok=True)

        bboxes = gt_bbox_all["bbox"]
        joints2d = gt_bbox_all["bbox"]
        wb_kps["bbox"] = bboxes.reshape(-1)
        # if there is no bbox detection 
        if len(gt_bbox_all["bbox"]) ==0:
            pred_results = {'pred_cam': None,
                            'orig_cam': None,
                            'orig_cam_t': None,
                            'verts': None,
                            'joints3d': None,
                            "pymafx_joints2d": None,
                            'joints2d': None,
                            'bboxes': [],
                            'frame_ids': None,
                            'person_ids': None,
                            'smplx_params': {}}
            
            print('==========NOT A SINGLE DETECTION IN SEQUENCE==========')
            total_time = time.time() - total_time
            
            print(f'Saving output results to \"{os.path.join(output_path, "output.pkl")}\".')


            joblib.dump(pred_results, os.path.join(output_path, "output.pkl"))
            
            # output video still 
            for h in range(len(sorted(glob.glob(os.path.join(image_folder, '*.jpg'))))):
                im_name = sorted(glob.glob(os.path.join(image_folder, '*.jpg')))[h]
                img = Image.open(im_name)
                img.save(f"{joints2d_image_folder}/{h:04d}.jpg")
                
            if args.out_vid_path is not None:
                cmd = f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{joints2d_image_folder}/*.jpg' -vcodec libx264 -pix_fmt yuv420p {args.out_vid_path} "
                subprocess.run(cmd, shell=True)
            
            return 

        else:
            if args.pre_load_imgs:
                dataset = Inference(image_folder=image_folder,
                                    frames=frames,
                                    bboxes=bboxes,
                                    joints2d=joints2d,
                                    scale=bbox_scale,
                                    pre_load_imgs=pre_load_imgs[frames],
                                    full_body=True,
                                    person_ids=person_id_list,
                                    wb_kps=wb_kps)
            else:
                
                #################################
                # for in-the-wild videos 
                if len(joints2d.shape) != 1:
                    if joints2d.shape[1] == 5:
                        joints2d_ = np.zeros((joints2d.shape[0], 2, 3))
                    
                        joints2d_[:, 0, :2] = joints2d[:, :2] 
                        joints2d_[:, 1, :2] = joints2d[:, 2:4] 
                        joints2d_[:, 0, -1] = joints2d[:, -1]
                        joints2d_[:, 1, -1] = joints2d[:, -1]
                        joints2d = joints2d_
                ##########################################
 

                dataset = Inference(image_folder=image_folder,
                                    frames=frames,
                                    bboxes=bboxes,
                                    joints2d=joints2d,
                                    scale=bbox_scale,
                                    full_body=True,   
                                    person_ids=person_id_list,
                                    wb_kps=wb_kps)

        bboxes = dataset.bboxes
        scales = dataset.scales
        frames = dataset.frames

        dataloader = DataLoader(dataset, batch_size=args.model_batch_size, num_workers=0)
        
        with torch.no_grad():

            pred_cam, pred_verts, pred_smplx_verts, pred_pose, pred_betas, pred_joints3d, norm_joints2d = [], [], [], [], [], [], []
            orig_height, orig_width = [], []
            person_ids = []
            
            images_rh = []  
            
            smplx_params = []

            for i, batch in enumerate(tqdm(dataloader)):
                
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k,v in batch.items()}
                
                orig_height.append(batch['orig_height'])
                orig_width.append(batch['orig_width'])
            
                images_rh.append(255 * batch["img_rhand_unnorm"].permute(0, 2, 3, 1).cpu().numpy())
                         
                preds_dict, _ = model(batch) 
                
 
                output = preds_dict['mesh_out'][-1]

                pred_cam.append(output['theta'][:, :3])
                pred_verts.append(output['verts_rh'])
      
                norm_joints2d.append(output['pred_rhand_kp2d'])
                pred_joints3d.append(output['pred_rhand_kp3d'])

                smplx_params.append({'shape' : output['pred_shape_rh'], 
                                    'root_orient' : output['pred_orient_rh_rotmat'],
                                    'body_pose' : output['pred_rhand_rotmat']})
   

            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            norm_joints2d = torch.cat(norm_joints2d, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)

            orig_height = torch.cat(orig_height, dim=0)
            orig_width = torch.cat(orig_width, dim=0)

            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        orig_height = orig_height.cpu().numpy()
        orig_width = orig_width.cpu().numpy()
        
        batch_size = pred_cam.shape[0]

        # Output 2D keypoints for both detected and non-detected cases. COnfidence is taken from bounding box detection.
        pymafx_joints2d_all = np.zeros((num_frames, 21, 3))                                  
        detection_cases = list(tracking_results.keys())

        pymafx_joints2d = bboxes[:, None, :2] + 112 * dataset.scales[:, None, None] * norm_joints2d.cpu().numpy()    

        conf = np.array(wb_kps['bbox'][4::5])                            # take every fifth element 
        conf = np.expand_dims(conf, axis=[1, 2])                         # add two dimensions
        conf = conf.repeat(pymafx_joints2d.shape[1], axis=1)             # repeat J times

        pymafx_joints2d_with_conf = np.concatenate([pymafx_joints2d, conf], axis=-1)
        pymafx_joints2d_all[detection_cases] = pymafx_joints2d_with_conf
            
        # Render 2d keypoints  
        openpose_skeleton = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19]
        circle_rad = 2

        raw_img_list = sorted([osp.join(image_folder, x) for x in os.listdir(image_folder) if x.endswith('.png') or x.endswith('.jpg')])
   
        for h in tqdm(range(len(raw_img_list))):
            right_hand_tmp = pymafx_joints2d_all[h]
            im_name = raw_img_list[h]
            img = Image.open(im_name)
            draw = ImageDraw.Draw(img)
            
            # draw hand bb 
            if h in detection_cases:
                h_ = detection_cases.index(h)

                bb = wb_kps['bbox'][5*h_:5*h_+4]
                draw.rectangle([(bb[0], bb[1]), (bb[2], bb[3])], outline=(0, 255, 0), width=2)    

            for k in range(21):        
            
                if openpose_skeleton[k] == -1:
                    continue

                if right_hand_tmp[k][2] == 0:
                    continue
            
                kps_parent = right_hand_tmp[openpose_skeleton[k]]
                kps_child = right_hand_tmp[k]

                draw.line([(kps_child[0], kps_child[1]), (kps_parent[0], kps_parent[1])], fill=(0, 0, 200), width=2)
                draw.ellipse((kps_child[0]-circle_rad, kps_child[1]-circle_rad, 
                                kps_child[0]+circle_rad, kps_child[1]+circle_rad), 
                                fill=(200, 0, 0))
     
            img.save(f"{joints2d_image_folder}/{h:04d}.jpg")


        if args.out_vid_path is not None:
            cmd = f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{joints2d_image_folder}/*.jpg' -vcodec libx264 -pix_fmt yuv420p {args.out_vid_path} "
            subprocess.run(cmd, shell=True)
         
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height)   
        camera_translation = convert_to_full_img_cam(
                                pare_cam=pred_cam,
                                bbox_height=scales * 200.,
                                bbox_center=bboxes[:, :2],
                                img_w=orig_width,
                                img_h=orig_height,
                                focal_length=5000.)

        # Do rendering in full image 
        if args.render:
            from smplx import MANO
            import open3d as o3d
            import open3d.visualization.rendering as rendering
 
            alpha_val = 0.8

            render = rendering.OffscreenRenderer(width=orig_width[0], height=orig_height[0])
            render.scene.set_background((1., 1., 1., 1.))
            render.scene.scene.set_sun_light([0.707, 0.0, -.707], [1.0, 1.0, 1.0], 75000)
            render.scene.scene.enable_sun_light(False)
            hand_model = MANO(model_path="./data/smpl/MANO_RIGHT.pkl", flat_hand_mean=True, use_pca=False, batch_size=1).to('cuda')
        
            # only gt s are availlable in images_rh
            for i, real_i in enumerate(gt_bbox_all["frame_id"]):
                
                root_orient = rotation_matrix_to_angle_axis(smplx_params[i//8]["root_orient"]).reshape(-1, 3)[i%8]
                poses = rotation_matrix_to_angle_axis(smplx_params[i//8]["body_pose"].reshape(-1, 3, 3)).reshape(-1, 15, 3)[i%8]
                betas = smplx_params[i//8]["shape"][i%8]
                
                if render.scene.has_geometry(f'smpl_body'):
                    render.scene.remove_geometry(f'smpl_body')
        
                mano_out = hand_model(betas=betas.unsqueeze(0),
                            global_orient=root_orient.view(-1, 3),
                            hand_pose=poses.reshape(-1, 45),
                            return_tips=True)
                            # transl=torch.tensor(camera_translation).to("cuda").unsqueeze(0))
            
                cam_extrinsics = np.eye(4)
                cam_extrinsics[:3, :3] = np.eye(3)
                cam_extrinsics[:3, 3] = camera_translation[i]
                
                # PyMAF-X assumess focal length of 5000 mm. 
                cam_int = np.array([[5000., 0, orig_width[0]/2],[0, 5000., orig_height[0]/2], [0, 0, 1.]])
                    
                o3d_pinhole_cam = o3d.camera.PinholeCameraIntrinsic(orig_width[0], orig_height[0], cam_int)
    
                render.setup_camera(o3d_pinhole_cam, cam_extrinsics.astype(np.float64))

                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.array(mano_out.vertices.detach().cpu()[0])), 
                                                    o3d.utility.Vector3iVector(hand_model.faces))
                mesh.compute_vertex_normals()
                mat = rendering.MaterialRecord()
                mat.base_color = [*np.array([0.65098039, 0.74117647, 0.85882353]).tolist(), 1.0]
                mat.shader = "defaultLit"
                
                render.scene.add_geometry(f'smpl_body', mesh, mat)

                seqname = image_folder.split("/")[-2]
                aligned_image_folder = f"./aligned_image_folder/{seqname}/{i:06d}.jpg"
                
                os.makedirs(os.path.dirname(aligned_image_folder), exist_ok=True)
                
                img_file_j = o3d.io.read_image(f"{joints2d_image_folder}/{real_i:04d}.jpg")  
    
                hand_rgb = np.asarray(render.render_to_image())
                img_rgb = np.asarray(img_file_j)
                
                valid_mask = (np.sum(hand_rgb, axis=-1) < 573)[:, :, np.newaxis]

                # blend the two images through masking alpha blending            
                img_overlay = valid_mask * hand_rgb * alpha_val + valid_mask * img_rgb * (1 - alpha_val) + img_rgb * (1 - valid_mask)
                
                img_overlay = o3d.geometry.Image((img_overlay).astype(np.uint8))
                
                o3d.io.write_image(aligned_image_folder, img_overlay)
 
            del render

        pred_results = {
            'pred_cam': pred_cam,
            'orig_cam_t': camera_translation,
            'verts': pred_verts,
            'joints3d': pred_joints3d,
            "pymafx_joints2d": pymafx_joints2d_all,
            'joints2d': joints2d,
            'bboxes': bboxes,
            'frame_ids': frames,
            'person_ids': person_ids,
            'smplx_params': smplx_params,
        }

        del model

        total_time = time.time() - total_time
        print(f'Total time spent for reconstruction: {total_time:.2f} seconds (including model loading time).')

        print(f'Saving output results to \"{os.path.join(output_path, "output.pkl")}\".')

        joblib.dump(pred_results, os.path.join(output_path, "output.pkl"))
    
    print('================= END =================')

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=CustomFormatter)
 
    parser.add_argument('--img_file', type=str, default=None,
                        help='Path to a single input image')
    parser.add_argument('--out_vid_path', type=str, default=None,
    help='Path to a single input image')
    parser.add_argument('--cam_K', type=str, default='',
                        help='Focal length of camera')
    parser.add_argument('--joints2d_image_folder', type=str, default=None,
                        help='Path to vis 2d keypoints on input image')
    parser.add_argument('--vid_file', type=str, default=None,
                        help='input video path or youtube link')
    parser.add_argument('--image_folder', type=str, default=None,
                        help='input image folder')
    parser.add_argument('--output_folder', type=str, default='output',
                        help='output folder to write results')
    parser.add_argument('--tracking_method', type=str, default='pose', choices=['bbox', 'pose'],
                        help='tracking method to calculate the tracklet of a subject from the input video')
    parser.add_argument('--detector_checkpoint', type=str, default='shufflenetv2k30-wholebody',
                        help='detector checkpoint for openpifpaf')
    parser.add_argument('--detector_batch_size', type=int, default=1,
                        help='batch size of person detection')
    parser.add_argument('--detection_threshold', type=float, default=0.55,
                        help='pifpaf detection score threshold.')
    parser.add_argument('--cfg_file', type=str, default='configs/pymafx_config.yaml',
                        help='config file path.')
    parser.add_argument('--pretrained_model', default=None,
                        help='Path to network checkpoint')
    parser.add_argument('--pretrained_body', default=None, help='Load a pretrained checkpoint for body at the beginning training') 
    parser.add_argument('--pretrained_hand', default=None, help='Load a pretrained checkpoint for hand at the beginning training') 
    parser.add_argument('--pretrained_face', default=None, help='Load a pretrained checkpoint for face at the beginning training') 

    parser.add_argument('--misc', default=None, type=str, nargs="*",
                        help='other parameters')
    parser.add_argument('--model_batch_size', type=int, default=8,
                        help='batch size for SMPL prediction')
    parser.add_argument('--display', action='store_true',
                        help='visualize the results of each step during demo')
    parser.add_argument('--render', action='store_true',
                        help='disable final rendering of output video.')
    parser.add_argument('--render_vis_ratio', type=float, default=1.,
                        help='transparency ratio for rendered results')
    parser.add_argument('--render_part', type=str, default='arm',
                        help='render part mesh')
    parser.add_argument('--render_model', type=str, default='smplx', choices=['smpl', 'smplx'],
                        help='render model type')
    parser.add_argument('--with_raw', action='store_true',
                        help='attach raw image.')
    parser.add_argument('--empty_bg', action='store_true',
                        help='render meshes on empty background.')
    parser.add_argument('--use_gt', action='store_true',
                        help='use the ground truth tracking annotations.')
    parser.add_argument('--anno_file', type=str, default='',
                        help='path to tracking annotation file.')
    parser.add_argument('--render_ratio', type=float, default=1.,
                        help='ratio for render resolution')
    parser.add_argument('--recon_result_file', type=str, default='',
                        help='path to reconstruction result file.')
    parser.add_argument('--pre_load_imgs', action='store_true',
                        help='pred-load input images.')
    parser.add_argument('--save_obj', action='store_true',
                        help='save results as .obj files.')

    args = parser.parse_args()
    parse_args(args)

    print('Running demo...')
    run_demo(args)
