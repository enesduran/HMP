
 # Copyright (c) OpenMMLab. All rights reserved.
import os
import cv2
import glob
import time
import joblib
import logging
import mimetypes
import subprocess
import numpy as np
from tqdm import tqdm 
import mediapipe as mp
import json_tricks as json

# mmpose related imports
import mmcv
import mmengine
from mmpose.registry import VISUALIZERS
from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline
from mmpose.apis import init_model as init_pose_estimator
from mmdet.apis import inference_detector, init_detector
from mmpose.structures import merge_data_samples, split_instances


 
def export_json(rh_joints, file_name):
    dic = {}
    dic['version'] = '1.5'
    dic["people"] = []
    person_dic = {}
 
    person_dic["person_id"] = [-1]
    person_dic["pose_keypoints_2d"] = []
    person_dic["face_keypoints_2d"] = []
    person_dic["hand_left_keypoints_2d"] = []        
    person_dic["hand_right_keypoints_2d"] = rh_joints
    
    dic["people"].append(person_dic)

    with open(file_name, 'w') as fp:
        json.dump(dic, fp)
    return


def expand_bbox(_bbox_, image_width, image_height, EXPAND_COEF=1.2):
        # expand the bbox by a factor of coef
        center = ((_bbox_[0]+_bbox_[2])/2, (_bbox_[1]+_bbox_[3])/2)
        assert (0 <= center[0] < image_width) and (0 <= center[1] <= image_height), "Center is out of image"
        width = abs(_bbox_[2] - _bbox_[0])
        height = abs(_bbox_[3] - _bbox_[1])

        new_width = width * EXPAND_COEF
        new_height = height * EXPAND_COEF
        
        x1 = int(max(0.0, center[0] - new_width/2))
        x2 = int(min(image_width, center[0] + new_width/2))
        y1 = int(max(0.0, center[1] - new_height/2))
        y2 = int(min(image_height, center[1] + new_height/2)) 
        
        return [x1, y1, x2, y2]

class MMPOSEJson:
    def __init__(self,
                det_config="./external/mmpose/demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py",
                det_checkpoint="./data/mmpose_models/cascade_rcnn_x101_64x4d_fpn_20e_onehand10k-dac19597_20201030.pth",
                pose_config="./external/mmpose/configs/hand_2d_keypoint/topdown_heatmap/onehand10k/td-hm_res50_8xb32-210e_onehand10k-256x256.py",
                pose_checkpoint="./data/mmpose_models/res50_onehand10k_256x256-e67998f6_20200813.pth",
                device='cuda:0',
                radius=3,
                alpha=0.8,
                thickness=1,
                bbox_thr=0.3,
                nms_thr=0.3,
                kpt_thr=0.3,
                wait_time = 0,
                det_cat_id=0,
                draw_bbox=True,
                draw_heatmap=False,
                show_kpt_idx=False,
                skeleton_style='mmpose',   # ['mmpose', 'openpose']
                 ):
            
        # build detector
        self.detector = init_detector(det_config, det_checkpoint, device=device)
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

        # build pose estimator
        self.pose_estimator = init_pose_estimator(pose_config, pose_checkpoint,device=device,
            cfg_options=dict(model=dict(test_cfg=dict(output_heatmaps=draw_heatmap))))
        

        # build visualizer
        self.pose_estimator.cfg.visualizer.radius = radius
        self.pose_estimator.cfg.visualizer.alpha = alpha
        self.pose_estimator.cfg.visualizer.line_width = thickness
        self.visualizer = VISUALIZERS.build(self.pose_estimator.cfg.visualizer)
        # the dataset_meta is loaded from the checkpoint and
        # then pass to the model in init_pose_estimator
        self.visualizer.set_dataset_meta(self.pose_estimator.dataset_meta, skeleton_style=skeleton_style)
        
        self.skeleton_style = skeleton_style
        self.show_kpt_idx = show_kpt_idx
        self.draw_heatmap = draw_heatmap
        self.det_cat_id = det_cat_id
        self.wait_time = wait_time
        self.draw_bbox = draw_bbox
        self.bbox_thr = bbox_thr
        self.nms_thr = nms_thr
        self.kpt_thr = kpt_thr
      

    def process_one_image(self, img, gt_bbox):
        """Visualize predicted keypoints (and heatmaps) of one image."""

        # predict bbox
        det_result = inference_detector(self.detector, img)
        pred_instance = det_result.pred_instances.cpu().numpy()

        if gt_bbox is None:
            bboxes = np.concatenate(
                (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
            bboxes = bboxes[np.logical_and(pred_instance.labels == self.det_cat_id,
                                        pred_instance.scores > self.bbox_thr)]
            bboxes = bboxes[nms(bboxes, self.nms_thr), :4]

        else: 
            bboxes = gt_bbox[None, :4]
 
        # predict keypoints
        pose_results = inference_topdown(self.pose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)

        # show the results
        if isinstance(img, str):
            img = mmcv.imread(img, channel_order='rgb')
        elif isinstance(img, np.ndarray):
            img = mmcv.bgr2rgb(img)

        self.visualizer.add_datasample(
            'result',
            img,
            data_sample=data_samples,
            draw_gt=False,
            draw_heatmap=self.draw_heatmap,
            draw_bbox=self.draw_bbox,
            show_kpt_idx=self.show_kpt_idx,
            skeleton_style=self.skeleton_style,
            show=False,
            wait_time=self.wait_time,
            kpt_thr=self.kpt_thr)

        # if there is no instance detected, return None
        return data_samples.get('pred_instances', None)

 
    def main(self, video_path, out_folder_path, json_folder, gt_dict):
    
        os.makedirs(out_folder_path, exist_ok=True)
        os.makedirs(json_folder, exist_ok=True)
        mmengine.mkdir_or_exist(out_folder_path)
        
        vid_out_path = os.path.join(os.path.dirname(video_path), "rgb_mmpose.mp4")

        frame_idx, valid_detection = 0, None
        is_itw = len(gt_dict["frame_id"]) == 0
        DIST_THR = 100

        vid_prefix = 'rgb_pseudo_right' if video_path.endswith('rgb_pseudo_raw.mp4') else 'rgb'
         
        if len(glob.glob(os.path.join(os.path.dirname(video_path), f"{vid_prefix}/*.jpg"))) > 0:
            frame_names = sorted(glob.glob(os.path.join(os.path.dirname(video_path), f"{vid_prefix}/*.jpg"))) 
            suffix = ".jpg"
        elif len(glob.glob(os.path.join(os.path.dirname(video_path), f"{vid_prefix}/*.png"))) > 0:
            frame_names = sorted(glob.glob(os.path.join(os.path.dirname(video_path), f"{vid_prefix}/*.png")))
            suffix = ".png"

        for frame_idx, frame_name in tqdm(enumerate(frame_names)): 

            frame = cv2.imread(frame_name)            
            bad_flag = False
            
            # if in_the_lab setting  
            
            if not is_itw:
                # and frame_idx not in gt_dict["frame_id"]:
                if frame_idx not in gt_dict["frame_id"]:
                    bad_flag = True
                    gt_bbox = None 
                else:
                    real_idx = gt_dict["frame_id"].index(frame_idx)
                    gt_bbox = gt_dict["bbox"][real_idx]
            else:
                gt_bbox = None 
                
            image_filename = os.path.join(out_folder_path, f"{frame_idx:04d}.jpg")
            json_filename = os.path.join(json_folder, f"{frame_idx:04d}_keypoints.json")
            pkl_filename = os.path.join(json_folder, f"{frame_idx:04d}.pkl")
            
            # topdown pose estimation
            pred_instances = self.process_one_image(frame, gt_bbox)
            
            # no detection or no gt present 
            if bad_flag or (pred_instances is None):
                bad_flag = True
            # detection 
            else:
                # multiple detections
                if len(pred_instances) > 1 and valid_detection is None:
                    
                    # forward it to mediapipe
                    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.7) as hands:
                        
                        bbox1 = pred_instances["bboxes"][0]
                        bbox2 = pred_instances["bboxes"][1]
                                        
                        bbox1 = expand_bbox(bbox1, EXPAND_COEF=1.2, 
                                    image_width=frame.shape[1], 
                                    image_height=frame.shape[0])
                        bbox2 = expand_bbox(bbox2, EXPAND_COEF=1.2, 
                                    image_width=frame.shape[1], 
                                    image_height=frame.shape[0])
                        
                        crop_img1 = frame.copy()[max(int(bbox1[1]), 0):min(int(bbox1[3]), frame.shape[0]), 
                                                max(int(bbox1[0]), 0):min(int(bbox1[2]), frame.shape[1])]
                        crop_img2 = frame.copy()[max(int(bbox2[1]), 0):min(int(bbox2[3]), frame.shape[0]), 
                                                max(int(bbox2[0]), 0):min(int(bbox2[2]), frame.shape[1])]
                        
                        results1 = hands.process(cv2.cvtColor(crop_img1, cv2.COLOR_BGR2RGB))
                        results2 = hands.process(cv2.cvtColor(crop_img2, cv2.COLOR_BGR2RGB))
                    
                        # if the first detection MediaPipe is valid, we use the second one. MediaPipe assumes reflected image. 
                        if results1.multi_hand_landmarks:
                            idx = 0 if results1.multi_handedness[0].classification[0].label == "Left" else 1
                        elif results2.multi_hand_landmarks:
                            idx = 0 if results2.multi_handedness[0].classification[0].label == "Left" else 1
                        # if both are not detected by MediaPipe, we select the rightmost one, hope that is the right hand. 
                        else:
                            idx = int(np.argmin([det["bboxes"][0, 0] for det in pred_instances]))
   
                elif len(pred_instances) > 1 and valid_detection is not None:
                    # if we have gt data, we can use it to select the best prediction
                    keyp0 = np.array(pred_instances[0]["keypoints"])
                    keyp1 = np.array(pred_instances[1]["keypoints"])
                    
                    dist0 = np.sum((rh_keypoints[:, :2] - keyp0)**2)
                    dist1 = np.sum((rh_keypoints[:, :2] - keyp1)**2)
                    
                    # if we don't have gt data, we can use the closest detection    
                    idx = 0 if dist0 < dist1 else 1
               
                # single detection
                else:
                    
                    assert len(pred_instances) == 1, "Something wrong with the detection"
                    
                    if valid_detection is None:
                        with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.8) as hands:
                            bbox1 = pred_instances["bboxes"][0]
                            bbox1 = expand_bbox(bbox1, EXPAND_COEF=1.2, 
                                        image_width=frame.shape[1], 
                                        image_height=frame.shape[0])
                            crop_img1 = frame.copy()[max(int(bbox1[1]), 0):min(int(bbox1[3]), frame.shape[0]), 
                                                    max(int(bbox1[0]), 0):min(int(bbox1[2]), frame.shape[1])]
                            results1 = hands.process(cv2.cvtColor(crop_img1, cv2.COLOR_BGR2RGB))
                            
                            # have a mediapipe detection
                            if results1.multi_hand_landmarks:
                                if results1.multi_handedness[0].classification[0].label == "Left":
                                    idx = 0
                                else:
                                    # print(f"{frame_idx}: MediaPipe detected a left hand, but we assume it is a right hand")
                                    bad_flag = True
                            else:
                                # print(f"{frame_idx}: MediaPipe did not detect any hand. Just skip it.")
                                bad_flag = True
    
                    else:
                        
                        # see if there is a hand jump 
                        a1, b1 = (valid_detection[0]["bboxes"][0, 0] + valid_detection[0]["bboxes"][0, 2])/2, \
                            (valid_detection[0]["bboxes"][0, 1] + valid_detection[0]["bboxes"][0, 3])/2
                        
                        a2, b2 = (pred_instances[0]["bboxes"][0, 0] + pred_instances[0]["bboxes"][0, 2])/2,   \
                            (pred_instances[0]["bboxes"][0, 1] + pred_instances[0]["bboxes"][0, 3])/2
                                
                        dist = np.sqrt((a1-a2)**2 + (b1-b2)**2)
                        
                        if dist > DIST_THR:
                            # print(f"{frame_idx}: Hand jump detected: {dist}")
                            bad_flag = True
                        else:
                            idx = 0 

            
            if bad_flag:
                rh_bbox = np.array([0, 0, frame.shape[0], frame.shape[1], 0])
                rh_keypoints = np.zeros((21, 3))
                valid_detection = None
            else:      

                bbox, = pred_instances[idx]["bboxes"]    # bbox values are float         
                bbox_score = pred_instances[idx]["bbox_scores"]
                rh_bbox = np.concatenate((bbox.astype(np.int32), bbox_score), axis=0)
                
                keyp = np.array(pred_instances[idx]["keypoints"])[0]
                keyp_scores = np.array(pred_instances[idx]["keypoint_scores"])[0, ..., None]
                rh_keypoints = np.concatenate((keyp, keyp_scores), axis=1)  
                valid_detection = pred_instances[idx]
                
            
            # save prediction results
            export_json(rh_keypoints.tolist(), json_filename)
            joblib.dump(np.array(rh_bbox), pkl_filename)

            # save image 
            frame_vis = cv2.cvtColor(self.visualizer.get_image(), cv2.COLOR_BGR2RGB)
           
            cv2.imwrite(f"{out_folder_path}/{frame_idx:04d}.jpg", frame_vis)            

        cmd = f"/usr/bin/ffmpeg -y -pattern_type glob -i '{out_folder_path}/*.jpg' -r 30 {os.path.join(os.path.dirname(out_folder_path), 'rgb_mmpose.mp4')}"
        subprocess.run(cmd, shell=True)
 
 