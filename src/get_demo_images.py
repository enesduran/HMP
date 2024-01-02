import os 
import cv2
import glob
import json
import torch
import joblib
import shutil
import tempfile
import subprocess 
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d_viz_overlay
from datasets.amass import openpose_skeleton


# motion prior figure cases cand_13(frames 425, 433, and 440) and cand_21(frames 353, 359, 380, 425)
# poster bbox failure cand_23 (frames 50, 55, 60, 65, 92)
# keypoint failure cases MDF11 (frames 150 481)(also can select 450-600) AP11(frames 40)(also can select 0-120)
# mediapipe std BB13 70
# bbox_fail cand23 and cand20

def get_images():
     

    # seqname = "cand_20"
    seqname = "MDF11"

    if "cand" in seqname:
        datasetname = "in_the_wild"
        cfg = "_in_the_wild_setting_mmpose"
   
    else:
        cfg = "stage2_ts_0_os_1_trans_1_orient_2_rot_0_beta_10_mp_300_pp_0_js_0_reproj_0.05_lr_0.0501_400_blend_std"
        if seqname == "AP11":
            datasetname = "HO3D_v3/evaluation"
        else:
            datasetname = "HO3D_v3/train" 
         
    
    pymafx_path = f"./optim/_encode_decode/{datasetname}/{seqname}/recon_000_30fps_pymafx.mp4"
    hmp_npz_path = f"./optim/{cfg}/{datasetname}/{seqname}/recon_000_30fps.npz"
    hmp_vid_path = os.path.join(os.path.dirname(hmp_npz_path), "recon_000_30fps_hmp.mp4")

    mmpose_rgb_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_mmpose"
    mmpose_bbox_path = f"./data/rgb_data/{datasetname}/{seqname}/mmpose_keypoints2d"
    mmpose_vid_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_mmpose.mp4"
    
    raw_image_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb"
    mediapipe_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_mediapipe"
    mediapipe_keyp_path = f"./data/rgb_data/{datasetname}/{seqname}/mediapipe_keypoints2d"
    
    blend_keyp_path = f"./data/rgb_data/{datasetname}/{seqname}/blend_1.0_keypoints2d"
    blend_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_blend_1.0"
    blend_vid_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_blend_1.0.mp4"
    
    raw_vid_path = f"./data/rgb_data/{datasetname}/{seqname}/rgb_raw.mp4"
    raw_images = sorted([os.path.join(raw_image_path, x) for x in os.listdir(raw_image_path) if x.endswith('.png') or x.endswith('.jpg')]) 
    mediapipe_images = sorted([os.path.join(mediapipe_path, x) for x in os.listdir(mediapipe_path) if x.endswith('.png') or x.endswith('.jpg')])
    blend_images = sorted([os.path.join(blend_path, x) for x in os.listdir(blend_path) if x.endswith('.png') or x.endswith('.jpg')])
    mediapipe_keypoints = sorted(glob.glob(mediapipe_keyp_path + "/*.json"))
    blend_keypoints = sorted(glob.glob(blend_keyp_path + "/*.json"))
    timesteps = np.arange(0, len(raw_images), 1)
    
    open3d_viz_overlay.vis_opt_results(pred_file_path=hmp_npz_path, gt_file_path="", img_dir=raw_image_path, white_background=False)     
    
    # save them to method_images
    cmd = f"/usr/bin/ffmpeg -y -i {hmp_vid_path} -r 30  {frame_dir_hmp}/%06d.png"
    # os.system(cmd)
    
    img_height, img_width = cv2.imread(raw_images[0]).shape[:2]
    
    video_path = f"./trial/{seqname}_pose_estimation.mp4"
    cmd = f"/usr/bin/ffmpeg -y -i {raw_vid_path} -i {hmp_vid_path}  -filter_complex \
                    '[0]drawtext=text=VIDEO INPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [0:v]; \
                    [1]drawtext=text=OUTPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [1:v]; \
                    [0:v]scale=-1:{img_height}[0v]; \
                    [1:v]scale=-1:{img_height}[1v]; \
                    [0v][1v]hstack=inputs=2[outv]' \
                    -map '[outv]' {video_path}"
    # os.system(cmd)
    
    
    video_path = f"./trial/{seqname}_keypoint_fail.mp4"
    cmd = f"/usr/bin/ffmpeg -y -i {blend_vid_path} -i {hmp_vid_path}  -filter_complex \
                    '[0]drawtext=text=VIDEO INPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [0:v]; \
                    [1]drawtext=text=OUTPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [1:v]; \
                    [0:v]scale=-1:{img_height}[0v]; \
                    [1:v]scale=-1:{img_height}[1v]; \
                    [0v][1v]hstack=inputs=2[outv]' \
                    -map '[outv]' {video_path}"
    os.system(cmd)
    import ipdb; ipdb.set_trace()
    
    video_path = f"./trial/{seqname}_bbox_fail.mp4"
    cmd = f"/usr/bin/ffmpeg -y -i {mmpose_vid_path} -i {hmp_vid_path}  -filter_complex \
                    '[0]drawtext=text=VIDEO INPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [0:v]; \
                    [1]drawtext=text=OUTPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=blue: fontsize=w/30: x=text_w/8: y=text_h [1:v]; \
                    [0:v]scale=-1:{img_height}[0v]; \
                    [1:v]scale=-1:{img_height}[1v]; \
                    [0v][1v]hstack=inputs=2[outv]' \
                    -map '[outv]' {video_path}"
    os.system(cmd)
    import ipdb; ipdb.set_trace()
  
    # concatentate pymafx and hmp videos


    # cmd = f"/usr/bin/ffmpeg -i {pymafx_path} -r 30  {frame_dir_pymafx}/%06d.jpg"
    # os.system(cmd)
    
    

    
    # mmpose_images = sorted([os.path.join(mmpose_rgb_path, x) for x in os.listdir(mmpose_rgb_path) if x.endswith('.png') or x.endswith('.jpg')])
    os.makedirs("method_images/raw", exist_ok=True)
    os.makedirs("method_images/blend", exist_ok=True)
 

 
    for t in tqdm(timesteps):    

        continue
        # shutil.copyfile(blend_images[t], f"method_images/blend/{t:04d}.jpg")
        shutil.copyfile(raw_images[t], f"method_images/raw/{t:04d}.jpg")
        
        # shutil.copyfile(mediapipe_images[t], f"method_images/mediapipe_{t}.jpg")
        # shutil.copyfile(pymafx_images[t], f"method_images/pymafx_{t}.jpg")
        # shutil.copyfile(hmp_images[t], f"method_images/hmp_{t}.jpg")

    bbox_list = sorted(glob.glob(mmpose_bbox_path + "/*.pkl"))
    
    # render_mmpose(bbox_list, raw_images)  
    # render_mediapipe(blend_keyp_filename_list, raw_images)
    render_blend(blend_keypoints, raw_images)
    
    print("Done")

 

def render_mediapipe(jts_filenames, raw_im_list):
    
    assert len(jts_filenames) == len(raw_im_list)
    
    for i in range(len(jts_filenames)):
        
        raw_img_i = cv2.imread(raw_im_list[i])
        jts_i = np.array(json.load(open(jts_filenames[i]))["people"][0]["hand_right_keypoints_2d"]).reshape(21, 3)[:, :2]
        
        for k in range(21):   
            start_point = (int(jts_i[openpose_skeleton[k]][0]), int(jts_i[openpose_skeleton[k]][1])) 
            end_point = (int(jts_i[k][0]), int(jts_i[k][1]))  
            
            cv2.circle(raw_img_i, (int(start_point[0]), int(start_point[1])), 1, (0, 0, 0), thickness=4)
            cv2.circle(raw_img_i, (int(end_point[0]), int(end_point[1])), 1, (0, 0, 0), thickness=4)
            
            if not openpose_skeleton[k] == -1:
                cv2.line(raw_img_i, start_point, end_point, color=(0, 150, 0), thickness=2) 
        cv2.imwrite(f"method_images/mediapipe/{i:04d}.jpg", raw_img_i)        
        
    return 

def render_blend(jts_filenames, raw_im_list):

    assert len(jts_filenames) == len(raw_im_list)

    for i in tqdm(range(len(jts_filenames))):
        
        raw_img_i = cv2.imread(raw_im_list[i])
        jts_i = np.array(json.load(open(jts_filenames[i]))["people"][0]["hand_right_keypoints_2d"]).reshape(21, 3)[:, :2]
        
        for k in range(21):   
            start_point = (int(jts_i[openpose_skeleton[k]][0]), int(jts_i[openpose_skeleton[k]][1])) 
            end_point = (int(jts_i[k][0]), int(jts_i[k][1]))  
            
            cv2.circle(raw_img_i, (int(start_point[0]), int(start_point[1])), 1, (0, 0, 0), thickness=4)
            cv2.circle(raw_img_i, (int(end_point[0]), int(end_point[1])), 1, (0, 0, 0), thickness=4)
            
            if not openpose_skeleton[k] == -1:
                cv2.line(raw_img_i, start_point, end_point, color=(0, 150, 0), thickness=2) 
        cv2.imwrite(f"method_images/blend/{i:04d}.jpg", raw_img_i)        
    return 

def render_mmpose(bbox_list, raw_im_list):
    
    os.makedirs("method_images/mmpose_bbox", exist_ok=True)
    
    for i in tqdm(range(len(bbox_list))):
        
        raw_img_i = cv2.imread(raw_im_list[i])
        
        bbox = joblib.load(bbox_list[i]).astype(int)
    
        
        if not (bbox[0] == 0 and bbox[1] == 0):
            cv2.rectangle(raw_img_i, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color=(255, 0, 0), thickness=4)
            
        cv2.imwrite(f"method_images/mmpose_bbox/{i:04d}_mmpose.jpg", raw_img_i)      
          
    return


if __name__ == "__main__":

    os.makedirs("method_images", exist_ok=True) 
    frame_dir_pymafx = "method_images/pymafx"
    frame_dir_hmp = "method_images/hmp"
    
    os.makedirs(frame_dir_pymafx, exist_ok=True)
    os.makedirs(frame_dir_hmp, exist_ok=True)
    
    get_images()