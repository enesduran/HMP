import os 
import cv2
import glob
import json
import subprocess
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from PIL import Image, ImageDraw
from google.protobuf.json_format import MessageToDict

from datasets.amass import openpose_skeleton


def detect_mp_(img, bbox):
    with mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.85) as hands:
        crop_img = img.copy()[max(int(bbox[1]), 0):min(int(bbox[3]), img.shape[0]), 
                            max(int(bbox[0]), 0):min(int(bbox[2]), img.shape[1])]
        
        results = hands.process(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks is None:
            return None, None, None
        else: 
            if len(results.multi_hand_landmarks) == 2:
                           
                # LEFT IS RIGHT, RIGHT IS LEFT 
                if results.multi_handedness[0].classification[0].label == "Left":
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_conf = results.multi_handedness[0].classification[0].score
                else:
                    hand_landmarks = results.multi_hand_landmarks[1]
                    hand_conf = results.multi_handedness[1].classification[0].score
            else:
                
                if results.multi_handedness[0].classification[0].label == "Left":
                    hand_landmarks = results.multi_hand_landmarks[0]
                    hand_conf = results.multi_handedness[0].classification[0].score
                else:
                    return None, None, None
                           
            mp_joints = []
            for i in range(21):
                mp_joints.append([hand_landmarks.landmark[i].x * crop_img.shape[1], 
                                hand_landmarks.landmark[i].y * crop_img.shape[0]])
        
            j2d_bbox_space = np.array(mp_joints)
            j2d_full_img = j2d_bbox_space + np.array([bbox[0], bbox[1]])
            return j2d_bbox_space, j2d_full_img, hand_conf
      

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


def draw_std(img, joints2d, std_list, frame_num=None):
    
    if frame_num is not None:
        cv2.putText(img, "frame: " + str(frame_num), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  
    
    str_format = ""
    
    for idx, j in enumerate(joints2d):
        str_format = str(idx) + ": " + "%.1f" % std_list[idx] 
        cv2.putText(img, str_format, (10, 20  * (idx+2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)  
        

def draw_bbox_joints(img, joints2d, bbox):
    for idx, j in enumerate(joints2d):
        
        if idx in [3, 4]:                   
            jts_color = (255, 0, 0)
        elif idx in [7, 8]:
            jts_color = (139,69,19)
        elif idx in [19, 20]:
            jts_color = (0, 255, 0)
        else:
            jts_color = (0, 0, 0)
        
        cv2.putText(img, str(idx), (int(j[0]), int(j[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.3, jts_color, 1)
        cv2.circle(img, (int(j[0]), int(j[1])), 1, (0, 0, 255), -1)
        # bbox is (x1, y1, x2, y2) draw it
    if bbox is not None:
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        

class MPJson:
    def __init__(self, gt_dict=None, expand_coef=1.5, verbose=False):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands   
        
        self.EXPAND_COEF = expand_coef
        self.KEYPOINT_CONF_THRESHOLD = 0.0
        self.AREA_RATIO = 0.4
        self.VIEW_NUM = 11  
        
        self.verbose = verbose
        
        
        try: 
            self.gt_bboxes, self.gt_frame_id = gt_dict["bbox"], gt_dict["frame_id"]
            self.handedness = gt_dict["handedness"]
        except:
            self.gt_frame_id, self.gt_bboxes = None, None
            print("No ground truth bounding box or frame id is provided for Mediapipe detection.")
        
    def check_bbox(self, bbox_joints, area_bbox):
        delta_w = abs(max(bbox_joints[:, 0]) - min(bbox_joints[:, 0]))
        delta_h = abs(max(bbox_joints[:, 1]) - min(bbox_joints[:, 1]))
        
        area_jts = delta_h * delta_w
        
        return area_jts < self.AREA_RATIO * area_bbox
        
         

    def export_json(self):
        dic = {}
        dic['version'] = '1.5'
        dic["people"] = []
        person_dic = {}
    
        person_dic["person_id"] = [-1]
        person_dic["pose_keypoints_2d"] = []
        person_dic["face_keypoints_2d"] = []
        person_dic["hand_left_keypoints_2d"] = []        
        person_dic["hand_right_keypoints_2d"] = self.right_hand_tmp
        
        dic["people"].append(person_dic)

        with open(self.json_filename, 'w') as fp:
            json.dump(dic, fp)


    def get_joint_list(self, _hand_landmarks_, image_cropped_width=None, image_cropped_height=None, 
                        start_point_x=0, start_point_y=0, class_conf=1.0):
        
        if image_cropped_height is None or image_cropped_width is None:
            image_cropped_height = self.image_height
            image_cropped_width = self.image_width
        
        jt_list = []
        for joint_idx, coords in enumerate(_hand_landmarks_):
            
            coords_dict = MessageToDict(coords)
            x = coords_dict['x'] * image_cropped_width + start_point_x
            y = coords_dict['y'] * image_cropped_height + start_point_y
            
            jt_list += [x, y, class_conf]
                
        return jt_list     
     
     
    def annotate_jts2d(self):
    
        img_draw = Image.fromarray(self.annotated_image)
        draw = ImageDraw.Draw(img_draw)
        circle_rad = 2
    
        for k in range(21):        
            if openpose_skeleton[k] == -1:
                continue

            # thumb 
            if k in [3, 4]:
                jts_color = (255, 0, 0)
            elif k in [7, 8]:
                jts_color = (139,69,19)
            # pinky 
            elif k in [19, 20]:
                jts_color = (0, 0, 0)
            else:
                jts_color = (100, 100, 100)

            kps_parent = self.right_hand_tmp[3*openpose_skeleton[k]: 3*openpose_skeleton[k] + 2]
            kps_child = self.right_hand_tmp[3*k: 3*k+2]
            draw.line([(kps_child[0], kps_child[1]), (kps_parent[0], kps_parent[1])], fill=(0, 0, 200), width=2)
            draw.ellipse((kps_child[0]-circle_rad, kps_child[1]-circle_rad, 
                            kps_child[0]+circle_rad, kps_child[1]+circle_rad), 
                            fill=jts_color)
            
        self.annotated_image = np.array(img_draw)
    
    def flip_bbox(self, img_idx, _bbox_):   
        ''' img_idx: 0, 1, 2, 3
            0: original, 1: x, 2: y, 3: xy relections 
            _bbox_: [x1, y1, x2, y2] 
        '''     
        bbox_2d_resh = np.array(_bbox_).reshape(-1, 2)
        m, n, _ = self.image_height, self.image_width, 3
        
        if img_idx == 0:
            pass 
        elif img_idx == 1:
            bbox_2d_resh[:, 0] = n - bbox_2d_resh[:, 0]
        elif img_idx == 2:
            bbox_2d_resh[:, 1] = m - bbox_2d_resh[:, 1]
        elif img_idx == 3: 
            bbox_2d_resh[:, :2] = np.array([n, m]) - bbox_2d_resh[:, :2]
        elif img_idx == 4:
            raise ValueError("img_idx should be between 0 and 4")    
        elif img_idx == 5:
            raise ValueError("img_idx should be between 0 and 4")    
        elif img_idx == 6:
            raise ValueError("img_idx should be between 0 and 4")    
        elif img_idx == 7:
            raise ValueError("img_idx should be between 0 and 4")    
        else:
            raise ValueError("img_idx should be between 0 and 4")    
        


        bbox_flat = bbox_2d_resh.reshape(-1)
        bbox_flat = [min(bbox_flat[0], bbox_flat[2]), min(bbox_flat[1], bbox_flat[3]), max(bbox_flat[0], bbox_flat[2]), max(bbox_flat[1], bbox_flat[3])]
        
        return bbox_flat
    

    def flip_joints(self, img_idx, jts2d):
        
        m, n, _ = self.image_height, self.image_width, 3
        
        if img_idx == 0:
            pass 
        elif img_idx == 1:
            jts2d[:, 0] = n - jts2d[:, 0]
        elif img_idx == 2:
            jts2d[:, 1] = m - jts2d[:, 1]
        else: 
            jts2d[:, :2] = np.array([n, m]) - jts2d[:, :2]
        return jts2d

    def annotate_bbox(self, _bbox_):
        img_draw = Image.fromarray(self.annotated_image)
        draw = ImageDraw.Draw(img_draw)
        draw.rectangle([(_bbox_[0], _bbox_[1]), (_bbox_[2], _bbox_[3])], outline=(0, 255, 0), width=2)
        self.annotated_image = np.array(img_draw)
        
        
    def expand_bbox(self, _bbox_):
        # expand the bbox by a factor of coef
        center = ((_bbox_[0]+_bbox_[2])/2, (_bbox_[1]+_bbox_[3])/2)
        assert (0 <= center[0] < self.image_width) and (0 <= center[1] <= self.image_height), "Center is out of image"
        width = abs(_bbox_[2] - _bbox_[0])
        height = abs(_bbox_[3] - _bbox_[1])

        new_width = width * self.EXPAND_COEF
        new_height = height * self.EXPAND_COEF
        
        x1 = int(max(0.0, center[0] - new_width/2))
        x2 = int(min(self.image_width, center[0] + new_width/2))
        y1 = int(max(0.0, center[1] - new_height/2))
        y2 = int(min(self.image_height, center[1] + new_height/2)) 
        
        return [x1, y1, x2, y2]


    def mp_bbox_conf(self, img_dir, out_dir):
        
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        
        IMAGE_FILES = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.png') or x.endswith('.jpg')])

        self.image_height, self.image_width, _ = cv2.imread(IMAGE_FILES[0]).shape
        
        BG_COLOR = (192, 192, 192) # gray
        tot_num = len(IMAGE_FILES)
        print("\nRUN MEDIAPIPE BBOX CONF:")
        confidences = []
        
        
        with self.mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=1, min_detection_confidence=0.3) as holistic: 
            
            for t_idx, file in enumerate(tqdm(IMAGE_FILES)): 

                image_raw = cv2.imread(file)
                                
                self.annotated_image = image_raw.copy()
                  
                gt_exists_flag = t_idx in self.gt_frame_id
                t_idx_gt, gt_bbox_t, bbox_delta_x, bbox_delta_y = -1, None, 0, 0
                
                if gt_exists_flag:
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    gt_bbox_t = self.gt_bboxes[t_idx_gt]
                    
                    # there are some cases with gt bbox has width or height of 0
                    if gt_bbox_t is not None:    
                        bbox_delta_x = abs(gt_bbox_t[2] - gt_bbox_t[0])
                        bbox_delta_y = abs(gt_bbox_t[3] - gt_bbox_t[1])
                
                if (gt_bbox_t is not None) and gt_exists_flag and (bbox_delta_x > 0) and (bbox_delta_y > 0):
                    
                    # find the corresponding ground truth frame index

                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    
                    raw_gt_bbox = gt_bbox_t.copy()

                    gt_bbox_t = self.expand_bbox(gt_bbox_t)
                    # think x and y in terms of cardial coordinate system 
                    x1, y1, x2, y2 = gt_bbox_t
                    
                    # image_raw_cropped = image_raw 
                    image_raw_cropped = image_raw[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2), :]
                    image_cropped_height, image_cropped_width, _ = image_raw_cropped.shape

                    results = holistic.process(cv2.cvtColor(image_raw_cropped, cv2.COLOR_BGR2RGB))

                    bg_image = np.zeros(image_raw_cropped.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                                    
                    if results.multi_hand_landmarks:                        
                        if len(results.multi_hand_landmarks) == 2: 
                            class_conf = max(results.multi_handedness[0].classification[0].score, results.multi_handedness[1].classification[0].score)                                
                        else:   
                            class_conf = results.multi_handedness[0].classification[0].score
 
                    # there is a ground truth but no detection
                    else:
                        class_conf = 0.0
                    
                     
                # no detection case
                else:
                    class_conf = 0.0   
                
                confidences.append(class_conf)
            
            assert len(confidences) == tot_num, "Confidences and total number of frames are not equal"

        # save confidences 
        np.savez(out_dir, np.array(confidences))
        return 
    
    def detect_mp(self, img_dir, out_dir, json_out_dir=None, video_out=False):
    
        os.makedirs(out_dir, exist_ok=True)
        
        # The joint order is the same as openpose but the skeleton is different
        IMAGE_FILES = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.png') or x.endswith('.jpg')])
        self.image_height, self.image_width, _ = cv2.imread(IMAGE_FILES[0]).shape
        
        BG_COLOR = (192, 192, 192) # gray
        tot_num = len(IMAGE_FILES)
        prev_result = None 

        # REFER TO THERE. DONT FORGET TO FIND SKELETON CORRESPONDENCE
        # https://github.dev/Atif-Anwer/Mediapipe_to_OpenPose_JSON/blob/main/src/mediapipe_JSON.py        

        print("\nRUN MEDIAPIPE:")
        with self.mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.6) as holistic: 
            for t_idx, file in enumerate(tqdm(IMAGE_FILES)): 
                
                image_raw = cv2.imread(file)
                self.annotated_image = image_raw.copy()
                  
                gt_exists_flag = t_idx in self.gt_frame_id
                t_idx_gt, gt_bbox_t, bbox_delta_x, bbox_delta_y = -1, None, 0, 0
                
                if gt_exists_flag:
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    gt_bbox_t = self.gt_bboxes[t_idx_gt]
                    
                    # there are some cases with gt bbox has width or height of 0
                    if gt_bbox_t is not None:    
                        bbox_delta_x = abs(gt_bbox_t[2] - gt_bbox_t[0])
                        bbox_delta_y = abs(gt_bbox_t[3] - gt_bbox_t[1])
                
                if (gt_bbox_t is not None) and gt_exists_flag and (bbox_delta_x > 0) and (bbox_delta_y > 0):
                    
                    # find the corresponding ground truth frame index
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    
                    raw_gt_bbox = gt_bbox_t.copy()

                    gt_bbox_t = self.expand_bbox(gt_bbox_t)
                    # think x and y in terms of cardial coordinate system 
                    x1, y1, x2, y2 = gt_bbox_t
                    start_point_x, start_point_y = min(x1, x2), min(y1, y2)
                    
                    # image_raw_cropped = image_raw 
                    image_raw_cropped = image_raw[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2), :]
                    image_cropped_height, image_cropped_width, _ = image_raw_cropped.shape

                    results = holistic.process(cv2.cvtColor(image_raw_cropped, cv2.COLOR_BGR2RGB))

                    bg_image = np.zeros(image_raw_cropped.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                                    
                    if results.multi_hand_landmarks:
                        
                        
                        if len(results.multi_hand_landmarks) == 2:
                           
                            # random selection 
                            if prev_result is None: 
                                hand_landmarks = results.multi_hand_landmarks[0]
                            # select the one with closest to the previous frame
                            else: 

                                # LEFT IS RIGHT, RIGHT IS LEFT 
                                if results.multi_handedness[0].classification[0].label == "Left":
                                    hand_landmarks = results.multi_hand_landmarks[0]
                                    class_conf = results.multi_handedness[0].classification[0].score
                                else:
                                    hand_landmarks = results.multi_hand_landmarks[1]
                                    class_conf = results.multi_handedness[1].classification[0].score
                                # first_detect = results.multi_hand_landmarks[0]
                                # second_detect = results.multi_hand_landmarks[1]
                                
                                # first_joints = self.get_joint_list(first_detect.landmark, 
                                #                    image_cropped_width=image_cropped_width,  image_cropped_height=image_cropped_height,
                                #                     start_point_x=start_point_x, start_point_y=start_point_y, 
                                #                     class_conf=results.multi_handedness[0].classification[0].score)  
                        
                                # second_joints = self.get_joint_list(second_detect.landmark, 
                                #                    image_cropped_width=image_cropped_width,  image_cropped_height=image_cropped_height,
                                #                     start_point_x=start_point_x, start_point_y=start_point_y, 
                                #                     class_conf=results.multi_handedness[1].classification[0].score)   
                                
                                # first_joints = np.array(first_joints).reshape(-1, 3)[:, :2]
                                # second_joints = np.array(second_joints).reshape(-1, 3)[:, :2]
                                
                                # first_dist = np.mean((prev_result - first_joints)**2)
                                # second_dist = np.mean((prev_result - second_joints)**2)
                                
                                # if first_dist < second_dist:
                                #     hand_landmarks = first_detect
                                #     class_conf = results.multi_handedness[0].classification[0].score
                                # else:
                                #     hand_landmarks = second_detect
                                #     class_conf = results.multi_handedness[1].classification[0].score
                                
                        else:   
                            hand_landmarks = results.multi_hand_landmarks[0]
                            class_conf = results.multi_handedness[0].classification[0].score

                        
                        # now skip the frame if regressed joints squeezed to very small area

                        self.right_hand_tmp = self.get_joint_list(hand_landmarks.landmark, 
                                                   image_cropped_width=image_cropped_width,  image_cropped_height=image_cropped_height,
                                                    start_point_x=start_point_x, start_point_y=start_point_y, 
                                                    class_conf=class_conf)   

            
                        bbox_joints = np.array(self.right_hand_tmp).reshape(-1, 3)[:, :2]
                        delta_w = abs(max(bbox_joints[:, 0]) - min(bbox_joints[:, 0]))
                        delta_h = abs(max(bbox_joints[:, 1]) - min(bbox_joints[:, 1]))
                        
                        area_jts = delta_h * delta_w
                        area_bbox = bbox_delta_x * bbox_delta_y

                        if area_jts < 0.01 * area_bbox:
                            if self.verbose:
                                print("JOINT AREA IS TOO SMALL")
                            self.right_hand_tmp = [0.0] * 63    
                        else:
                            prev_result = np.array(self.right_hand_tmp.copy()).reshape(-1, 3)[:, :2]
                        
                    # there is a ground truth but no detection
                    else:
                        self.right_hand_tmp = [0.0] * 63    
                    
                    self.annotate_bbox(gt_bbox_t)
                    self.annotate_jts2d()
                     
                # no detection case
                else:
                    self.right_hand_tmp = [0.0] * 63    
                
                if json_out_dir:      
                    os.makedirs(json_out_dir, exist_ok=True)    
                    
                    if file.endswith('.png'):
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".png", "_keypoints.json"))
                        suffix = ".png"
                    else:
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".jpg", "_keypoints.json"))
                        suffix = ".jpg"
            
                    self.export_json()
            
                
                fname_single_view = os.path.join(out_dir, os.path.basename(file))
                cv2.imwrite(fname_single_view, self.annotated_image)
            
            if video_out:
                cmd = f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{out_dir}/*{suffix}' -vcodec libx264 -pix_fmt yuv420p {os.path.join(os.path.dirname(out_dir), 'rgb_mediapipe.mp4')} "
                subprocess.run(cmd, shell=True)
                
                
    def detect_mp_std(self, img_dir, out_dir, json_out_dir=None, video_out=False):
    
        os.makedirs(out_dir, exist_ok=True)
        
        # The joint order is the same as openpose but the skeleton is different
        IMAGE_FILES = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.png') or x.endswith('.jpg')])
        self.image_height, self.image_width, _ = cv2.imread(IMAGE_FILES[0]).shape
        
        BG_COLOR = (192, 192, 192) # gray
        tot_num = len(IMAGE_FILES)
        
        std_dict = {}
        std_dict["frame_id"] = []
        std_dict["std"] = []
        
 
        print("\nRUN MEDIAPIPE WITH CONF:")
        with self.mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.6) as holistic: 
            for t_idx, file in enumerate(tqdm(IMAGE_FILES)): 
                
                image_raw = cv2.imread(file)
                self.annotated_image = image_raw.copy()
                  
                gt_exists_flag = t_idx in self.gt_frame_id
                t_idx_gt, gt_bbox_t, bbox_delta_x, bbox_delta_y = -1, None, 0, 0
                
                if gt_exists_flag:
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    gt_bbox_t = self.gt_bboxes[t_idx_gt]
                    
                    # there are some cases with gt bbox has width or height of 0
                    if gt_bbox_t is not None:    
                        bbox_delta_x = abs(gt_bbox_t[2] - gt_bbox_t[0])
                        bbox_delta_y = abs(gt_bbox_t[3] - gt_bbox_t[1])
                
                if (gt_bbox_t is not None) and gt_exists_flag and (bbox_delta_x > 0) and (bbox_delta_y > 0):
                    
                    # find the corresponding ground truth frame index
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    
                    raw_gt_bbox = gt_bbox_t.copy()

                    gt_bbox_t = self.expand_bbox(gt_bbox_t)
                    # think x and y in terms of cardial coordinate system 
                    x1, y1, x2, y2 = gt_bbox_t
                    start_point_x, start_point_y = min(x1, x2), min(y1, y2)
                    
                    # image_raw_cropped = image_raw 
                    image_raw_cropped = image_raw[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2), :]
                    image_cropped_height, image_cropped_width, _ = image_raw_cropped.shape

                    results = holistic.process(cv2.cvtColor(image_raw_cropped, cv2.COLOR_BGR2RGB))

                    bg_image = np.zeros(image_raw_cropped.shape, dtype=np.uint8)
                    bg_image[:] = BG_COLOR
                                    
                    if results.multi_hand_landmarks:
                        
                        if len(results.multi_hand_landmarks) == 2:
                            
                            if results.multi_handedness[0].classification[0].label == "Left":
                                hand_landmarks = results.multi_hand_landmarks[0]
                            else:
                                hand_landmarks = results.multi_hand_landmarks[1]
                                            
                        else:   
                            hand_landmarks = results.multi_hand_landmarks[0]

                        
                        # now skip the frame if regressed joints squeezed to very small area
                        self.right_hand_tmp = self.get_joint_list(hand_landmarks.landmark, 
                                                   image_cropped_width=image_cropped_width,  image_cropped_height=image_cropped_height,
                                                    start_point_x=start_point_x, start_point_y=start_point_y)   

                        area_bbox = bbox_delta_x * bbox_delta_y
                       
                        if self.check_bbox(bbox_joints=np.array(self.right_hand_tmp).reshape(-1, 3)[:, :2],
                                        area_bbox=area_bbox):
                            if self.verbose:
                                print("JOINT AREA IS TOO SMALL")
                            self.right_hand_tmp = [0.0] * 63       
                        else:
                            
                            joints2d_preds, views = [], [image_raw.copy()]
                            
                            valid_view_num = 0
                            
                            # now everything is good, we can run views 
                            for ra in range(self.VIEW_NUM):
                                
                                center = (image_raw.shape[1] // 2, image_raw.shape[0] // 2)
                                angle = np.random.randint(15, 45)
                                if np.random.rand(1) > 0.5:
                                    angle *= -1

                                scale = float(np.random.rand(1) * 0.2 + 0.9)

                                rot_mat = cv2.getRotationMatrix2D(center, angle, scale)
                                
                                bbox = np.array([[gt_bbox_t[0], gt_bbox_t[1]], 
                                [gt_bbox_t[0], gt_bbox_t[3]], 
                                [gt_bbox_t[2], gt_bbox_t[3]], 
                                [gt_bbox_t[2], gt_bbox_t[1]]])
                                
                                orient_bbox = transform_2d_pts(bbox, rot_mat)
                                
                                bbox = [np.min(orient_bbox[:, 0]), np.min(orient_bbox[:, 1]), 
                                np.max(orient_bbox[:, 0]), np.max(orient_bbox[:, 1])]

                                img_t = cv2.warpAffine(image_raw.copy(), rot_mat, (image_raw.shape[1], image_raw.shape[0]))
                                
                                _, j2d_full_img, hand_conf = detect_mp_(img_t, bbox)
                
                                if j2d_full_img is None:
                                    views.append(img_t)
                                    if self.verbose:
                                        print(f'ra: {ra}, no hand detected')
                                    continue
                                # dont check bbox area, because if a view is bad, the whole picture may be bad as well.
                                valid_view_num += 1
                                views.append(img_t)
                                
                                # cv2.rectangle(img_t, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                                # cv2.imwrite("mommi.jpg", img_t)
                     
                                                
                                rot_mat_inv = cv2.invertAffineTransform(rot_mat)
                                full_img_j2d = transform_2d_pts(j2d_full_img, rot_mat_inv)

                                joints2d_preds.append(full_img_j2d)
                                
                            if valid_view_num > 1:    
                                joints2d_preds = np.stack(joints2d_preds)
                                per_joint_std = joints2d_preds.std(0).std(-1)
                                
                                if self.verbose:
                                    print(per_joint_std)
                                
                                # image_concat = np.concatenate([canvas, final_img], axis=1)
                                # draw_std(image_concat, initial_j2d, per_joint_std, frame_num=data_idx) 
                                # cv2.imwrite(f'trial/{seqname}/{data_idx:06d}_j2d_orig.jpg', image_concat)
                                
                                std_dict["std"].append(per_joint_std)
                                std_dict["frame_id"].append(t_idx)
                                          
                                conf_ = obtain_conf(per_joint_std)
                                self.right_hand_tmp = np.concatenate((np.array(self.right_hand_tmp).reshape(-1, 3)[:, :2], conf_[:, None]), axis=1)
                                self.right_hand_tmp = self.right_hand_tmp.reshape(-1).tolist()
                                                                
                       
                    # there is a ground truth but no detection
                    else:
                        if self.verbose:
                            print("No detection")
                        self.right_hand_tmp = [0.0] * 63    
                    
                    self.annotate_bbox(gt_bbox_t)
                    self.annotate_jts2d()
                     
                # no detection case
                else:
                    self.right_hand_tmp = [0.0] * 63    
                
                if json_out_dir:      
                    os.makedirs(json_out_dir, exist_ok=True)    
                    
                    if file.endswith('.png'):
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".png", "_keypoints.json"))
                        suffix = ".png"
                    else:
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".jpg", "_keypoints.json"))
                        suffix = ".jpg"
            
                    self.export_json()
            
                fname_single_view = os.path.join(out_dir, os.path.basename(file))
                cv2.imwrite(fname_single_view, self.annotated_image)
                
                # write keypoints with std 
            
            np.savez(os.path.join(os.path.dirname(out_dir), 'confidences/mediapipe_std.npz'), **std_dict)

            if video_out:
                cmd = f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{out_dir}/*{suffix}' -vcodec libx264 -pix_fmt yuv420p {os.path.join(os.path.dirname(out_dir), 'rgb_mediapipe.mp4')} "
                subprocess.run(cmd, shell=True)




    def detect_mp_multiview(self, img_dir, out_dir, json_out_dir=None, video_out=False, multi_view=True):
    
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(os.path.join(os.path.join(os.path.dirname(out_dir), 'rgb_mediapipe')), exist_ok=True)
        
        # read gt bboxes if they exist 

        # The joint order is the same as openpose but the skeleton is different
        IMAGE_FILES = sorted([os.path.join(img_dir, x) for x in os.listdir(img_dir) if x.endswith('.png') or x.endswith('.jpg')])
        BG_COLOR = (192, 192, 192) # gray
        tot_num = len(IMAGE_FILES)

        img_temp = cv2.imread(IMAGE_FILES[0]) 
        height, width, _ = img_temp.shape     
        
        self.image_width, self.image_height = width, height
        
        # max change in two consecutive timesteps 
        APPLY_THRESHOLDING = False
        pix_diff_max = 20
        
        # there are 4 views in total
        prev_res_dict = {0: {"prev_result_idx": 0, "prev_result": None},
                         1: {"prev_result_idx": 0, "prev_result": None}, 
                         2: {"prev_result_idx": 0, "prev_result": None},
                         3: {"prev_result_idx": 0, "prev_result": None},
                         4: {"prev_result_idx": 0, "prev_result": None},
                         5: {"prev_result_idx": 0, "prev_result": None},
                         6: {"prev_result_idx": 0, "prev_result": None},
                         7: {"prev_result_idx": 0, "prev_result": None}}

        # REFER TO THERE. DONT FORGET TO FIND SKELETON CORRESPONDENCE
        # https://github.dev/Atif-Anwer/Mediapipe_to_OpenPose_JSON/blob/main/src/mediapipe_JSON.py    

        print("\nRUN MEDIAPIPE MULTIVIEW:")

        
        # model complexity: 0, 1
        with self.mp_hands.Hands(static_image_mode=True, model_complexity=1, max_num_hands=2, min_detection_confidence=0.6) as holistic: 
            for t_idx, file in enumerate(tqdm(IMAGE_FILES)): 
                
                image_raw = cv2.imread(file)
                
                setting_data_dict = {"max_conf": 0.0, "max_conf_idx": 0.0, "bbox": [],
                                        "landmarks": [], "keypoints2d": [], "image": image_raw} 

                gt_exists_flag = t_idx in self.gt_frame_id
                t_idx_gt, gt_bbox_t, bbox_delta_x, bbox_delta_y = -1, None, 0, 0
                if gt_exists_flag:
                    t_idx_gt = self.gt_frame_id.index(t_idx)
                    gt_bbox_t = self.gt_bboxes[t_idx_gt]
                    
                    # there are some cases with gt bbox has width or height of 0
                    if gt_bbox_t is not None:    
                        bbox_delta_x = abs(gt_bbox_t[2] - gt_bbox_t[0])
                        bbox_delta_y = abs(gt_bbox_t[3] - gt_bbox_t[1])
                
                
                # check if there is a ground truth, crop the image and detect the hand
                if (gt_bbox_t is not None) and gt_exists_flag and (bbox_delta_x > 0) and (bbox_delta_y > 0):
                    
                    image_rgb_T = cv2.transpose(image_raw)

                    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
                    image_T = cv2.cvtColor(image_rgb_T, cv2.COLOR_BGR2RGB)

                    image_height, image_width, _ = image_raw.shape

                    # reflections 
                    img_x_raw, img_x = cv2.flip(image_raw, 0), cv2.flip(image, 0)
                    img_y_raw, img_y = cv2.flip(image_raw, 1), cv2.flip(image, 1)
                    img_xy_raw, img_xy = cv2.flip(image_raw, -1), cv2.flip(image, -1)
                    four_view_img = np.zeros((image_height * 2, image_width * 2, 3), dtype=np.uint8)

                    # reflections for transposed images
                    img_x_raw_T, img_x_T = cv2.flip(image_rgb_T, 0), cv2.flip(image_T, 0)
                    img_y_raw_T, img_y_T = cv2.flip(image_rgb_T, 1), cv2.flip(image_T, 1)
                    img_xy_raw_T, img_xy_T = cv2.flip(image_rgb_T, -1), cv2.flip(image_T, -1)
                    four_view_img_T = np.zeros((image_width * 2, image_height * 2, 3), dtype=np.uint8)

                    image_raw_list = [image_raw, img_y_raw, img_x_raw, img_xy_raw] if multi_view else [image_raw]
                    image_raw_list_T = [image_rgb_T, img_y_raw_T, img_x_raw_T, img_xy_raw_T] if multi_view else [image_rgb_T]
                    image_raw_list_extend = image_raw_list + image_raw_list_T

                    image_list = [image, img_y, img_x, img_xy] if multi_view else [image]
                    image_list_T = [image_T, img_y_T, img_x_T, img_xy_T] if multi_view else []
                    image_list_extend = image_list + image_list_T

                    # left is right, right is left
                    transpose_flag = [False, False, False, False, True, True, True, True] if multi_view else [False]
                    handedness_extend = ["Left", "Right", "Right", "Left"] if multi_view else ["Left"]

                    # dont use Transpose for now
                    for img_idx, img_elem in enumerate(image_list):
                        
                        # first expand the gt bbox, then get corresponding reflected bbox. 
                        flipped_bbox = self.flip_bbox(img_idx, self.expand_bbox(gt_bbox_t))
                        
                        # Finally, obtain the bbox indices  
                        x1, y1, x2, y2 = flipped_bbox 
                        
                        start_point_x, start_point_y = min(x1, x2), min(y1, y2)
                        
                        image_elem_cropped = img_elem[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2), :]
                        image_cropped_height, image_cropped_width, _ = image_elem_cropped.shape
                        
                        results = holistic.process(image_elem_cropped)
                        
                        self.annotated_image = image_raw_list_extend[img_idx].copy()
                        
                        # copy no matter we detect a hand or not
                        bg_image = np.zeros(self.annotated_image.shape, dtype=np.uint8)
                        bg_image[:] = BG_COLOR
                        best_indx = 0    # in case of a single hand, this is always 0
                     
                        wrong_hand, confidence = False, 0.0
                        
                        if results.multi_handedness:  
                            if len(results.multi_handedness) == 2:
                                if self.verbose:
                                    print(f"TWO HANDS DETECTED")
                                confidence = max(results.multi_handedness[0].classification[0].score, results.multi_handedness[1].classification[0].score)
                            else:   
                                confidence = results.multi_handedness[0].classification[0].score
                                
                                # import ipdb; ipdb.set_trace()

                                if results.multi_handedness[0].classification[0].label != handedness_extend[img_idx]:
                                
                                    if self.verbose: 
                                        print("OPPOSITE HAND DETECTED")
                                    wrong_hand = True

                        if (confidence < self.KEYPOINT_CONF_THRESHOLD):
                            if self.verbose:
                                print("DETECTION UNSUCCESSFUL")
                        
                        # only enters if there is a detection and timestep has ground truth
                        # if results.multi_hand_landmarks and (not Left_hand):
                        if results.multi_hand_landmarks and (not wrong_hand) and (confidence > self.KEYPOINT_CONF_THRESHOLD):
                            
                            # find the corresponding ground truth frame index
                            t_idx_gt = self.gt_frame_id.index(t_idx)
                        
                            # FINGERS CROSSED: HOPE THAT THERE IS NO TWO DETECTION. if there are two hands, select the one with the same handedness or closest to gt frame
                            if len(results.multi_hand_landmarks) == 2:
                                
                                print(f"TWO HANDS DETECTED, {results.multi_handedness[0].classification[0].label}, {results.multi_handedness[1].classification[0].label}")
                                    
                                # no previous detection, two detections in the first stage 
                                if prev_res_dict[img_idx]["prev_result"] is None: 
                                    img_conf = max(results.multi_handedness[0].classification[0].score, results.multi_handedness[1].classification[0].score)
                                    
                                    if results.multi_handedness[0].classification[0].score > results.multi_handedness[1].classification[0].score:
                                        hand_landmarks = results.multi_hand_landmarks[0]
                                    else:
                                        hand_landmarks = results.multi_hand_landmarks[1]
                                
                                # prev detection exists, two detections in the second stage
                                else:
                                    
                                    jts_cand_0 = self.get_joint_list(_hand_landmarks_=results.multi_hand_landmarks[0].landmark, 
                                                                    image_cropped_width=image_cropped_width, image_cropped_height=image_cropped_height, 
                                                                    start_point_x=start_point_x, start_point_y=start_point_y, 
                                                                    class_conf=results.multi_handedness[0].classification[0].score)
            
                                    jts_cand_1 = self.get_joint_list(_hand_landmarks_=results.multi_hand_landmarks[1].landmark, 
                                                                    image_cropped_width=image_cropped_width, image_cropped_height=image_cropped_height, 
                                                                    start_point_x=start_point_x, start_point_y=start_point_y, 
                                                                    class_conf=results.multi_handedness[1].classification[0].score)
        
        
                                    # reflect joints based on the view
                                    # jts_cand_0 = self.flip_joints(img_idx, jts2d=np.array(jts_cand_0).reshape(-1, 3)[:,:2])
                                    # jts_cand_1 = self.flip_joints(img_idx, jts2d=np.array(jts_cand_1).reshape(-1, 3)[:,:2])
                                    jts_cand_0 = np.array(jts_cand_0).reshape(-1, 3)[:,:2]
                                    jts_cand_1 = np.array(jts_cand_1).reshape(-1, 3)[:,:2]
                                    
                                
                                    prev_result_ = prev_res_dict[img_idx]["prev_result"].reshape(-1, 3)[:,:2] 
                                    
                                    first_dist = np.mean((prev_result_ - jts_cand_0)**2)
                                    second_dist = np.mean((prev_result_ - jts_cand_1)**2)
                                    
                                    if first_dist < second_dist:
                                        hand_landmarks = results.multi_hand_landmarks[0]
                                        img_conf = results.multi_handedness[0].classification[0].score
                                    else:
                                        hand_landmarks = results.multi_hand_landmarks[1]
                                        img_conf = results.multi_handedness[1].classification[0].score                        
                            
                            # single detection case
                            else:
                                hand_landmarks = results.multi_hand_landmarks[0]
                                img_conf = results.multi_handedness[0].classification[0].score    
                                
                            
                            # find which view has the most confident detection  
                            if img_conf > setting_data_dict["max_conf"]:
                                setting_data_dict["bbox"] = gt_bbox_t
                                setting_data_dict["max_conf"] = img_conf
                                setting_data_dict["max_conf_idx"] = img_idx

                            # After detecting the landmarks, we need to convert them to the openpose format
                            self.right_hand_tmp = self.get_joint_list(hand_landmarks.landmark, 
                                                   image_cropped_width=image_cropped_width,  image_cropped_height=image_cropped_height,
                                                    start_point_x=start_point_x, start_point_y=start_point_y, 
                                                    class_conf=results.multi_handedness[0].classification[0].score)   
                                                
                            bbox_joints = np.array(self.right_hand_tmp).reshape(-1, 3)[:, :2]
                            delta_w = abs(max(bbox_joints[:, 0]) - min(bbox_joints[:, 0]))
                            delta_h = abs(max(bbox_joints[:, 1]) - min(bbox_joints[:, 1]))
                            
                            area_jts = delta_h * delta_w
                            area_bbox = bbox_delta_x * bbox_delta_y

                            if area_jts < self.AREA_RATIO * area_bbox:
                                if self.verbose:
                                    print("JOINT AREA IS TOO SMALL")
                                self.right_hand_tmp = [0.0] * 63    
                                
                            prev_res_dict[img_idx]["prev_result_idx"] = t_idx
                            prev_res_dict[img_idx]["prev_result"] = np.array(self.right_hand_tmp.copy())
                            
                            
                            self.annotate_bbox(flipped_bbox)
                            self.annotate_jts2d()

                        # no detection case although there is a ground truth
                        else:
                            self.right_hand_tmp = [0.0] * 63            
                        
                        setting_data_dict["landmarks"].append(self.annotated_image.copy())
                        setting_data_dict["keypoints2d"].append(self.right_hand_tmp.copy())

                # no ground truth case, no need to detect
                else:
                    self.right_hand_tmp = [0.0] * 63    
                    setting_data_dict["landmarks"].append(image_raw.copy())
                    setting_data_dict["keypoints2d"].append(self.right_hand_tmp.copy())
                    

                compare_settings_img, merged_single_view_img = self.return_setting_images(setting_data_dict)

                if json_out_dir:      
                    os.makedirs(json_out_dir, exist_ok=True)  
                    
                    if file.endswith('.png'):
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".png", "_keypoints.json"))
                        suffix = ".png"
                    else:
                        self.json_filename = os.path.join(json_out_dir, os.path.basename(file).replace(".jpg", "_keypoints.json"))
                        suffix = ".jpg"
                        
                    self.export_json()

                fname_multi_view = os.path.join(out_dir, os.path.basename(file))
                fname_single_view = os.path.join(os.path.join(os.path.dirname(out_dir), 'rgb_mediapipe'), os.path.basename(file))
                           
                cv2.imwrite(fname_multi_view, compare_settings_img)
                cv2.imwrite(fname_single_view, merged_single_view_img)
            
            if video_out:  
                cmd = f"/usr/bin/ffmpeg -y -framerate 30 -pattern_type glob -i '{out_dir}/*{suffix}' -vcodec libx264 -pix_fmt yuv420p {os.path.join(os.path.dirname(out_dir), 'rgb_mediapipe_mv.mp4')} "
                subprocess.run(cmd, shell=True)

 
    def return_setting_images(self, data_dict):
        """
            Draw images to compare different settings. 
                setting1: single view detection (only first view), 
                setting2: select the detection with highest confidence, 
                setting3: select the detection with highest confidence if first view detection does not exist)


                return concatenated image and the final single image based on the merging strategy
        """
        img_orig = data_dict["image"]
        bbox = data_dict["bbox"]
        
        m, n, _ = img_orig.shape
        # no detection in any frames 
        if data_dict["max_conf"] == 0:
            return np.concatenate((img_orig, img_orig, img_orig, img_orig), axis=1), img_orig
        else:
            # call convert handlandmark for reflecting back to the original image
            max_conf_idx = data_dict["max_conf_idx"]
            img_set1 = data_dict["landmarks"][0]
           
            # conf automatically saved for the best setting
            rh_tmp = np.array(data_dict["keypoints2d"][max_conf_idx]).reshape(-1, 3)  
            rh_tmp = self.flip_joints(max_conf_idx, rh_tmp)
            
            # qualitative evaluation shows that this setting works best. Therefore keypoints is saved. 
            if max_conf_idx == 0:
                img_set2 = data_dict["landmarks"][0]
            elif max_conf_idx == 1:
                img_set2 = cv2.flip(data_dict["landmarks"][max_conf_idx], 1) 
                # rh_tmp[:, 0] = m - rh_tmp[:, 0]
            elif max_conf_idx == 2:
                img_set2 = cv2.flip(data_dict["landmarks"][max_conf_idx], 0)
                # rh_tmp[:, 1] = n - rh_tmp[:, 1]
            else: 
                img_set2 = cv2.flip(data_dict["landmarks"][max_conf_idx], -1)
                # rh_tmp[:, :2] = np.array([m, n]) - rh_tmp[:, :2]    

            # cast back to list for JSON recording

            if data_dict["landmarks"][0] is not None:
                img_set3 = data_dict["landmarks"][0]
            else:
                map_dict = {1: 1, 2: 0, 3: -1}
                img_set3 = cv2.flip(data_dict["landmarks"][max_conf_idx], map_dict[max_conf_idx]) 
                
                # if max_conf_idx == 1:
                #     img_set3 = cv2.flip(data_dict["landmarks"][max_conf_idx], 1) 
                # elif max_conf_idx == 2:
                #     img_set3 = cv2.flip(data_dict["landmarks"][max_conf_idx], 0)
                # else: 
                #     img_set3 = cv2.flip(data_dict["landmarks"][max_conf_idx], -1)

            # As for the last frame, draw the self.right_hand_tmp variable to the image 
            # img_draw = Image.fromarray(img_orig)
            # draw = ImageDraw.Draw(img_draw)
            # circle_rad = 2
            
            
            self.right_hand_tmp = list(rh_tmp.reshape(-1))
            self.annotated_image = img_orig.copy()
            self.annotate_jts2d()
            self.annotate_bbox(bbox)
                            
            # draw.rectangle([(_bbox_[0], _bbox_[1]), (_bbox_[2], _bbox_[3])], outline=(0, 255, 0), width=2)

            # for k in range(21):        
                
            #     if openpose_skeleton[k] == -1:
            #         continue
                
            #     kps_parent = self.right_hand_tmp[3 * openpose_skeleton[k]: 3 * openpose_skeleton[k]+2]
            #     kps_child = self.right_hand_tmp[3*k: 3*k+2]
                 
            #     draw.line([(kps_child[0], kps_child[1]), (kps_parent[0], kps_parent[1])], fill=(0, 0, 200), width=2)
            
            #     draw.ellipse((kps_child[0]-circle_rad, kps_child[1]-circle_rad, 
            #                     kps_child[0]+circle_rad, kps_child[1]+circle_rad), 
            #                     fill=(200, 0, 0))

            return np.concatenate((img_set1, img_set2, img_set3, self.annotated_image), axis=1), img_set2

    
    # if APPLY_THRESHOLDING and t_diff == 1 and prev_res_dict[img_idx]["prev_result"] is not None:
                                                            
    #                             # find diff 
    #                             joint_diff = cur_result - prev_res_dict[img_idx]["prev_result"]
    #                             # cast diff
    #                             joint_diff[joint_diff > pix_diff_max] = pix_diff_max  
    #                             joint_diff[joint_diff < -pix_diff_max] = -pix_diff_max 

    #                             self.right_hand_tmp = list(joint_diff + prev_res_dict[img_idx]["prev_result"])             
                                
    #                             # write to landmarks in case of a change 
    #                             for joint_idx, coords in enumerate(hand_landmarks.landmark):
    #                                 coords.x = self.right_hand_tmp[joint_idx][0] / image_width
    #                                 coords.y = self.right_hand_tmp[joint_idx][1] / image_height