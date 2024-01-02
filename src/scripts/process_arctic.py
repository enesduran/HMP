import os
import glob
import time
import torch
import argparse
import numpy as np

# file imports
import utils.body_utils as body_utils
from body_model import BodyModel
from utils.renderer import HTMLRenderer
from utils.vis_utils import colors, viz_smpl_seq
from datasets.data_augmentation import reflect_body
from utils.torch_utils import copy2cuda, copy2cpu
from utils.amass_utils import ALL_AMASS_DATASETS, ALL_DATASETS
from utils.transform_utils import batch_rodrigues, axisangle2matrots, matrot2axisangle

from process_utils import (device, flat_hand_flag, use_finger_tips, 
            MANO_NUM_BETAS, BODY_NUM_BETAS, HAND_KEYPOINT_VERTICES_DICT, 
            KEYPOINT_VERTICES, NUM_KEYPOINT_VERTS, 
            get_mesh_sequence, get_hand_model_sequence, get_wrist_orientations, 
            estimate_velocity, estimate_angular_velocity, regress_hand_params)


 
def process_mocap(body_data_dict, paths, settings, is_augment):
    input_file_path, output_file_path, bm_path, rh_path, lh_path = paths

    assert body_data_dict["gender"] == "neutral", gender
    
    # already tried extracting them. Should be no problem.
    fps = body_data_dict['mocap_frame_rate']
    num_frames = body_data_dict['poses'].shape[0]
    pose_body = copy2cuda(body_data_dict['poses'])                  # smplx body joint rotations (55 joints)
    trans = copy2cuda(body_data_dict['transl'][:])                  # global translation
    body_root_orient = copy2cuda(body_data_dict['poses'][:, :3])    # global orientation of the pelvis wrt canonical coordinate system (1 joint)  
    pose_left_hand = copy2cuda(body_data_dict['poses'][:, 75:120])  # finger articulation joint rotations
    pose_right_hand = copy2cuda(body_data_dict['poses'][:, 120:])   # finger articulation joint rotations
    # body_betas = copy2cuda(body_data_dict['betas'][:])              # body shape parameters
    body_betas = regress_hand_params()              # body shape parameters
    hand_betas = regress_hand_params()
    
    
    print(f"Number of frames before crop: {num_frames}")
    # only keep middle 80% of sequences to avoid redundant static poses (except TCDHands)
    trim_data = [trans, body_root_orient, pose_body, pose_right_hand, pose_left_hand]

    first_index, last_index = 0, num_frames

    if input_file_path.split("/")[-3] != "TCDHands":
        for i, data_seq in enumerate(trim_data):
 
            first_index = int((1 - settings["CROP_RATIO"]) * 0.5 * num_frames) 
            last_index = int((1 + settings["CROP_RATIO"])  * 0.5 * num_frames)
            
            trim_data[i] = data_seq[first_index:last_index]
        trans, body_root_orient, pose_body, pose_right_hand, pose_left_hand = trim_data

    print("\n\nStart-end " + str(first_index) + "," + str(last_index) + "\n\n")

    num_frames = trans.shape[0]
    print(f"Number of frames after crop: {num_frames}")

    assert pose_right_hand.shape == pose_left_hand.shape, f"Hand pose shapes should be equal. RH shape:{pose_right_hand.shape}, LH shape:{pose_left_hand.shape}"

    # must do MANO forward pass to get joints. Split into manageable chunks to avoid running out of GPU memory.
    right_j_seq, left_j_seq, body_j_seq = [], [], []
    rh_vtx_seq, lh_vtx_seq, body_vtx_seq = [], [], []
    process_inds = [0, min([num_frames, settings["SPLIT_FRAME_LIMIT"]])]   
        
    rh_wrist_orientation_seq,  lh_wrist_orientation_seq = [], []
    # Save the vertices for visualization acc
    body_vert_seq, rh_vert_seq, lh_vert_seq = [], [], []
    
    while process_inds[0] < num_frames:
        print("Indices after cropping: " + str(process_inds))
        sidx, eidx = process_inds
        
        body_output, rhand_output, lhand_output = get_mesh_sequence(body_path=bm_path, mano_rh_path=rh_path, mano_lh_path=lh_path, 
                    num_frames=process_inds[1] - process_inds[0], body_pose=pose_body[sidx:eidx], pose_right_hand=pose_right_hand[sidx:eidx],
                     pose_left_hand=pose_left_hand[sidx:eidx], body_betas=body_betas, hand_betas=hand_betas, body_root_orient=body_root_orient[sidx:eidx],
                         body_trans=trans[sidx:eidx])
                    
        body_vert_seq.append(body_output.vertices)
        rh_vert_seq.append(rhand_output.vertices)
        lh_vert_seq.append(lhand_output.vertices)
        
        body_j_seq.append(body_output.joints)
        right_j_seq.append(rhand_output.joints)
        left_j_seq.append(lhand_output.joints)
        
        rh_wrist_orientation_seq.append(rhand_output.root_orient)
        lh_wrist_orientation_seq.append(lhand_output.root_orient)
        
        # save specific vertices if desired
        if settings["SAVE_KEYPOINT_VERTICES"]:
            rh_vtx_seq.append(rhand_output.vertices[:, settings["KEYPOINT_VERTICES"], :])
            lh_vtx_seq.append(lhand_output.vertices[:, settings["KEYPOINT_VERTICES"], :])            

        process_inds[0] = process_inds[1]
        process_inds[1] = min([num_frames, process_inds[1] + settings["SPLIT_FRAME_LIMIT"]])

    # concatenate lists 
    body_joint_seq = torch.cat(body_j_seq, dim=0)
    rh_joint_seq = torch.cat(right_j_seq, dim=0)
    lh_joint_seq = torch.cat(left_j_seq, dim=0)    

    body_vert_seq = torch.cat(body_vert_seq, dim=0)
    rh_vert_seq = torch.cat(rh_vert_seq, dim=0)
    lh_vert_seq = torch.cat(lh_vert_seq, dim=0)
      
    rh_wrist_orient_seq = torch.cat(rh_wrist_orientation_seq, dim=0) 
    lh_wrist_orient_seq = torch.cat(lh_wrist_orientation_seq, dim=0) 

    assert rh_joint_seq.shape[0] == rh_joint_seq.shape[0] == body_vert_seq.shape[0] == rh_vert_seq.shape[0], f"Error with data dimensions. 1: {rh_joint_seq.shape[0]}, 2: {rh_joint_seq.shape[0]}, 3: {lh_joint_seq.shape[0]}, 4: {rh_vert_seq.shape[0]}"

    body_vtx_seq, right_vtx_seq = None, None
    if settings["SAVE_KEYPOINT_VERTICES"]:
        right_vtx_seq = torch.cat(rh_vtx_seq, dim=0)
        left_vtx_seq = torch.cat(lh_vtx_seq, dim=0)

    # estimate various velocities based on full frame rate with second order central difference.
    right_hand_joint_vel_seq = left_hand_joint_vel_seq = None
    pose_body_vel_seq = body_joints_vel_seq = None
    body_vtx_vel_seq = right_vtx_vel_seq = left_vtx_vel_seq = None
    trans_vel_seq = root_orient_vel_seq = None
    if settings["SAVE_VELOCITIES"]:
        h = 1.0 / fps
        # joints
        right_hand_joint_vel_seq = estimate_velocity(rh_joint_seq, h)
        left_hand_joint_vel_seq = estimate_velocity(lh_joint_seq, h)

        if settings["SAVE_KEYPOINT_VERTICES"]:
            # vertices
            right_vtx_vel_seq = estimate_velocity(right_vtx_seq, h)
            left_vtx_vel_seq = estimate_velocity(left_vtx_seq, h)

        # translation
        trans_vel_seq = estimate_velocity(trans, h)

        # root orient
        root_orient_mat = axisangle2matrots(body_root_orient.reshape(num_frames, 1, 3)).reshape((num_frames, 3, 3))
        root_orient_vel_seq = estimate_angular_velocity(root_orient_mat, h)

        # body pose
        if settings["SAVE_BODY_DATA"]:
            body_joints_vel_seq = estimate_velocity(body_joint_seq, h)
        
        pose_body_mat = axisangle2matrots(pose_body[:, 3:].reshape(num_frames, 
            body_utils.NUM_BODY_JOINTS - 1, 3)).reshape((num_frames, body_utils.NUM_BODY_JOINTS - 1, 3, 3))
        
        pose_right_hand_mat = axisangle2matrots(pose_right_hand.reshape(num_frames, 
            body_utils.NUM_HAND_JOINTS - 1, 3)).reshape((num_frames, body_utils.NUM_HAND_JOINTS - 1, 3, 3))
        
        pose_left_hand_mat = axisangle2matrots(pose_left_hand.reshape(num_frames, 
            body_utils.NUM_HAND_JOINTS - 1, 3)).reshape(num_frames, body_utils.NUM_HAND_JOINTS - 1, 3, 3)

        pose_body_vel_seq = estimate_angular_velocity(pose_body_mat, h)
        pose_rh_vel_seq = estimate_angular_velocity(pose_right_hand_mat, h)
        pose_lh_vel_seq = estimate_angular_velocity(pose_left_hand_mat, h)
 
        # throw out edge frames for other data so velocities are accurate
        num_frames = num_frames - 2
        trans = trans[1:-1]
        body_root_orient = body_root_orient[1:-1]
        pose_body = pose_body[1:-1]
        pose_left_hand = pose_left_hand[1:-1]
        pose_right_hand = pose_right_hand[1:-1]
        
        body_joint_seq = body_joint_seq[1:-1]
        rh_joint_seq = rh_joint_seq[1:-1]
        lh_joint_seq = lh_joint_seq[1:-1]
        

        rh_wrist_orient_seq = rh_wrist_orient_seq[1:-1]
        lh_wrist_orient_seq = lh_wrist_orient_seq[1:-1]

        if settings["SAVE_KEYPOINT_VERTICES"]:
            right_vtx_seq = right_vtx_seq[1:-1]
            left_vtx_seq = left_vtx_seq[1:-1]

    # downsample before saving
    if settings["OUT_FPS"] != fps:
        if settings["OUT_FPS"] > fps:
            print('Cannot supersample data, saving at data rate!')
        else:
            fps_ratio = float(settings["OUT_FPS"]) / fps
            print('Downsamp ratio: %f' % (fps_ratio))
            new_num_frames = int(fps_ratio * num_frames)
            print('Downsamp num frames: %d' % (new_num_frames))
            downsamp_inds = np.linspace(0, num_frames - 1, num=new_num_frames, dtype=int)

            # update data to save
            fps = settings["OUT_FPS"]
            num_frames = new_num_frames
            trans = trans[downsamp_inds]
            body_root_orient = body_root_orient[downsamp_inds]

            pose_body = pose_body[downsamp_inds]
            pose_left_hand = pose_left_hand[downsamp_inds]
            pose_right_hand = pose_right_hand[downsamp_inds]

            body_joint_seq = body_joint_seq[downsamp_inds] 
            rh_joint_seq = rh_joint_seq[downsamp_inds]
            lh_joint_seq = lh_joint_seq[downsamp_inds]
            
            pose_rh_vel_seq = pose_rh_vel_seq[downsamp_inds]
            pose_lh_vel_seq = pose_lh_vel_seq[downsamp_inds]

            rh_wrist_orient_seq = rh_wrist_orient_seq[downsamp_inds]
            lh_wrist_orient_seq = lh_wrist_orient_seq[downsamp_inds]

            # No need to do this for left. We only are to visualize body along with the right hand. 
            body_vert_seq = body_vert_seq[downsamp_inds] 
            rh_vert_seq = rh_vert_seq[downsamp_inds]
            

            if settings["SAVE_KEYPOINT_VERTICES"]:
                right_vtx_seq = right_vtx_seq[downsamp_inds]
                left_vtx_seq = left_vtx_seq[downsamp_inds]

            if settings["SAVE_VELOCITIES"]:
                right_hand_joint_vel_seq = right_hand_joint_vel_seq[downsamp_inds]
                left_hand_joint_vel_seq = left_hand_joint_vel_seq[downsamp_inds]

                if settings["SAVE_BODY_DATA"]:
                    pose_body_vel_seq = pose_body_vel_seq[downsamp_inds]

                    # joint up-axis angular velocity (need to compute joint frames first...)
                    body_joints_vel_seq = body_joints_vel_seq[downsamp_inds]

                if settings["SAVE_KEYPOINT_VERTICES"]:
                    right_vtx_vel_seq = right_vtx_vel_seq[downsamp_inds]
                    left_vtx_vel_seq = left_vtx_vel_seq[downsamp_inds]

                    if settings["SAVE_BODY_DATA"]:
                        # body_vtx_vel_seq = body_vtx_vel_seq[downsamp_inds]
                        pass 

                trans_vel_seq = trans_vel_seq[downsamp_inds]
                root_orient_vel_seq = root_orient_vel_seq[downsamp_inds]
    else:
        downsamp_inds = np.arange(num_frames)
    
    # For pseudo right hand part 
    if is_augment:
        output_file_path_list = output_file_path[:-4].split('/')
        output_file_path =  "/".join(output_file_path_list[:5] + ["pseudo_" + output_file_path_list[5]])
    else:
        output_file_path = output_file_path[:-4]
        
    # Visualize right hand motion along with the body. (better for debugging.) 
    if settings["VIS_SEQ"]:  

        crop_ind = 500 if num_frames > 500 else num_frames
        
        # notice that we need to visualize the processed mocap. Therefore, we create a new body model instance and 

        body_vis = BodyModel(model_path=bm_path, model_type="smplx", device=device, batch_size=crop_ind, name="body_processed", mesh_color=colors["lavender"], **{"flat_hand_mean":flat_hand_flag})
        rh_vis = BodyModel(model_path=rh_path, model_type="mano", device=device, batch_size=crop_ind, **{"is_rhand":True, "flat_hand_mean":flat_hand_flag}, name="right_hand", mesh_color=colors["red"])
        lh_vis = BodyModel(model_path=rh_path, model_type="mano", device=device, batch_size=crop_ind, **{"is_rhand":False, "flat_hand_mean":flat_hand_flag}, name="left_hand", mesh_color=colors["blue"])
 
        body_vis_out = body_vis({"root_orient": body_root_orient[:crop_ind],
                                 "body_pose": pose_body[:crop_ind, 3:66],
                                 "jaw_pose": pose_body[:crop_ind, 66:69],
                                 "leye_pose": pose_body[:crop_ind, 69:72],
                                 "reye_pose": pose_body[:crop_ind, 72:75],
                                 "left_hand_pose": pose_body[:crop_ind, 75:120],
                                 "right_hand_pose": pose_body[:crop_ind, 120:],
                                 "transl": trans[:crop_ind]})

        rh_vis_out = rh_vis({"hand_pose": pose_right_hand[:crop_ind],
                                "global_orient": rh_wrist_orient_seq[:crop_ind], 
                                "transl": rh_joint_seq[:crop_ind, 0]})
 
        # eliminate .npz extension in naming.        
        html_name = "/".join(output_file_path.split("/")[-3:-1]) + "/" + output_file_path.split("/")[-1].split(".")[0]
        scene = HTMLRenderer(wandb=True, save_html=True, html_name=html_name, wandb_scene_title="3D Motion", wandb_project_name="amass_after_process", wandb_note="flat_hand_mean=" + str(flat_hand_flag))  
        scene(body_output_list=[body_vis_out], camera_rotation=np.eye(3), translate_origin=np.array([0, 0, 0]), show_ground_floor=True)
           
    print(output_file_path)

    # Last control before saving the data, CURRENTLY DONT CONSIDER BODY VERTEXES
    assert trans.shape[0] == trans_vel_seq.shape[0] == body_root_orient.shape[0] ==  root_orient_vel_seq.shape[0] == pose_body.shape[0] == \
             pose_body_vel_seq.shape[0] == body_joint_seq.shape[0] == body_joints_vel_seq.shape[0] == pose_right_hand.shape[0] == pose_left_hand.shape[0] == rh_wrist_orient_seq.shape[0] == \
                    lh_wrist_orient_seq.shape[0] == pose_rh_vel_seq.shape[0] == pose_lh_vel_seq.shape[0] ==  rh_joint_seq.shape[0] == \
                        lh_joint_seq.shape[0] == right_hand_joint_vel_seq.shape[0] == left_hand_joint_vel_seq.shape[0] == right_vtx_seq.shape[0] == \
                            left_vtx_seq.shape[0] == right_vtx_vel_seq.shape[0] == left_vtx_vel_seq.shape[0] == downsamp_inds.shape[0], "DIMENSIONS ARE WRONG!!!"

    # assert torch.allclose(body_joint_seq[:, body_utils.RIGHT_WRIST_INDEX], rh_joint_seq[:, 0], rtol=0, atol=4e-5), 'Error with RH wrist location'
    # assert torch.allclose(body_joint_seq[:, body_utils.LEFT_WRIST_INDEX], lh_joint_seq[:, 0], rtol=0, atol=4e-5), 'Error with LH wrist location'
    
    # add additional whole joint assertion check after recovering hand betas 

    # add number of frames and frame rate to file path for each of loading
    output_file_path = output_file_path + '_%d_frames_%d_fps.npz' % (num_frames, int(fps))
    
    
    
    processed_num_frames = downsamp_inds.shape[0]

    np.savez(output_file_path,
            
            # general info
            fps=fps,
            trans=copy2cpu(trans),
            trans_vel=copy2cpu(trans_vel_seq),
            body_root_orient=copy2cpu(body_root_orient),
            root_orient_vel=copy2cpu(root_orient_vel_seq),
            augment=is_augment,
            downsamp_inds=downsamp_inds,
            
            # crop indices
            orig_start_ind = first_index,
            orig_end_ind = last_index,

            # body-related info
            body_betas=copy2cpu(body_betas),
            pose_body=copy2cpu(pose_body),
            pose_body_vel=copy2cpu(pose_body_vel_seq),
            body_joints=copy2cpu(body_joint_seq),
            body_joints_vel=copy2cpu(body_joints_vel_seq),
            mojo_verts=body_vtx_seq,
            mojo_verts_vel=body_vtx_vel_seq,

            # hand-related
            hand_betas=copy2cpu(hand_betas),
            pose_right_hand=copy2cpu(pose_right_hand),
            pose_left_hand=copy2cpu(pose_left_hand),
            
            r_wrist_location = copy2cpu(body_joint_seq[:, body_utils.RIGHT_WRIST_INDEX, :]),
            l_wrist_location = copy2cpu(body_joint_seq[:, body_utils.LEFT_WRIST_INDEX, :]),
            r_wrist_orient = copy2cpu(rh_wrist_orient_seq),   
            l_wrist_orient = copy2cpu(lh_wrist_orient_seq),

            # hand angular velocities
            pose_right_hand_vel = copy2cpu(pose_rh_vel_seq),
            pose_left_hand_vel = copy2cpu(pose_lh_vel_seq),

            # joint locations 
            right_hand_joints = copy2cpu(rh_joint_seq),
            left_hand_joints = copy2cpu(lh_joint_seq),

            # velocity of joints
            right_hand_joint_vel=copy2cpu(right_hand_joint_vel_seq),
            left_hand_joint_vel=copy2cpu(left_hand_joint_vel_seq),

            # selected vertex points 
            right_hand_vtx=copy2cpu(right_vtx_seq),
            left_hand_vtx=copy2cpu(left_vtx_seq),

            # velocity of selected vertices
            right_hand_vtx_vel=copy2cpu(right_vtx_vel_seq),
            left_hand_vtx_vel=copy2cpu(left_vtx_vel_seq),
            
            joint_orient_vel_seq=None,
            contacts=None)
          
    return processed_num_frames

def process_seq(data_paths, exception_file, eligible_mocap_file, settings):
    start_t = time.time()

    input_file_path = data_paths[0]

    # ARCTIC 
    try: 
        bdata = np.load(input_file_path, allow_pickle=True).item()
        gender = "neutral"
    except:
        import ipdb; ipdb.set_trace()

    assert gender == "neutral", gender
    
    mocap_data_dict = {}
    data_names = ['body_pose', 'transl']

    sequence_num_frames = 0 
    
        
    # check if there is an error
    try:
        for dname in data_names:
            mocap_data_dict[dname] = bdata[dname]
        
        num_frames = bdata['body_pose'].shape[0]
        pose_left_hand = bdata['left_hand_pose'] # finger articulation joint rotations
        pose_right_hand = bdata['right_hand_pose'] # finger articulation joint rotations
        mocap_data_dict['hand_betas'] = regress_hand_params()
        mocap_data_dict['mocap_frame_rate'] = 30     # for ARCTIC 
        mocap_data_dict['gender'] = gender
        
        mocap_data_dict['poses'] = np.concatenate((bdata["global_orient"], 
                        bdata["body_pose"], 
                        bdata["jaw_pose"],
                        bdata["leye_pose"],
                        bdata["reye_pose"],
                        bdata["left_hand_pose"],
                        bdata["right_hand_pose"]), axis=1)
    except:
        exception_file.write(input_file_path + "\n")
        exception_file.flush()
        print("input_file_path: ", input_file_path)
        return sequence_num_frames
    
    # discard if shorter than threshold
    if num_frames < settings["DISCARD_SHORTER_THAN"] * mocap_data_dict['mocap_frame_rate']:
        print('Sequence shorter than %f s, discarding...' % (settings["DISCARD_SHORTER_THAN"]))
        return sequence_num_frames
    

    # now divide the sequence into smaller chunks, chunks of 128 frames
    num_of_chuncks = int(np.ceil(num_frames / settings["SPLIT_FRAME_LIMIT"]))
    
    for k in range(num_of_chuncks):
        
        trim = slice(settings["SPLIT_FRAME_LIMIT"] * k , settings["SPLIT_FRAME_LIMIT"] * (k+1))
    
        temp_right = pose_right_hand[trim]
        temp_left = pose_left_hand[trim]
        
        # append zeros if the last chunk is not full
        if k == num_of_chuncks - 1:
            temp_right = np.concatenate((pose_right_hand[trim], np.zeros((settings["SPLIT_FRAME_LIMIT"]*(k+1) - pose_right_hand.shape[0], 
                                                                          pose_right_hand.shape[1]))), axis=0)
            temp_left = np.concatenate((pose_left_hand[trim], np.zeros((settings["SPLIT_FRAME_LIMIT"]*(k+1) - pose_left_hand.shape[0], 
                                                                          pose_left_hand.shape[1]))), axis=0)            
        # convert aa to 6d and calculate variance
        temp_right_6d = batch_rodrigues(torch.tensor(temp_right).reshape(-1, 3))[:, :2, :].reshape(settings["SPLIT_FRAME_LIMIT"], -1, 6)
        temp_left_6d = batch_rodrigues(torch.tensor(temp_left).reshape(-1, 3))[:, :2, :].reshape(settings["SPLIT_FRAME_LIMIT"], -1, 6)
    
        rh_var_k_6d = torch.var(temp_right_6d, axis=0).flatten().mean().item()
        lh_var_k_6d = torch.var(temp_left_6d, axis=0).flatten().mean().item()
    
        rh_var_k_aa = np.var(temp_right, axis=0).flatten().mean()
        lh_var_k_aa = np.var(temp_left, axis=0).flatten().mean()
        
        eligible_mocap_file.write(input_file_path +"_{} AA (l,r): {},{} 6D (l,r): {},{} \n".format(k, lh_var_k_aa, rh_var_k_aa, lh_var_k_6d, rh_var_k_6d))
        eligible_mocap_file.flush()
     
     
    mean_rh_var = np.var(pose_right_hand, axis=0).flatten().mean()
    mean_lh_var = np.var(pose_left_hand, axis=0).flatten().mean()
    
    right_var_flag = mean_rh_var > settings["VARIANCE_THRESHOLD"]
    left_var_flag = mean_lh_var > settings["VARIANCE_THRESHOLD"]
    

    if right_var_flag:
        print("Sufficient right hand articulation. Variance: {:2.2e}".format(mean_rh_var))
        sequence_num_frames += process_mocap(mocap_data_dict, paths=data_paths, settings=settings, is_augment=False) 

    if left_var_flag:
        print("Sufficient left hand articulation. Variance: {:2.2e}".format(mean_lh_var))
        
        # get together    
        mocap_data_dict['poses'], mocap_data_dict['trans'] = reflect_body(b_pose=mocap_data_dict['poses'], root_transl=mocap_data_dict["transl"], root_rot=True)
        sequence_num_frames += process_mocap(mocap_data_dict, paths=data_paths, settings=settings, is_augment=True)
    
    if not (right_var_flag or left_var_flag):
        print("Insignificant hand articulation FOR BOTH. SKIPPING")
        return sequence_num_frames

    
    print('Seq process time: %f s Seq num frames: %d' % (time.time() - start_t, sequence_num_frames))
    return sequence_num_frames


def main(config):
    # Delete before starting. 
    
    os.system("rm -rf " + config.out_path + "/ARCTIC")
    
    start_time = time.time()
    out_folder = config.out_path
    if not os.path.exists(out_folder):
        if not os.path.exists("data"):
            os.mkdir("data")
        os.mkdir(out_folder)

    # get all available datasets
    dataset_dirs, all_dataset_dirs, dataset_names = [], [], [] 
   
    if os.path.isdir(config.arctic_root):
        all_dataset_dirs += [config.arctic_root]
        dataset_dirs += [os.path.join(config.arctic_root, "ARCTIC")]
        dataset_names += ["ARCTIC"]

    all_dataset_dirs = [f for f in all_dataset_dirs if os.path.isdir(f)]
    all_dataset_dirs.reverse()
    
    print('Found %d available datasets from raw AMASS data source.' % (len(all_dataset_dirs)))
    all_dataset_names = [f.split('/')[-1] for f in all_dataset_dirs]
    print(f"Names: {all_dataset_names}")

    # requested datasets
    print('\nRequested datasets:')
    print(dataset_dirs)
    print(dataset_names)
    

    # go through each dataset to set up directory structure before processing
    all_seq_in_files = []
    all_seq_out_files = []
    for data_dir, data_name in zip(dataset_dirs, dataset_names):        
        if not os.path.exists(data_dir):
            print('Could not find dataset %s in available raw AMASS data!' % (data_dir))
            return

        data_name = os.path.basename(data_dir)
        cur_output_dir = os.path.join(out_folder, data_name)
        
        if not os.path.exists(cur_output_dir):
            os.mkdir(cur_output_dir)

        # first create subject structure in output
        cur_subject_dirs = [f for f in sorted(os.listdir(data_dir)) if f[0] != '.' and os.path.isdir(os.path.join(data_dir, f))]
        print(cur_subject_dirs)
        
        for subject_dir in cur_subject_dirs:
            cur_subject_out = os.path.join(cur_output_dir, subject_dir)
            if not os.path.exists(cur_subject_out):
                os.mkdir(cur_subject_out)


        input_seqs = glob.glob(os.path.join(data_dir, '*/*smplx.npy'))
        input_seqs_mano = glob.glob(os.path.join(data_dir, '*/*mano.npy'))
        # np.load(input_seqs_mano[0], allow_pickle=True).item()

        # and create output sequence file names
        output_file_names = ['/'.join(f.split('/')[-2:]) for f in input_seqs]
        output_seqs = [os.path.join(cur_output_dir, f) for f in output_file_names]
        print(f"Total motion number in input/output data: {len(input_seqs), len(output_seqs)}")

        already_processed = [i for i in range(len(output_seqs)) if len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 1]
        already_processed_output_names = [output_file_names[i] for i in already_processed]
        print(f'Already processed these sequences, skipping: {already_processed_output_names}')
        
        not_already_processed = [i for i in range(len(output_seqs)) if
                                 len(glob.glob(output_seqs[i][:-4] + '*.npz')) == 0]
        input_seqs = [input_seqs[i] for i in not_already_processed]
        output_seqs = [output_seqs[i] for i in not_already_processed]

        all_seq_in_files += input_seqs
        all_seq_out_files += output_seqs
        
    smplx_paths = [config.smplx_path] * len(all_seq_in_files)
    rh_paths = [config.right_hand_path] * len(all_seq_in_files)
    lh_paths = [config.left_hand_path] * len(all_seq_in_files)
    data_paths = list(zip(all_seq_in_files, all_seq_out_files, smplx_paths, rh_paths, lh_paths))

    if not os.path.exists("logs"):
        os.mkdir("logs")
    if not os.path.exists("logs/hands_after_process"):
         os.mkdir("logs/hands_after_process")
         
    exception_file = open("logs/hands_after_process/amass_exception_files.txt", "w+")
    eligible_mocaps = open("logs/hands_after_process/amass_eligible_mocaps.txt", "w+")

    settings = {"OUT_FPS": config.out_fps,
                "SAVE_KEYPOINT_VERTICES": config.save_keypoint_vertices,
                "SAVE_VELOCITIES": config.save_velocities,
                "KEYPOINT_VERTICES": config.keypoint_vertices,
                "SAVE_BODY_DATA": config.save_body_data,
                "SAVE_ALIGN_ROT": config.save_align_rot,
                "VIS_SEQ": config.visualize_sequence,
                "DISCARD_SHORTER_THAN": config.discard_shorter_than,
                "MANO_NUM_BETAS": config.mano_num_betas,
                "BODY_NUM_BETAS": config.body_num_betas,
                "SPLIT_FRAME_LIMIT": config.split_frame_limit,
                "VARIANCE_THRESHOLD": config.variance_threshold,
                "RECORD_STATS": config.record_stats,
                "CROP_RATIO": config.crop_ratio}

    tot_frames = 0

    for data_in in data_paths:
        tot_frames += process_seq(data_in, exception_file, eligible_mocaps, settings)


    total_time = time.time() - start_time
    exception_file.close()
    eligible_mocaps.close()
    print('TIME TO PROCESS: %f min  TOTAL FRAMES AFTER PROCESSING: %d ' % (total_time / 60.0, tot_frames))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arctic_root', type=str, default="./data/amass_raw", help='Root directory of raw ARCTIC dataset.')
    parser.add_argument('--interhands_root', type=str, default="./data/interhands_raw", help='Root directory of raw InterHands dataset.')
    parser.add_argument('--datasets', type=str, nargs='+', default=ALL_DATASETS, help='Which datasets to process. By default processes all.')
    parser.add_argument('--amass_datasets', type=str, nargs='+', default=ALL_AMASS_DATASETS, help='Which datasets to process. By default processes all.')
    parser.add_argument('--out_path', type=str, default='./data/amass_processed_hist', help='Root directory to save processed output to.')
    parser.add_argument('--smplx_path', type=str, default='./data/body_models/smplx/SMPLX_NEUTRAL.npz', help='Root directory of the SMPLX body model.')
    parser.add_argument('--right_hand_path', type=str, default='./data/body_models/mano/MANO_RIGHT.pkl', help='Root directory of the MANO right hand model')
    parser.add_argument('--left_hand_path', type=str, default='./data/body_models/mano/MANO_LEFT.pkl', help='Root directory of the MANO left hand model.')
    parser.add_argument('--out_fps', type=int, default=30, help='Output fps of the processed data.')
    parser.add_argument('--save_keypoint_vertices', type=bool, default=True, help='Save the given vertices of meshes')
    parser.add_argument('--keypoint_vertices', type=list, default=KEYPOINT_VERTICES, help='Vertex indices to record')
    parser.add_argument('--save_velocities', type=bool, default=True, help='Save the linear and angular velocities')
    parser.add_argument('--save_align_rot', type=bool, default=False, help='Save rotation mats that go from world root orient to aligned root orient')
    parser.add_argument('--save_body_data', type=bool, default=True, help='Save everything about the body: pose, joint velocity and angular velocities')
    parser.add_argument('--mano_num_betas', type=int, default=10, help='PCA size for hand models')
    parser.add_argument('--body_num_betas', type=int, default=10, help='PCA size for body model')
    parser.add_argument('--variance_threshold', type=float, default=5e-3, help='Threshold variance of the hand pose for discarding. Discard if lower than the threshold.')
    parser.add_argument('--split_frame_limit', type=int, default=128, help='if sequence is longer than this, splits into sequences of this size to avoid running out of memory.')
    parser.add_argument('--visualize_sequence', type=bool, default=False, help='Visualize sequence')
    parser.add_argument('--record_stats', type=bool, default=False, help='Calculate and record articulation statistics')
    parser.add_argument('--crop_ratio', type=float, default=0.9, help='Crop the given ratio of data from middle')
    parser.add_argument('--discard_shorter_than', type=float, default=1.0, help='discard mocaps shorter than __ seconds')
    
    config = parser.parse_known_args()
    config = config[0]

    main(config)


 
