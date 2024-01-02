import os
import cv2
import sys
import math
import glob
import torch
import tempfile
import numpy as np
import scenepic as sp
from matplotlib import cm
from PIL import ImageColor
from smplx import SMPL, MANO
import matplotlib.pyplot as plt


cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))


from kornia_transform import rotation_matrix_to_angle_axis, angle_axis_to_rotation_matrix
from datasets.amass import get_stage2_res, read_camera_intrinsics, get_ho3d_v3_gt, get_dexycb_gt



CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]
])

AUGMENTED_MANO_CHAIN = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9])   
RIGHT_WRIST_BASE_LOC = torch.tensor([[0.0957, 0.0064, 0.0062]])
LEFT_WRIST_BASE_LOC = torch.tensor([[-0.0957, 0.0064, 0.0062]])


rh = {21: 'R_Wrist', 40: 'rindex0', 41: 'rindex1', 42: 'rindex2', 43: 'rmiddle0', 44: 'rmiddle1', 45: 'rmiddle2',
                     46: 'rpinky0', 47: 'rpinky1', 48: 'rpinky2', 49: 'rring0', 50: 'rring1', 51: 'rring2',
                     52: 'rthumb0', 53: 'rthumb1', 54: 'rthumb2'}

CAM_LINES = np.array([[1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

SMPL_MODEL_DIR = "./data/body_models/smpl/SMPL_MALE.pkl"
MANO_RIGHT_MODEL_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"
MANO_LEFT_MODEL_DIR = "./data/body_models/mano/MANO_LEFT.pkl"


def mat4x4_inverse(mat4x4):
    R = mat4x4[:, :3, :3].clone()
    t = mat4x4[:, :3, 3].clone()
    R = R.transpose(-1, -2)
    t = -R @ t[..., None]
    mat4x4[:, :3, :3] = R
    mat4x4[:, :3, 3] = t[..., 0]
    return mat4x4


def rotation_about_x(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def gl_camera_to_world(worldtocam: torch.Tensor) -> torch.Tensor:
    worldtocam = mat4x4_inverse(worldtocam)
    return worldtocam @ rotation_about_x(math.pi)[None].to(worldtocam)



def get_camera_lineset():
    start, end = [], []
    for s,e in CAM_LINES:
        start.append(CAM_POINTS[s])
        end.append(CAM_POINTS[e])
    return np.array(start)/10, np.array(end)/10


def make_checker_board_texture(color1='black', color2='white', width=2, height=10, n_tile=15):
    c1 = np.asarray(ImageColor.getcolor(color1, 'RGB')).astype(np.uint8)
    c2 = np.asarray(ImageColor.getcolor(color2, 'RGB')).astype(np.uint8)
    hw = width // 2
    hh = height // 2
    c1_block = np.tile(c1, (hh, hw, 1))
    c2_block = np.tile(c2, (hh, hw, 1))
    tex = np.block([[[c1_block], [c2_block]],
                    [[c2_block], [c1_block]]])
    tex = np.tile(tex, (n_tile, n_tile, 1))  # [:,:,:3]
    # import matplotlib.pyplot as plt; plt.imshow(tex); plt.show()
    return tex

def get_ground_truth_seq(img_dir):
    
    if "DexYCB" in img_dir:
        gt_dict = get_dexycb_gt(img_dir)
        
    elif "HO3D" in img_dir:
        gt_path = os.path.join(os.path.dirname(img_dir), "meta")
        gt_dict = get_ho3d_v3_gt(gt_path) 
        
    else:
        pass 

    # change keynames 
    gt_dict["rh_pose"] = torch.tensor(gt_dict.pop("poses"))
    gt_dict["rh_shape"] = torch.tensor(gt_dict.pop("betas"))
    gt_dict["rh_trans"] = torch.tensor(gt_dict.pop("trans"))

    
    # from openGL to openCV. Converting global rotation is sufficient.
    rotmat = angle_axis_to_rotation_matrix(gt_dict["rh_pose"][:, :3])
    # change y and z axis
    rotmat[:, 1, :] *= -1
    rotmat[:, 2, :] *= -1
    
    gt_dict["rh_pose"][:, :3] = rotation_matrix_to_angle_axis(rotmat)

    return gt_dict


class HTMLRenderer():
    def __init__(self, render_body=False, device=torch.device('cpu'), frame_size=32):
        self.render_body = render_body
 
        # self.mano_right = MANO(MANO_RIGHT_MODEL_DIR, is_rhand=True, batch_size=frame_size, flat_hand_mean=True, use_pca=False)
        self.mano_right = MANO(MANO_RIGHT_MODEL_DIR, is_rhand=True, flat_hand_mean=True, use_pca=False)
        self.mano_left = MANO(MANO_LEFT_MODEL_DIR, is_rhand=False, batch_size=frame_size, flat_hand_mean=True, use_pca=False)
        self.rh_faces = self.mano_right.faces
        self.lh_faces = self.mano_left.faces
        
        self.smpl_joint_parents = AUGMENTED_MANO_CHAIN
        

    def load_default_camera(self, intrinsics):
        # this function loads an "OpenCV"-style camera representation
        # and converts it to a GL style for use in ScenePic
        # location = np.array(camera_info["location"], np.float32)
        # euler_angles = np.array(camera_info["rotation"], np.float32)
        # rotation = sp.Transforms.euler_angles_to_matrix(euler_angles, "XYZ")
        # translation = sp.Transforms.translate(location)
        # extrinsics = translation @ rotation

        img_width = intrinsics[0, 2] * 2
        img_height = intrinsics[1, 2] * 2

        aspect_ratio = img_width / img_height
       
        return sp.Camera(center=(5, 0, 2), look_at=(0, 0, 1), up_dir=(0, 0, 1), 
                         fov_y_degrees=45.0, aspect_ratio=aspect_ratio)

    def load_camera_from_ext_int(self, extrinsics, intrinsics):
        # this function loads an "OpenCV"-style camera representation
        # and converts it to a GL style for use in ScenePic
        # location = np.array(camera_info["location"], np.float32)
        # euler_angles = np.array(camera_info["rotation"], np.float32)
        # rotation = sp.Transforms.euler_angles_to_matrix(euler_angles, "XYZ")
        # translation = sp.Transforms.translate(location)
        # extrinsics = translation @ rotation

        img_width = intrinsics[0, 2] * 2
        img_height = intrinsics[1, 2] * 2
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]

        # pred_f_pix = orig_img_h / 2. / np.tan(pred_vfov / 2.)
        
        vfov = 2. * np.arctan(img_height / (2. * fy))
        # ft = img_height / (2. * np.tan(vfov))
        # print(f"{img_width=} {img_height=}")
        # print(f"{fy=} {vfov=} {np.degrees(vfov)=}")
        # breakpoint()
        world_to_camera = extrinsics  # sp.Transforms.gl_world_to_camera(extrinsics)
        aspect_ratio = img_width / img_height
        projection = sp.Transforms.gl_projection(np.degrees(vfov), aspect_ratio, 0.01, 100)

        return sp.Camera(world_to_camera, projection)
    
    def create_canvas_2d(self, scene, joints2d, img_width, img_height, is_gt=False):
        canvas2 = scene.create_canvas_2d(width=img_width, height=img_height)

        for i in range(joints2d.shape[0]):
            frame = canvas2.create_frame()
            for j in range(joints2d[i].shape[0]):
                frame.add_circle(
                    joints2d[i][j][0], 
                    joints2d[i][j][1], 
                    5, 
                    fill_color=sp.Colors.Green if is_gt else sp.Colors.Red, 
                    line_width=2, 
                    line_color=sp.Colors.Green if is_gt else sp.Colors.Red
                )

            for j, pa in enumerate(AUGMENTED_MANO_CHAIN):
                if pa >= 0:
                
                    frame.add_line(
                        coordinates=np.stack([joints2d[i][j], joints2d[i][pa]], axis=0),
                        line_color=sp.Colors.Green if is_gt else sp.Colors.Red,
                        line_width=3,
                    )
        return canvas2

    def create_canvas_2d_seq(self, scene, joints2d_list, img_width, img_height):
        canvas2 = scene.create_canvas_2d(width=img_width, height=img_height)
        seq_len = joints2d_list[0].shape[0]
        color_list = [
            sp.Colors.Cyan, # cyan
            sp.Colors.Yellow, # yellow
            sp.Colors.Orange, # orange
            sp.Colors.Purple, # purple
            sp.Colors.Blue, # Blue
            sp.Colors.Red, # red
            sp.Colors.Brown, 
        ]
        color_list = color_list[:len(joints2d_list)-1][::-1]
        color_list.append(sp.Colors.Green)

        for i in range(seq_len):
            frame = canvas2.create_frame()
            
            for seq_id, _ in enumerate(joints2d_list):
                if seq_id == len(joints2d_list) - 1:
                    text = '-- GT'
                else:
                    text = f'-- step-{seq_id}'
                size = img_height.item() // 40
                loc = seq_id+1
                frame.add_text(
                    text, left=5, bottom=size*2 + size*loc + seq_id*5, 
                    color=color_list[seq_id], size_in_pixels=size, 
                    font_family='sans-serif', layer_id='label',
                )

            for seq_id, joints2d in enumerate(joints2d_list):
                for j in range(joints2d[i].shape[0]):
                    frame.add_circle(
                        joints2d[i][j][0], 
                        joints2d[i][j][1], 
                        5, 
                        fill_color=color_list[seq_id], 
                        line_width=2, 
                        line_color=color_list[seq_id]
                    )

                for j, pa in enumerate(AUGMENTED_MANO_CHAIN):
                    if pa >= 0:
                        frame.add_line(
                            coordinates=np.stack([joints2d[i][j], joints2d[i][pa]], axis=0),
                            line_color=color_list[seq_id],
                            line_width=3,
                        )
        return canvas2

    def create_canvas(self, scene, smpl_seq, gt_seq, img_width, img_height, cam_intrinsics, img_dir=None, label="", slam_cam_poses=None):
        canvas = scene.create_canvas_3d(width=img_width, height=img_height)
 
        rh_verts, rh_joints = self.get_mano_right_params(smpl_seq)
  
        rh_verts_gt, rh_joints_gt = None, None
        if gt_seq is not None:
            rh_verts_gt, rh_joints_gt = self.get_mano_right_params(gt_seq)
        
        # lh_verts, lh_joints = self.get_mano_left_params(smpl_seq)
        
        # j2d = smpl_seq['rh_joints2d']
        # j3d = smpl_seq['rh_joints3d'] # + RIGHT_WRIST_BASE_LOC
        
        # print(sum(abs(smpl_seq['rh_joints3d']-rh_joints)))

        # import ipdb; ipdb.set_trace()
        # c2c = lambda x: x.detach().cpu().numpy()
        # _, rvec, init_cam_t = cv2.solvePnP(
        #     objectPoints=c2c(j3d[0].float()), 
        #     imagePoints=c2c(j2d[0, :, :2].float()), 
        #     cameraMatrix=c2c(cam_intrinsics.float()),
        #     distCoeffs=None,
        #     rvec=None,
        #     tvec=None,
        #     useExtrinsicGuess=False,
        #     flags=cv2.SOLVEPNP_EPNP,
        # )
        # # cam_R = np.repeat(np.eye(3)[None, ...], repeats=j3d.shape[0], axis=0)
        # cam_R = cv2.Rodrigues(rvec)[0][None].repeat(j3d.shape[0], axis=0) 
        # cam_t = init_cam_t[:, 0][None].repeat(j3d.shape[0], axis=0)
      
        cam_R = smpl_seq["cam_rot"]
        cam_t = smpl_seq["cam_trans"]

        coord_ax = scene.create_mesh(layer_id="coord")
        coord_ax.add_coordinate_axes()
  
        label = scene.create_label(text=label, color=sp.Colors.White, size_in_pixels=80, offset_distance=0.0, camera_space=True)
        
        cam_lineset = get_camera_lineset()
        in_joint_pos = None
        
        if 'in_joint_pos' in smpl_seq.keys():
            in_joint_pos = smpl_seq['in_joint_pos']
            
        if 'slam_cam_poses' in smpl_seq.keys():
            slam_cam_poses = smpl_seq['slam_cam_poses']
            
        if slam_cam_poses is not None:
            # slam_cam_lines = []
            # slam_cam_frustums = []
            # slam_cam_datasets = []
            for i in range(slam_cam_poses.shape[0]):
                
                slam_cam_ext = slam_cam_poses[i]
                slam_cam_ext[1:3] *= -1

                slam_cam_dataset = self.load_camera_from_ext_int(
                    extrinsics=slam_cam_ext,
                    intrinsics=cam_intrinsics
                )
                
                color = cm.jet(int((i/slam_cam_poses.shape[0]) * 255))[:3]

                # slam_cam_line = scene.create_mesh(layer_id="slam_cam_lines")
                # slam_cam_line.add_lines(cam_lineset[0], cam_lineset[1], color=sp.Color(*color), transform=gl_camera_to_world(slam_cam_ext[None])[0])
                # slam_cam_frustum = scene.create_mesh(layer_id="slam_cam_frustum")
                # slam_cam_frustum.add_camera_frustum(slam_cam_dataset, color=sp.Color(*color))
                
                # slam_cam_lines.append(slam_cam_line)
                # slam_cam_frustums.append(slam_cam_frustum)
                # slam_cam_datasets.append(slam_cam_dataset)
        
        gt_cam_lines = []
        for j in range(rh_verts.shape[0]):
            color = cm.jet(int((j/rh_verts.shape[0]) * 255))[:3]
            
            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3,:3] = cam_R[j] # smpl_seq["cam_rot"][j]
            cam_extrinsics[:3,3] = cam_t[j] # smpl_seq["cam_trans"][j]
            # cam_extrinsics[1:3] *= -1
            # cam_extrinsics[:3,:3] = cam_extrinsics[:3,:3].T
            
            # cam_pos_e = (-cam_extrinsics[:3,:3] @ cam_extrinsics[:3,3][..., None])[..., 0] # WTF transpose?
            
            # cam_pos = smpl_seq["cam_eye"][j]
            
            cam_line = scene.create_mesh(layer_id="cam_traj")
            cam_line.add_lines(cam_lineset[0], 
                               cam_lineset[1], 
                               color=sp.Color(*color), 
                               transform=gl_camera_to_world(torch.from_numpy(cam_extrinsics)[None])[0])
            
            # sphere = scene.create_mesh(layer_id="gt_cam_traj")
            # sphere.add_sphere(color=sp.Color(*color), transform = np.dot(sp.Transforms.Scale(0.06), h_traj))
            # sphere.add_sphere(color=sp.Color(*color), transform = np.dot(sp.Transforms.translate(cam_pos_e), sp.Transforms.Scale(0.005)))

            # gt_cam_lines.append(sphere)

            # gt_cam_lines.append(cam_line)
        
        # compatible with DexYCB
        sorted_images = sorted(glob.glob(f'{img_dir}/*.jpg'))
        if len(sorted_images)== 0:
            sorted_images = sorted(glob.glob(f'{img_dir}/*.png'))

        for i in range(rh_verts.shape[0]):
            frame = canvas.create_frame()

            if self.render_body:
                smpl_mesh = scene.create_mesh(shared_color=(0.7, 0.7, 0.7), layer_id="rh_mesh")
                rh_gt_mesh = scene.create_mesh(shared_color=(0.7, 0.2, 0.2), layer_id="rh_gt_mesh")
                
                smpl_mesh.add_mesh_without_normals(rh_verts[i].contiguous().numpy(), self.rh_faces)
                frame.add_mesh(smpl_mesh)
                
                # add ground truth mesh for only the frames that are in the ground truth
                if not rh_verts_gt is None: 
                    if i in gt_seq["frame_id"]:
                        rh_gt_mesh.add_mesh_without_normals(rh_verts_gt[gt_seq["frame_id"].index(i)].contiguous().numpy(), self.rh_faces)
                        frame.add_mesh(rh_gt_mesh)
                
                # lh_mesh = scene.create_mesh(shared_color=(0.7, 0.7, 0.7), layer_id="lh_mesh")
                # lh_mesh.add_mesh_without_normals(lh_verts[i].contiguous().numpy(), self.lh_faces)
                # frame.add_mesh(lh_mesh)
                

            joints_mesh = scene.create_mesh(shared_color = sp.Color(0.0, 1.0, 0.0), layer_id="joints")
            joint_colors = np.zeros_like(rh_joints[i, :24])
            
            joint_colors[:, 1] = 1.0
            # joint_colors[contacts[i].numpy().astype(int)] = (1.0, 0.0, 0.0)
            joints_mesh.add_sphere(transform = sp.Transforms.Scale(0.005))
            joints_mesh.enable_instancing(positions=rh_joints[i, :24], colors = joint_colors)
            
            for j, pa in enumerate(self.smpl_joint_parents):
                if pa >= 0:
                    bone_mesh = scene.create_mesh(shared_color=(1.0, 0.5, 0.0), layer_id="bones")
                    bone_mesh.add_lines(rh_joints[i][j][None], rh_joints[i][pa][None])
                    frame.add_mesh(bone_mesh)
                    
            if in_joint_pos is not None:
                in_joints_mesh = scene.create_mesh(shared_color = sp.Color(64./255., 224./255., 208./255.), layer_id="in_joints")
                in_joints_mesh.add_sphere(transform = sp.Transforms.Scale(0.005))
                in_joints_mesh.enable_instancing(positions=in_joint_pos[i])
                
                frame.add_mesh(in_joints_mesh)
                
                for j, pa in enumerate(self.smpl_joint_parents[:22]):
                    if pa >= 0:
                        in_bone_mesh = scene.create_mesh(shared_color=(0.5, 0.0, 0.5), layer_id="in_joints")
                        in_bone_mesh.add_lines(in_joint_pos[i][j][None], in_joint_pos[i][pa][None])
                        frame.add_mesh(in_bone_mesh)
            

            if 'joint_vel' in smpl_seq.keys():
                
                for j_id, j_vel in enumerate(smpl_seq['joint_vel'][i, :22]):
                    p1 = rh_joints[i, j_id]
                    p2 = rh_joints[i, j_id] + j_vel*0.1
                    vel_mesh = scene.create_mesh(shared_color=(1.0, 0.0, 0.0), layer_id="vel")
                    vel_mesh.add_lines(p1[None], p2[None])
                    frame.add_mesh(vel_mesh)


            cam_extrinsics = np.eye(4)
            cam_extrinsics[:3,:3] = cam_R[i] # smpl_seq["cam_rot"][i]
            cam_extrinsics[:3,3] = cam_t[i] # smpl_seq["cam_trans"][i]
            cam_extrinsics[1:3] *= -1
            
            cam_dataset = self.load_camera_from_ext_int(cam_extrinsics, cam_intrinsics)
            cam_frustum = scene.create_mesh(layer_id="camera")
            cam_frustum.add_camera_frustum(cam_dataset, sp.Colors.Orange)
            
            if img_dir is not None:
                    
                # cam_image = scene.create_image(f'{i+1:06d}.png')
                img_length = len(sorted_images)
                image_index = i if i < img_length else img_length-1
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
            
            frame.camera = cam_dataset # self.load_default_camera(cam_intrinsics)
            # frame.add_mesh(joints_mesh)
            # frame.add_mesh(floor_mesh)
            # frame.add_mesh(cam_frustum)
            frame.add_mesh(coord_ax)
            frame.add_label(label=label, position=[-1.0, 1.0, -5.0])
        
            for m_id in range(len(gt_cam_lines)):
                frame.add_mesh(gt_cam_lines[m_id])
            
            # if slam_cam_poses is not None:
            #     for dc in range(len(slam_cam_lines)):
            #         frame.add_mesh(slam_cam_lines[dc])
            #         frame.add_mesh(slam_cam_frustums[dc])
                    
        layer_settings = {}
        layer_settings["cam_traj"] = {"filled":False} 
        layer_settings["gt_cam_traj"] = {"filled":False} 
        layer_settings["coord"] = {"filled":False}
        
        if slam_cam_poses is not None:
            layer_settings["slam_cam_frustum"] = {"filled":False}
            layer_settings["slam_cam_lines"] = {"filled":False}
        
        if in_joint_pos is not None:
            layer_settings["in_joints"] = {"filled":False}
            
        canvas.set_layer_settings(layer_settings)
        return canvas

    @torch.no_grad()
    def create_animation(self, smpl_seq_list, filename="demo.html", show_2d_kpts=False, 
                         single_2d_kpts=False, show_2d_loss=False, slam_cam_poses=None,
                         img_dir=None, fps=10):
        
        scene = sp.Scene()
        cam_intrinsics = smpl_seq_list[0]['cam_intrinsics']

        # A Scene can contain many canvases
        # For correct operation, you should create these using scene1.create_canvas() (rather than constructing directly using sp.Canvas(...)) 
        img_width = cam_intrinsics[0,2] * 2
        img_height = cam_intrinsics[1,2] * 2
        fx = cam_intrinsics[0,0]
        fy = cam_intrinsics[1,1]
        all_canvases = []

        for seq_id, smpl_seq in enumerate(smpl_seq_list): 
            
            label = f"ours"
            gt_seq = None
            
            if seq_id == len(smpl_seq_list) - 1:
                label = "gt&pymafx"
                if img_dir is not None:
                    gt_seq = get_ground_truth_seq(img_dir)
                
        
            canvas_pred = self.create_canvas(scene, smpl_seq, gt_seq, img_width, img_height, 
                                             cam_intrinsics, img_dir, label=label, 
                                             slam_cam_poses=slam_cam_poses)
            all_canvases.append(canvas_pred)

        if show_2d_kpts:
            import ipdb; ipdb.set_trace()
            if single_2d_kpts:
                joints2d_list = [smpl_seq["rh_joints2d"] for smpl_seq in smpl_seq_list]
                canvas_pred_2d = self.create_canvas_2d_seq(
                    scene, joints2d_list, img_width, img_height)
                all_canvases.append(canvas_pred_2d)
            else:
                for seq_id, smpl_seq in enumerate(smpl_seq_list):
                    canvas_pred_2d = self.create_canvas_2d(
                        scene, smpl_seq["rh_joints2d"], img_width, img_height, 
                        is_gt=seq_id == len(smpl_seq_list) - 1)
                    all_canvases.append(canvas_pred_2d)
        
        if show_2d_loss:
            if 'reprojection_loss' in smpl_seq_list[0].keys():
                plt.clf(); plt.cla()
                loss_2d = smpl_seq_list[0]['reprojection_loss']
                
                # plot the loss curve
                plt.plot(loss_2d)
                plt.xlabel('Num SupGD iters')
                plt.ylabel('2d loss')
            
                # convert matplotlib plot to np arr
                cn = plt.gca().figure.canvas
                cn.draw()
                data = np.frombuffer(cn.tostring_rgb(), dtype=np.uint8)
                loss_image = data.reshape(cn.get_width_height()[::-1] + (3,))
                canvas2 = scene.create_canvas_2d(width=img_width, height=img_height)
                frame = canvas2.create_frame()
                sp_img = scene.create_image(image_id = "joint_loss_image")
                sp_img.from_numpy(loss_image)
      
                frame.add_image(sp_img, 'fit')
                all_canvases.append(canvas2)

            if 'contact_vel_loss' in smpl_seq_list[0].keys():
                plt.clf(); plt.cla()
                loss_2d = smpl_seq_list[0]['contact_vel_loss']
                
                # plot the loss curve
                plt.plot(loss_2d)
                plt.xlabel('Num SupGD iters')
                plt.ylabel('Vel loss')
            
                # convert matplotlib plot to np arr
                cn = plt.gca().figure.canvas
                cn.draw()
                data = np.frombuffer(cn.tostring_rgb(), dtype=np.uint8)
                loss_image = data.reshape(cn.get_width_height()[::-1] + (3,))
                canvas2 = scene.create_canvas_2d(width=img_width, height=img_height)
                frame = canvas2.create_frame()
                sp_img = scene.create_image(image_id = "vel_loss_image")
                sp_img.from_numpy(loss_image)
                # import ipdb; ipdb.set_trace()
                frame.add_image(sp_img, 'fit')
                all_canvases.append(canvas2)            


        scene.grid(
            width=f"{img_width*len(smpl_seq_list)}px", 
            grid_template_rows=f"{img_height}px {img_height}px", 
            grid_template_columns=" ".join([f"{img_width}px"] * len(smpl_seq_list))
        )

        scene.link_canvas_events(*all_canvases)
        scene.framerate = fps
        scene.save_as_html(filename, title=filename.replace(".html", ""))
        return scene

    def get_mano_right_params(self, smpl_seq):
        
        pose = smpl_seq[f'rh_pose'].float()
        trans = smpl_seq[f'rh_trans'].float()
        shape = smpl_seq['rh_shape'].float()
        orig_pose_shape = pose.shape
        
        
        # adjust beta shape and pose shape accoording to the batch size.
        # No need to input batch size in the forward pass 
        smpl_motion = self.mano_right.forward(
            global_orient=pose[..., :3].view(-1, 3),
            betas = shape.view(-1, 10),
            hand_pose=pose[..., 3:48].view(-1, 45),
            return_tips=True,
            transl=trans.view(-1, 3))
        
        mano_verts = smpl_motion.vertices
        mano_joints = smpl_motion.joints 
        return mano_verts, mano_joints   
    
    def get_mano_left_params(self, smpl_seq):
        
        pose = smpl_seq[f'lh_pose'].float()
        trans = smpl_seq[f'lh_trans'].float()
        shape = smpl_seq['lh_shape'].float()
        orig_pose_shape = pose.shape
        
        smpl_motion = self.mano_left.forward(
            global_orient=pose[..., :3].view(-1, 3),
            betas = shape.view(-1, 10),
            hand_pose=pose[..., 3:48].view(-1, 45),
            transl=trans.view(-1, 3))

        mano_verts = smpl_motion.vertices
        mano_joints = smpl_motion.joints 
        return mano_verts, mano_joints 
    

def get_mano_seq(stage_res, nemf_out=False):
 
    for k, v in stage_res.items():    
        if isinstance(v, np.ndarray) and not k in ['gender', 'img_dir', 'save_path', 'config_type', 'source', 'handedness']:  
            stage_res[k] = torch.from_numpy(v)
    
    floor_R = np.eye(3)
    floor_t = np.array([0, 0, 0])
    seqlen = stage_res['trans'].shape[0]

    smpl_seq = {}
    
    # this means they are given separately in stage_res dictionary 
    if "root_orient" in stage_res.keys():
        smpl_seq['rh_pose'] = torch.cat([stage_res['root_orient'], stage_res['poses']], dim=-1) # stage_res['poses'][:, :48] 
    # this means they are given together in stage_res dictionary 
    else:
        
        print("Root rotation and hand pose given together.")
        # The output of the NeMF model is of different shape than stage III optimization. 
        if nemf_out:
            smpl_seq['rh_pose'] = stage_res['poses'][:, :48]        
        else:
            assert stage_res['poses'].shape[1] == 48
            smpl_seq['rh_pose'] = stage_res['poses']        
        
    # smpl_seq['rh_pose'] = torch.cat([stage_res['right_root_orient'], stage_res['poses']], dim=-1) # stage_res['poses'][:, :48] 
    # smpl_seq['lh_pose'] = torch.cat([stage_res['left_root_orient'], stage_res['pose_left']], dim=-1) # stage_res['poses'][:, :48] 
    smpl_seq['rh_trans'] = stage_res['trans']
    # smpl_seq['lh_trans'] = stage_res['lh_trans']

    if stage_res["betas"].shape[0] == seqlen:
        smpl_seq['rh_shape'] = stage_res['betas']    
    else:
        smpl_seq['rh_shape'] = stage_res['betas'].unsqueeze(0).repeat_interleave(seqlen, dim=0)
    # smpl_seq['lh_shape'] = stage_res['lh_betas'].unsqueeze(0).repeat_interleave(seqlen, dim=0)
    smpl_seq['gender'] = 'neutral'
        
    if 'cam_R' in stage_res.keys():
        smpl_seq["cam_rot"] = stage_res['cam_R']
        smpl_seq["cam_trans"] = stage_res['cam_t']
    else:
        smpl_seq["cam_rot"] = torch.eye(3).unsqueeze(0).repeat_interleave(smpl_seq['rh_pose'].shape[0], dim=0)
        smpl_seq["cam_trans"] = torch.zeros((smpl_seq['rh_pose'].shape[0], 3))

    
    if 'cam_f' in stage_res.keys():
        smpl_seq["cam_intrinsics"] = torch.tensor(
            [[stage_res['cam_f'][0, 0], 0, stage_res['cam_center'][0,0]], 
            [0, stage_res['cam_f'][0,1], stage_res['cam_center'][0,1]], 
            [0, 0, 1]]
        )#torch.tensor(gt['cam_mtx'])
    else:
        print("No camera intrinsics given. Using default values.")
        smpl_seq["cam_intrinsics"] = torch.tensor(
            [[1060.0, 0, 960], 
            [0, 1060, 540], 
            [0, 0, 1]]
        )
    if 'joints_2d' in stage_res.keys():
        smpl_seq['rh_joints2d'] = stage_res['joints_2d']
    if 'lh_joints2d' in stage_res.keys():
        smpl_seq['lh_joints2d'] = stage_res['lh_joints2d']
    if 'joints_3d' in stage_res.keys():
        smpl_seq['rh_joints3d'] = stage_res['joints_3d']
    if 'lh_joints3d' in stage_res.keys():
        smpl_seq['lh_joints3d'] = stage_res['lh_joints3d']
        
    smpl_seq["floor_R"] = floor_R
    smpl_seq["floor_t"] = floor_t

    return smpl_seq


def load_res(res_path):
    '''
    Load np result from our model or GT
    '''
    # res_path = os.path.join(result_dir, file_name)
    if not os.path.exists(res_path):
        return None
    res = np.load(res_path, allow_pickle=True)
    res_dict = {k : res[k] for k in res.files}
    return res_dict


def vis_opt_results(npz_file_path, gt_npz_file_path, img_dir):

    frame_size = np.load(npz_file_path)["poses"].shape[0]
    renderer = HTMLRenderer(render_body=True, frame_size=frame_size)
    
    html_fname = npz_file_path.replace('npz', 'html')
    smpl_seqs = []
    
    smpl_seq = get_mano_seq(load_res(npz_file_path)) 
    smpl_seqs.append(smpl_seq)
    
    # try adding the gt if exists
    try:
        smpl_seq = get_mano_seq(load_res(gt_npz_file_path)) 
        smpl_seqs.append(smpl_seq)

    except:
        print("No gt file found.")
    
    renderer.create_animation(smpl_seqs, html_fname, fps=30, img_dir=img_dir, show_2d_kpts=False, single_2d_kpts=False)
    
    return 


def vis_static_results():
     
    # stage3 case 
    if len(sys.argv) == 4:
        npz_file_path = sys.argv[1]
        gt_npz_file_path = sys.argv[2]
        img_dir = sys.argv[3]

        is_gt = "gt" in npz_file_path 
    
        frame_size = np.load(npz_file_path)["poses"].shape[0]

        renderer = HTMLRenderer(render_body=True, frame_size=frame_size)
        
        html_fname = npz_file_path.replace('npz', 'html')
        smpl_seqs = []
        
        smpl_seq = get_mano_seq(load_res(npz_file_path)) 
        smpl_seqs.append(smpl_seq)
        
        # try adding the gt if exists
        try:
            smpl_seq = get_mano_seq(load_res(gt_npz_file_path)) 
            smpl_seqs.append(smpl_seq)
    
        except:
            print("No gt file found.")
        
        renderer.create_animation(smpl_seqs, html_fname, fps=30, img_dir=img_dir, show_2d_kpts=True, single_2d_kpts=False)
    
    
    # for visualization of train, motion inbetween, train basic etc.
    else:
        fnames = sys.argv[1]
    
        npz_file_paths = []
        for root, dirs, files in os.walk(fnames):
            for _file in files:
                if _file.endswith(".npz"):
                    # only load gt and 30 fps reconstruction files
                    if (not _file.endswith("gt.npz") and _file.endswith("30fps.npz")) or _file.endswith("gt.npz"):
                        npz_file_paths.append(os.path.join(root, _file))
        # pair them up
        npz_file_paths = sorted(npz_file_paths)
        assert len(npz_file_paths) % 2 == 0
        pair_number = len(npz_file_paths) // 2
        # have for loop for every file on the folder. Read 30 fps files only. 
        for i in range(pair_number):
            print(f"{i}/{pair_number}")
            # first prediction then gt
            if "gt" in npz_file_paths[2*i]:
                pairs = [npz_file_paths[2*i+1], npz_file_paths[2*i]]
            else:
                pairs = [npz_file_paths[2*i], npz_file_paths[2*i+1]]
            
            _npz_file_path = pairs[0]
            try:
                frame_size = int(_npz_file_path.split('/')[2].split('_')[1])
            except:
                # normal train test part 
                frame_size = np.load(_npz_file_path)["poses"].shape[0]

            renderer = HTMLRenderer(render_body=True, frame_size=frame_size)
            
            html_fname = _npz_file_path.replace('npz', 'html')
            smpl_seqs = []
            for fname in pairs:
                smpl_seq = get_mano_seq(load_res(fname), nemf_out=True) 

                # since the model does not trained on any translation loss its not eliglible to input it to body model. 
                smpl_seq["rh_trans"] = torch.zeros_like(smpl_seq["rh_trans"])

                smpl_seqs.append(smpl_seq)
                
            renderer.create_animation(smpl_seqs, html_fname, fps=30)
    
    

if __name__=="__main__":
    vis_static_results()
    
