import os
import cv2
import sys
import glob
import math
import time
import torch
import joblib
import shutil
import tempfile
import argparse
import colorsys
import subprocess
import numpy as np
import open3d as o3d
from tqdm import tqdm
import os.path as osp
from tqdm import tqdm
from matplotlib import cm
from PIL import ImageColor
from smplx import SMPL, MANO
import matplotlib.pyplot as plt
import open3d.visualization.rendering as rendering

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from fitting_utils import RIGHT_WRIST_BASE_LOC

AUGMENTED_MANO_CHAIN = np.array([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 15, 3, 6, 12, 9]) 


COLORS = {
    # colorbline/print/copy safe:
    'purple': np.array([118/255, 42/255, 131/255]),
    'turkuaz': np.array([50/255, 134/255, 204/255]),
    'light_blue': np.array([0.65098039, 0.74117647, 0.85882353]),
    'light_pink': np.array([.9, .7, .7]),  # This is used to do no-3d
    'light_green': np.array([i/255 for i in [120, 198, 121]])
}

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


CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]
])

def get_mano_skeleton():
    return np.array([[0 , 1],
                    [1 , 2],
                    [2 , 3],
                    [3 , 4],
                    [0 , 5],
                    [5 , 6],
                    [6 , 7],
                    [7 , 8],
                    [0 , 9],
                    [9 ,10],
                    [10,11],
                    [11,12],
                    [0 ,13],
                    [13,14],
                    [14,15],
                    [15,16],
                    [0 ,17],
                    [17,18],
                    [18,19],
                    [19,20],
                    [20,21]])


SMPL_MODEL_DIR = "data/body_models/SMPL"


def mat4x4_inverse(mat4x4):
    R = mat4x4[:3, :3]
    t = mat4x4[:3, 3]
    R = R.transpose(-1, -2)
    t = -R @ t[..., None]
    mat4x4[:3, :3] = R
    mat4x4[:3, 3] = t[..., 0]
    return mat4x4


def batch_look_at_np(camera_position, look_at, camera_up_direction):
    r"""Generate transformation matrix for given camera parameters.
    Formula is :math:`\text{P_cam} = \text{P_world} * \text{transformation_mtx}`,
    with :math:`\text{P_world}` being the points coordinates padded with 1.
    Args:
        camera_position (torch.FloatTensor):
            camera positions of shape :math:`(\text{batch_size}, 3)`,
            it means where your cameras are
        look_at (torch.FloatTensor):
            where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
        camera_up_direction (torch.FloatTensor):
            camera up directions of shape :math:`(\text{batch_size}, 3)`,
            it means what are your camera up directions, generally [0, 1, 0]
    Returns:
        (torch.FloatTensor):
            The camera transformation matrix of shape :math:`(\text{batch_size}, 4, 3)`.
    """
    z_axis = (camera_position - look_at)
    z_axis /= np.linalg.norm(z_axis, axis=1, keepdims=True)
    x_axis = np.cross(camera_up_direction, z_axis, axis=1)
    x_axis /= np.linalg.norm(x_axis, axis=1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis, axis=1)
    rot_part = np.stack([x_axis, y_axis, z_axis], axis=2)
    trans_part = (np.expand_dims(-camera_position, axis=1) @ rot_part)
    trans_part = trans_part.transpose(0, 2, 1)
    return np.concatenate([rot_part, trans_part], axis=2)


def get_material(color=(0.7, 0.7, 0.7, 1.0), shader="defaultLit"):
    mat = rendering.MaterialRecord()
    mat.base_color = color
    # red.base_color = [0.7, 0.0, 0.0, 1.0]
    mat.shader = shader
    return mat


def rotation_about_x(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def gl_camera_to_world(worldtocam: torch.Tensor) -> torch.Tensor:
    worldtocam = mat4x4_inverse(worldtocam)
    return worldtocam @ rotation_about_x(math.pi)[None].to(worldtocam)


def simplify_mesh(v, f, num_tris=500):
    
    mesh_in = o3d.geometry.TriangleMesh()
    mesh_in.vertices = o3d.utility.Vector3dVector(v)
    mesh_in.triangles = o3d.utility.Vector3iVector(f)
    mesh_smp = mesh_in.simplify_quadric_decimation(target_number_of_triangles=500)

    return mesh_smp.vertices, mesh_smp.triangles


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


def images_to_video(img_dir, out_path, img_fmt="%06d.jpg", fps=30, crf=25, verbose=False):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    ffmpeg_path = '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    cmd = [ffmpeg_path, '-y', '-r', f'{fps}', '-f', 'image2', '-start_number', '0',
            '-i', f'{img_dir}/{img_fmt}', '-vcodec', 'libx264', 
            '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", '-crf', f'{crf}', '-pix_fmt', 'yuv420p', out_path]
    if not verbose:
        cmd += ['-hide_banner', '-loglevel', 'error']
    subprocess.run(cmd)
    
    
def images_to_video_glob(img_dir, out_path, img_fmt="%06d.png", fps=30, crf=25, verbose=False):
    os.makedirs(osp.dirname(out_path), exist_ok=True)
    ffmpeg_path = 'ffmpeg' # '/usr/bin/ffmpeg' if osp.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    cmd = f"/usr/bin/ffmpeg -framerate 30 -pattern_type glob -i '{img_dir}/*.jpg' -y -c:v libx264 -pix_fmt yuv420p {out_path}"
    subprocess.run(cmd, shell=True)
    cmd = f"/usr/bin/ffmpeg -framerate 30 -pattern_type glob -i '{img_dir}/*.png' -y -c:v libx264 -pix_fmt yuv420p {out_path}"
    subprocess.run(cmd, shell=True)


class Open3DRenderer():
    def __init__(
        self, 
        device=torch.device('cpu'), 
        use_floor=True,
        add_cube=False, 
        cam_distance=5,
        verbose=False,
        headless=True,
        show_axes=True,
        enable_shadow=True, 
        enable_ambient_occlusion=True,
        enable_antialiasing=True,
        enable_post_processing=False,
        img_quality=0,
        bg_color=(1., 1., 1., 1.),
        debug_n_frames=1,
        view_type='camera', #'world',  
        trails=False,
        flip_flag=False):
        
        MANO_RIGHT_MODEL_DIR = "./data/body_models/mano/MANO_RIGHT.pkl"
        self.mano_right = MANO(MANO_RIGHT_MODEL_DIR, is_rhand=True, flat_hand_mean=True, use_pca=False)
        self.rh_faces = self.mano_right.faces
        
        self.smpl_joint_parents = AUGMENTED_MANO_CHAIN
        
        self.window_size = None # window_size
        self.view_type = view_type
        self.verbose = verbose
        self.trails = trails
        
        # scene objects
        self.add_cube = add_cube
        self.use_floor = use_floor
        self.show_axes = show_axes
        self.bg_color = bg_color
        
        # animation control
        self.fr = 0
        self.num_fr = debug_n_frames
        
        # camera
        self.cam_distance = cam_distance
        self.cam_intrinsics = None
        
        # render settings
        self.headless = headless
        self.show_axes = show_axes
        self.img_quality = img_quality
        self.enable_shadow = enable_shadow
        self.enable_antialiasing = enable_antialiasing
        self.enable_post_processing = enable_post_processing
        self.enable_ambient_occlusion = enable_ambient_occlusion

        self.flip_flag = flip_flag
        
        
    def init_scene(self):
        
        if hasattr(self, 'render'):
            del self.render
            print('deleting render')
            
        self.render = rendering.OffscreenRenderer(width=self.window_size[0], height=self.window_size[1])
            
        self.render.scene.view.set_ambient_occlusion(self.enable_ambient_occlusion)
        self.render.scene.view.set_antialiasing(self.enable_antialiasing)
        self.render.scene.view.set_shadowing(self.enable_shadow)
        self.render.scene.view.set_post_processing(self.enable_post_processing)
        
        self.render.scene.set_background(self.bg_color)
        # self.render.scene.scene.set_sun_light([-0.707, 0.0, .707], [1.0, 1.0, 1.0], 150000)
        self.render.scene.scene.set_sun_light([-np.sqrt(1/3), np.sqrt(1/3), np.sqrt(1/3)], [1.0, 1.0, 1.0], 150000)        
        self.render.scene.scene.enable_sun_light(True)
         
        
        # floor
        if self.use_floor:
            print('adding floor')
            
            gp_mat = rendering.MaterialRecord()
            
            # tex = make_checker_board_texture('#81C6EB', '#D4F1F7', width=100, height=100)
            tex = make_checker_board_texture('#FFFFFF', '#E5E4E2', width=100, height=100, n_tile=50)
            gp_mat.albedo_img = o3d.geometry.Image(tex)
            gp_mat.aspect_ratio = 1.0
            gp_mat.shader = "defaultLit"

            extent = 40.0
            h = 0.08
            ground_plane = o3d.geometry.TriangleMesh.create_box(
                extent, extent, h, create_uv_map=True, 
                map_texture_to_each_face=True)
            ground_plane.compute_vertex_normals()
            
            h += 0.1
            ground_plane.translate([-extent // 2, -extent // 2, -h])
            self.render.scene.add_geometry("ground_plane", ground_plane, gp_mat)
            
        if self.show_axes: 
            print('show axes')
            # self.render.scene.show_axes(True)
            coords = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
            coords.compute_vertex_normals()
            self.render.scene.add_geometry("coords", coords, rendering.MaterialRecord())
    
    def get_mano_right_params(self, smpl_seq):
        
        if "rh_verts" in smpl_seq:
            return smpl_seq["rh_verts"].squeeze(1), smpl_seq["rh_joints3d"].squeeze(1)
        
        pose = smpl_seq[f'rh_pose'].float()
        trans = smpl_seq[f'rh_trans'].float()
        shape = smpl_seq['rh_shape'].float()
        orig_pose_shape = pose.shape
         
        # adjust beta shape and pose shape according to the batch size.
        # No need to input batch size in the forward pass 
 
        mano_out = self.mano_right.forward(
            global_orient=pose[..., :3].view(-1, 3),
            betas = shape.view(-1, 10),
            hand_pose=pose[..., 3:48].view(-1, 45),
            return_tips=True,
            transl=trans.view(-1, 3))

        return mano_out.vertices, mano_out.joints   
   
   
    def render_video(self, smpl_seq_list, img_width, img_height, white_background, 
        video_path="demo.mp4", 
        img_dir=None, 
        fps=30,
        cleanup=True,
        frame_dir=None,
        cam_eye=(0.0, 0.0, 1.5),
        cam_look_at=(0.0, -1.0, 0.0),
        cam_up=(0.0, 0.0, 1.0),
        method='ours',
        alpha_val=0.90):        
        # since there is a appension to the last frames to make the number divisible by 128. 
        # smpl_seq_list[0]['rh_pose'].shape[0] != len(smpl_seq_list)

        imgfnames = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        N_frames = len(imgfnames)

        if N_frames == 0:
            imgfnames = sorted(glob.glob(os.path.join(img_dir, "*.png")))
            N_frames = len(imgfnames)

        N_person = len(smpl_seq_list)
     
        if method == 'hmp':
            mesh_colors = {0: COLORS["light_green"]}   
            # mesh_colors = {0: COLORS["turkuaz"]}   
        elif method == 'pymafx':
            mesh_colors = {0: COLORS["light_blue"]} 
        else:
            mesh_colors = {0: COLORS["light_green"],           # hmp  
                        1: COLORS["light_blue"]}    	       # pymafx
         
        scene_items = {'cam': [], 'cam_lines': [], 'cam_frustums': []}
        
        for i in range(N_person):
            scene_items[f'body_meshes_{i}'] = []
            scene_items[f'root_traj_{i}'] = []
        
        smpl_seq = smpl_seq_list[0]
        smpl_verts_list = []
        smpl_joints_list = []
        
        min_x, min_y, max_x, max_y = np.inf, np.inf, -np.inf, -np.inf
        
        self.window_size = (int(img_width), int(img_height))
        
        for sid in range(N_person):
            smpl_verts, smpl_joints = self.get_mano_right_params(smpl_seq_list[sid])
         
            smpl_verts_list.append(smpl_verts)
            smpl_joints_list.append(smpl_joints)
            
            if smpl_joints[..., 0].min() < min_x:
                min_x = smpl_joints[..., 0].min()
            if smpl_joints[..., 0].max() > max_x:
                max_x = smpl_joints[..., 0].max()
            if smpl_joints[..., 1].min() < min_y:
                min_y = smpl_joints[..., 1].min()
            if smpl_joints[..., 1].max() > max_y:
                max_y = smpl_joints[..., 1].max() 

        if isinstance(smpl_seq['cam_rot'], np.ndarray):
            smpl_seq["cam_rot"] = torch.from_numpy(smpl_seq["cam_rot"])
            smpl_seq['cam_trans'] = torch.from_numpy(smpl_seq['cam_trans'])
            
        cam_points = (-smpl_seq["cam_rot"].transpose(-1, -2) @ smpl_seq['cam_trans'][..., None])[..., 0].numpy()
        
        if self.view_type == 'camera':           
            self.window_size = (int(img_width), int(img_height))
              
        self.init_scene()
  
        if frame_dir is None:
            frame_dir = tempfile.mkdtemp(prefix="visualizer3d-")
        else:
            if osp.exists(frame_dir):
                shutil.rmtree(frame_dir, ignore_errors=True)
            os.makedirs(frame_dir)

        # the only_gt_frames is for displaying frames with gt annotations 
        only_gt_frames = "render_gt_only" in smpl_seq.keys()
        render_timesteps = np.array(smpl_seq["frame_id"]) if only_gt_frames  else range(0, N_frames, 1)
        
        for j in tqdm(render_timesteps, desc='Rendering'): 
             
            for sid in range(N_person):
      
                v = smpl_verts_list[sid][render_timesteps.tolist().index(j)] if only_gt_frames else smpl_verts_list[sid][j] 
        
             
                if self.render.scene.has_geometry(f'smpl_body_{sid}'):
                    self.render.scene.remove_geometry(f'smpl_body_{sid}')
        
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), 
                                                 o3d.utility.Vector3iVector(self.mano_right.faces))
                mesh.compute_vertex_normals()
                
                mat = rendering.MaterialRecord()
                mat.base_color = [*mesh_colors[sid].tolist(), 1.0]
                mat.shader = "defaultLit"
                
                self.render.scene.add_geometry(f'smpl_body_{sid}', mesh, mat)
        
            cam_extrinsics = np.eye(4)
            cam_int = smpl_seq_list[sid]["cam_intrinsics"]
            o3d_pinhole_cam = o3d.camera.PinholeCameraIntrinsic(self.img_width, self.img_height, cam_int)
            self.render.setup_camera(o3d_pinhole_cam, cam_extrinsics.astype(np.float64))
        
            img_file_j = o3d.io.read_image(imgfnames[j])   

            img_path = f'{frame_dir}/{j:06d}.jpg'
            hand_rgb = self.render.render_to_image()

            if white_background:
                o3d.io.write_image(img_path, hand_rgb)
                continue
            else:
                img_rgb = np.asarray(img_file_j)
  
            hand_rgb = np.array(hand_rgb)
            valid_mask = (np.sum(hand_rgb, axis=-1) < 765)[:, :, np.newaxis]
            
            
            try:
                # blend the two images through masking alpha blending            
                img_overlay = valid_mask * hand_rgb * alpha_val + valid_mask * img_rgb * (1 - alpha_val) + img_rgb * (1 - valid_mask)
            except:
                import ipdb; ipdb.set_trace()
            
            
            img_overlay = o3d.geometry.Image((img_overlay).astype(np.uint8))
            
            o3d.io.write_image(img_path, img_overlay)
         
    
        images_to_video(frame_dir, video_path, fps=fps, crf=25, verbose=self.verbose)
        return
         
    
    @torch.no_grad()
    def create_animation(self, smpl_seq_list, img_dir, video_path, white_background, fps=30, method='ours'):
        
        self.cam_intrinsics = smpl_seq_list[0]['cam_intrinsics']
        
        if len(sorted(glob.glob(os.path.join(img_dir, '*.jpg')))) != 0:
            img_height, img_width, _ = cv2.imread(sorted(glob.glob(os.path.join(img_dir, '*.jpg')))[0]).shape
        else:
            img_height, img_width, _ = cv2.imread(sorted(glob.glob(os.path.join(img_dir, '*.png')))[0]).shape

        # find possible pymafx path 
        vp_split = os.path.dirname(video_path).split("/")
        # vp_split[2] = "_encode_decode"
        vp_split[2] = "_pymafx_raw"
       
        encode_decode_path = "/".join(vp_split)
        encode_decode_pymafx = os.path.join(encode_decode_path, "recon_000_30fps_pymafx.mp4")
        
        self.img_width = img_width
        self.img_height = img_height
        
        raw_vid_path = os.path.join(os.path.dirname(img_dir), "rgb_raw.mp4")
        # raw_images = sorted([os.path.join(img_dir, x) for x in os.listdir(raw_images_path) if x.endswith('.png') or x.endswith('.jpg')])
        
        frame_dir = os.path.join(os.path.dirname(video_path), "open3d_img")
        vid_paths = []
    
        for i in range(len(smpl_seq_list)):
    
            # check if the video already exists in encode-decode folder
            if method[i] == "pymafx" and os.path.exists(encode_decode_pymafx):
                video_path_inp = encode_decode_pymafx
            else:
                seq_list = [smpl_seq_list[i]] if i !=2 else smpl_seq_list 
                video_path_inp = video_path.replace('.mp4', f'_{method[i]}.mp4')
                print(video_path_inp)
   
                self.render_video(smpl_seq_list=seq_list, 
                            img_width=img_width, 
                            img_height=img_height,  
                            img_dir=img_dir, 
                            video_path=video_path_inp,
                            fps=fps,
                            frame_dir=frame_dir,
                            method=method[i],
                            white_background=white_background)
            
            vid_paths.append(video_path_inp) 

        flip_command = '-vf hflip' if self.flip_flag else ''

        if len(vid_paths) == 2:
            # call ffmpeg to create video 
            # cmd = f"/usr/bin/ffmpeg -y -i {vid_paths[0]} -i {vid_paths[0]} -i {vid_paths[1]} -filter_complex \
            cmd = f"/usr/bin/ffmpeg -y -i {raw_vid_path} -i {vid_paths[1]} -i {vid_paths[0]} -filter_complex \
                    '[0]drawtext=text=VIDEO INPUT: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=black: fontsize=w/30: x=text_w/8: y=text_h [0:v]; \
                    [1]drawtext=text={method[1].upper()}: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=black: fontsize=w/30: x=text_w/8: y=text_h [1:v]; \
                    [2]drawtext=text={method[0].upper()}: fontfile=/usr/share/fonts/truetype/msttcorefonts/arial.ttf: fontcolor=black: fontsize=w/30: x=text_w/8: y=text_h [2:v]; \
                    [0:v]scale=-1:{img_height}[0v]; \
                    [1:v]scale=-1:{img_height}[1v]; \
                    [2:v]scale=-1:{img_height}[2v]; \
                    [0v][1v][2v]hstack=inputs=3[outv]' \
                    -map '[outv]' {video_path} -y"
        
        else:             
            # convert to video without any filtering. For MeTro visualization
            cmd = f"/usr/bin/ffmpeg -pattern_type glob -y -i '{frame_dir}/*.jpg' {flip_command} {video_path}"

        subprocess.run(cmd, shell=True)
     
    
def get_mano_seq(stage_res, nemf_out=False):


    for k, v in stage_res.items():    
        if isinstance(v, np.ndarray) and not k in ['gender', 'img_dir', 'save_path', 'config_type', 'source', 'handedness']:  
            stage_res[k] = torch.from_numpy(v)
       
    floor_R = np.eye(3)
    floor_t = np.array([0, 0, 0])
    
    try: 
        seqlen = stage_res['trans'].shape[0]
    except:
        seqlen = stage_res['vertices'].shape[0]

    smpl_seq = {}

    # this means they are given separately in stage_res dictionary 
    if "root_orient" in stage_res.keys() and "poses" in stage_res.keys():
        smpl_seq['rh_pose'] = torch.cat([stage_res['root_orient'], stage_res['poses']], dim=-1) # stage_res['poses'][:, :48] 
    # this means they are given together in stage_res dictionary under the key value of poses 
    elif "poses" in stage_res.keys():
        print("Root rotation and hand pose given together.")
        # The output of the NeMF model is of different shape than stage III optimization. 
        if nemf_out:
            smpl_seq['rh_pose'] = stage_res['poses'][:, :48]        
        else:
            assert stage_res['poses'].shape[1] == 48
            smpl_seq['rh_pose'] = stage_res['poses']        
    # this means we dont have any pose information, read vertices and joints  
    else:
        smpl_seq['rh_verts'] = stage_res['vertices']
        smpl_seq['rh_joints3d'] = stage_res['joints']
 
    if 'trans' in stage_res.keys():
        smpl_seq['rh_trans'] = stage_res['trans']

    if 'betas' in stage_res.keys():
        if stage_res["betas"].shape[0] == seqlen:
            smpl_seq['rh_shape'] = stage_res['betas']    
        else:
            smpl_seq['rh_shape'] = stage_res['betas'].unsqueeze(0).repeat_interleave(seqlen, dim=0)

    smpl_seq['gender'] = 'neutral'

    if 'cam_R' in stage_res.keys():
        smpl_seq["cam_rot"] = stage_res['cam_R']
        smpl_seq["cam_trans"] = stage_res['cam_t']
    else:
        smpl_seq["cam_rot"] = torch.eye(3).unsqueeze(0).repeat_interleave(seqlen, dim=0)
        smpl_seq["cam_trans"] = torch.zeros((seqlen, 3))
    
    if 'cam_f' in stage_res.keys():
        smpl_seq["cam_intrinsics"] = torch.tensor(
            [[stage_res['cam_f'][0, 0], 0, stage_res['cam_center'][0,0]], 
            [0, stage_res['cam_f'][0,1], stage_res['cam_center'][0,1]], 
            [0, 0, 1]]
        )
    else:
        print("No camera intrinsics given. Using default values.")
        smpl_seq["cam_intrinsics"] = torch.tensor(
            [[1060.0, 0, 960], 
            [0, 1060, 540], 
            [0, 0, 1]])

    if 'render_gt_only' in stage_res.keys():
        smpl_seq["render_gt_only"] = stage_res['render_gt_only']
    if "frame_id" in stage_res.keys():
        smpl_seq["frame_id"] = stage_res["frame_id"]
    if 'rh_verts' in stage_res.keys():
        smpl_seq['rh_verts'] = stage_res['rh_verts']
    if 'joints_2d' in stage_res.keys():
        smpl_seq['rh_joints2d'] = stage_res['joints_2d']
    if 'lh_joints2d' in stage_res.keys():
        smpl_seq['lh_joints2d'] = stage_res['lh_joints2d']
    if 'joints_3d' in stage_res.keys():
        smpl_seq['rh_joints3d'] = stage_res['joints_3d']
    if 'lh_joints3d' in stage_res.keys():
        smpl_seq['lh_joints3d'] = stage_res['lh_joints3d']
    if 'frame_id' in stage_res.keys():
        smpl_seq['frame_id'] = stage_res['frame_id']

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

    if res_path.endswith(".pkl"):
        res_dict = joblib.load(res_path)
    else:
        res = np.load(res_path, allow_pickle=True)
        res_dict = {k : res[k] for k in res.files}
    
    return res_dict


def vis_opt_results(pred_file_path, gt_file_path, img_dir, post_process_flag=False, 
                    img_quality=1, trails=False, white_background=False, fps=30, flip_flag=False):
     
    if pred_file_path.endswith(".pkl"):
        frame_size = joblib.load(pred_file_path)["joints"].shape[0]
    else:
        frame_size = np.load(pred_file_path)["poses"].shape[0]
  
    smpl_seqs = []
    
    methods = ["hmp"]
    
    # load vertices directly if exists, if not take pose and shape to forward to the MANO model
    smpl_seq = get_mano_seq(load_res(pred_file_path)) 
    smpl_seqs.append(smpl_seq)

    # try adding the gt if exists (pymafx to be honest)
    try:
        smpl_seq = get_mano_seq(load_res(gt_file_path)) 
        smpl_seqs.append(smpl_seq)
        methods.append("pymafx")
        # methods.append("together")
    except:
        print("No gt file found.")
        
    file_extension = os.path.splitext(pred_file_path)[1]
    video_fname = pred_file_path.replace(file_extension, '.mp4')        
    
    os.makedirs(os.path.dirname(video_fname), exist_ok=True)

    renderer = Open3DRenderer(
        device=torch.device('cpu'), 
        use_floor=True,
        add_cube=False, 
        cam_distance=5,
        verbose=False,
        headless=True,
        show_axes=False,
        enable_shadow=False, 
        enable_ambient_occlusion=True,
        enable_antialiasing=True,
        enable_post_processing=post_process_flag,
        img_quality=img_quality,
        bg_color=(1., 1., 1., 1.),
        debug_n_frames=1,    
        trails=trails,
        flip_flag=flip_flag)
    
    renderer.create_animation(smpl_seqs, video_path=video_fname, img_dir=img_dir, fps=fps, method=methods, white_background=white_background)
    
     

    

    