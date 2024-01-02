import os
import sys
import wandb
import torch
import numpy as np
import scenepic as sp
from PIL import ImageColor
from typing import List, Any

cur_file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cur_file_path, '..'))

from utils.vis_utils import colors
from utils.logger_utils import mkdir


class HTMLRenderer(object):
    def __init__(self, width: int =1200, height: int = 800, wandb: bool = True, save_html: bool = True, 
                 html_name: str ="unnamed", wandb_project_name: str = "unnamed",  wandb_note: str = None, 
                 wandb_scene_title:str = "", wandb_logs: dict = {}, caller_func: str = "", skel_color: np.array = colors["red"], wandb_obj=None): 
        super().__init__()

        self._width = width
        self._height = height
        
        self.skeleton_color = skel_color
        self.wandb_scene_title = wandb_scene_title
        self.wandb_note = wandb_note
        self.wandb_logs = wandb_logs
        self.mocap_html = html_name
        self.save_html_flag = save_html
        self.save_wandb = wandb
        self.wandb_project_name = wandb_project_name

        # wandb object instance 
        self.wandb_obj = wandb_obj 
        
        assert caller_func in ["data_loader", "vis_log", "test_sampling", "single", "eval_recon", "recon_sampling", ""], "Invalid caller function!"
        
        if caller_func == "":
            self.caller_func = "vis_log"
        else:
            self.caller_func = caller_func

    def __call__(self, body_output_list: List, 
                 camera_rotation: torch.tensor,  
                 translate_origin: np.array = np.array([0, 0, 0]), 
                 show_ground_floor: bool = True) -> Any:

        

        # Convert to numpy in case the vertices & faces are tensors.
        for elem in body_output_list:
            if torch.is_tensor(elem.vertices):
                elem.vertices = elem.vertices.detach().cpu().numpy() 
        
            if torch.is_tensor(elem.faces):
                elem.faces = elem.faces.cpu().numpy()
            
        if torch.is_tensor(camera_rotation):
            camera_rotation = camera_rotation.detach().cpu().numpy()
        self.scene = sp.Scene()

        timestep = len(body_output_list[0].vertices)
        mesh_num = len(body_output_list)

        canvas = self.scene.create_canvas_3d(width=self._width, height=self._height)

        layer_settings = {}
        for mii in range(timestep):

            next_frame = canvas.create_frame(focus_point=sp.FocusPoint([0, 0, 0]))
            
            if show_ground_floor:
                next_frame.add_mesh(self.make_checkerboard_texture())

            for diff_parts in range(mesh_num):
                layer_settings[body_output_list[diff_parts].mesh_name] = {"opacity": 1.0}
                mesh = self.scene.create_mesh(shared_color=body_output_list[diff_parts].mesh_color, layer_id=body_output_list[diff_parts].mesh_name)
                # we need to have tensors/arrays stored in cpu

                mesh.add_mesh_without_normals(body_output_list[diff_parts].vertices[mii], body_output_list[diff_parts].faces)
                next_frame.add_mesh(mesh)

                # Add skeleton. Notice that the skeleton and mesh numbers do not have to match. There may be some meshes w/o skeleton or vice versa. 
                if body_output_list[diff_parts].skeleton_flag:
                    # input the skeletons for the timestep mii                       
                    skel_mesh = self.add_skeleton(body_output_list[diff_parts].skeleton[mii], body_output_list[diff_parts].skeleton_name)
                    next_frame.add_mesh(skel_mesh) 

            coord = self.scene.create_mesh(layer_id='camera')

            scale = 0.4
            matrix = scale * np.eye(4, 4)

            if translate_origin.any():
                matrix[:3, 3] = translate_origin

            coord.add_coordinate_axes(transform=matrix)

            camera_pose = np.eye(4)
            camera_pose[:3, :3] = camera_rotation

            coord.apply_transform(camera_pose)
            next_frame.add_mesh(coord)

        canvas.set_layer_settings(dict(camera={}, **layer_settings))    

        if self.save_html_flag:
            self.save_html_wandb()

        return

    def add_skeleton(self, joint_locs, mesh_name):
        """ Indicate the skeleton and the joint positions for better grasp. 
            
            b_out.skeleton: numpy array (N, 6) ---> In the second dimension, entries are the x,y,z coordinates of the two connected joints   
            m_name = name of the skeleton mesh eg. body_skeleton, hand_skeleton etc.
            """ 

        assert type(joint_locs) == np.ndarray, "Wrong data type!"

        # for bones
        mesh = self.scene.create_mesh(shared_color=self.skeleton_color, layer_id=mesh_name)
        mesh.add_lines(start_points=joint_locs[:, :3], end_points=joint_locs[:, 3:])
        
        
        skel_cat = np.concatenate([joint_locs[:, :3], joint_locs[:, 3:]], axis=0)
        unique_joints = np.unique(skel_cat, axis=0) # avoid repeating joints

        # create spheres for joints
        for jloc in unique_joints:
            transform_mat = np.eye(4)/75
            transform_mat[:3, 3] = jloc
            mesh.add_sphere(transform=transform_mat)

        return mesh

    def make_checkerboard_texture(self, color1='gray', color2='white', width=1, height=1, n_tile=50):
        c1 = np.asarray(ImageColor.getcolor(color1, 'RGB')).astype(np.uint8)
        c2 = np.asarray(ImageColor.getcolor(color2, 'RGB')).astype(np.uint8)
        hw = width
        hh = height
        c1_block = np.tile(c1, (hh, hw, 1))
        c2_block = np.tile(c2, (hh, hw, 1))
        tex = np.block([
            [[c1_block], [c2_block]],
            [[c2_block], [c1_block]]
        ])
        tex = np.tile(tex, (n_tile, n_tile, 1))

        # image and texture id should be the same for matching
        floor_img = self.scene.create_image(image_id="ground")
        floor_img.from_numpy(tex)
        floor_mesh = self.scene.create_mesh(texture_id="ground", layer_id="floor")
        floor_mesh.add_image(transform=sp.Transforms.Scale(20.))

        return floor_mesh
    
    def save_html_wandb(self):
        
        # Hierarchial order of the folders 
        if self.caller_func == "test_sampling":
            local_wandb_path = "./out/" + self.mocap_html.split("_")[0] + "/" + self.wandb_project_name
            local_html_folder_path = local_wandb_path 
            local_mocap_html_path = local_wandb_path + "/"  + self.mocap_html + ".html"
        
        elif self.caller_func == "eval_recon":
            local_wandb_path = os.path.join("./out/", self.mocap_html.split("/")[2], self.wandb_project_name)
            local_html_folder_path = local_wandb_path 
            local_mocap_html_path = self.mocap_html + ".html"
            
        elif self.caller_func == "single":
            local_wandb_path = "./logs/" + self.wandb_project_name
            local_html_folder_path = local_wandb_path + "/html" 
            local_mocap_html_path = local_html_folder_path + "/" + self.mocap_html + ".html"    
            
        elif self.caller_func == "vis_log":
            local_wandb_path = "./logs/" + self.wandb_project_name
            local_html_folder_path = local_wandb_path + "/html" 
            local_dataset_folder_path = local_html_folder_path + "/" + self.mocap_html.split("/")[0]  
            # directory of the html path
            local_mocap_html_path = local_html_folder_path + "/" + self.mocap_html + ".html"
        
        elif self.caller_func == "recon_sampling":
            local_wandb_path = os.path.join("./out", self.mocap_html.split("/")[2], self.wandb_project_name)
            local_html_folder_path = local_wandb_path 
            local_mocap_html_path = local_wandb_path + "/" + self.mocap_html.split("/")[-1] + ".html"
            # god knows what to do here
        
        else:
            raise ValueError("Invalid caller function")
            exit()
        
        # create logs/PROJECTNAME folder 
        mkdir(local_wandb_path)
        # create logs/PROJECTNAME/html folder 
        mkdir(local_html_folder_path)
            
        if self.caller_func == "vis_log":
            # create logs/PROJECTNAME/html/DATASETNAME folder
            if not os.path.exists(local_dataset_folder_path):
                os.mkdir(local_dataset_folder_path)
        
            # create logs/PROJECTNAME/html/DATASETNAME/SUBJECTNAME folder if not visualizing for the data_loader
            local_subject_name_path = local_dataset_folder_path + "/" + self.mocap_html.split("/")[1]
        
            if not os.path.exists(local_subject_name_path):
                os.mkdir(local_subject_name_path)    
    
        # import ipdb; ipdb.set_trace()
        self.scene.framerate = 30
        # Now we are ready to save the html file     
        self.scene.save_as_html(local_mocap_html_path, title="processed mocap")
        
        
        if self.save_wandb:
            
            mkdir(local_wandb_path)
            
            if self.wandb_obj is None:   
                wandb_instance = wandb.init(project=self.wandb_project_name, entity="hmp", dir=local_wandb_path, name="/".join(self.mocap_html.split("/")[-3:]), notes=self.wandb_note)
                wandb_instance.log({self.wandb_scene_title: wandb.Html(open(local_mocap_html_path))})
                wandb_instance.log({"log": self.wandb_logs})
                wandb_instance.finish()
                return

            wandb_instance = self.wandb_obj
            wandb_instance.log({self.wandb_scene_title: wandb.Html(open(local_mocap_html_path))})
            wandb_instance.log({"log": self.wandb_logs})
        
        return 