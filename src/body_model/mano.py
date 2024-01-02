import os 
import sys 
import torch
import torch.nn as nn
from smplx import SMPLX, MANO
from smplx.utils import Struct


RIGHT_WRIST_BASE_LOC = torch.tensor([[0.0957, 0.0064, 0.0062]])
LEFT_WRIST_BASE_LOC = torch.tensor([[-0.0957, 0.0064, 0.0062]])
HAND_TIP_IDS = {'thumb':744, 'index':320, 'middle': 443, 'ring': 554, 'pinky': 671}

class BodyModel(nn.Module):
    '''
    Wrapper around SMPLX body model class.
    '''

    def __init__(self,
                 model_path,
                 model_type,
                 device,
                 num_betas=10,
                 batch_size=1,
                 num_expressions=10,
                 name = "",
                 skeleton_flag = False,
                 mesh_color = [0.5, 0.5, 0.5],
                 **keyword_args):
        super(BodyModel, self).__init__()
        '''
        Creates the body model object at the given path.
        :param bm_path: path to the body model pkl file
        :param num_expressions: only for smplx
        :param model_type: one of [smpl, smplh, smplx]
        :param use_vtx_selector: if true, returns additional vertices as joints that correspond to OpenPose joints
        '''
        kwargs = {
                'model_type' : model_type,              
                'num_betas': num_betas,
                'batch_size' : batch_size,
                'num_expression_coeffs' : num_expressions,
                'use_pca' : False,
                ** keyword_args
        }
        assert(model_type in ['smplx', 'mano'])
        
        self.hand_argument_set = {"hand_pose", "betas", "global_orient", "transl", "no_shift", "return_finger_tips"}
        self.smplx_argument_set = {"root_orient", "betas", "body_pose", "right_hand_pose", "left_hand_pose", "transl", "jaw_pose", "reye_pose", "leye_pose"}

        self.model = eval(model_type.upper())(model_path, **kwargs).to(device)
        self.NUM_JOINTS = eval(model_type.upper()).NUM_JOINTS
        # add pelvis to the joint number 
        self.NUM_JOINTS += (model_type == "smplx") * 1

        self.mesh_name = name 
        self.device = device
        self.mesh_color = mesh_color
        self.model_type = model_type
        self.skeleton_flag = skeleton_flag
        self.skeleton_name = self.mesh_name + "_skeleton"

        if "is_rhand" in keyword_args.keys():
            self.is_rhand = keyword_args["is_rhand"]

        if skeleton_flag:
            self.set_skeleton_flag()
        

    def forward(self, input_dict, **kwargs):
        '''
        Note dmpls are not supported.
        '''
        for k_name in input_dict.keys():
            if type(input_dict[k_name]) != torch.Tensor:
                input_dict[k_name] = torch.tensor(input_dict[k_name], dtype=torch.float32).to(self.device)

        if self.model_type == "smplx":
            
            assert set(input_dict.keys()).issubset(self.smplx_argument_set), "Wrong input arguments for SMPLX body model."    
            
            out_obj = self.model(
                betas=input_dict["betas"] if "betas" in input_dict.keys() else None,
                global_orient=input_dict["root_orient"] if "root_orient" in input_dict.keys() else None,
                body_pose=input_dict["body_pose"] if "body_pose" in input_dict.keys() else None,
                right_hand_pose= input_dict["right_hand_pose"] if "right_hand_pose" in input_dict.keys() else None,
                left_hand_pose=input_dict["left_hand_pose"] if "left_hand_pose" in input_dict.keys() else None,
                transl=input_dict["transl"] if "transl" in input_dict.keys() else None,
                jaw_pose=input_dict["jaw_pose"] if "jaw_pose" in input_dict.keys() else None,
                leye_pose=input_dict["leye_pose"] if "leye_pose" in input_dict.keys() else None,
                reye_pose=input_dict["reye_pose"] if "reye_pose" in input_dict.keys() else None,
                return_full_pose=True,
                **kwargs
        )
        # mano case
        else:            
            out_obj = self.model(
                hand_pose=input_dict["hand_pose"] if "hand_pose" in input_dict.keys() else None,
                betas=input_dict["betas"] if "betas" in input_dict.keys() else None,
                global_orient=input_dict["global_orient"] if "global_orient" in input_dict.keys() else None,
                transl=input_dict["transl"] if "transl" in input_dict.keys() else None,
                **kwargs)  
            
            # import ipdb; ipdb.set_trace()
            assert set(input_dict.keys()).issubset(self.hand_argument_set), "Wrong input arguments for MANO hand model."    
            
            # shift the vertices and the joints of the output so that the root joint is on the origin if there is no translation.
            if "no_shift" not in input_dict.keys():
                WRIST_DEFAULT_POSE_LOC = RIGHT_WRIST_BASE_LOC if self.is_rhand else LEFT_WRIST_BASE_LOC
                out_obj.joints -= torch.repeat_interleave(WRIST_DEFAULT_POSE_LOC.unsqueeze(0).to(self.device), self.model.batch_size, dim=0)
                out_obj.vertices -= torch.repeat_interleave(WRIST_DEFAULT_POSE_LOC.unsqueeze(0).to(self.device), self.model.batch_size, dim=0)
             
            # add finger tips to the joints
            if "return_finger_tips" in input_dict.keys():
                if input_dict["return_finger_tips"]:
                    # concatenate the finger tips to the end columns of the joints matrix.
                    out_obj.joints = torch.cat([out_obj.joints, out_obj.vertices[:, list(HAND_TIP_IDS.values())]], dim=1)

        out = {
            'mesh_name': self.mesh_name,
            'mesh_color': self.mesh_color,

            'vertices': out_obj.vertices,
            'faces': self.model.faces_tensor,
            'betas': out_obj.betas,
            'joints': out_obj.joints,
            'full_pose': out_obj.full_pose,
            'root_orient': out_obj.global_orient,
            'transl': self.model.transl,

            'skeleton_flag': self.skeleton_flag,
            'skeleton': self.produce_skeleton(out_obj.joints) if self.skeleton_flag else None, 
            'skeleton_name': self.skeleton_name}

        if self.model_type == 'smplx':            
            out['pose_right_hand'] = out_obj.right_hand_pose
            out['left_hand_pose'] = out_obj.left_hand_pose
            out['pose_jaw'] = out_obj.jaw_pose


        return Struct(**out)


    def produce_skeleton(self, joints):
        """ joints: Torch tensor of shape (Timestep, JointNum, 3)"""
        joint_loc = torch.cat([joints[:, self.directed_graph[:, 0]], joints[:, self.directed_graph[:, 1]]], dim=2)

        return joint_loc.detach().cpu().numpy()

    
    def set_skeleton_flag(self):
        func = eval(self.model_type + "_full_chain")
        self.directed_graph = func(self.is_rhand) if self.model_type == "mano" else func()