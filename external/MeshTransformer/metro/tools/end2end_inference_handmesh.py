"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

End-to-end inference codes for 
3D hand mesh reconstruction from an image
"""

from __future__ import absolute_import, division, print_function
import os
import sys
import cv2
import json
import torch
import joblib
import argparse
import numpy as np
import os.path as op
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torchvision.utils import make_grid
from metro.utils.miscellaneous import mkdir, set_seed

# file related imports 
from metro.utils.image_ops import crop 
import metro.modeling.data.config as cfg
from metro.modeling._mano import MANO, Mesh
from metro.utils.logger import setup_logger
from metro.modeling.bert import BertConfig, METRO
from metro.modeling.hrnet.hrnet_cls_net import get_cls_net
from metro.modeling.hrnet.config import config as hrnet_config
from metro.utils.geometric_layers import orthographic_projection
from metro.modeling.bert import METRO_Hand_Network as METRO_Network
from metro.modeling.hrnet.config import update_config as hrnet_update_config
from metro.utils.renderer import Renderer, visualize_reconstruction, visualize_reconstruction_test, visualize_reconstruction_no_text, visualize_reconstruction_and_att_local




transform = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])

transform_visualize = transforms.Compose([           
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor()])

def read_gt_bbox(args):

    if "DexYCB" in args.image_file_or_path or "HO3D" in args.image_file_or_path:
        bb_path = os.path.join(os.path.dirname(args.image_file_or_path), "pymafx_out", "gt_bbox.npz")
    else:
        bb_path = os.path.join(os.path.dirname(args.image_file_or_path), "pymafx_out", "gt_bbox.npz")
  
    try:
        gt_dict = np.load(bb_path, allow_pickle=True)  
    except:
        gt_dict = None
        print("No gt bbox found for PYMAFX, using mmpose bbox instead") 
 
    return gt_dict

def run_inference(args, image_list, _metro_network, mano, renderer, mesh_sampler):
    # switch to evaluate mode
    _metro_network.eval()
    
    focal_length = 1000
    res = 224
    
    # make dir for cropped images
    crop_img_folder = os.path.join(os.path.dirname(image_list[0]), "..", "metro_out", "rgb_crop")
    os.makedirs(crop_img_folder, exist_ok=True)
    
        
    gt_bboxes = read_gt_bbox(args)
    
    output_dict = {
                "normalized_vertices": [], 
                "vertices": [],
                "scales": [],
                "expanded_bbox_hw": [], 
                "joints": [], 
                "pred_cam": [],
                "crop_cam_t": [],
                "cam_t": [],
                "cam_R": [], 
                "bbox_center": [],
                    }
    
    exception_frame_id = []
    
    # take gts only 
    for idx, image_file in enumerate(np.array(image_list)[gt_bboxes["frame_id"]]):
        
        temp_fname = image_file[:-4] + '_metro_pred.jpg'
        temp_fname = temp_fname.replace('/rgb/', '/rgb_metro/')
        
        # continue
        
        att_all = []
        img = Image.open(image_file)
        img_height, img_width, _ = np.asarray(img).shape 

        # if gt_bboxes["bbox"][idx] is not None:
        #     if gt_bboxes["bbox"][idx][0] != gt_bboxes["bbox"][idx][2] and gt_bboxes["bbox"][idx][1] != gt_bboxes["bbox"][idx][3]:
        #         gt_bbox = expand_bbox(gt_bboxes["bbox"][idx], img_height=img_height, img_width=img_width, EXPAND_COEF=3.0)
    
        gt_bbox = gt_bboxes["bbox"][idx]
    
        if gt_bbox is None:
            exception_frame_id.append(idx) 
            continue
        
        x1, y1, x2, y2 = gt_bbox[:4]
        
        if x1 == x2 or y1 == y2:
            exception_frame_id.append(idx) 
            continue
          
        start_point_x, start_point_y = min(x1, x2), min(y1, y2)
        end_point_x, end_point_y = max(x1, x2), max(y1, y2)

        expanded_width, expanded_height = end_point_x - start_point_x, end_point_y - start_point_y
        
        kw, kh =  abs(x2-x1)/200, abs(y2-y1)/200
        k_scale = 1.2 * max(kw, kh)
        output_dict["scales"].append(k_scale)
        # k_width, k_height = res/expanded_width, res/expanded_height
       
        bbox_center = [(start_point_x + end_point_x) / 2, (start_point_y + end_point_y) / 2]

        # bbox cropping and scaling

        try: 
            resulting_img = rgb_processing(img_res=224, rgb_img=np.asarray(img), center=bbox_center, scale=k_scale, rot=0, pn=[0, 0, 0])
        except:
            import ipdb; ipdb.set_trace()
        
        cv2.imwrite(f"mm/{idx}.png", resulting_img.transpose(1, 2, 0))
        resulting_img = Image.fromarray(resulting_img.transpose(1, 2, 0).astype(np.uint8))
        
     
        crop_flag = False

        if crop_flag:
            img_cropped_bbox = Image.fromarray(np.array(img)[start_point_y:end_point_y, start_point_x:end_point_x, :])
            img_tensor = transform(img_cropped_bbox)
            img_visual = transform_visualize(img_cropped_bbox)
            
            # save cropped image to visualize
            crop_image_filename = os.path.join(crop_img_folder, os.path.basename(image_file))
            Image.fromarray((img_visual * 255).numpy().transpose(1,2,0).astype(np.uint8)).save(crop_image_filename)
                  
            
        else:
            img_tensor = transform(resulting_img)
            img_visual = transform_visualize(resulting_img)

        # img_cropped_bbox.save(crop_image_filename)
        _, crop_img_width, crop_img_height = img_tensor.shape  

        batch_imgs = torch.unsqueeze(img_tensor, 0).cuda()
        batch_visual_imgs = torch.unsqueeze(img_visual, 0).cuda()
        pred_camera, pred_3d_joints, pred_vertices_sub, pred_vertices, hidden_states, att = _metro_network(batch_imgs, mano, mesh_sampler)     
            
            
        # obtain 3d joints from full mesh
        pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
        pred_3d_pelvis = pred_3d_joints_from_mesh[:,cfg.J_NAME.index('Wrist'),:]
        pred_3d_joints_from_mesh = pred_3d_joints_from_mesh - pred_3d_pelvis[:, None, :]
        pred_vertices = pred_vertices - pred_3d_pelvis[:, None, :]
        
        # save attantion
        att_max_value = att[-1]
        att_cpu = np.asarray(att_max_value.cpu().detach())
        att_all.append(att_cpu)
        
        # orig_camera = convert_crop_cam_to_orig_img(pred_camera[None, ...].cpu().detach(), np.array([[x1, y1, bbox_hw_expanded, bbox_hw_expanded]]), img_width, img_height)
        # pred_camera = orig_camera
        
        # obtain 3d joints, which are regressed from the full mesh
        pred_3d_joints_from_mesh = mano.get_3d_joints(pred_vertices)
        # obtain 2d joints, which are proj)ected from 3d joints of mesh
        pred_2d_joints_from_mesh = orthographic_projection(pred_3d_joints_from_mesh.contiguous(), pred_camera.contiguous())
        pred_2d_coarse_vertices_from_mesh = orthographic_projection(pred_vertices_sub.contiguous(), pred_camera.contiguous())


        # visual_imgs = visualize_mesh_no_text(renderer,
        #                     batch_visual_imgs[0],
        #                     pred_vertices[0].detach(),  
        #                     pred_camera.detach())
        
        # cv2.imwrite(os.path.join(os.getcwd(), temp_fname), np.array(visual_imgs*255).astype(np.uint8).transpose(1,2,0))
        print("Saved to {}".format(temp_fname))
        
        pred_camera_array = pred_camera.cpu().detach().numpy()
        pred_camera_t = np.array([pred_camera_array[1], pred_camera_array[2], 2*focal_length/(res * pred_camera_array[0] +1e-9)])
        
        # visual_imgs_att = visualize_mesh_and_attention(renderer, batch_visual_imgs[0],
        #                                             pred_vertices[0].detach(), 
        #                                             pred_vertices_sub[0].detach(), 
        #                                             pred_2d_coarse_vertices_from_mesh[0].detach(),
        #                                             pred_2d_joints_from_mesh[0].detach(),
        #                                             pred_camera.detach(),
        #                                             att[-1][0].detach())
        # visual_imgs = np.asarray(visual_imgs_att.transpose(1,2,0))
        
        output_dict["normalized_vertices"].append(pred_vertices.cpu().detach().numpy())
        # output_dict["expand_coef"].append([k_width, k_height])
        output_dict["expanded_bbox_hw"].append([expanded_width, expanded_height])
        
        output_dict["joints"].append(pred_3d_joints_from_mesh.cpu().detach().numpy())
        output_dict["pred_cam"].append(pred_camera.cpu().detach().numpy())
        output_dict["crop_cam_t"].append(pred_camera_t)
        output_dict["bbox_center"].append(bbox_center)
        output_dict['cam_R'].append(np.eye(3))
        
        # cv2.imwrite(os.path.join(os.getcwd(), temp_fname), np.array(visual_imgs[:,:,::-1]*255))

    # make video from images
    # video_outname = os.path.join(os.path.dirname(temp_fname), "..", "rgb_metro.mp4")
    
    # # convert images to video using ffmpeg with framerate 30 
    # pred_images = os.path.join(os.path.dirname(temp_fname), "%04d_metro_pred.jpg")
    # os.system(f"/usr/bin/ffmpeg -y -framerate 30 -i {pred_images} -pattern_type glob -c:v libx264 -pix_fmt yuv420p {video_outname}")

    # output_dict["cropped_cam_t"] = np.array(output_dict["cam_t"])
    one_to_T = np.arange(gt_bboxes["frame_id"].shape[0])
    valid_inds = np.setdiff1d(one_to_T, np.array(exception_frame_id), assume_unique=True)

    output_dict["frame_id"] = valid_inds
    output_dict["render_gt_only"] = True
    output_dict["cam_R"] = np.array(output_dict["cam_R"])
    output_dict["joints"] = np.array(output_dict["joints"])
    output_dict["pred_cam"] = np.array(output_dict["pred_cam"])
    output_dict['cam_f'] = np.array([[focal_length, focal_length]])
    output_dict["bbox_center"] = np.array(output_dict["bbox_center"])
    output_dict["scales"] = np.array(output_dict["scales"])
    output_dict['cam_center'] = np.array([[img_width//2, img_height//2]])
    output_dict["normalized_vertices"] = np.array(output_dict["normalized_vertices"])
    output_dict['crop_cam_center'] = np.array([[crop_img_height//2, crop_img_width//2]])
    output_dict['crop_cam_t'] = np.array(output_dict["crop_cam_t"])
    output_dict["expanded_bbox_hw"] = np.array(output_dict["expanded_bbox_hw"])

    output_dict["cam_t"] = convert_to_full_img_cam(pare_cam=output_dict["pred_cam"],
                            # bbox_height=np.maximum(output_dict["expanded_bbox_hw"][:,0], output_dict["expanded_bbox_hw"][:,1]),
                            bbox_height= 200 * output_dict["scales"],
                            bbox_center=output_dict["bbox_center"],
                            img_w=img_width, 
                            img_h=img_height, 
                            focal_length=focal_length)
    
    output_dict["vertices"] = output_dict["normalized_vertices"].squeeze(1) + output_dict['cam_t'][:, None, :]
 
    joblib.dump(output_dict, os.path.join(os.path.dirname(temp_fname.replace("rgb_metro", "metro_out")), "output.pkl"))
      
    return 

def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''

    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return torch.tensor(orig_cam)


def convert_to_full_img_cam(
        pare_cam, bbox_height, bbox_center,
        img_w, img_h, focal_length):
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    r = bbox_height / res
    tz = 2 * focal_length / (r * res * s)

    cx = 2 * (bbox_center[:, 0] - (img_w / 2.)) / (s * bbox_height)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.)) / (s * bbox_height)

    if torch.is_tensor(pare_cam):
        cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    else:
        cam_t = np.stack([tx + cx, ty + cy, tz], axis=-1)

    return cam_t


def expand_bbox(_bbox_, img_width, img_height, EXPAND_COEF=1.2):
        # expand the bbox by a factor of coef
        center = ((_bbox_[0]+_bbox_[2])/2, (_bbox_[1]+_bbox_[3])/2)

        try: 

            assert (0 <= center[0] <= img_width) and (0 <= center[1] <= img_height), "Center is out of image"
        except:
            import ipdb; ipdb.set_trace()
        
        width = abs(_bbox_[2] - _bbox_[0])
        height = abs(_bbox_[3] - _bbox_[1])

        new_width = width * EXPAND_COEF
        new_height = height * EXPAND_COEF
        
        x1 = int(max(0.0, center[0] - new_width/2))
        x2 = int(min(img_width, center[0] + new_width/2))
        y1 = int(max(0.0, center[1] - new_height/2))
        y2 = int(min(img_height, center[1] + new_height/2)) 
        
        return [x1, y1, x2, y2]

def rgb_processing(img_res, rgb_img, center, scale, rot, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [img_res, img_res], rot=rot)
        
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1)) 
        
        return rgb_img


def visualize_mesh_and_attention( renderer, images,
                    pred_vertices_full,
                    pred_vertices, 
                    pred_2d_vertices,
                    pred_2d_joints,
                    pred_camera,
                    attention):

    """Tensorboard logging."""
    
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices_full = pred_vertices_full.cpu().numpy() 
    vertices = pred_vertices.cpu().numpy()
    vertices_2d = pred_2d_vertices.cpu().numpy()
    joints_2d = pred_2d_joints.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    att = attention.cpu().numpy()
    # Visualize reconstruction and attention
    rend_img = visualize_reconstruction_and_att_local(img, 224, vertices_full, vertices, vertices_2d, cam, renderer, joints_2d, att, color='pink')
    rend_img = rend_img.transpose(2,0,1)

    return rend_img


def visualize_mesh_no_text( renderer,
                    images,
                    pred_vertices, 
                    pred_camera):
    """Tensorboard logging."""
    img = images.cpu().numpy().transpose(1,2,0)
    # Get predict vertices for the particular example
    vertices = pred_vertices.cpu().numpy()
    cam = pred_camera.cpu().numpy()
    # Visualize reconstruction only
    rend_img = visualize_reconstruction_no_text(img, 224, vertices, cam, renderer, color='hand')
    rend_img = rend_img.transpose(2,0,1)
    return rend_img


def parse_args():
    parser = argparse.ArgumentParser()
    #########################################################
    # Data related arguments
    #########################################################
    parser.add_argument("--image_file_or_path", default='./test_images/hand', type=str, 
                        help="test data")
    #########################################################
    # Loading/saving checkpoints
    #########################################################
    parser.add_argument("--model_name_or_path", default='metro/modeling/bert/bert-base-uncased/', type=str, required=False,
                        help="Path to pre-trained transformer model or model type.")
    parser.add_argument("--resume_checkpoint", default=None, type=str, required=False,
                        help="Path to specific checkpoint for inference.")
    parser.add_argument("--output_dir", default='output/', type=str, required=False,
                        help="The output directory to save checkpoint and test results.")
    #########################################################
    # Model architectures
    #########################################################
    parser.add_argument('-a', '--arch', default='hrnet-w64',
                    help='CNN backbone architecture: hrnet-w64, hrnet, resnet50')
    parser.add_argument("--num_hidden_layers", default=4, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--hidden_size", default=-1, type=int, required=False, 
                        help="Update model config if given")
    parser.add_argument("--num_attention_heads", default=4, type=int, required=False, 
                        help="Update model config if given. Note that the division of "
                        "hidden_size / num_attention_heads should be in integer.")
    parser.add_argument("--intermediate_size", default=-1, type=int, required=False, 
                        help="Update model config if given.")
    parser.add_argument("--input_feat_dim", default='2051,512,128', type=str, 
                        help="The Image Feature Dimension.")          
    parser.add_argument("--hidden_feat_dim", default='1024,256,64', type=str, 
                        help="The Image Feature Dimension.")   
    #########################################################
    # Others
    #########################################################
    parser.add_argument("--device", type=str, default='cuda', 
                        help="cuda or cpu")
    parser.add_argument('--seed', type=int, default=88, 
                        help="random seed for initialization.")


    args = parser.parse_args()
    return args


def main(args):
    global logger
    # Setup CUDA, GPU & distributed training
    args.num_gpus = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = args.num_gpus > 1
    args.device = torch.device(args.device)
    
    # create rgb_metro folder
    os.makedirs(os.path.join(os.path.dirname(args.image_file_or_path), "rgb_metro"), exist_ok=True)    

    mkdir(args.output_dir)
    logger = setup_logger("METRO Inference", args.output_dir, 0)
    set_seed(args.seed, args.num_gpus)
    logger.info("Using {} GPUs".format(args.num_gpus))

    # Mesh and MANO utils
    mano_model = MANO().to(args.device)
    mano_model.layer = mano_model.layer.cuda()
    mano_model.layer.flat_hand_mean = True
    mesh_sampler = Mesh()
    # Renderer for visualization
    renderer = Renderer(faces=mano_model.face)


    # Load pretrained model    
    logger.info("Inference: Loading from checkpoint {}".format(args.resume_checkpoint))

    if args.resume_checkpoint!=None and args.resume_checkpoint!='None' and 'state_dict' not in args.resume_checkpoint:
        logger.info("Evaluation: Loading from checkpoint {}".format(args.resume_checkpoint))
        _metro_network = torch.load(args.resume_checkpoint)
    else:
        # Build model from scratch, and load weights from state_dict.bin
        trans_encoder = []
        input_feat_dim = [int(item) for item in args.input_feat_dim.split(',')]
        hidden_feat_dim = [int(item) for item in args.hidden_feat_dim.split(',')]
        output_feat_dim = input_feat_dim[1:] + [3]
        # init three transformer encoders in a loop
        for i in range(len(output_feat_dim)):
            config_class, model_class = BertConfig, METRO
            config = config_class.from_pretrained(args.model_name_or_path)

            config.output_attentions = False
            config.img_feature_dim = input_feat_dim[i] 
            config.output_feature_dim = output_feat_dim[i]
            args.hidden_size = hidden_feat_dim[i]
            args.intermediate_size = int(args.hidden_size*4)

            # update model structure if specified in arguments
            update_params = ['num_hidden_layers', 'hidden_size', 'num_attention_heads', 'intermediate_size']

            for idx, param in enumerate(update_params):
                arg_param = getattr(args, param)
                config_param = getattr(config, param)
                if arg_param > 0 and arg_param != config_param:
                    logger.info("Update config parameter {}: {} -> {}".format(param, config_param, arg_param))
                    setattr(config, param, arg_param)

            # init a transformer encoder and append it to a list
            assert config.hidden_size % config.num_attention_heads == 0
            model = model_class(config=config) 
            logger.info("Init model from scratch.")
            trans_encoder.append(model)

        # init ImageNet pre-trained backbone model
        if args.arch=='hrnet':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w40_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w40_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w40 model')
        elif args.arch=='hrnet-w64':
            hrnet_yaml = 'models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml'
            hrnet_checkpoint = 'models/hrnet/hrnetv2_w64_imagenet_pretrained.pth'
            hrnet_update_config(hrnet_config, hrnet_yaml)
            backbone = get_cls_net(hrnet_config, pretrained=hrnet_checkpoint)
            logger.info('=> loading hrnet-v2-w64 model')
        else:
            print("=> using pre-trained model '{}'".format(args.arch))
            backbone = models.__dict__[args.arch](pretrained=True)
            # remove the last fc layer
            backbone = torch.nn.Sequential(*list(backbone.children())[:-2])

        trans_encoder = torch.nn.Sequential(*trans_encoder)
        total_params = sum(p.numel() for p in trans_encoder.parameters())
        logger.info('Transformers total parameters: {}'.format(total_params))
        backbone_total_params = sum(p.numel() for p in backbone.parameters())
        logger.info('Backbone total parameters: {}'.format(backbone_total_params))

        # build end-to-end METRO network (CNN backbone + multi-layer transformer encoder)
        _metro_network = METRO_Network(args, config, backbone, trans_encoder)

        logger.info("Loading state dict from checkpoint {}".format(args.resume_checkpoint))
        cpu_device = torch.device('cpu')
        state_dict = torch.load(args.resume_checkpoint, map_location=cpu_device)
        _metro_network.load_state_dict(state_dict, strict=False)
        del state_dict

    # update configs to enable attention outputs
    setattr(_metro_network.trans_encoder[-1].config,'output_attentions', True)
    setattr(_metro_network.trans_encoder[-1].config,'output_hidden_states', True)
    _metro_network.trans_encoder[-1].bert.encoder.output_attentions = True
    _metro_network.trans_encoder[-1].bert.encoder.output_hidden_states =  True
    for iter_layer in range(4):
        _metro_network.trans_encoder[-1].bert.encoder.layer[iter_layer].attention.self.output_attentions = True
    for inter_block in range(3):
        setattr(_metro_network.trans_encoder[-1].config,'device', args.device)

    _metro_network.to(args.device)
    logger.info("Run inference")

    image_list = []
    if not args.image_file_or_path:
        raise ValueError("image_file_or_path not specified")
    if op.isfile(args.image_file_or_path):
        image_list = [args.image_file_or_path]
    elif op.isdir(args.image_file_or_path):
        # should be a path with images only
        for filename in os.listdir(args.image_file_or_path):
            if filename.endswith(".png") or filename.endswith(".jpg") and 'pred' not in filename:
                image_list.append(args.image_file_or_path+'/'+filename) 
    else:
        raise ValueError("Cannot find images at {}".format(args.image_file_or_path))

    image_list = sorted(image_list)
    run_inference(args, image_list, _metro_network, mano_model, renderer, mesh_sampler)    

if __name__ == "__main__":
    args = parse_args()
    main(args)
