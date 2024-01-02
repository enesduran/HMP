# This code snippet is entirely taken from https://github.com/shreyashampali/ho3d
from __future__ import print_function, unicode_literals
import os 
import cv2
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm


""" General util functions. """
def _assert_exist(p):
    msg = 'File does not exists: %s' % p
    assert os.path.exists(p), msg


def db_size(set_name, version='v2'):
    """ Hardcoded size of the datasets. """
    if set_name == 'training':
        if version == 'v2':
            return 66034  # number of unique samples (they exists in multiple 'versions')
        elif version == 'v3':
            return 78297
        else:
            raise NotImplementedError
    elif set_name == 'evaluation':
        if version == 'v2':
            return 11524
        elif version == 'v3':
            return 20137
        else:
            raise NotImplementedError
    else:
        assert 0, 'Invalid choice.'


def read_RGB_img(base_dir, seq_name, file_id, split):
    """Read the RGB image in dataset"""
    if os.path.exists(os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')):
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.png')
    else:
        img_filename = os.path.join(base_dir, split, seq_name, 'rgb', file_id + '.jpg')

    _assert_exist(img_filename)

    img = cv2.imread(img_filename)

    return img


def load_pickle_data(f_name):
    """ Loads the pickle data """
    if not os.path.exists(f_name):
        raise Exception('Unable to find annotations picle file at %s. Aborting.'%(f_name))
    with open(f_name, 'rb') as f:
        try:
            pickle_data = pickle.load(f, encoding='latin1')
        except:
            pickle_data = pickle.load(f)

    return pickle_data


def read_annotation(base_dir, seq_name, file_id, split):
    meta_filename = os.path.join(base_dir, split, seq_name, 'meta', file_id + '.pkl')

    _assert_exist(meta_filename)

    pkl_data = load_pickle_data(meta_filename)

    return pkl_data

def main(base_path, pred_out_path, pred_func, version, set_name=None):
    """
        Main eval loop: Iterates over all evaluation samples and saves the corresponding predictions.
    """
    # default value
    if set_name is None:
        set_name = 'evaluation'

    # init output containers
    xyz_pred_list, verts_pred_list = list(), list()


    # read list of evaluation files
    with open(os.path.join(base_path, set_name+'.txt')) as f:
        file_list = f.readlines()
    file_list = [f.strip() for f in file_list]

    assert len(file_list) == db_size(set_name, version), '%s.txt is not accurate. Aborting'%set_name

    # iterate over the dataset once
    for idx in tqdm(range(db_size(set_name, version))):
        if idx >= db_size(set_name, version):
            break

        seq_name = file_list[idx].split('/')[0]
        file_id = file_list[idx].split('/')[1]

        # load input image
        img = read_RGB_img(base_path, seq_name, file_id, set_name)
        aux_info = read_annotation(base_path, seq_name, file_id, set_name)
        

        # use some algorithm for prediction
        xyz, verts = pred_func(
            img,
            aux_info,
            rgb_img_info = (seq_name, file_id),
        )

        # simple check if xyz and verts are in opengl coordinate system
        if np.all(xyz[:,2]>0) or np.all(verts[:,2]>0):
            raise Exception('It appears the pose estimates are not in OpenGL coordinate system. Please read README.txt in dataset folder. Aborting!')

        xyz_pred_list.append(xyz)
        verts_pred_list.append(verts)

    # dump results
    dump(pred_out_path, xyz_pred_list, verts_pred_list)


def dump(pred_out_path, xyz_pred_list, verts_pred_list):
    """ Save predictions into a json file. """
    # make sure its only lists
    xyz_pred_list = [x.tolist() for x in xyz_pred_list]
    verts_pred_list = [x.tolist() for x in verts_pred_list]

    # save to a json
    with open(pred_out_path, 'w') as fo:
        json.dump([xyz_pred_list, verts_pred_list], fo)
    print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))

    pred_out_path_zip = pred_out_path.replace(".json", ".zip")

    exec(f"zip -j ./../../{pred_out_path_zip} {pred_out_path}")


def pred_template(img, aux_info, rgb_img_info):
    """ Predict joints and vertices from a given sample.
        img: (640, 480, 3) RGB image.
        aux_info: dictionary containing hand bounding box, camera matrix and root joint 3D location
    
        TODO: Put your algorithm here, which computes (metric) 3D joint coordinates and 3D vertex positions
        xyz = np.zeros((21, 3))  # 3D coordinates of the 21 joints
        verts = np.zeros((778, 3)) # 3D coordinates of the shape vertices
    
    """


    wrist_loc = aux_info['handJoints3D']
    delta_xyz = np.zeros((21, 3))  # 3D coordinates of the 21 joints
    timestep = int(rgb_img_info[1])

    if args.method == "pymafx":
        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        cfg_name = "_pymafx_raw"
    elif args.method == "metro":
        coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        cfg_name = "_metro_raw"
    else:
        coord_change_mat = np.array([[1., 0., 0.], [0, 1., 0.], [0., 0., 1.]], dtype=np.float32)
        cfg_name = "stage2_ts_0_os_1_trans_1_orient_2_rot_0_beta_10_mp_300_pp_0_js_0_reproj_0.05_lr_0.0501_400_blend_std"
        # cfg_name = "stage2_ts_3_os_1_trans_1_orient_2_rot_0_beta_10_mp_200_pp_0.1_js_0_reproj_0.03_lr_0.0501_400_blend"
    
    pred_filepath = os.path.join("./optim", cfg_name, f"HO3D_v3/evaluation/{rgb_img_info[0]}/recon_000_30fps.npz")
    pred_dict = dict(np.load(pred_filepath, allow_pickle=True))

    jts3d = pred_dict["joints_3d"][timestep] @ coord_change_mat
    verts = pred_dict["vertices"][timestep] @ coord_change_mat

    # now align according to ground truth data 
    delta_xyz = wrist_loc - jts3d[0]

    xyz = jts3d + delta_xyz
    verts = verts + delta_xyz
   
    return xyz, verts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Show some samples from the dataset.')
    parser.add_argument('base_path', type=str,
                        help='Path to where the HO3D dataset is located.')
    parser.add_argument('--out', type=str, default='pred.json',
                        help='File to save the predictions.')
    parser.add_argument('--version', type=str, choices=['v2', 'v3'],
                        help='version number')
    parser.add_argument('--method', type=str, default="hmp", choices=['hmp', 'pymafx', 'metro'], help='which method to use')

    args = parser.parse_args()

    # call with a predictor function
    main(
        args.base_path,
        args.out,
        pred_func=pred_template,
        set_name='evaluation',
        version=args.version
    )


# python src/scripts/ho3d_eval_server.py ./data/rgb_data/HO3D_v3 --out ho3d_preds/metro/pred.json  --version v3  --method metro
# zip -j pred.zip pred.json