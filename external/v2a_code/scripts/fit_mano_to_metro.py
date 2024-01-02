import os
import sys
import copy
import torch
import joblib
import smplx
import numpy as np
from glob import glob
from tqdm import tqdm
from smplx import MANO
from external.v2a_code.scripts.fitting import optimize_mano_shape

MANO_DIR_L = "./data/body_models/mano/MANO_LEFT.pkl"
MANO_DIR_R = "./data/body_models/mano/MANO_RIGHT.pkl"

mano_layers = {"right": MANO(MANO_DIR_R, is_rhand=True, flat_hand_mean=True, use_pca=False),
                "left": MANO(MANO_DIR_L, is_rhand=False, flat_hand_mean=True, use_pca=False)}


def fit_frame(input_vert, save_mesh, iteration, init_params=None):
    # target_v3d = np.load(input_p)
    target_v3d = input_vert
    target_v3d = torch.FloatTensor(target_v3d.reshape(1, -1, 3)).cuda()
    
    vis_dir = f"./data/rgb_data/metro_fit_vis/"

    tip_sem_idx = [12, 11, 4, 5, 6]

    optim_specs = {
        "epoch_coarse": 10000,    # 10000
        "epoch_fine": 10000,       # 10000
        "is_right": True,
        "save_mesh": save_mesh,
        "criterion": torch.nn.MSELoss(reduction="none"),
        "seed": 0,
        "vis_dir": vis_dir,
        "sem_idx": tip_sem_idx,
    }

    # os.makedirs(optim_specs["vis_dir"], exist_ok=True)

    params = optimize_mano_shape(
        target_v3d, mano_layers, optim_specs, iteration, init_params=init_params
    )
    
    return params

def main(dataset):
    seq_name = dataset
    
    pkl_path = os.path.join(os.path.dirname(dataset), "metro_out/output.pkl") 
    mesh_dict = joblib.load(pkl_path)

    fnames = mesh_dict["frame_id"]

    pbar = tqdm(enumerate(fnames))
    prev_out = None
    
    final_results_metro = {"pose": [], "betas": [], "trans": [], "orient": []}
     
    for iteration, input_idx in pbar:
        # pbar.set_description(
        #     "Processing %s [%d/%d]" % (seq_name, iteration + 1, len(fnames)))
        
        real_idx = list(fnames).index(input_idx) 
        vert_i = mesh_dict["vertices"][real_idx]
        
        out = fit_frame(input_vert=vert_i, save_mesh=False, 
                        init_params=prev_out, iteration=iteration)
        prev_out = out
        
        final_results_metro["trans"].append(copy.deepcopy(out["transl"]))
        final_results_metro["orient"].append(copy.deepcopy(out["global_orient"]))
        final_results_metro["pose"].append(copy.deepcopy(out["hand_pose"]))
        final_results_metro["betas"].append(copy.deepcopy(out["betas"]))

    final_results_metro["orient"] = torch.cat(final_results_metro["orient"]).cpu().detach().numpy()
    final_results_metro["pose"] = torch.cat(final_results_metro["pose"]).cpu().detach().numpy()
    final_results_metro["betas"] = torch.cat(final_results_metro["betas"]).cpu().detach().numpy()
    final_results_metro["trans"] = torch.cat(final_results_metro["trans"]).cpu().detach().numpy()
    
    # take the remaining key values from the original pkl file
    for k in set(mesh_dict.keys()).difference(set(final_results_metro.keys())):
        final_results_metro[k] = mesh_dict[k] 

    joblib.dump(final_results_metro, pkl_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="flat_rotate")
    args = parser.parse_args()
    
    main(args.dataset)
