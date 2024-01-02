import os 

FPS = 30

AMASS_RAW_PATH = "./data/amass_raw/smplx_neutral"
AMASS_PROCESSED_PATH = "./data/amass_processed"
INTERHANDS_RAW_PATH = "./data/interhands_raw/Interhands_30fps"
INTERHANDS_PROCESSED_PATH = "./data/interhands_processed"
ROLLOUT_PROCESSED_PATH = "./data/rollout_processed"

ALL_AMASS_DATASETS = ["TCDHands", "SAMP", "GRAB"]  # Everything with hand articulation.
ARCTIC_DATASET = ["ARCTIC"]

AMASS_TRAIN_DATASETS =  ["GRAB"]
AMASS_TEST_DATASETS = ["SAMP"] 
AMASS_VAL_DATASETS = ["TCDHands"] 

ALL_INTERHANDS_DATASETS = ["ih_train", "ih_test", "ih_val"]
INTERHANDS_TRAIN_DATASETS = []
INTERHANDS_TEST_DATASETS = ["ih_test"]
INTERHANDS_VAL_DATASETS = ["ih_val"]

TRAIN_DATASETS = AMASS_TRAIN_DATASETS + INTERHANDS_TRAIN_DATASETS + ARCTIC_DATASET
TEST_DATASETS = AMASS_TEST_DATASETS + INTERHANDS_TEST_DATASETS
VAL_DATASETS = AMASS_VAL_DATASETS + INTERHANDS_VAL_DATASETS

ALL_DATASETS = TRAIN_DATASETS + TEST_DATASETS + VAL_DATASETS


def find_dataset_path(dataset_name, split):
    
    if split == "custom":
        return os.path.join(ROLLOUT_PROCESSED_PATH, dataset_name)
    else:
        if dataset_name in ALL_AMASS_DATASETS:
            return os.path.join(AMASS_PROCESSED_PATH, dataset_name)
        elif dataset_name in ALL_INTERHANDS_DATASETS:
            return os.path.join(INTERHANDS_PROCESSED_PATH, dataset_name)
        else:
            return None

ROT_REPS = ['mat', 'aa', '6d']
ROT_REP_SIZE = {'aa' : 3, '6d' : 6, 'mat' : 9, '9d' : 9}

# FLAT_HAND_DICT = {"SAMP":False, "GRAB":False, "TCDHands" : True}

SPLITS = ['train', 'val', 'test', 'custom']
SPLIT_BY = [ 
             'single',   # the data path is a single .npz file. Don't split: train and test are same
             'sequence', # the data paths are directories of subjects. Collate and split by sequence.
             'subject',  # the data paths are directories of datasets. Collate and split by subject.
             'dataset'   # a single data path to the amass data root is given. The predefined datasets will be used for each split.
            ]

DATA_NAMES = ['pose_body', 'pose_body_vel', 'body_joints', 'body_joints_vel', 'trans', 'body_trans_vel', 'body_root_orient', 'root_orient_vel',
                 'rh_trans', 'lh_trans', 'r_wrist_orient', 'l_wrist_orient', 'pose_right_hand', 'pose_left_hand', 'pose_right_hand_vel', 'pose_left_hand_vel', 
                'right_hand_joints', 'left_hand_joints', 'right_hand_joint_vel', 'right_hand_vtx', 'right_hand_vtx_vel', 'contacts']


SMPLX_JOINTS = ['L_Wrist', 'R_Wrist', 'lindex0', 'lindex1', 'lindex2', 'lmiddle0', 'lmiddle1', 'lmiddle2', 'lpinky0', 'lpinky1', 'lpinky2', 
                'lring0', 'lring1', 'lring2', 'lthumb0', 'lthumb1', 'lthumb2', 'rindex0', 'rindex1', 'rindex2', 'rmiddle0', 'rmiddle1', 'rmiddle2', 
                'rpinky0', 'rpinky1', 'rpinky2', 'rring0', 'rring1', 'rring2', 'rthumb0', 'rthumb1', 'rthumb2']

# For now lets assume any joint can contact
CONTACT_INDS = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]


MANO_JOINTS_RETURN_CONFIG = {
    'pose_body' : False,
    'pose_body_vel' : False,
    'body_joints' : False,
    'body_joints_vel': False,
    'trans' : False,
    'body_trans_vel' : False,
    'body_root_orient' : False,
    'root_orient_vel' : False,
    'rh_trans' : False,
    'lh_trans' : False,
    'r_wrist_orient' : False,
    'l_wrist_orient' : False,
    'pose_right_hand' : True,
    'pose_left_hand' : False,
    'pose_right_hand_vel' : False,
    'pose_left_hand_vel' : False,
    'right_hand_joints' : True,
    'left_hand_joints' : False,
    'right_hand_joint_vel' : True,
    'right_hand_vtx' : False,
    'right_hand_vtx_vel' : False,
    'contacts' : False
}

MANO_JOINTS_CONTACTS_RETURN_CONFIG = {
    'pose_body' : False,
    'pose_body_vel' : False,
    'body_joints' : False,
    'body_joints_vel': False,
    'trans' : False,
    'body_trans_vel' : False,
    'body_root_orient' : False,
    'root_orient_vel' : False,
    'rh_trans' : True,
    'lh_trans' : False,
    'r_wrist_orient' : True,
    'l_wrist_orient' : False,
    'pose_right_hand' : True,
    'pose_left_hand' : False,
    'pose_right_hand_vel' : False,
    'pose_left_hand_vel' : False,
    'right_hand_joints' : True,
    'left_hand_joints' : False,
    'right_hand_joint_vel' : True,
    'right_hand_vtx' : False,
    'right_hand_vtx_vel' : False,
    'contacts' : False
}

ALL_RETURN_CONFIG = {
    'pose_body' : False,
    'pose_body_vel' : False,
    'body_joints' : False,
    'body_joints_vel': False,
    'trans' : False,
    'body_trans_vel' : False,
    'body_root_orient' : False,
    'root_orient_vel' : False,
    'rh_trans' : True,
    'lh_trans' : False,
    'r_wrist_orient' : True,
    'l_wrist_orient' : False,    
    'pose_right_hand' : True,
    'pose_left_hand' : False,
    'pose_right_hand_vel' : False,
    'pose_left_hand_vel' : False,
    'right_hand_joints' : True,
    'left_hand_joints' : False,
    'right_hand_joint_vel' : True,
    'right_hand_vtx' : True,
    'right_hand_vtx_vel' : True,
    'contacts' : False                  # change it to True after development 
}

RETURN_CONFIGS = {
                  'mano+joints+contacts' : MANO_JOINTS_CONTACTS_RETURN_CONFIG,
                  'mano+joints' : MANO_JOINTS_RETURN_CONFIG,
                  'all' : ALL_RETURN_CONFIG
                  }


def data_name_list(return_config):
    '''
    returns the list of data values in the given configuration
    '''
    assert DATA_NAMES == list(RETURN_CONFIGS[return_config].keys()), "Missing data names!!!"
    cur_ret_cfg = RETURN_CONFIGS[return_config]
    data_names = [k for k in DATA_NAMES if cur_ret_cfg[k]]
    return data_names


def data_dim(dname, rot_rep_size=9, add_finger_tips=True):
    '''
    returns the dimension of the data with the given name. If the data is a rotation, returns the size with the given representation.
    '''
    import body_utils
    
    NUM_BODY_JOINTS = body_utils.NUM_BODY_JOINTS
    NUM_HAND_JOINTS = (body_utils.NUM_HAND_JOINTS + body_utils.NUM_HAND_TIP) if add_finger_tips else body_utils.NUM_HAND_JOINTS
    NUM_KEYPOINT_VERTS = body_utils.NUM_KEYPOINT_VERTS

    if dname in ['trans', 'rh_trans', 'lh_trans',  'body_trans_vel', 'root_orient_vel']:
        return 3
    elif dname in ['root_orient', 'r_wrist_orient', 'l_wrist_orient']:
        return rot_rep_size    
    elif dname in ['pose_body']:
        return (NUM_BODY_JOINTS-1) * rot_rep_size
    elif dname in ['body_joints', 'body_joints_vel']:
        return NUM_BODY_JOINTS * 3
    elif dname in ['pose_right_hand', 'pose_left_hand']:
        return  (body_utils.NUM_HAND_JOINTS-1) * rot_rep_size
    elif dname in ['pose_right_hand_vel', 'pose_left_hand_vel']:
        return (body_utils.NUM_HAND_JOINTS-1) * 3
    elif dname in ['pose_body_vel']:
        return (NUM_BODY_JOINTS-1) * 3
    elif dname in ['right_hand_joints', 'right_hand_joint_vel', 'left_hand_joints']:
        return NUM_HAND_JOINTS * 3    
    elif dname in ['right_hand_vtx', 'right_hand_vtx_vel']:
        return NUM_KEYPOINT_VERTS * 3
    elif dname in ['contacts']:
        # FURTHER WORK
        return None
    else:
        print('The given data name %s is not valid!' % (dname))
        exit()

