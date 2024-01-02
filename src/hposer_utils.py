import os
import torch
import glob
import importlib
from configer import Configer

def get_device(gpu_idx=0):
    '''
    Returns the pytorch device for the given gpu index.
    '''
    gpu_device_str = 'cuda:%d' % (gpu_idx)
    device_str = gpu_device_str if torch.cuda.is_available() else 'cpu'
    if device_str == gpu_device_str:
        print('Using detected GPU...')
        device_str = 'cuda:0'
    else:
        print('No detected GPU...using CPU.')
    device = torch.device(device_str)
    return device


def load_hposer(expr_dir='./data/body_models/hposer', vp_model='snapshot'):
    '''
    :param expr_dir:
    :param vp_model: either 'snapshot' to use the experiment folder's code or a VPoser imported module, e.g.
    from human_body_prior.train.vposer_smpl import VPoser, then pass VPoser to this function
    :param if True will load the model definition used for training, and not the one in current repository
    :return:
    '''

    ps, trained_model_right_fname = expid2model(expr_dir)
    if vp_model == 'snapshot':

        # hposer_path = sorted(glob.glob(os.path.join(expr_dir, 'hposer_mano.py')), key=os.path.getmtime)[-1]
        hposer_path = os.path.join(expr_dir, 'hposer_mano.py')

        spec = importlib.util.spec_from_file_location('HPoser', hposer_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        hposer_right_pt = getattr(module, 'HPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
        hposer_left_pt = getattr(module, 'HPoser')(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)
    else:
        hposer_right_pt = vp_model(num_neurons=ps.num_neurons, latentD=ps.latentD, data_shape=ps.data_shape)

    hposer_right_pt.load_state_dict(torch.load(trained_model_right_fname, map_location=get_device()))
    hposer_right_pt.eval()
    
    # hposer_left_pt.load_state_dict(torch.load(trained_model_left_fname, map_location=get_device()))
    # hposer_left_pt.eval()

    return hposer_right_pt, ps


def expid2model(expr_dir):
    
    if not os.path.exists(expr_dir): raise ValueError('Could not find the experiment directory: %s' % expr_dir)

    best_model_right_fname = os.path.join(expr_dir, 'hposer_right.pt')

    print(('Found Trained Model: %s' % best_model_right_fname))

    default_ps_fname = glob.glob(os.path.join(expr_dir,'*.ini'))[0]
    if not os.path.exists(default_ps_fname): raise ValueError('Could not find the appropriate vposer_settings: %s' % default_ps_fname)
    ps = Configer(default_ps_fname=default_ps_fname, work_dir = expr_dir, best_model_fname=best_model_right_fname)
    return ps, best_model_right_fname


if __name__ == '__main__':

    expr_dir = './data/body_models/hposer'
    hposer_right_pt, ps = load_hposer(expr_dir)
    print(hposer_right_pt)
    print(ps)