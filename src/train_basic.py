import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

from arguments import Arguments
from nemf.basic import Architecture


def load_data(data_path, index, frames):
    data = {}
    for key in ['pos', 'global_xform', 'root_orient', 'root_vel', 'trans']:
        data[key] = torch.load(os.path.join(data_path, f'{key}_{index}.pt'))
        if frames != -1:
            # data[key] = data[key][350:frames + 350].unsqueeze(0)
            data[key] = data[key][:frames].unsqueeze(0)
        else: 
            data[key] = data[key].unsqueeze(0)

    return data


def train_basic(frames, save_dir, steps):
    data_path = os.path.join(args.dataset_dir, 'train')
    file_indices = list(range(16)) if args.amass_data else [22]

    statistics = dict()
    for index in file_indices:
        print(f'Fitting sequence {index} with {frames} frames:')
        args.save_dir = os.path.join(save_dir, f'frame_{frames}', f'sequence_{index}')
        model = Architecture(args, ngpu)
        model.setup()
        model.print()
        print(f'# of parameters: {model.count_params()}')
    
        model.set_input(load_data(data_path, index, frames))
        start_time = time.time()

        iterations = args.iterations * (frames // 32) if args.amass_data else args.iterations
        if args.is_train:
            model.train()
            loss_min = None
            for iter in range(iterations):
                model.optimize_parameters()

                model.epoch()
                res = model.verbose()

                if args.verbose:
                    print(f'Iteration {iter}/{iterations}:')
                    print(json.dumps(res, sort_keys=True, indent=4))

                if loss_min is None or res['total_loss']['train'] < loss_min['total_loss']['train']:
                    loss_min = res
                    model.save(optimal=True)

            end_time = time.time()
            print(f'Training finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
            print('Final Loss:')
            print(json.dumps(loss_min, sort_keys=True, indent=4))

        model.eval()
        model.load(optimal=True)

        for step in steps:
            model.super_sampling(step)

            if step == 1.0:
                errors = model.report_errors()
                if not statistics:
                    statistics = {
                        'iterations': [iterations],
                        'rotation_error': [errors['rotation'] * 180.0 / np.pi],
                        'position_error': [errors['position'] * 100.0],
                        'orientation_error': [errors['orientation'] * 180.0 / np.pi],
                        'translation_error': [errors['translation'] * 100.0]
                    }
                else:
                    statistics['iterations'].append(iterations)
                    statistics['rotation_error'].append(errors['rotation'] * 180.0 / np.pi)
                    statistics['position_error'].append(errors['position'] * 100.0)
                    statistics['orientation_error'].append(errors['orientation'] * 180.0 / np.pi)
                    statistics['translation_error'].append(errors['translation'] * 100.0)

    if statistics:
        df = pd.DataFrame.from_dict(statistics)
        df.to_csv(os.path.join(save_dir, f'recon_{frames}.csv'), index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        args = Arguments('./configs', filename='basic.yaml')
    else:
        args = Arguments('./configs', filename=sys.argv[1])
    print(json.dumps(args.json, sort_keys=True, indent=4))

    torch.set_default_dtype(torch.float32)

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    ngpu = 1
    if args.multi_gpu is True:
        ngpu = torch.cuda.device_count()
        if ngpu == 1:
            args.multi_gpu = False
    print(f'Number of GPUs: {ngpu}')

    save_dir = args.save_dir
    frames = [32, 64, 128, 256, 512] if args.amass_data else [-1]
    steps = [1.0, 0.5, 0.25, 0.125] if args.amass_data else [1.0]
    for f in frames:
        train_basic(frames=f, save_dir=save_dir, steps=steps)
