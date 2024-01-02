import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets.amass import AMASS
from arguments import Arguments
from nemf.generative import Architecture


def train():
    model = Architecture(args, ngpu, len(train_data_loader))
    model.print()
    model.setup()

    loss_min = None
    if args.epoch_begin != 0:
        model.load(epoch=args.epoch_begin)
        model.eval()
        for data in valid_data_loader:
            model.set_input(data)
            model.validate()
        loss_min = model.verbose()

    epoch_begin = args.epoch_begin + 1
    epoch_end = epoch_begin + args.epoch_num
    start_time = time.time()

    for epoch in range(epoch_begin, epoch_end):
        model.train()
        with tqdm(train_data_loader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                model.set_input(data)
                model.optimize_parameters()

        model.eval()
        for data in valid_data_loader:
            model.set_input(data)
            model.validate()

        model.epoch()
        res = model.verbose()

        if args.verbose:
            print(f'Epoch {epoch}/{epoch_end - 1}:')
            print(json.dumps(res, sort_keys=True, indent=4))

        if loss_min is None or res['total_loss']['val'] < loss_min['total_loss']['val']:
            loss_min = res
            model.save(optimal=True)

        if epoch % args.checkpoint == 0 or epoch == epoch_end - 1:
            model.save()

    end_time = time.time()
    print(f'Training finished in {time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))}')
    print('Final Loss:')
    print(json.dumps(loss_min, sort_keys=True, indent=4))
    df = pd.DataFrame.from_dict(loss_min)
    df.to_csv(os.path.join(args.save_dir, f'{args.filename}.csv'), index=False)


def test_decoder():
    model = Architecture(args, ngpu, len(test_data_loader))
    model.load(optimal=True)
    model.eval()
    model.test_index = 0
    for data in test_data_loader:
        model.set_input(data)
        z_local, _, _ = model.encode_local()
        z_global, _, _ = model.encode_global()
        decoder_out = model.decode(z_l=z_local, z_g=z_global, length=args.data.clip_length)
        model.recon_data = decoder_out
        model.compute_test_result()
    return     

def test(steps):
    model = Architecture(args, ngpu, len(test_data_loader))
    model.load(optimal=True)
    model.eval()

    for step in steps:
        statistics = dict()
        model.test_index = 0
        for data in test_data_loader:
            model.set_input(data)
            
            model.super_sampling(step=step)
        
            if step == 1.0:
                errors = model.report_errors()
                if not statistics:
                    statistics = {
                        'rotation_error': [errors['rotation'] * 180.0 / np.pi],
                        'position_error': [errors['position'] * 100.0],
                        'orientation_error': [errors['orientation'] * 180.0 / np.pi],
                        'translation_error': [errors['translation'] * 100.0]
                    }
                else:
                    statistics['rotation_error'].append(errors['rotation'] * 180.0 / np.pi)
                    statistics['position_error'].append(errors['position'] * 100.0)
                    statistics['orientation_error'].append(errors['orientation'] * 180.0 / np.pi)
                    statistics['translation_error'].append(errors['translation'] * 100.0)

        if step == 1.0:
            df = pd.DataFrame.from_dict(statistics)
            df.to_csv(os.path.join(args.save_dir, f'{args.filename}_test.csv'), index=False)


if __name__ == '__main__':
    if len(sys.argv) == 1:
        args = Arguments('./configs', filename='generative.yaml')
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

    # dataset definition
    train_dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'train'))
    train_data_loader = DataLoader(train_dataset, batch_size=ngpu * args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'valid'))
    valid_data_loader = DataLoader(valid_dataset, batch_size=ngpu * args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)
    test_dataset = AMASS(dataset_dir=os.path.join(args.dataset_dir, 'test'))
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=True)

    if args.is_train:
        train()
        
    args.is_train = False
    test_decoder()
    # test(steps=[0.5, 1.0])
