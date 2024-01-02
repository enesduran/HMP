import os

import numpy as np
import yaml


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Arguments:
    def __init__(self, config_path, filename='default.yaml'):
        with open(os.path.join(config_path, 'mano.yaml'), 'r') as f:
            smpl = yaml.safe_load(f)

        self.smpl = Struct(**smpl)
        self.smpl.offsets['right'] = np.array(self.smpl.offsets['right'])
        self.smpl.offsets['left'] = np.array(self.smpl.offsets['left'])
        self.smpl.parents = np.array(self.smpl.parents).astype(np.int32)
        self.smpl.joint_num = 16 # len(self.smpl.joints_to_use)
        self.smpl.joints_to_use = np.array(self.smpl.joints_to_use)
        self.smpl.joints_to_use = np.arange(0, 63).reshape((-1, 3))[self.smpl.joints_to_use].reshape(-1)

        self.filename = os.path.splitext(filename)[0]
        with open(os.path.join(config_path, filename), 'r') as f:
            config = yaml.safe_load(f)

        for key, value in config.items():
            if isinstance(value, dict):
                setattr(self, key, Struct(**value))
            else:
                setattr(self, key, value)

        self.json = config
