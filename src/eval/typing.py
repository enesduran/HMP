# Copyright (c) ETH Zurich and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
Our code started from the original code in https://github.com/vchoutas/expose; We have made significant modification to the original code in developing TempCLR.
"""
from typing import NewType, List, Union, Tuple, Optional
from dataclasses import dataclass, fields
import numpy as np
import torch
from yacs.config import CfgNode as CN

from abc import ABC, abstractmethod
from loguru import logger


class AbstractStructure(ABC):
    def __init__(self):
        super(AbstractStructure, self).__init__()
        self.extra_fields = {}

    def __del__(self):
        if hasattr(self, 'extra_fields'):
            self.extra_fields.clear()

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def delete_field(self, field):
        if field in self.extra_fields:
            del self.extra_fields[field]

    def shift(self, vector, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.shift(vector)
            self.add_field(k, v)
        self.add_field('motion_blur_shift', vector)
        return self

    def transpose(self, method):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.transpose(method)
            self.add_field(k, v)
        self.add_field('is_flipped', True)
        return self

    def normalize(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.normalize(*args, **kwargs)
        return self

    def rotate(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.rotate(*args, **kwargs)
        self.add_field('rot', kwargs.get('rot', 0))
        return self

    def crop(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.crop(*args, **kwargs)
        return self

    def resize(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v = v.resize(*args, **kwargs)
            self.add_field(k, v)
        return self

    def to_tensor(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if isinstance(v, AbstractStructure):
                v.to_tensor(*args, **kwargs)
            self.add_field(k, v)

    def to(self, *args, **kwargs):
        for k, v in self.extra_fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
        return self

__all__ = [
    'CN',
    'Tensor',
    'Array',
    'IntList',
    'IntTuple',
    'IntPair',
    'FloatList',
    'FloatTuple',
    'StringTuple',
    'StringList',
    'TensorTuple',
    'TensorList',
    'DataLoader',
    'BlendShapeDescription',
    'AppearanceDescription',
]


Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)
IntList = NewType('IntList', List[int])
IntTuple = NewType('IntTuple', Tuple[int])
IntPair = NewType('IntPair', Tuple[int, int])
FloatList = NewType('FloatList', List[float])
FloatTuple = NewType('FloatTuple', Tuple[float])
StringTuple = NewType('StringTuple', Tuple[str])
StringList = NewType('StringList', List[str])

TensorTuple = NewType('TensorTuple', Tuple[Tensor])
TensorList = NewType('TensorList', List[Tensor])
StructureList = NewType('StructureList', List[AbstractStructure])
DataLoader = torch.utils.data.DataLoader


@dataclass
class BlendShapeDescription:
    dim: int
    mean: Optional[Tensor] = None

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        for f in fields(self):
            if f.name == key:
                return getattr(self, key)


@dataclass
class AppearanceDescription:
    dim: int
    mean: Optional[Tensor] = None

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        for f in fields(self):
            if f.name == key:
                return getattr(self, key)
