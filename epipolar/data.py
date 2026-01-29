import h5py
import sys
import re
import numpy as np


def read_dsp5(disp_path):
    def _readDsp5Disp(filename):
        with h5py.File(filename, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]

    disp_img = _readDsp5Disp(disp_path)
    disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)[::2, ::2]
    occ_mask = np.zeros_like(disp_img, dtype=bool)
    valid_mask = disp_img < 512
    return disp_img, occ_mask, valid_mask


def readpfm(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if (sys.version[0]) == '3':
        header = header.decode('utf-8')
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    if (sys.version[0]) == '3':
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    else:
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    if (sys.version[0]) == '3':
        scale = float(file.readline().rstrip().decode('utf-8'))
    else:
        scale = float(file.readline().rstrip())

    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale



import os
import numpy as np
import h5py
import cv2
import torch
from PIL import Image
from pathlib import Path
from torchvision import transforms, datasets
import random


class SpringDataset(torch.utils.data.Dataset):
    def __init__(self, root, split_file, image_size):
        super().__init__()
        self.image_size = image_size
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])
        self.data_list = []
        with open(split_file, 'r') as fp:
            self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')

        right_img = Image.open(right_path).convert('RGB')

        disp_img = self._readDsp5Disp(disp_path)
        disp_img = np.ascontiguousarray(disp_img, dtype=np.float32)[::2, ::2]
        occ_mask = np.zeros_like(self.transform(disp_img), dtype=bool)

        scale = self.image_size / disp_img.shape[-1]

        sample = {
            'left': self.transform(left_img),
            'right': self.transform(right_img),
            'disp': self.transform(disp_img) * scale,
            'occ_mask': occ_mask
        }

        sample['valid'] = (sample['disp'] < 512) & np.isfinite(sample['disp'])
        sample['index'] = idx
        sample['name'] = left_path

        return sample

    def _readDsp5Disp(self, filename):
        with h5py.File(filename, "r") as f:
            if "disparity" not in f.keys():
                raise IOError(f"File {filename} does not have a 'disparity' key. Is this a valid dsp5 file?")
            return f["disparity"][()]


class SceneFlowDataset(torch.utils.data.Dataset):
    def __init__(self, root, split_file, image_size):
        super().__init__()
        self.image_size = image_size
        self.root = root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((image_size, image_size)),
        ])
        self.data_list = []
        with open(split_file, 'r') as fp:
            self.data_list.extend([x.strip().split(' ') for x in fp.readlines()])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths
        left_img = Image.open(left_path).convert('RGB')
        right_img = Image.open(right_path).convert('RGB')

        disp_img = readpfm(disp_path)[0].astype(np.float32)
        assert not np.isnan(disp_img).any(), 'disp_img has nan'
        occ_mask = np.zeros_like(disp_img, dtype=bool)

        scale = self.image_size / disp_img.shape[-1]

        sample = {
            'left': self.transform(left_img),
            'right': self.transform(right_img),
            'disp': self.transform(disp_img) * scale,
            'occ_mask': self.transform(occ_mask)
        }

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
