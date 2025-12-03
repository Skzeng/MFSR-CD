# modified from https://github.com/desimone/vision/blob/fb74c76d09bcc2594159613d5bdadd7d4697bb11/torchvision/datasets/folder.py

import os
import os.path

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image
import numpy as np


class FileFloder(data.Dataset):
    def __init__(self, root, transform=None):
        files = []
        for file in os.listdir(root):
            files.append('{}'.format(file))
        self.root = root
        self.files = files
        self.transform = transform

    def __getitem__(self, index):
        filename = self.files[index]
        # print(filename)
        try:
            das_data = np.load(os.path.join(self.root, filename)).reshape((1, 32, 32))
        except:
            return torch.zeros((1, 32, 32))
        if self.transform is not None:
            das_data = self.transform(das_data)
        # print('归一化前', das_data)
        norm_das_data = (das_data - np.min(das_data))/(np.max(das_data) - np.min(das_data))
        # print('归一化后', das_data)
        # return norm_das_data, filename, das_data
        # return filename, das_data
        return filename, norm_das_data

    def __len__(self):
        return len(self.files)


class DASBigFileDataset(data.Dataset):
    def __init__(self, file_path, transform=None):
        """
        Used for loading a single large.npy file after merging
        file_path: The absolute path of the large-.npy file, and the data shape should be (N, 32, 32).
        """
        super().__init__()

        try:
            self.data = np.load(file_path)
        except FileNotFoundError:
            print(f"Error: THE FILE CANNOT BE FOUND {file_path}")

            self.data = np.zeros((1, 32, 32), dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index):
        # 1.
        das_data = self.data[index]  #  (32, 32)

        # 2. INCREASE THE CHANNEL DIMENSION -> (1, 32, 32)
        #
        das_data = das_data.reshape(1, 32, 32).astype(np.float32)

        # 3.
        if self.transform is not None:
            das_data = self.transform(das_data)

        # 4. NORMALIZATION
        #
        min_val = np.min(das_data)
        max_val = np.max(das_data)
        norm_das_data = (das_data - min_val) / (max_val - min_val + 1e-8)

        # 5.
        filename = f"sample_{index}.npy"

        return filename, norm_das_data

    def __len__(self):
        return self.data.shape[0]

class NpzFileFloder(data.Dataset):
    def __init__(self, root, transform=None):
        files = []
        for file in os.listdir(root):
            files.append('{}'.format(file))
        self.root = root
        self.files = files
        self.transform = transform

    def __getitem__(self, index):
        filename = self.files[index]
        try:
            das_data = np.load(os.path.join(self.root, filename))
            codes = np.unpackbits(das_data['codes'])
            codes = np.reshape(codes, das_data['shape']).astype(np.float32) * 2 - 1
        except:
            return torch.zeros((1, 128, 512))
        if self.transform is not None:
            codes = self.transform(codes)
        return (codes, filename)

    def __len__(self):
        return len(self.files)


class CAEDATSET(data.Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.data = np.load(self.root)
        self.transform = transform
        self.row = self.data.shape[0]
        self.col = self.data.shape[1] // 1600

    def __getitem__(self, index):
        # filename = self.files[index]
        # print(filename)
        row_index = index // self.col
        col_index = index % self.col
        das_data = self.data[row_index, col_index * 1600:(col_index+1) * 1600].reshape(1, 1600)
        if self.transform is not None:
            das_data = self.transform(das_data)
        # print('归一化前', das_data)
        # norm_das_data = (das_data - np.min(das_data))/(np.max(das_data) - np.min(das_data))
        # print('归一化后', das_data)
        return das_data
        # return norm_das_data, das_data

    def __len__(self):
        return self.row * self.col


class CAEDATSET_60000(data.Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.data = np.load(self.root)
        self.transform = transform
        # self.row = self.data.shape[0]
        # self.col = self.data.shape[1] // 1600

    def __getitem__(self, index):
        # filename = self.files[index]
        # print(filename)
        # row_index = index // self.col
        # col_index = index % self.col
        das_data = self.data[index, 0:60000].reshape(1, 60000)
        if self.transform is not None:
            das_data = self.transform(das_data)
        # print('归一化前', das_data)
        # norm_das_data = (das_data - np.min(das_data))/(np.max(das_data) - np.min(das_data))
        # print('归一化后', das_data)
        return das_data
        # return norm_das_data, das_data

    def __len__(self):
        return self.data.shape[0]


class CAEDATSET_1920(data.Dataset):
    def __init__(self, root, transform=None):

        self.root = root
        self.data = np.load(self.root)
        self.transform = transform
        # self.row = self.data.shape[0]
        # self.col = self.data.shape[1] // 1600

    def __getitem__(self, index):
        # filename = self.files[index]
        # print(filename)
        # row_index = index // self.col
        # col_index = index % self.col
        das_data = self.data[index, 0:1920].reshape(1, 1920)
        if self.transform is not None:
            das_data = self.transform(das_data)
        # print('归一化前', das_data)
        # norm_das_data = (das_data - np.min(das_data))/(np.max(das_data) - np.min(das_data))
        # print('归一化后', das_data)
        return das_data
        # return norm_das_data, das_data

    def __len__(self):
        return self.data.shape[0]
