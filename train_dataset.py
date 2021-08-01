"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class SourceImageDataset(Dataset):
    def __init__(self, source_data_root, input_size):
        self.source_data_root = source_data_root
        self.source_files = []
        for dirPath, dirNames, fileNames in os.walk(source_data_root):
            for f in fileNames:
                self.source_files.append(os.path.join(dirPath, f))
        self.input_size = input_size

    def __len__(self):
        return len(self.source_files)

    def __getitem__(self, index):
        image_path = self.source_files[index]
        image1 = cv2.imread(image_path)
        image1 = cv2.resize(image1, self.input_size)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        # gray image2
        image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image2 = np.asarray([image2, image2, image2])
        image2 = image2.transpose((1, 2, 0))

        # only the upper half of face is cropped for training as mask data
        if random.random() > 0.5:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
        image1 = (image1 / 255. - 0.5) / 0.5
        image1 = image1.transpose((2, 0, 1))
        image2 = (image2 / 255. - 0.5) / 0.5
        image2 = image2.transpose((2, 0, 1))
        image1 = torch.from_numpy(image1.astype(np.float32))
        image2 = torch.from_numpy(image2.astype(np.float32))
        return image1, image2


class TargetImageDataset(Dataset):
    def __init__(self, target_data_root, input_size):
        self.target_data_root = target_data_root
        self.target_data_list = []
        self.target_smooth_data_list = []
        self.target_label_list = []

        _, self.dirNames, _ = next(os.walk(self.target_data_root))
        print('dirNames:', self.dirNames)

        for dirPath, _, fileNames in os.walk(self.target_data_root):
            key = dirPath.split('/')[-2]

            for f in fileNames:
                if key not in self.dirNames:
                    continue

                f = os.path.join(dirPath, f)
                label_index = self.dirNames.index(key)

                if dirPath.split('/')[-1] == "style":
                    self.target_data_list.append(f)
                elif dirPath.split('/')[-1] == "smooth":
                    self.target_smooth_data_list.append(f)
                self.target_label_list.append(label_index)

        self.input_size = input_size

    def get_class_size(self):
        return len(self.dirNames)

    def __len__(self):
        return len(self.target_data_list)

    def __getitem__(self, index):
        image_path = self.target_data_list[index]
        image1 = cv2.imread(image_path)
        image1 = cv2.resize(image1, self.input_size)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

        # gray image2
        image2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image2 = np.asarray([image2, image2, image2])
        image2 = image2.transpose((1, 2, 0))

        image_smooth_path = self.target_smooth_data_list[index]
        image_smooth1 = cv2.imread(image_smooth_path)
        image_smooth1 = cv2.resize(image_smooth1, self.input_size)
        image_smooth1 = cv2.cvtColor(image_smooth1, cv2.COLOR_BGR2RGB)

        # gray image2
        image_smooth2 = cv2.imread(image_smooth_path, cv2.IMREAD_GRAYSCALE)
        image_smooth2 = np.asarray([image_smooth2, image_smooth2, image_smooth2])
        image_smooth2 = image_smooth2.transpose((1, 2, 0))

        # only the upper half of face is cropped for training as mask data
        if random.random() > 0.5:
            image1 = cv2.flip(image1, 1)
            image2 = cv2.flip(image2, 1)
            image_smooth1 = cv2.flip(image_smooth1, 1)
            image_smooth2 = cv2.flip(image_smooth2, 1)
        image1 = (image1 / 255. - 0.5) / 0.5
        image1 = image1.transpose((2, 0, 1))
        image2 = (image2 / 255. - 0.5) / 0.5
        image2 = image2.transpose((2, 0, 1))
        image1 = torch.from_numpy(image1.astype(np.float32))
        image2 = torch.from_numpy(image2.astype(np.float32))

        image_smooth1 = (image_smooth1 / 255. - 0.5) / 0.5
        image_smooth1 = image_smooth1.transpose((2, 0, 1))
        image_smooth2 = (image_smooth2 / 255. - 0.5) / 0.5
        image_smooth2 = image_smooth2.transpose((2, 0, 1))
        image_smooth1 = torch.from_numpy(image_smooth1.astype(np.float32))
        image_smooth2 = torch.from_numpy(image_smooth2.astype(np.float32))

        return image1, image2, image_smooth1, image_smooth2, self.target_label_list[index]


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import sys
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_size = tuple([256, 256])
    train_data_src = SourceImageDataset('../../AnimeGAN/dataset/train_photo/', input_size=input_size)
    train_data_tgt = TargetImageDataset('../style_dataset', input_size=input_size)
    test_data_tgt = SourceImageDataset('../../places365/val/val_256', input_size=input_size)

    batch_size = 12
    train_loader_src = torch.utils.data.DataLoader(train_data_src
                                                   , batch_size=batch_size
                                                   , shuffle=True,
                                                   drop_last=False)
    train_loader_tgt = torch.utils.data.DataLoader(train_data_tgt
                                                   , batch_size=batch_size
                                                   , shuffle=True,
                                                   drop_last=False)

    test_loader_src = torch.utils.data.DataLoader(test_data_tgt
                                                  , batch_size=1
                                                  , shuffle=True
                                                  , drop_last=False)

    train_loader_src_iterator = iter(train_loader_src)
    train_loader_tgt_iterator = iter(train_loader_tgt)
    max_iter = len(train_data_tgt) // batch_size

    for current_iter in range(max_iter):
        try:
            x, x_gray = next(train_loader_src_iterator)
            y, y_gray, y_smooth, y_smooth_gray, labels = next(train_loader_tgt_iterator)
        except StopIteration:
            train_loader_src_iterator = iter(train_loader_src)
            train_loader_tgt_iterator = iter(train_loader_tgt)
            x, x_gray = next(train_loader_src_iterator)
            y, y_gray, y_smooth, y_smooth_gray, labels = next(train_loader_tgt_iterator)

        pos_result = torch.cat((x[0], y[0]), 2)
        plt.imsave('sample.png', (pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

        pos_result = torch.cat((y_smooth[0], y_smooth_gray[0]), 2)
        plt.imsave('sample_smooth.png', (pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)

        pos_result = torch.cat((x_gray[0], y_gray[0]), 2)
        plt.imsave('sample_gray.png', (pos_result.cpu().numpy().transpose(1, 2, 0) + 1) / 2)
