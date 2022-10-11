import os
import glob
import numpy as np
import cv2
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root, dataset_type, fig_type='*', img_size=160, seed=123, num=128, transform=None):
        self.transform = transform
        self.img_size = img_size
        self.file_names = [f for f in glob.glob(os.path.join(root, fig_type, '*.npz')) if dataset_type in f]
        print(len(self.file_names))
        # random.seed(seed)
        # random.shuffle(self.file_names)
        # self.file_names = self.file_names[:num]
        # print(len(self.file_names))

        # # few samples
        # if dataset_type == 'train':
        #     random.seed(seed)
        #     random.shuffle(self.file_names)
        #     self.file_names = self.file_names[:num]
        #     print(len(self.file_names))
        # if dataset_type == 'test':
        #     random.seed(seed)
        #     random.shuffle(self.file_names)
        #     self.file_names = self.file_names[:20000]
        #     print(len(self.file_names))


        # ####
        # if dataset_type == 'train':
        #     self.file_names = [f for f in self.file_names if f.split('_')[-1][-5] not in ['x', 'y', 'g']]
        # print(len(self.file_names))
        # ####

        # #######
        # if dataset_type == 'train':
        #     self.file_names = [f for f in self.file_names if f.split('_')[-1][-5] not in ['x', 'y', 'g'] or f.split('_')[-1][-6:-4] == 'xy']
        # print(len(self.file_names))
        # #######

        
        #self.file_names = [self.file_names[i] for i in idx[0:21000]]   # for few-shot learning
        
    def __len__(self):
        return len(self.file_names)
    

    def __getitem__(self, idx):
        data = np.load(self.file_names[idx])
        image = data['image'].reshape(16, 160, 160)
        target = data['target']
        
        del data
        
        resize_image = image
        # if self.img_size is not None:
        #     resize_image = []
        #     for idx in range(0, 16):
        #         resize_image.append(cv2.resize(image[idx, :], (self.img_size, self.img_size), interpolation = cv2.INTER_NEAREST))
        #     resize_image = np.stack(resize_image)

        if self.transform:
            resize_image = self.transform(resize_image)
            # print(resize_image)
            target = torch.tensor(target, dtype=torch.long)
                    
        return resize_image/255., target
        
