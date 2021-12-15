from PIL import Image
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import pdb

class CISIA_SURF(Dataset):

    def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/code/metadata/', 
                        data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/data/' , datatxt = '', transform=None):
        self.metadata_root = metadata_root
        self.data_root = data_root
        self.transform = transform
        self.rgb_paths = []
        self.labels = []

        lines_in_txt = open(os.path.join(metadata_root, datatxt),'r')

        for line in lines_in_txt:
            line = line.rstrip() 
            split_str = line.split()

            rgb_path = os.path.join(data_root, split_str[0])
            label = split_str[1] 
            self.rgb_paths.append(rgb_path)
            self.labels.append(label)

    def __getitem__(self,index):
        rgb_path = self.rgb_paths[index]
        rgb_img = Image.open(rgb_path).convert('RGB')

        if self.transform is not None:
            rgb_img = self.transform(rgb_img)

        label = torch.as_tensor(int(self.labels[index]))
        
        return rgb_img, label

    def __len__(self):
        return len(self.rgb_paths)

def load_cisia_surf(train_size=128, test_size=64, mode=1): 
    
    train_transforms = transforms.Compose([
        transforms.Resize((124,124)),
        # transforms.CenterCrop((112,112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((124,124)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(112, padding=5),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    train_data=CISIA_SURF(datatxt='MakeTextFileCode_RGB/train_data_list.txt', transform=train_transforms)
    test_data=CISIA_SURF(datatxt='MakeTextFileCode_RGB/test_data_list.txt', transform=train_transforms)

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=32)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=32)
    
    # print('train dataset count :',len(train_data))
    # print('test dataset count :',len(test_data))
    # print('train loader count :',len(train_loader))
    # print('test loader count :',len(test_loader))

    return train_loader, test_loader
