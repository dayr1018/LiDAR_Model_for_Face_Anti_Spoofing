from PIL import Image
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
import pdb

class Face_Data(Dataset):

    def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_code/metadata/', 
                        data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/data/' , datatxt = '', transform=None):
            self.metadata_root = metadata_root
            self.data_root = data_root
            self.transform = transform
            self.rgb_paths = []
            self.depth_paths = []
            self.labels = []

            lines_in_txt = open(os.path.join(metadata_root, datatxt),'r')

            for line in lines_in_txt:
                line = line.rstrip() 
                split_str = line.split()

                rgb_path = os.path.join(data_root, split_str[0])
                depth_path = os.path.join(data_root, split_str[1])
                label = split_str[2] 
                self.rgb_paths.append(rgb_path)
                self.depth_paths.append(depth_path)
                self.labels.append(label)

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]

        rgb_image = Image.open(rgb_path).convert('RGB')
        depth_image = Image.open(depth_path).convert('L')

        transform_RGB = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0,0,0], std=[1,1,1])
        ])

        transform_Depth = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0], std=[1])
        ])

        rgb_image = transform_RGB(rgb_image)
        depth_image = transform_Depth(depth_image)

        label = torch.as_tensor(int(self.labels[index]))
        
        return rgb_image, depth_image, label

    def __len__(self):
        return len(self.rgb_paths)
        
def Facedata_Loader(train_size=64, test_size=64, use_lowdata=True, dataset=0): 
    
    # 기존 데이터 (trian셋은 동일)
    if dataset == 0 : 
        print("***** Data set's type is 0 (original).")
        if use_lowdata:
            train_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/train_data_list.txt')
            valid_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/valid_data_list.txt')
            test_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/test_data_list.txt') 
            print("***** Low data is included to data set.")
        else:
            train_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/train_data_list_wo_low.txt")
            valid_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/valid_data_list_wo_low.txt")
            test_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/test_data_list_wo_low.txt")
            print("***** Low data is not included to data set.")

    # 추가된 데이터(trian셋은 동일)
    elif dataset == 1:
        print("***** Data set's type is 1 (added otherthings).")
        if use_lowdata:
            train_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/train_data_list_w_etc.txt')
            valid_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/valid_data_list_w_etc.txt')
            test_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/test_data_list_w_etc.txt') 
            print("***** Low data is included to data set")
        else:
            train_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/train_data_list_w_etc_wo_low.txt")
            valid_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/valid_data_list_w_etc_wo_low.txt")
            test_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/test_data_list_w_etc_wo_low.txt")  
            print("***** Low data is not included to data set")

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=8)
    valid_loader = DataLoader(dataset=valid_data, batch_size=train_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=8)

    return train_loader, valid_loader, test_loader