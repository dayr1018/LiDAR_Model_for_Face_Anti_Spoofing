import os
import os.path as osp
import random

import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
import seaborn as sns

import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T 

from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Face_Data(Dataset):

    def __init__(self, data_paths):
        self.data_paths = data_paths
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])        
            

    def __getitem__(self, index):
        rgb_path = self.data_paths[index][0]
        cloud_path = self.data_paths[index][1]
        
        # crop setting
        crop_width = 90
        crop_height = 140
        mid_x, mid_y = 90, 90
        offset_x, offset_y = crop_width//2, crop_height//2
        
        # RGB open and crop 
        rgb_data = cv2.imread(rgb_path)
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = cv2.resize(rgb_data, (180,180), interpolation=cv2.INTER_CUBIC)
        rgb_data = rgb_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]     
        
        if self.transforms is not None :
            rgb_data = self.transforms(rgb_data)
            
        # Point Cloud(192, 256, 3) open and crop 
        cloud_data = np.load(cloud_path)
        cloud_data = cv2.resize(cloud_data, (180,180), interpolation=cv2.INTER_CUBIC)
        cloud_data += 5
        cloud_data = cloud_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]
        
        # Point Cloud and Depth Scaling
        shift_value = 0
        xcoor = np.array(cloud_data[:, :, 0] + shift_value)
        ycoor = np.array(cloud_data[:, :, 1] + shift_value)
        zcoor = np.array(cloud_data[:, :, 2] + shift_value)
        depth = np.array(cloud_data[:, :, 3] + shift_value)
        
        xcoor = (xcoor-xcoor.mean())/xcoor.std()
        ycoor = (ycoor-ycoor.mean())/ycoor.std()
        zcoor = (zcoor-zcoor.mean())/zcoor.std()
        depth = (depth-depth.mean())/depth.std()     
        
        scaled_cloud_data = np.concatenate([xcoor[np.newaxis,:],ycoor[np.newaxis,:],zcoor[np.newaxis,:]]) 
        scaled_depth_data = depth[np.newaxis,:]
        
  
        # label - { 0 : real , 1 : mask }
        if 'bonafide' in rgb_path :
            label = 0
        elif 'attack_mask' in rgb_path :
            label = 1
        elif 'attack_replay' in rgb_path :
            label = 1
        elif 'attack_paper' in rgb_path :
            label = 1
        return rgb_data, scaled_cloud_data, scaled_depth_data, label
        
    def __len__(self):
        return len(self.data_paths)
        
def Facedata_Loader(batch_size=4, num_workers=4, attack_type="", dataset_type=12, traindata_ratio=1): 
    
    ## Input : RGB(3-channel) + Depth(1-channel) + Point_Cloud(3-channel)
     
    data_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS'
    npy_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/NPY_Files'

    # person_number = [int(i) for i in os.listdir(data_path) if i.isdigit()]
    if dataset_type == 12:
        # 1~12 dataset 
        print("dataset_12 is used")
        person_number = [i for i in range(1,10)] # 1~9
        test_number = [i for i in range(10,13)]  # 10~12
    elif dataset_type == 15:
        # 1~15 dataset
        print("dataset_15 is used")
        person_number = [i for i in range(1,13)] # 1~12
        test_number = [i for i in range(13,16)]  # 13~15

    traindata_portion = traindata_ratio

    train_img_paths, test_img_paths = [],[]
    for i in person_number :
        img_path = osp.join(data_path,str(i),'bonafide')
        files = os.listdir(img_path)
        files = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        random.shuffle(files)
        
        bonafide_files = [osp.join(data_path,str(i),'bonafide',j) for j in files]
        paper_files= [osp.join(data_path,str(i),'attack_paper',j) for j in files]
        replay_files= [osp.join(data_path,str(i),'attack_replay',j) for j in files]
        mask_files= [osp.join(data_path,str(i),'attack_mask',j) for j in files]
        

        bonafide_cloud_files = [osp.join(npy_path, 'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        paper_cloud_files = [osp.join(npy_path, 'paper_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in paper_files]
        replay_cloud_files = [osp.join(npy_path, 'replay_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in replay_files]
        mask_cloud_files = [osp.join(npy_path, 'mask_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in mask_files]
        
        # bonafide
        train_img_paths += list(zip(bonafide_files,bonafide_cloud_files))[:int(len(bonafide_files)*traindata_portion)]
        test_img_paths += list(zip(bonafide_files,bonafide_cloud_files))[int(len(bonafide_files)*traindata_portion):]
        
        # PAs
        if "p" in attack_type:
            train_img_paths += list(zip(paper_files,paper_cloud_files))[:int(len(paper_files)*traindata_portion)]
            test_img_paths += list(zip(paper_files,paper_cloud_files))[int(len(paper_files)*traindata_portion):]
        if "r" in attack_type:
            train_img_paths += list(zip(replay_files,replay_cloud_files))[:int(len(replay_files)*traindata_portion)]
            test_img_paths += list(zip(replay_files,replay_cloud_files))[int(len(replay_files)*traindata_portion):]
        if "m" in attack_type:
            train_img_paths += list(zip(mask_files,mask_cloud_files))[:int(len(mask_files)*traindata_portion)]
            test_img_paths += list(zip(mask_files,mask_cloud_files))[int(len(mask_files)*traindata_portion):]
        
    for i in test_number :
        img_path = osp.join(data_path,str(i),'bonafide')
        files = os.listdir(img_path)
        files = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        random.shuffle(files)
    
        bonafide_files = [osp.join(data_path,str(i),'bonafide',j) for j in files]
        paper_files= [osp.join(data_path,str(i),'attack_paper',j) for j in files]
        replay_files= [osp.join(data_path,str(i),'attack_replay',j) for j in files]
        mask_files= [osp.join(data_path,str(i),'attack_mask',j) for j in files]
        
        
        bonafide_cloud_files = [osp.join(npy_path,'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        paper_cloud_files = [osp.join(npy_path,'paper_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in paper_files]
        replay_cloud_files = [osp.join(npy_path,'replay_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in replay_files]
        mask_cloud_files = [osp.join(npy_path,'mask_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in mask_files]

        # bonafide
        test_img_paths += list(zip(bonafide_files, bonafide_cloud_files))[:]

        # PAs
        if "p" in attack_type:
            test_img_paths += list(zip(paper_files, paper_cloud_files))[:]
        if "r" in attack_type:
            test_img_paths += list(zip(replay_files, replay_cloud_files))[:]
        if "m" in attack_type:
            test_img_paths += list(zip(mask_files, mask_cloud_files))[:]
        
    random.shuffle(train_img_paths)
    random.shuffle(test_img_paths)

    print(len(train_img_paths))
    print(len(test_img_paths))
    
    train_dataset=Face_Data(train_img_paths)
    test_dataset=Face_Data(test_img_paths) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
