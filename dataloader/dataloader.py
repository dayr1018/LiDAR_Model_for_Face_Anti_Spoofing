import os
import os.path as osp
import random
from re import S

import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T 

np.random.seed(1)
random.seed(1)

class Face_Data(Dataset):

    def __init__(self, data_paths, crop=False):
        self.data_paths = data_paths
        self.crop = crop
        
        normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std = [0.229, 0.224, 0.225])
        self.transforms = T.Compose([
            T.ToTensor(),
            normalize
        ])   
        self.transforms2 = T.Compose([
            T.ToTensor()
        ])               
    
    def __getitem__(self, index):
        rgb_path = self.data_paths[index][0]
        cloud_path = self.data_paths[index][1]
        depth_path = self.data_paths[index][2]   
             
        # crop setting
        crop_width = 90
        crop_height = 150
        mid_x, mid_y = 90, 90
        offset_x, offset_y = crop_width//2, crop_height//2
        
        # RGB open and crop 
        rgb_data = cv2.imread(rgb_path)
        rgb_data = cv2.cvtColor(rgb_data, cv2.COLOR_BGR2RGB)
        rgb_data = cv2.resize(rgb_data, (180,180), interpolation=cv2.INTER_CUBIC)
        if self.crop == True:
            rgb_data = rgb_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]     
        if self.transforms is not None :
            rgb_data = self.transforms(rgb_data)
            
        # Depth open 
        depth_data = cv2.imread(depth_path)
        depth_data = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)
        depth_data = cv2.resize(depth_data, (180,180), interpolation=cv2.INTER_CUBIC)        
        if self.transforms2 is not None :
            depth_data = self.transforms2(depth_data)            
            
        # Point Cloud(192, 256, 3) open and crop 
        cloud_data = np.load(cloud_path)
        cloud_data = cv2.resize(cloud_data, (180,180), interpolation=cv2.INTER_CUBIC)
        cloud_data += 5
        if self.crop == True:
            cloud_data = cloud_data[mid_y-offset_y:mid_y+offset_y, mid_x-offset_x:mid_x+offset_x]
        
        # Point Cloud and Depth Scaling
        shift_value = 0
        xcoor = np.array(cloud_data[:, :, 0] + shift_value)
        ycoor = np.array(cloud_data[:, :, 1] + shift_value)
        zcoor = np.array(cloud_data[:, :, 2] + shift_value)
        # depth = np.array(cloud_data[:, :, 3] + shift_value)

        # Stadardivation
        # xcoor = (xcoor-xcoor.mean())/xcoor.std()
        # ycoor = (ycoor-ycoor.mean())/ycoor.std()
        # zcoor = (zcoor-zcoor.mean())/zcoor.std()
        # depth = (depth-depth.mean())/depth.std()    
        
        # Min Max         
        xcoor = (xcoor-xcoor.min())/(xcoor.max()-xcoor.min())
        ycoor = (ycoor-ycoor.min())/(ycoor.max()-ycoor.min())
        zcoor = (zcoor-zcoor.min())/(zcoor.max()-zcoor.min())
        # depth = (depth-depth.min())/(depth.max()-depth.min())  
        
        scaled_cloud_data = np.concatenate([xcoor[np.newaxis,:],ycoor[np.newaxis,:],zcoor[np.newaxis,:]]) 
        # scaled_depth_data = depth[np.newaxis,:]
        
        # label - { 0 : real , 1 : mask }
        if 'bonafide' in rgb_path :
            label = 0
        elif 'attack_mask' in rgb_path :
            label = 1
        elif 'attack_replay' in rgb_path :
            label = 1
        elif 'attack_paper' in rgb_path :
            label = 1
        # return rgb_data, scaled_cloud_data, scaled_depth_data, label
        return rgb_data, scaled_cloud_data, depth_data, label
    def __len__(self):
        return len(self.data_paths)



def load_dataset(args): 
        
    ## Input : RGB(3-channel) + Depth(1-channel) + Point_Cloud(3-channel)
    data_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/1. Indoor'
    npy_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/NPY_Files/1. Indoor'

    traindata_count = [i for i in range(1,10)] # 1~9
    testdata_count = [i for i in range(10,13)]  # 10~12    
 
    train_img_paths = []
    for i in traindata_count :
        img_path = osp.join(data_path, str(i), 'bonafide')
        files = os.listdir(img_path)
        rgbs = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        depths = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='depth')]
        random.shuffle(rgbs)
        random.shuffle(depths)
        
        # RGB 
        bonafide_files = [osp.join(data_path, str(i), 'bonafide', j) for j in rgbs]
        paper_files= [osp.join(data_path, str(i), 'attack_paper', j) for j in rgbs]
        replay_files= [osp.join(data_path, str(i), 'attack_replay', j) for j in rgbs]
        mask_files= [osp.join(data_path, str(i), 'attack_mask', j) for j in rgbs]
        
        # Depth
        bonafide_depths = [osp.join(data_path, str(i), 'bonafide', j) for j in depths]
        paper_depths= [osp.join(data_path, str(i), 'attack_paper', j) for j in depths]
        replay_depths= [osp.join(data_path, str(i), 'attack_replay', j) for j in depths]
        mask_depths= [osp.join(data_path, str(i), 'attack_mask', j) for j in depths]       
        
        # Point Cloud
        bonafide_cloud_files = [osp.join(npy_path, 'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        paper_cloud_files = [osp.join(npy_path, 'paper_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in paper_files]
        replay_cloud_files = [osp.join(npy_path, 'replay_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in replay_files]
        mask_cloud_files = [osp.join(npy_path, 'mask_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in mask_files]
        
        # bonafide
        train_img_paths += list(zip(bonafide_files,bonafide_cloud_files,bonafide_depths))[:]
        
        # PAs
        if "p" in args.attacktype:
            train_img_paths += list(zip(paper_files,paper_cloud_files,paper_depths))[:]
        if "r" in args.attacktype:
            train_img_paths += list(zip(replay_files,replay_cloud_files,replay_depths))[:]
        if "m" in args.attacktype:
            train_img_paths += list(zip(mask_files,mask_cloud_files,mask_depths))[:]
 
    test_img_paths = []
    for i in testdata_count :
        img_path = osp.join(data_path, str(i), 'bonafide')
        files = os.listdir(img_path)
        rgbs = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        depths = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='depth')]
        random.shuffle(rgbs)
        random.shuffle(depths)
        
        # RGB
        bonafide_files = [osp.join(data_path, str(i), 'bonafide', j) for j in rgbs]
        paper_files= [osp.join(data_path, str(i), 'attack_paper', j) for j in rgbs]
        replay_files= [osp.join(data_path, str(i), 'attack_replay', j) for j in rgbs]
        mask_files= [osp.join(data_path, str(i), 'attack_mask', j) for j in rgbs]
        
        # Depth
        bonafide_depths = [osp.join(data_path, str(i), 'bonafide', j) for j in depths]
        paper_depths= [osp.join(data_path, str(i), 'attack_paper', j) for j in depths]
        replay_depths= [osp.join(data_path, str(i), 'attack_replay', j) for j in depths]
        mask_depths= [osp.join(data_path, str(i), 'attack_mask', j) for j in depths]        
        
        # Point Cloud
        bonafide_cloud_files = [osp.join(npy_path, 'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        paper_cloud_files = [osp.join(npy_path, 'paper_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in paper_files]
        replay_cloud_files = [osp.join(npy_path, 'replay_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in replay_files]
        mask_cloud_files = [osp.join(npy_path, 'mask_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in mask_files]
        
        # bonafide
        test_img_paths += list(zip(bonafide_files,bonafide_cloud_files,bonafide_depths))[:]
        
        # PAs
        if "p" in args.attacktype:
            test_img_paths += list(zip(paper_files,paper_cloud_files,paper_depths))[:]
        if "r" in args.attacktype:
            test_img_paths += list(zip(replay_files,replay_cloud_files,replay_depths))[:]
        if "m" in args.attacktype:
            test_img_paths += list(zip(mask_files,mask_cloud_files,mask_depths))[:]
        
    random.shuffle(train_img_paths)
    random.shuffle(test_img_paths)

    print(len(train_img_paths))
    print(len(test_img_paths))
    
    train_dataset=Face_Data(train_img_paths, False)
    test_dataset=Face_Data(test_img_paths, False) 

    return train_dataset, test_dataset


def load_test_dataset(args, dir_name): 
        
    ## Input : RGB(3-channel) + Depth(1-channel) + Point_Cloud(3-channel)
    LDFAS_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/'
    data_path = osp.join(LDFAS_path, dir_name)
    npy_path = osp.join(LDFAS_path, "NPY_Files", dir_name)
         
    testdata_count = [i for i in range(1,13)]  # 1~12    
    test_img_paths = []
    for i in testdata_count :
        img_path = osp.join(data_path, str(i), 'bonafide')
        files = os.listdir(img_path)
        rgbs = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        depths = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='depth')]
        random.shuffle(rgbs)
        random.shuffle(depths)
        
        # RGB
        bonafide_files = [osp.join(data_path, str(i), 'bonafide', j) for j in rgbs]
        paper_files= [osp.join(data_path, str(i), 'attack_paper', j) for j in rgbs]
        replay_files= [osp.join(data_path, str(i), 'attack_replay', j) for j in rgbs]
        mask_files= [osp.join(data_path, str(i), 'attack_mask', j) for j in rgbs]
        
        # Depth
        bonafide_depths = [osp.join(data_path, str(i), 'bonafide', j) for j in depths]
        paper_depths= [osp.join(data_path, str(i), 'attack_paper', j) for j in depths]
        replay_depths= [osp.join(data_path, str(i), 'attack_replay', j) for j in depths]
        mask_depths= [osp.join(data_path, str(i), 'attack_mask', j) for j in depths]        
               
        # Point Cloud
        bonafide_cloud_files = [osp.join(npy_path, 'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        paper_cloud_files = [osp.join(npy_path, 'paper_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in paper_files]
        replay_cloud_files = [osp.join(npy_path, 'replay_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in replay_files]
        mask_cloud_files = [osp.join(npy_path, 'mask_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in mask_files]
        
        # bonafide
        test_img_paths += list(zip(bonafide_files,bonafide_cloud_files,bonafide_depths))[:]
        
        # PAs
        if "p" in args.attacktype:
            test_img_paths += list(zip(paper_files,paper_cloud_files,paper_depths))[:]
        if "r" in args.attacktype:
            test_img_paths += list(zip(replay_files,replay_cloud_files,replay_depths))[:]
        if "m" in args.attacktype:
            test_img_paths += list(zip(mask_files,mask_cloud_files,mask_depths))[:]
     
    random.shuffle(test_img_paths)
    print(len(test_img_paths))
    
    test_dataset=Face_Data(test_img_paths, False) 

    return test_dataset
