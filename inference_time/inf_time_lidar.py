from models.Network import rgbdp_v3_twostep_model
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T 
import os
import os.path as osp
import cv2

fix_device = 'cuda:0'

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

def load_dataset(): 
        
    ## Input : RGB(3-channel) + Depth(1-channel) + Point_Cloud(3-channel)
    data_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/1. Indoor'
    npy_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/NPY_Files/1. Indoor'

    traindata_count = [i for i in range(1,6)]
 
    train_img_paths = []
    for i in traindata_count :
        img_path = osp.join(data_path, str(i), 'bonafide')
        files = os.listdir(img_path)
        rgbs = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        depths = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='depth')]
        
        # RGB 
        bonafide_files = [osp.join(data_path, str(i), 'bonafide', j) for j in rgbs]
        
        # Depth
        bonafide_depths = [osp.join(data_path, str(i), 'bonafide', j) for j in depths]     
        
        # Point Cloud
        bonafide_cloud_files = [osp.join(npy_path, 'real_cloud_data',j.split('/')[-3], 
                                (('pc_'+j.split('/')[-1].split('_')[-1]).split('.')[0]+'.npy')) for j in bonafide_files]
        
        # bonafide
        train_img_paths += list(zip(bonafide_files,bonafide_cloud_files,bonafide_depths))[:]
 
    train_dataset=Face_Data(train_img_paths, False)

    return train_dataset

lidar_model = rgbdp_v3_twostep_model(device=fix_device).eval()

train_dataset = load_dataset()
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

for step, data in enumerate(train_loader) :  
    rgb, cloud, depth, label = data
    rgb = rgb.float().to(fix_device) 
    cloud = cloud.float().to(fix_device) 
    depth = depth.float().to(fix_device) 
    label = label.float().to(fix_device) 

    _ = lidar_model(rgb, depth, cloud) 