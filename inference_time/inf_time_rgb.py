from models.Network import Face_Detection_Model
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
        rgb_path = self.data_paths[index]  
             
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
            
        # label - { 0 : real , 1 : mask }
        if 'bonafide' in rgb_path :
            label = 0
        elif 'attack_mask' in rgb_path :
            label = 1
        elif 'attack_replay' in rgb_path :
            label = 1
        elif 'attack_paper' in rgb_path :
            label = 1

        return rgb_data, label
    def __len__(self):
        return len(self.data_paths)


def load_dataset(): 
        
    ## Input : RGB(3-channel) + Depth(1-channel) + Point_Cloud(3-channel)
    data_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/1. Indoor'

    traindata_count = [i for i in range(1,6)]
    train_img_paths = []
    for i in traindata_count :
        img_path = osp.join(data_path, str(i), 'bonafide')
        files = os.listdir(img_path)
        rgbs = [j for j in files if (j.split('.')[-1]=='jpg') and (j.split('_')[0]=='rgb')]
        
        # RGB 
        bonafide_files = [osp.join(data_path, str(i), 'bonafide', j) for j in rgbs]
     
        # bonafide
        train_img_paths += list(bonafide_files)[:]
 
    train_dataset=Face_Data(train_img_paths, False)

    return train_dataset

rgb_model = Face_Detection_Model(3).to(fix_device).eval()

train_dataset = load_dataset()
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

for step, data in enumerate(train_loader) :  
    rgb, label = data
    rgb = rgb.float().to(fix_device) 
    label = label.float().to(fix_device) 

    _ = rgb_model(rgb) 