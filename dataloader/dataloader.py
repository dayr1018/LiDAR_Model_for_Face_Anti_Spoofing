
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms 
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from PIL import Image
import os
import cv2
import torch
import numpy as np
from plyfile import PlyData
import pandas as pd 

class Face_Data(Dataset):

    def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_code/metadata/', 
                        data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/' , data_txt = '', transform=None, rgb_norm='nothing', depth_norm='nothing', histogram_stretching=False):
            self.metadata_root = metadata_root
            self.data_root = data_root
            self.transform = transform
            self.rgb_paths = []
            self.depth_paths = []
            self.pointcloud_paths = []
            self.labels = []
            self.rgb_norm = rgb_norm
            self.depth_norm = depth_norm
            self.histogram_stretching = histogram_stretching
            self.croped_height = 112
            self.croped_width = 112

            lines_in_txt = open(os.path.join(metadata_root, data_txt),'r')
            for line in lines_in_txt:
                line = line.rstrip() 
                split_str = line.split()

                rgb_path = os.path.join(data_root, split_str[0])
                depth_path = os.path.join(data_root, split_str[1])
                pointcloud_path = os.path.join(data_root, split_str[2])
                label = split_str[3] 
                
                self.rgb_paths.append(rgb_path)
                self.depth_paths.append(depth_path)
                self.pointcloud_paths.append(pointcloud_path)
                self.labels.append(label)

    def __getitem__(self, index):
        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        pointcloud_path = self.pointcloud_paths[index]
        label = int(self.labels[index])
 
        size_transformer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.CenterCrop((self.croped_height, self.croped_width))
        ])   

        ############ RGB open 
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Hisgoram Stetching 할 경우 
        if self.histogram_stretching == True:
            rgb_image = histogram_stretching(rgb_image)
            
        # 다양한 정규화 조합 적용 
        if self.rgb_norm == 'nothing':
            rgb_numpy = np.array(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            
        elif self.rgb_norm == 'std':
            # swaped_rgb_np = np.array(rgb_image).swapaxes(0,2)
            # rgb_array[0] = StandardScaler().fit_transform(swaped_rgb_np[0])
            # rgb_array[1] = StandardScaler().fit_transform(swaped_rgb_np[1])
            # rgb_array[2] = StandardScaler().fit_transform(swaped_rgb_np[2])
            # shapeback_rgb_np = rgb_array.swapaxes(0,2)
            # # rgb_tensor = torch.from_numpy(shapeback_rgb_np)
            # rgb_image = Image.fromarray(shapeback_rgb_np.astype(np.uint8))   
            
            # rgb numpy를 tensor로 바꾸고 (H,W,C) -> (C,H,W) 로 변경  
            rgb_numpy = np.array(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            
            # 표준화 수행
            rgb_numpy = rgb_tensor.numpy() 
             
            rgb_numpy[0] = StandardScaler().fit_transform(rgb_numpy[0])
            rgb_numpy[1] = StandardScaler().fit_transform(rgb_numpy[1])
            rgb_numpy[2] = StandardScaler().fit_transform(rgb_numpy[2])
            
            rgb_tensor = torch.from_numpy(rgb_numpy)
            
        elif self.rgb_norm == 'minmax':           
            minmax_transformer = transforms.Compose([
                transforms.ToTensor()
            ])   
            rgb_tensor = minmax_transformer(rgb_image) 
            
        elif self.rgb_norm == 'stdminmax':
            
            # rgb numpy를 tensor로 바꾸고 (H,W,C) -> (C,H,W) 로 변경  
            rgb_numpy = np.array(rgb_image)            
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            
            # 표준화 수행
            rgb_numpy = rgb_tensor.numpy() 
        
            rgb_numpy[0] = StandardScaler().fit_transform(rgb_numpy[0])
            rgb_numpy[1] = StandardScaler().fit_transform(rgb_numpy[1])
            rgb_numpy[2] = StandardScaler().fit_transform(rgb_numpy[2])
            
            # minmax 수행 
            rgb_numpy[0] = MinMaxScaler().fit_transform(rgb_numpy[0])
            rgb_numpy[1] = MinMaxScaler().fit_transform(rgb_numpy[1])
            rgb_numpy[2] = MinMaxScaler().fit_transform(rgb_numpy[2])
            
            # 텐서로 변경 (permute 필요없음)
            rgb_tensor = torch.from_numpy(rgb_numpy)
   
        elif self.rgb_norm == 'minmaxstd':            
            minmaxstd_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0,0,0], [1,1,1])
            ])   
            rgb_tensor = minmaxstd_transformer(rgb_image)   
 
         # 이미지 크기 조정 
        rgb_tensor = size_transformer(rgb_tensor)
           
        ############ Depth open
        depth_image = Image.open(depth_path).convert('L')

        # 다양한 정규화 조합 적용 
        if self.depth_norm == 'nothing':
            
            # depth numpy를 (H,W) -> (H,W,C) 로 바꾸고, tensor로 변경((H,W,C) -> (C,H,W) 도 변경)  
            depth_numpy = np.array(depth_image)[:, :, np.newaxis]
            depth_tensor = torch.from_numpy(depth_numpy).permute(2,0,1)

        elif self.depth_norm == 'std':
                        
            # depth numpy 표준화 수행 
            depth_numpy = np.array(depth_image)
            depth_numpy = StandardScaler().fit_transform(depth_numpy)
            
            # (H,W) -> (H,W,C) 로 바꾸고, tensor로 변경 (shapeh도 (H,W,C) -> (C,H,W) 도 변경)  
            depth_numpy = depth_numpy[:, :, np.newaxis]
            depth_tensor = torch.from_numpy(depth_numpy).permute(2,0,1)
            
        elif self.depth_norm == 'minmax':   
            minmax_transformer = transforms.Compose([
                transforms.ToTensor()
            ])   
            depth_tensor = minmax_transformer(depth_image)                     
            
        elif self.depth_norm == 'stdminmax':
            
            # depth numpy 표준화, MinMax 정규화 수행 
            depth_numpy = np.array(depth_image)
            depth_numpy = StandardScaler().fit_transform(depth_numpy)
            depth_numpy = MinMaxScaler().fit_transform(depth_numpy)
            
            # (H,W) -> (H,W,C) 로 바꾸고, tensor로 변경 (shapeh도 (H,W,C) -> (C,H,W) 도 변경)  
            depth_numpy = depth_numpy[:, :, np.newaxis]
            depth_tensor = torch.from_numpy(depth_numpy).permute(2,0,1)
   
        elif self.depth_norm == 'minmaxstd':            
            minmaxstd_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0], [1])
            ])   
            depth_tensor = minmaxstd_transformer(depth_image)   
            
         # 이미지 크기 조정 
        depth_tensor = size_transformer(depth_tensor)
        
        ############ Point Cloud(PLY) open 
        plydata = PlyData.read(pointcloud_path)
        ply_pd = pd.DataFrame({key_: plydata['vertex'][key_] for key_ in ['x', 'y', 'z', 'red', 'green', 'blue', 'cx', 'cy', 'depth', 'alpha']})
        
        # cx, cy 
        list_width = plydata['vertex']['cx']
        list_height = plydata['vertex']['cy']
        
        # Point Cloud 정규화 적용 
        if self.depth_norm == 'nothing':
            list_x = plydata['vertex']['x']
            list_y = plydata['vertex']['y']
            list_z = plydata['vertex']['z']
            list_depth = plydata['vertex']['depth']
            
        elif self.depth_norm == 'std':
            standard_result = StandardScaler().fit_transform(ply_pd)
            standard_pd = pd.DataFrame(standard_result)
            standard_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = standard_pd['x']
            list_y = standard_pd['y']
            list_z = standard_pd['z']
            list_depth = standard_pd['depth']
            
        elif self.depth_norm == 'minmax':   
            minmax_result = MinMaxScaler().fit_transform(ply_pd)
            minmax_pd = pd.DataFrame(minmax_result)
            minmax_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = minmax_pd['x']
            list_y = minmax_pd['y']
            list_z = minmax_pd['z']
            list_depth = minmax_pd['depth']
            
        elif self.depth_norm == 'stdminmax':
            standard_result = StandardScaler().fit_transform(ply_pd)
            minmax_result = MinMaxScaler().fit_transform(standard_result)
            stdminmax_pd = pd.DataFrame(minmax_result)
            stdminmax_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = stdminmax_pd['x']
            list_y = stdminmax_pd['y']
            list_z = stdminmax_pd['z']
            list_depth = stdminmax_pd['depth']
   
        elif self.depth_norm == 'minmaxstd':      
            minmax_result = MinMaxScaler().fit_transform(ply_pd)
            standard_result = StandardScaler().fit_transform(minmax_result)
            minmaxstd_pd = pd.DataFrame(standard_result)
            minmaxstd_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = minmaxstd_pd['x']
            list_y = minmaxstd_pd['y']
            list_z = minmaxstd_pd['z']
            list_depth = minmaxstd_pd['depth']
    
        # (C, H, W) 인 Point Cloud Array 에 저장
        pointcloud_array = np.zeros((3, 192, 256)) 

        for idx in range(49151):
            pointcloud_array[0][int(list_height[idx])][int(list_width[idx])] = list_x[idx]  # [c, h, w]  그리고 h = cy, w = cx
            pointcloud_array[1][int(list_height[idx])][int(list_width[idx])] = list_y[idx]
            pointcloud_array[2][int(list_height[idx])][int(list_width[idx])] = list_z[idx]
            # depth_array[0][int(list_height[idx])][int(list_width[idx])] = list_depth[idx]

        # Tensor로 변경 후 이미지 크기 조정 
        pointcloud_tensor = torch.from_numpy(pointcloud_array)
        pointcloud_tensor = size_transformer(pointcloud_tensor)   
    
        # label 텐서화 
        label_tensor = torch.as_tensor(label)
        
        return rgb_tensor, depth_tensor, pointcloud_tensor, label_tensor

    def __len__(self):
        return len(self.rgb_paths)
        
def Facedata_Loader(train_size=64, test_size=64, use_lowdata=True, dataset=0, rgb_norm='nothing', depth_norm='nothing', histogram_stretching=False): 
    
    # dataset 종류, use_lowdata 여부 상관 없음 ! 
    print("**** LDFAS Datset is using.")
    train_data=Face_Data(data_txt='MakeTextFileCode_RGB_Depth_PointCloud/train_data.txt', rgb_norm=rgb_norm, depth_norm=depth_norm, histogram_stretching=histogram_stretching)
    test_data=Face_Data(data_txt='MakeTextFileCode_RGB_Depth_PointCloud/test_data.txt', rgb_norm=rgb_norm, depth_norm=depth_norm, histogram_stretching=histogram_stretching) 

    train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=8)

    return train_loader, test_loader

# class Face_Data(Dataset):

#     def __init__(self, metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_code/metadata/', 
#                         data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/data/' , datatxt = '', transform=None, histogram_stretching=False):
#             self.metadata_root = metadata_root
#             self.data_root = data_root
#             self.transform = transform
#             self.rgb_paths = []
#             self.depth_paths = []
#             self.labels = []
#             self.histogram_stretching = histogram_stretching

#             print(f"histogram stretching:{histogram_stretching}")

#             lines_in_txt = open(os.path.join(metadata_root, datatxt),'r')

#             for line in lines_in_txt:
#                 line = line.rstrip() 
#                 split_str = line.split()

#                 rgb_path = os.path.join(data_root, split_str[0])
#                 depth_path = os.path.join(data_root, split_str[1])
#                 label = split_str[2] 
#                 self.rgb_paths.append(rgb_path)
#                 self.depth_paths.append(depth_path)
#                 self.labels.append(label)

#     def __getitem__(self, index):
#         rgb_path = self.rgb_paths[index]
#         depth_path = self.depth_paths[index]

#         rgb_image = Image.open(rgb_path).convert('RGB')
#         depth_image = Image.open(depth_path).convert('L')

#         # Hisgoram Stetching 할 경우 
#         if self.histogram_stretching == True:
#             rgb_image = histogram_stretching(rgb_image)

#         transform_RGB = transforms.Compose([
#             transforms.Resize((128,128)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0,0,0], std=[1,1,1])
#         ])

#         transform_Depth = transforms.Compose([
#             transforms.Resize((128,128)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0], std=[1])
#         ])

#         rgb_image = transform_RGB(rgb_image)
#         depth_image = transform_Depth(depth_image)

#         label = torch.as_tensor(int(self.labels[index]))
        
#         return rgb_image, depth_image, label

#     def __len__(self):
#         return len(self.rgb_paths)
        
# def Facedata_Loader(train_size=64, test_size=64, use_lowdata=True, dataset=0, histogram_stretching=False): 
    
#     # 기존 데이터 (trian셋은 동일)
#     if dataset == 0 : 
#         print("***** Data set's type is 0 (original).")
#         if use_lowdata:
#             train_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/train_data_list.txt', histogram_stretching=histogram_stretching)
#             test_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/test_data_list.txt', histogram_stretching=histogram_stretching) 
#             print("***** Low data is included to data set.")
#         else:
#             train_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/train_data_list_wo_low.txt", histogram_stretching=histogram_stretching)
#             test_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/test_data_list_wo_low.txt", histogram_stretching=histogram_stretching)
#             print("***** Low data is not included to data set.")

#     # 추가된 데이터(trian셋은 동일)
#     elif dataset == 1:
#         print("***** Data set's type is 1 (added otherthings).")
#         if use_lowdata:
#             train_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/train_data_list_w_etc.txt', histogram_stretching=histogram_stretching)
#             test_data=Face_Data(datatxt='MakeTextFileCode_RGB_Depth/test_data_list_w_etc.txt', histogram_stretching=histogram_stretching) 
#             print("***** Low data is included to data set")
#         else:
#             train_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/train_data_list_w_etc_wo_low.txt", histogram_stretching=histogram_stretching)
#             test_data=Face_Data(datatxt="MakeTextFileCode_RGB_Depth/test_data_list_w_etc_wo_low.txt", histogram_stretching=histogram_stretching)  
#             print("***** Low data is not included to data set")

#     train_loader = DataLoader(dataset=train_data, batch_size=train_size, shuffle=True, num_workers=8)
#     test_loader = DataLoader(dataset=test_data, batch_size=test_size, shuffle=True, num_workers=8)

#     return train_loader, test_loader


# 10, 200  ->
# 50, 150  -> _1
# 80, 110  -> _2
# 60, 140  -> _3

minValue = 60
maxValue = 140

def normalizeRed(intensity):

    iI      = intensity
    minI    = minValue # 86 
    maxI    = maxValue #230
    minO    = 0
    maxO    = 255

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeGreen(intensity):

    iI      = intensity
    minI    = minValue # 90
    maxI    = maxValue #225
    minO    = 0
    maxO    = 255

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

def normalizeBlue(intensity):

    iI      = intensity
    minI    = minValue # 100
    maxI    = maxValue # 210
    minO    = 0
    maxO    = 255

    iO      = (iI-minI)*(((maxO-minO)/(maxI-minI))+minO)
    return iO

# imageObject is PIL 
def histogram_stretching(imageObject):
    # Split the red, green and blue bands from the Image
    multiBands      = imageObject.split()

    # Apply point operations that does contrast stretching on each color band
    normalizedRedBand      = multiBands[0].point(normalizeRed)
    normalizedGreenBand    = multiBands[1].point(normalizeGreen)
    normalizedBlueBand     = multiBands[2].point(normalizeBlue)

    # Create a new image from the contrast stretched red, green and blue brands
    normalizedImage = Image.merge("RGB", (normalizedRedBand, normalizedGreenBand, normalizedBlueBand))
    
    return normalizedImage
