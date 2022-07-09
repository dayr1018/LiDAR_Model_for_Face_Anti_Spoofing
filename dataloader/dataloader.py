
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
            
            self.image_height = 112
            self.image_width = 112
            

            print(f"histogram stretching:{histogram_stretching}")

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
        
        rgb_array = np.zeros((3, self.image_height, self.image_width))
        pointcloud_array = np.zeros((3,192,256))
        depth_array = np.zeros((1,192,256))

        size_transformer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.CenterCrop((self.image_height, self.image_width))
        ])   

        ############ RGB open 
        rgb_image = Image.open(rgb_path).convert('RGB')
        
        # Hisgoram Stetching 할 경우 
        if self.histogram_stretching == True:
            rgb_image = histogram_stretching(rgb_image)
            
        # Size 조정 
        rgb_image = size_transformer(rgb_image)
        
        # 다양한 정규화 조합 적용 
        if self.rgb_norm == 'nothing':
            print("[rgb] nothing ")
            rgb_numpy = np.array(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            
        elif self.rgb_norm == 'std':
            print("[rgb] std ")
            # swaped_rgb_np = np.array(rgb_image).swapaxes(0,2)
            # rgb_array[0] = StandardScaler().fit_transform(swaped_rgb_np[0])
            # rgb_array[1] = StandardScaler().fit_transform(swaped_rgb_np[1])
            # rgb_array[2] = StandardScaler().fit_transform(swaped_rgb_np[2])
            # shapeback_rgb_np = rgb_array.swapaxes(0,2)
            # # rgb_tensor = torch.from_numpy(shapeback_rgb_np)
            # rgb_image = Image.fromarray(shapeback_rgb_np.astype(np.uint8))   
            
            rgb_numpy = np.array(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            rgb_numpy = rgb_tensor.numpy() # C, H, W 인 numpy 생성 
        
            # C, H, W 인 rgb_array
            rgb_array[0] = StandardScaler().fit_transform(rgb_numpy[0])
            rgb_array[1] = StandardScaler().fit_transform(rgb_numpy[1])
            rgb_array[2] = StandardScaler().fit_transform(rgb_numpy[2])
            
            rgb_tensor = torch.from_numpy(rgb_array)
            
        elif self.rgb_norm == 'minmax':
            print("[rgb] minmax ")
            
            minmax_transformer = transforms.Compose([
                transforms.ToTensor()
            ])   
            rgb_tensor = minmax_transformer(rgb_image) 
            
        elif self.rgb_norm == 'stdminmax':
            print("[rgb] stdminmax ")

            # std 수행 
            rgb_numpy = np.array(rgb_image)
            rgb_tensor = torch.from_numpy(rgb_numpy).permute(2,0,1)
            rgb_numpy = rgb_tensor.numpy() # C, H, W 인 numpy 생성 
        
            # C, H, W 인 rgb_array
            rgb_array[0] = StandardScaler().fit_transform(rgb_numpy[0])
            rgb_array[1] = StandardScaler().fit_transform(rgb_numpy[1])
            rgb_array[2] = StandardScaler().fit_transform(rgb_numpy[2])
            
            # minmax 수행 
            rgb_array[0] = MinMaxScaler().fit_transform(rgb_array[0])
            rgb_array[1] = MinMaxScaler().fit_transform(rgb_array[1])
            rgb_array[2] = MinMaxScaler().fit_transform(rgb_array[2])
            
            # 텐서로 변경 (permute 필요없음)
            rgb_tensor = torch.from_numpy(rgb_array)
   
        elif self.rgb_norm == 'minmaxstd':
            print("[rgb] minmaxstd ")
            
            minmaxstd_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0,0,0], [1,1,1])
            ])   
            rgb_tensor = minmaxstd_transformer(rgb_image)   
 
        
        ############ Depth open
        depth_image = Image.open(depth_path).convert('L')

        if self.depth_norm == 'std':
            print("[depth] std_norm do it")
            swaped_depth_np = np.array(depth_image)#.swapaxes(0,2)
            depth_array = StandardScaler().fit_transform(swaped_depth_np)
            shapeback_depth_np = depth_array#.swapaxes(0,2)
            depth_image = Image.fromarray(shapeback_depth_np.astype(np.uint8))         

        # Depth 텐서화
        depth_transfomer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.CenterCrop((112,112)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.5], std=[1]) #
        ])
        depth_tensor = depth_transfomer(depth_image)


        ############ Point Cloud(PLY) open 
        plydata = PlyData.read(pointcloud_path)
        ply_pd = pd.DataFrame({key_: plydata['vertex'][key_] for key_ in ['x', 'y', 'z', 'red', 'green', 'blue', 'cx', 'cy', 'depth', 'alpha']})
        
        # Point Cloud 정규화 (1.MinMax 2.Std 3.nothing)
        # 1. Std 정규화 
        if self.depth_norm == 'std':
            print("[point cloud] std_norm do it")
            standard_result = StandardScaler().fit_transform(ply_pd)
            standard_pd = pd.DataFrame(standard_result)
            standard_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = standard_pd['x']
            list_y = standard_pd['y']
            list_z = standard_pd['z']
            list_depth = standard_pd['depth']
            list_width = plydata['vertex']['cx']
            list_height = plydata['vertex']['cy']
        
        # 2. MinMax 정규화         
        elif self.depth_norm == 'minmax':  
            print("[point cloud] minmax_norm do it")               
            minmax_result = MinMaxScaler().fit_transform(ply_pd)
            minmax_pd = pd.DataFrame(minmax_result)
            minmax_pd.columns = ["x","y","z","red","blue","green","cx","cy","depth","alpha"]

            list_x = minmax_pd['x']
            list_y = minmax_pd['y']
            list_z = minmax_pd['z']
            list_depth = minmax_pd['depth']
            list_width = plydata['vertex']['cx']
            list_height = plydata['vertex']['cy']
            
        # 3. Nothing 
        else:
            list_x = plydata['vertex']['x']
            list_y = plydata['vertex']['y']
            list_z = plydata['vertex']['z']
            list_depth = plydata['vertex']['depth']
            list_width = plydata['vertex']['cx']
            list_height = plydata['vertex']['cy']

        for idx in range(49151):
            pointcloud_array[0][int(list_height[idx])][int(list_width[idx])] = list_x[idx]  # [c, h, w]  그리고 h = cy, w = cx
            pointcloud_array[1][int(list_height[idx])][int(list_width[idx])] = list_y[idx]
            pointcloud_array[2][int(list_height[idx])][int(list_width[idx])] = list_z[idx]
            depth_array[0][int(list_height[idx])][int(list_width[idx])] = list_depth[idx]

        # Point Cloud 텐서화 
        pointcloud_tensor = torch.from_numpy(pointcloud_array)
        pointcloud_tensor.resize_(3,112,112) # 128, 128 이어야 함 
        # pointcloud_tensor.centercrop_(3,112,112)
        
        depth_transfomer = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.CenterCrop((112,112)),
            transforms.ToTensor()
            # transforms.Normalize(mean=[0.5], std=[1]) #
        ])
        
    
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
