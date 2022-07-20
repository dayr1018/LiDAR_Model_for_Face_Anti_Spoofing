from torch.utils.data import Dataset, DataLoader
import numpy as np
from plyfile import PlyData
import pandas as pd 
import os

if __name__ == "__main__":
    metadata_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_code/metadata/'
    train_path = 'MakeTextFileCode_RGB_Depth_PointCloud/train_data.txt'
    test_path = 'MakeTextFileCode_RGB_Depth_PointCloud/test_data.txt'
    data_root = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/'
    
    lines_in_train = open(os.path.join(metadata_root, train_path),'r')
    for line in lines_in_train:
        line = line.rstrip() 
        split_line = line.split()
        
        pointcloud_path = os.path.join(data_root, split_line[2])
             
        npy_path = split_line[2].split('.')[0]
        npy_path = npy_path.split('/')
        
        npy_name = npy_path[3] + ".npy"
        npy_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/NPY_Files/" + npy_path[1] + "/" + npy_path[2] + "/" 
        
        if not os.path.exists(npy_path): 
            os.makedirs(npy_path)   
        
        npy_file = npy_path + npy_name
        plydata = PlyData.read(pointcloud_path)
        data = np.array(plydata['vertex'])
        np.save(npy_file, data)
        

    lines_in_test = open(os.path.join(metadata_root, test_path),'r')
    for line in lines_in_test:
        line = line.rstrip() 
        split_line = line.split()
        
        pointcloud_path = os.path.join(data_root, split_line[2])
             
        npy_path = split_line[2].split('.')[0]
        npy_path = npy_path.split('/')
        
        npy_name = npy_path[3] + ".npy"
        npy_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/NPY_Files/" + npy_path[1] + "/" + npy_path[2] + "/" 
        
        if not os.path.exists(npy_path): 
            os.makedirs(npy_path)   
        
        npy_file = npy_path + npy_name
        plydata = PlyData.read(pointcloud_path)
        data = np.array(plydata['vertex'])
        np.save(npy_file, data)
        
        
        
        