import os
import os.path as osp
from tqdm import tqdm

from plyfile import PlyData
import numpy as np
import matplotlib.pyplot as plt
import cv2

import torch

def read_ply_xyzrgbnormal(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert (os.path.isfile(filename))
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:, 0] = plydata['vertex'].data['x']
        vertices[:, 1] = plydata['vertex'].data['y']
        vertices[:, 2] = plydata['vertex'].data['z']
        vertices[:, 3] = plydata['vertex'].data['cx']
        vertices[:, 4] = plydata['vertex'].data['cy']
        vertices[:, 5] = plydata['vertex'].data['depth']

    return vertices

if __name__ == "__main__":
    data_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS'
    save_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/LDFAS/NPY_Files'

    kind = ["1. Indoor", "2. Outdoor", "3. Indoor_dark"]
    for idx, dir_name in enumerate(kind):
        dir_path = osp.join(data_path, dir_name)
        
        person_number = [int(i) for i in os.listdir(dir_path) if i.isdigit()]
        for i in person_number:
            if not osp.exists(osp.join(save_path, dir_name, 'real_cloud_data', str(i))) :
                os.makedirs(osp.join(save_path, dir_name, 'real_cloud_data', str(i)))
            if not osp.exists(osp.join(save_path, dir_name, 'mask_cloud_data', str(i))) :
                os.makedirs(osp.join(save_path, dir_name, 'mask_cloud_data', str(i)))
            if not osp.exists(osp.join(save_path, dir_name, 'replay_cloud_data', str(i))) :
                os.makedirs(osp.join(save_path, dir_name, 'replay_cloud_data', str(i)))
            if not osp.exists(osp.join(save_path, dir_name, 'paper_cloud_data', str(i))) :
                os.makedirs(osp.join(save_path, dir_name, 'paper_cloud_data', str(i)))
    
  
# 여기까지 함 
                        
            
    # bonafide
    for i in tqdm(person_number) :
        for dir_name in kind:
            num_path = osp.join(data_path, dir_name, str(i))
            files = os.listdir(osp.join(num_path,'bonafide'))
            files = [j for j in files if j.split('_')[0]=='pc']
            for j in files : 
                ply = osp.join(osp.join(num_path,'bonafide',j))
                real = read_ply_xyzrgbnormal(ply)
        #         proj_depth = np.full((256, 192), -1,dtype=np.float32)
        #         proj_xyz = np.full((3, 256, 192), -1,dtype=np.float32)
                proj_xyzd = np.full((4, 192, 256), -1,dtype=np.float32)
                for k in range(len(real)):
        #             proj_depth[int(real[k, 3]), int(real[k, 4])] = real[k, 5]
                    proj_xyzd[0, int(real[k, 4]), int(real[k, 3])] = real[k, 0]
                    proj_xyzd[1, int(real[k, 4]), int(real[k, 3])] = real[k, 1]
                    proj_xyzd[2, int(real[k, 4]), int(real[k, 3])] = real[k, 2]
                    proj_xyzd[3, int(real[k, 4]), int(real[k, 3])] = real[k, 5]
                proj_xyzd = proj_xyzd.transpose(1,2,0)

                save_name = osp.join(save_path, dir_name, 'real_cloud_data',str(i),j.split('.')[0]+'.npy')
                np.save(save_name, proj_xyzd)

    # data check
    for dir_name in kind:
        for i in person_number :
            # org_path = osp.join(data_path,str(i))
            num_path = osp.join(data_path, dir_name, str(i))
            org_files = os.listdir(osp.join(num_path,'bonafide'))
            org_files = [j for j in org_files if j.split('_')[0]=='pc']

            new_path = osp.join(save_path, dir_name, 'real_cloud_data', str(i))
            new_files = os.listdir(new_path)
            print(i,f' Org Data Count : {len(org_files)}  New Data Count : {len(new_files)} ==> {len(org_files)==len(new_files)}')

            
    # attack_mask        
    for i in tqdm(person_number) :
        for dir_name in kind:
            num_path = osp.join(data_path, dir_name, str(i))
            files = os.listdir(osp.join(num_path,'attack_mask'))
            files = [j for j in files if j.split('_')[0]=='pc']
            for j in files : 
                ply = osp.join(osp.join(num_path,'attack_mask',j))
                real = read_ply_xyzrgbnormal(ply)
                proj_xyzd = np.full((4, 192, 256), -1,dtype=np.float32)
                for k in range(len(real)):
                    proj_xyzd[0, int(real[k, 4]), int(real[k, 3])] = real[k, 0]
                    proj_xyzd[1, int(real[k, 4]), int(real[k, 3])] = real[k, 1]
                    proj_xyzd[2, int(real[k, 4]), int(real[k, 3])] = real[k, 2]
                    proj_xyzd[3, int(real[k, 4]), int(real[k, 3])] = real[k, 5]
                proj_xyzd = proj_xyzd.transpose(1,2,0)

                save_name = osp.join(save_path, dir_name, 'mask_cloud_data',str(i),j.split('.')[0]+'.npy')
                np.save(save_name, proj_xyzd)
        
    # data check
    for dir_name in kind:
        for i in person_number :
            # org_path = osp.join(data_path,str(i))
            num_path = osp.join(data_path, dir_name, str(i))
            org_files = os.listdir(osp.join(num_path,'attack_mask'))
            org_files = [j for j in org_files if j.split('_')[0]=='pc']

            new_path = osp.join(save_path, dir_name, 'mask_cloud_data', str(i))
            new_files = os.listdir(new_path)
            print(i,f' Org Data Count : {len(org_files)}  New Data Count : {len(new_files)} ==> {len(org_files)==len(new_files)}')

    # attack_replay
    for i in tqdm(person_number) :
        for dir_name in kind:
            num_path = osp.join(data_path, dir_name, str(i))
            files = os.listdir(osp.join(num_path,'attack_replay'))
            files = [j for j in files if j.split('_')[0]=='pc']
            for j in files : 
                ply = osp.join(osp.join(num_path,'attack_replay',j))
                real = read_ply_xyzrgbnormal(ply)
                proj_xyzd = np.full((4, 192, 256), -1,dtype=np.float32)
                for k in range(len(real)):
                    proj_xyzd[0, int(real[k, 4]), int(real[k, 3])] = real[k, 0]
                    proj_xyzd[1, int(real[k, 4]), int(real[k, 3])] = real[k, 1]
                    proj_xyzd[2, int(real[k, 4]), int(real[k, 3])] = real[k, 2]
                    proj_xyzd[3, int(real[k, 4]), int(real[k, 3])] = real[k, 5]
                proj_xyzd = proj_xyzd.transpose(1,2,0)

                save_name = osp.join(save_path, dir_name, 'replay_cloud_data',str(i),j.split('.')[0]+'.npy')
                np.save(save_name, proj_xyzd)

    # data check
    for dir_name in kind:
        for i in person_number :
            # org_path = osp.join(data_path,str(i))
            num_path = osp.join(data_path, dir_name, str(i))
            org_files = os.listdir(osp.join(num_path,'attack_replay'))
            org_files = [j for j in org_files if j.split('_')[0]=='pc']

            new_path = osp.join(save_path, dir_name, 'replay_cloud_data', str(i))
            new_files = os.listdir(new_path)
            print(i,f' Org Data Count : {len(org_files)}  New Data Count : {len(new_files)} ==> {len(org_files)==len(new_files)}')

    # attack_paper
    for i in tqdm(person_number) :
        for dir_name in kind:
            num_path = osp.join(data_path, dir_name, str(i))
            files = os.listdir(osp.join(num_path,'attack_paper'))
            files = [j for j in files if j.split('_')[0]=='pc']
            for j in files : 
                ply = osp.join(osp.join(num_path,'attack_paper',j))
                real = read_ply_xyzrgbnormal(ply)
                proj_xyzd = np.full((4, 192, 256), -1,dtype=np.float32)
                for k in range(len(real)):
                    proj_xyzd[0, int(real[k, 4]), int(real[k, 3])] = real[k, 0]
                    proj_xyzd[1, int(real[k, 4]), int(real[k, 3])] = real[k, 1]
                    proj_xyzd[2, int(real[k, 4]), int(real[k, 3])] = real[k, 2]
                    proj_xyzd[3, int(real[k, 4]), int(real[k, 3])] = real[k, 5]
                proj_xyzd = proj_xyzd.transpose(1,2,0)

                save_name = osp.join(save_path, dir_name, 'paper_cloud_data', str(i),j.split('.')[0]+'.npy')
                np.save(save_name, proj_xyzd)

    # data check
    for dir_name in kind:
        for i in person_number :
            # org_path = osp.join(data_path,str(i))
            num_path = osp.join(data_path, dir_name, str(i))
            org_files = os.listdir(osp.join(num_path,'attack_paper'))
            org_files = [j for j in org_files if j.split('_')[0]=='pc']

            new_path = osp.join(save_path, dir_name, 'paper_cloud_data',str(i))
            new_files = os.listdir(new_path)
            print(i,f' Org Data Count : {len(org_files)}  New Data Count : {len(new_files)} ==> {len(org_files)==len(new_files)}')

