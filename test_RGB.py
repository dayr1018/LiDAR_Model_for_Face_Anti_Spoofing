import time
import argparse
import os
import matplotlib

matplotlib.use('Agg')
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

from dataloader.dataloader_RGB import load_cisia_surf
# from data.dataloader_RGB_Depth import load_cisia_surf
# from dataloader.dataloader_RGB_Depth_IR import load_cisia_surf

from models.model_RGB import Model
# from models.model_RGB_Depth import Model
# from models.model_RGB_Depth_IR import Model

from loger import Logger
from evalution import eval_model
from centerloss import CenterLoss
from utils import plot_roc_curve, plot_eval_metric

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='face anti-spoofing test')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size')
parser.add_argument('--test-size', default='64', type=int, help='test batch size')
parser.add_argument('--save-path', default='./logs/RGB/Test/', type=str, help='logs save path')
parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')
parser.add_argument('--mode', default=1, type=int, help='dataset protocol_mode')
args = parser.parse_args()

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'

if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(args.message)

_, test_data = load_cisia_surf(train_size=args.batch_size,test_size=args.test_size, mode=args.mode)

eval_history = []

def val(epoch=0, data_set=test_data,flag=1, weight_dir=''):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = Model(pretrained=False, num_classes=2)

    logger.Print(f"weight dir: {weight_dir}")
    if use_cuda:
        model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) 
        model = model.cuda()
        model.load_state_dict(
            torch.load(weight_dir))

    y_true = []
    y_pred = []
    y_prob = []
    
    model.eval()   
  
    with torch.no_grad():
        for batch, data in enumerate(data_set, 1):

            rgb_img = data[0]
            labels = data[1]

            if use_cuda:
                rgb_img = rgb_img.to(device)
                labels = labels.to(device)

            # 예측 오류 계산
            outputs, features = model(rgb_img)
            _, pred_outputs = torch.max(outputs, 1)
            prob_outputs = F.softmax(outputs,1)[:,1]
            
            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())


    eval_result, score, acer = eval_model(y_true, y_pred, y_prob)
    eval_history.append(eval_result)
    logger.Print(eval_result)

    auc_value = plot_roc_curve(save_path, epoch, y_true, y_prob)
    plot_eval_metric(save_path, epoch, y_true, y_pred)

    logger.Print(f"auc_value : {auc_value}")

if __name__ == '__main__':

    for i in range(50):
        if (i+1) % 10 == 0:
            local_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/checkpoint/RGB/checkpoint_v' + str(args.mode) + '_0/' +'Cycle_' + str(i) + '_min_acer_model.pth'
            val(epoch=i, weight_dir=local_dir)