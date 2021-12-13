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

# from data.dataloader_RGB import load_cisia_surf
# from data.dataloader_RGB_Depth import load_cisia_surf
from dataloader.dataloader_RGB_Depth_IR import load_cisia_surf

# from models.model_RGB import Model
# from models.model_RGB_Depth import Model
from models.model_RGB_Depth_IR import Model

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
parser.add_argument('--save-path', default='./logs/RGB_Depth_IR/Test/', type=str, help='logs save path')
parser.add_argument('--message', default='test', type=str, help='pretrained model checkpoint')
parser.add_argument('--mode', default=1, type=int, help='dataset protocol_mode')
args = parser.parse_args()

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'

if not os.path.exists(save_path):
    os.makedirs(save_path)
   #os.mkdir(save_path)
logger = Logger(f'{save_path}/logs.logs')
logger.Print(args.message)

_, test_data = load_cisia_surf(train_size=args.batch_size,test_size=args.test_size, mode=args.mode)

eval_history = []
eval_loss = []
eval_score = []
test_score = []

def val(epoch=0, data_set=test_data,flag=1, weight_dir=''):

    model = Model(pretrained=False, num_classes=2)
    criterion = nn.CrossEntropyLoss()
    ct_loss = CenterLoss(num_classes=2, feat_dim=512, use_gpu=use_cuda)

    logger.Print(f"weight dir: {weight_dir}")
    if use_cuda:
        model = torch.nn.DataParallel(model)
        model = model.cuda()
        model.load_state_dict(
            torch.load(weight_dir))

        criterion = criterion.cuda()
        ct_loss = ct_loss.cuda()

    y_true = []
    y_pred = []
    y_prob = []
    
    model.eval()

    total_loss = 0
    total_batch = 0

    with open(save_path+f'/prob_{epoch}.txt', 'w') as fb:
        with torch.no_grad():
            for batch, data in enumerate(data_set, 1):

                rgb_img = data[0]
                depth_img = data[1]
                ir_img = data[2]
                hsv_img = data[3]
                ycb_img = data[4]
                labels = data[5]

                if use_cuda:
                    rgb_img = rgb_img.cuda()
                    depth_img = depth_img.cuda()
                    ir_img = ir_img.cuda()
                    hsv_img = hsv_img.cuda()
                    ycb_img = ycb_img.cuda()
                    labels = labels.cuda()

                
                # 예측 오류 계산
                outputs, features = model(rgb_img, depth_img, ir_img, hsv_img, ycb_img)
                _, pred_outputs = torch.max(outputs, 1)
                prob_outputs = F.softmax(outputs,1)[:,1]
                
                loss_anti = criterion(outputs,labels)
                loss_ct = ct_loss(features,labels)
                lamda = 0.001
                loss = loss_anti + lamda * loss_ct

                y_true.extend(labels.data.cpu().numpy())
                y_pred.extend(pred_outputs.data.cpu().numpy())
                y_prob.extend(prob_outputs.data.cpu().numpy())
                
                total_loss += loss.item()
                total_batch = batch

        fb.close()

    eval_result, score, acer = eval_model(y_true, y_pred, y_prob)
    eval_history.append(eval_result)
    logger.Print(eval_result)

    plot_roc_curve(save_path, epoch, y_true, y_prob)
    plot_eval_metric(save_path, epoch, y_true, y_pred)

    if flag == 0 :
        eval_score.append(score)
        avg_loss = total_loss/total_batch
        eval_loss.append(avg_loss)
        message = f'|eval|loss:{avg_loss:.6f}|'
        logger.Print(message)
    else:
        test_score.append(score)

    with open(save_path+f'/val_{epoch}.txt', 'w') as f:
        for i in range(len(y_true)):
            message = f'{y_prob[i]:.6f} {y_pred[i]} {y_true[i]}'
            f.write(message)
            f.write('\n')
        f.close()

if __name__ == '__main__':
    global_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/code/models/output/RGB_Depth_IR/checkpoint_v'+ str(args.mode) +'_0/global_min_acer_model.pth'
    print("--global_dir start--")
    val(epoch=49, weight_dir=global_dir)
    print("--global_dir end--")

    local_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/code/models/output/RGB_Depth_IR/checkpoint_v' + str(args.mode) + '_0/' +'Cycle_1_min_acer_model.pth'
    print("--local_dir_1 start--")
    val(epoch=1, weight_dir=local_dir)
    print("--local_dir_1 end--")

    local_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/code/models/output/RGB_Depth_IR/checkpoint_v' + str(args.mode) + '_0/' +'Cycle_25_min_acer_model.pth'
    print("--local_dir_25 start--")
    val(epoch=25, weight_dir=local_dir)
    print("--local_dir_25 end--")

    local_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/code/models/output/RGB_Depth_IR/checkpoint_v' + str(args.mode) + '_0/' +'Cycle_49_min_acer_model.pth'
    print("--local_dir_49 start--")
    val(epoch=49, weight_dir=local_dir)
    print("--local_dir_49 end--")