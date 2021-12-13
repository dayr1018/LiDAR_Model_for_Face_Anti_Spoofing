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

from evalution import eval_model
from centerloss import CenterLoss
from utils import plot_figure, plot_roc_curve, plot_eval_metric
from loger import Logger

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)
use_cuda = True if torch.cuda.is_available() else False

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size')
parser.add_argument('--test-size', default='64', type=int, help='test batch size')
parser.add_argument('--save-path', default='./logs/RGB_Depth_IR/Train/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='message', type=str, help='pretrained model checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='train epochs')
parser.add_argument('--train', default=True, type=bool, help='train')
parser.add_argument('--mode', default=1, type=int, help='dataset protocol_mode')
parser.add_argument('--tryout', default=0, type=int, help='dataset protocol_tryout')

args = parser.parse_args()

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'

if not os.path.exists(save_path):
    os.makedirs(save_path)
logger = Logger(f'{save_path}/logs.logs')
logger.Print(args.message)

train_data, test_data= load_cisia_surf(train_size=args.batch_size,test_size=args.test_size, mode=args.mode)

model = Model(pretrained=False,num_classes=2)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

ct_loss = CenterLoss(num_classes=2, feat_dim=512, use_gpu=use_cuda)
optimzer4ct = optim.SGD(ct_loss.parameters(), lr =0.01, momentum=0.9,weight_decay=5e-4)
scheduler4ct = lr_scheduler.ExponentialLR(optimzer4ct, gamma=0.95)

if use_cuda:
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    criterion = criterion.cuda()
    ct_loss = ct_loss.cuda()

eval_history = []
train_history = []
test_history = []

train_loss = []
eval_loss = []
train_score = []
eval_score = []
test_score = []

def train(epochs):
    out_dir = '../checkpoint/RGB_Depth_IR'

    if args.mode == 1:
        if not os.path.exists(out_dir +'/checkpoint_v1_'+str(args.tryout)):
            os.makedirs(out_dir +'/checkpoint_v1_'+str(args.tryout))
    elif args.mode == 2:
        if not os.path.exists(out_dir +'/checkpoint_v2_'+str(args.tryout)):
            os.makedirs(out_dir +'/checkpoint_v2_'+str(args.tryout))
    else:
        print("wrong mode!")

    global_min_acer = 1.0

    logger.Print(f"dataset info: mode v{args.mode}_{args.tryout}")
    logger.Print(f"max epochs : {args.epochs}")
    logger.Print(f"batch_size: {args.batch_size}, test_size: {args.test_size}")

    for epoch in range(epochs):
        min_acer = 1.0

        logger.Print(f"|---Training epoch:{epoch}---|")  
        print("Epoch " + str(epoch))   
        model.train()     

        y_true = []
        y_pred = []
        y_prob = []
        
        total_loss = 0
        total_batch = 0

        for batch, data in enumerate(train_data, 1):

            size = len(train_data.dataset)

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

            loss_anti = criterion(outputs, labels)
            loss_ct = ct_loss(features, labels)
            lamda = 0.001
            loss = loss_anti + lamda * loss_ct

            # 역전파 
            optimizer.zero_grad()
            optimzer4ct.zero_grad()
            loss.backward()
            optimizer.step()
            optimzer4ct.step()  

            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

            total_loss += loss.item()
            total_batch = batch

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] [{batch}:{len(data[0])}]")

        scheduler.step()
        scheduler4ct.step()

        eval_result, score, acer = eval_model(y_true, y_pred, y_prob)
        train_history.append(eval_result)
        train_score.append(score)        
        logger.Print(eval_result)
        avg_loss = total_loss/total_batch
        train_loss.append(avg_loss)

        if acer < min_acer and epoch > 0:
            min_acer = acer
            ckpt_name = out_dir + '/checkpoint_v'+str(args.mode)+'_'+str(args.tryout)+'/Cycle_' + str(epoch) + '_min_acer_model.pth'
            torch.save(model.state_dict(), ckpt_name)

        if acer < global_min_acer and epoch > 0:
            global_min_acer = acer
            ckpt_name = out_dir + '/checkpoint_v'+str(args.mode)+'_'+str(args.tryout)+'/global_min_acer_model.pth'
            torch.save(model.state_dict(), ckpt_name)
            logger.Print('save global min acer model: ' + str(min_acer) + '\n')

        message = f'|epoch:{epoch}-iter:{total_batch}|loss:{loss:.6f}|loss_anti:{loss_anti:.6f}|loss_ct:{loss_ct:.6f}'
        logger.Print(message)

        # if (epoch+1) % args.epochs == 0:    
            # plot_figure(save_path, train_loss, eval_loss)
        plot_roc_curve(save_path, epoch, y_true, y_prob)
        plot_eval_metric(save_path, epoch, y_true, y_pred)

    logger.Print(f"|---train history---|")
    for i in range(len(train_history)):
        logger.Print(train_history[i])

if __name__ == '__main__':
    torch.manual_seed(999)
    if args.train == True:
        train(args.epochs)