# python train.py --depth true --model layer4_4to4 --cuda 0 --gr 50 --dr 0.5 --message 0202_layer4_4to4_g50 

import os
import os.path as osp
import sys
import time 
from datetime import timedelta
from datetime import datetime
import random
import time
from tqdm import tqdm
from collections import Counter

import pandas as pd
import numpy as np
import argparse
from IPython.display import display
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.optim import lr_scheduler
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.AutoEncoder import AutoEncoder_RGB, AutoEncoder_Depth
from models.AutoEncoder import AutoEncoder_Intergrated_Basic, AutoEncoder_Intergrated_Proposed
from models.Network import Face_Detection_Model
from dataloader.dataloader import load_dataset, load_test_dataset
from utility import draw_train_and_test_loss, draw_accuracy_and_f1_during_training
from loger import Logger

def booltype(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError("Boolean value expected")

def model_save(model, epoch, optimizer, train_loss, val_loss, train_f1, valid_f1, path) :
    torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss' : val_loss,
                'train_f1' : train_f1,
                'val_f1' : valid_f1
                }, path)
    print('Model Save ! > ', path)

def train(args, train_loader, test_loader):

    # Tensorboard 
    global writer
    # writer = SummaryWriter(f"runs/{args.message}")

    # args 출력
    logger.Print(args)

    # Model 생성 및 아키텍쳐 출력 
    model = Face_Detection_Model(args.inputchannel).to(args.device)
    # summary(model)
    
    # Loss, 옵티마이저, 스케줄러 생성 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
    # Train 
    train_performs, test_performs = {'ACC':[],'F1':[]},{'ACC':[],'F1':[], 'Info':[]}

    total_train_loss = []
    total_test_loss = []
    start_epoch = 0
    epochs = args.epochs
    sigmoid = nn.Sigmoid()
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, epochs) :
        model.train()
        train_loss = []
        train_probs, train_labels = [],[]
        train_bar = tqdm(enumerate(train_loader))
        for step, data in train_bar :  
            rgb, cloud, depth, label = data
            rgb = rgb.float().to(args.device)
            cloud = cloud.float().to(args.device)
            depth = depth.float().to(args.device)
            label = label.float().to(args.device)

            features = rgb
            if args.model == "rgbd":
                features = torch.cat([rgb, depth], dim=1)
            elif args.model == "rgbp":
                features = torch.cat([rgb, cloud], dim=1)
            elif args.model == "rgbdp":
                features = torch.cat([rgb, depth, cloud], dim=1)  
     
            optimizer.zero_grad()
            logits, _ = model(features)
            
            logits = logits[:,0]  # logtit (4.2) 텐서이던데 첫째는 batch 이겠고. 근데 왜 0번째 값만 쓸까?
            loss = loss_fn(logits, label.float())
            loss.backward()
            optimizer.step()
            
            probs = sigmoid(logits)
            train_loss.append(loss.item())
            train_probs += probs.cpu().detach().tolist()
            train_labels += label.cpu().detach().tolist()
            
            train_bar.set_description("[Train] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch+1,epochs,step+1,len(train_loader),
                                                        round(loss.item(),5),round(np.array(train_loss).mean(),5)
                                                                ))
        
        logger.Print("[Train] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch+1,epochs,step+1,len(train_loader),
                                                        round(loss.item(),5),round(np.array(train_loss).mean(),5)))    
        total_train_loss.append(round(np.array(train_loss).mean(),5))
        train_acc = accuracy_score(np.array(train_labels), np.round(train_probs))
        train_f1 = f1_score(np.array(train_labels), np.round(train_probs), average='macro')
        logger.Print(f'Train Accuracy : {train_acc:.4f}')
        logger.Print(f'Train F1-score : {train_f1:.4f}')
        train_performs['ACC'].append(train_acc)
        train_performs['F1'].append(train_f1)
        logger.Print(f'  > Counter(train_labels) : {Counter(train_labels)}')
        logger.Print(f'  > Counter(train_preds) : {Counter(np.round(train_probs))}')
        cf = confusion_matrix(np.array(train_labels), np.round(train_probs))
        cf = pd.DataFrame(cf)
        cf.columns = ['Predicted:0','Predicted:1']
        cf.index = ['Label:0','Label:1']    
        logger.Print(' --- [Train] Confustion_Matrix & Classification_Report --- ')
        
        # display(cf)
        logger.Print(cf.to_string())
        report = classification_report(np.array(train_labels), np.round(train_probs))
        logger.Print(report)
        
        model.eval()
        test_loss = []
        test_probs, test_labels = [],[]
        test_bar = tqdm(enumerate(test_loader))
        for step, data in test_bar :  
            rgb, cloud, depth, label = data
            rgb = rgb.float().to(args.device)
            cloud = cloud.float().to(args.device)
            depth = depth.float().to(args.device)
            label = label.float().to(args.device)
            
            features = rgb
            if args.model == "rgbd":
                features = torch.cat([rgb, depth], dim=1)
            elif args.model == "rgbp":
                features = torch.cat([rgb, cloud], dim=1)
            elif args.model == "rgbdp":
                features = torch.cat([rgb, depth, cloud], dim=1)  

            logits,_ = model(features)
            logits = logits[:,0]
            loss = loss_fn(logits, label.float())
            
            probs = sigmoid(logits)
            test_loss.append(loss.item())
            test_probs += probs.cpu().detach().tolist()
            test_labels += label.cpu().detach().tolist()        
            test_bar.set_description("[Test] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch+1,epochs,step+1,len(test_loader),
                                                        round(loss.item(),5),round(np.array(test_loss).mean(),5)
                                                                ))
        logger.Print("[Test] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch+1,epochs,step+1,len(test_loader),
                                                        round(loss.item(),5),round(np.array(test_loss).mean(),5)))
        total_test_loss.append(round(np.array(test_loss).mean(),5))
        test_acc = accuracy_score(np.array(test_labels), np.round(test_probs))
        test_f1 = f1_score(np.array(test_labels), np.round(test_probs), average='macro')
        logger.Print(f'Test Accuracy : {test_acc:.4f}')
        logger.Print(f'Test F1-score : {test_f1:.4f}')
        test_performs['ACC'].append(test_acc)
        test_performs['F1'].append(test_f1)
        logger.Print(f'  > Counter(test_labels) : {Counter(test_labels)}')
        logger.Print(f'  > Counter(test_probs) : {Counter(np.round(test_probs))}')
        test_cf = confusion_matrix(np.array(test_labels), np.round(test_probs))
        test_cf = pd.DataFrame(test_cf)
        test_cf.columns = ['Predicted:0','Predicted:1']
        test_cf.index = ['Label:0','Label:1']    
        logger.Print(' --- [Test] Confustion_Matrix & Classification_Report --- ')
        # display(test_cf)
        logger.Print(test_cf.to_string())
        test_performs['Info'].append(test_cf.to_string())
        test_report = classification_report(np.array(test_labels), np.round(test_probs))
        logger.Print(test_report)
        
        
        if (epoch+1)%5 == 0:
            logger.Print(f'Saving Model ..... ')
            model_save(model,epoch+1,optimizer,np.array(train_loss).mean(),np.array(test_loss).mean(),
                    train_f1,test_f1,
                    osp.join(args.model_path, f"epoch_{epoch+1}_model"+'.pth'))  
                    
        # if best_test_f1 <= test_f1 :
        #     logger.Print(' @@ New Best test_f1 !! @@ ')
        #     best_test_f1 = test_f1 
        #     best_test_epoch = epoch
        #     logger.Print(f' @@ Best Test Accuracy : {np.array(test_performs["ACC"]).max()}')
        #     logger.Print(f' @@ Best Test F1-Score : {best_test_f1}')
        #     logger.Print(f' @@ Best Test Epoch : {epoch}')
        #     logger.Print(f'Saving Model ..... ')
        #     model_save(model,epoch+1,optimizer,np.array(train_loss).mean(),np.array(test_loss).mean(),
        #             train_f1,test_f1,
        #             osp.join(args.model_path, f"epoch_{epoch}_model"+'.pth'))          
    
    draw_train_and_test_loss(args, total_train_loss, total_test_loss)
    draw_accuracy_and_f1_during_training(args, test_performs["ACC"], test_performs["F1"])
    
    accu_max = np.array(test_performs["ACC"]).max()
    accu_index = test_performs["ACC"].index(accu_max)
    f1_max = np.array(test_performs["F1"]).max()
    f1_index = test_performs["F1"].index(f1_max)
            
    logger.Print(' @@ THE END @@ ')
    logger.Print(f'  > Train Best Accuracy : {np.array(train_performs["ACC"]).max():.4f}')
    logger.Print(f'  > Train Best F1-Score : {np.array(train_performs["F1"]).max():.4f}')
    logger.Print(f'  > Test Best Accuracy : {np.array(test_performs["ACC"]).max():.4f}')
    logger.Print(f'  > Test Epoch (Best Accuracy): {accu_index+1}')   
    logger.Print(f'  > Test Best F1-Score : {np.array(test_performs["F1"]).max():.4f}')
    logger.Print(f'  > Test Epoch (Best F1-Score): {f1_index+1}')
    logger.Print(f'  > Test Best CF (std: Accuracy') 
    logger.Print(f'  > {np.array(test_performs["Info"][accu_index])}')
    logger.Print(f'  > Test Best CF (std: F1') 
    logger.Print(f'  > {np.array(test_performs["Info"][f1_index])}')
    logger.Print(f'')

if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')        
    parser.add_argument('--trainratio', default=1.0, type=float, help='ratio to divide train dataset')                               
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
    
    parser.add_argument('--attacktype', default='prm', type=str, help='Kinds of Presentation Attacks: r, p, m, prm')
    parser.add_argument('--dataset', default=12, type=int, help='dataset type: 12 or 15')
    parser.add_argument('--model', default='', type=str, help='rgb, rgbd, rgbp, rgbdp')   
    parser.add_argument('--inputchannel', default=3, type=int, help='inputchannel')
    parser.add_argument('--crop', default=False, type=booltype, help='use crop (default: False)')
    
    parser.add_argument('--ae-path', default='', type=str, help='Pretrained AutoEncoder path')
    parser.add_argument('--save-path', default='../bc_output/logs/', type=str, help='train logs path')
    parser.add_argument('--model-path', default='', type=str, help='model parameter path')
    parser.add_argument('--message', default='', type=str, help='parameter file name')                     

    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                         
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')
                                                  
    args = parser.parse_args()

    if args.model == 'rgb':
        args.inputchannel = 3
    elif args.model == 'rgbd':
        args.inputchannel = 4
    elif args.model == 'rgbp':
        args.inputchannel = 6
    elif args.model == 'rgbdp':
        args.inputchannel = 7
    else:
        print("You need to checkout option 'model' [rgb, rgbd, rgbp, rgbpc]")
        sys.exit(0)
        
    # 결과 파일 path 설정 
    args.save_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/logs'
    args.save_path = osp.join(args.save_path, args.message)
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)    
    
    # weight 파일 path
    args.model_path = f'/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    if not os.path.exists(args.model_path): 
        os.makedirs(args.model_path)

    # cuda 관련 코드
    use_cuda = True if torch.cuda.is_available() else False
    if use_cuda : 
        if args.cuda == 0:
            args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        elif args.cuda == 1:
            args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        print('device :', args.device)
    
    # random 요소 없애기 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    # logger 
    global logger 
    logger = Logger(f'{args.save_path}/Train_logs.logs')

    # Autoencoder's path
    # RGB, Depth
    # args.ae_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/0421_Both_3_dr01_gr0/epoch_90_model.pth"
    # RGB 
    # args.ae_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/0421_RGB_3_dr0_gr001/epoch_10_model.pth"

    # Pretrained 된 AutoEncoder 생성 (layer3)
    # global autoencoder
    # autoencoder = AutoEncoder_RGB(3, False, 0.1).to(args.device)
    # ### autoencoder = AutoEncoder_Intergrated_Basic(3, False, 0.1).to(args.device)
    # autoencoder.load_state_dict(torch.load(args.ae_path))
    # autoencoder.eval()

    train_dataset, test_dataset = load_dataset(args)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=args.workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)

    # train 코드
    train_start = time.time()
    train(args, train_loader, test_loader)

    train_time = str(timedelta(seconds=time.time()-train_start)).split(".")
    logger.Print(f"Train Execution Time: {train_time}")  