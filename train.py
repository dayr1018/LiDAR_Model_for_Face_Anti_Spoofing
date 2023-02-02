# python train.py --depth true --model layer4_4to4 --cuda 0 --gr 50 --dr 0.5 --message 0202_layer4_4to4_g50 

# from curses import init_color
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
import pickle
from IPython.display import display
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary

from models.Network import Face_Detection_Model, pointcloud_model, depth_model, rgbp_v1_twostep_model, rgbp_v2_twostep_model, rgbd_v1_twostep_model, rgbd_v2_twostep_model, rgbdp_v1_twostep_model, rgbdp_v2_twostep_model, rgbdp_v3_twostep_model
# models 
from dataloader.dataloader import load_dataset, load_test_dataset
from utility import draw_train_and_test_loss, draw_train_and_test_loss1, draw_accuracy_and_f1_during_training, draw_accuracy_and_f1_during_training1
from loger import Logger

use_saveinit = False

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

def train(args, train_loader, test_loader, outdoor_loader, dark_loader):

    # Tensorboard 
    global writer
    writer = SummaryWriter(f"runs/{args.message}")           
            
    # args 출력
    logger.Print(args)

    # Model 생성 및 아키텍쳐 출력   
    model=""
    if args.model == "rgb":
        model = Face_Detection_Model(3).to(args.device) 
        # model = rgb_model(device=args.device)         
    elif args.model == "pointcloud":
        model = pointcloud_model(device=args.device)   
    elif args.model == "depth":
        model = depth_model(device=args.device)    
    elif args.model == "rgbp_v1":
        model = rgbp_v1_twostep_model(device=args.device)
    elif args.model == "rgbp_v2":
        model = rgbp_v2_twostep_model(device=args.device)
    elif args.model == "rgbd_v1":
        model = rgbd_v1_twostep_model(device=args.device)
    elif args.model == "rgbd_v2":
        model = rgbd_v2_twostep_model(device=args.device)
    elif args.model == "rgbdp_v1":
        model = rgbdp_v1_twostep_model(device=args.device)
    elif args.model == "rgbdp_v2":
        model = rgbdp_v2_twostep_model(device=args.device)
    elif args.model == "rgbdp_v3":
        model = rgbdp_v3_twostep_model(device=args.device)
    elif args.model == "mobilenet_v3":
        model = mobile(device=args.device)
    # summary(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model.to(device)

    # Loss, 옵티마이저, 스케줄러 생성 
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # if args.scheduler:
    #     scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)
        
    # Train 
    train_performs = {'ACC':[],'F1':[]}
    test_performs = {'ACC':[],'F1':[], 'Info':[]}
    outdoor_performs = {'ACC':[],'F1':[], 'Info':[]}
    dark_performs = {'ACC':[],'F1':[], 'Info':[]}

    # Variables 
    total_train_loss = []
    total_test_loss = []
    total_outdoor_loss = []
    total_dark_loss = []
    
    epochs = args.epochs + 1
    sigmoid = nn.Sigmoid()
    loss_fn = nn.BCEWithLogitsLoss()

    indoor_check = {'train_loss':[],'test_loss':[],'train_f1':[],'test_f1':[],'train_acc':[],'test_acc':[]}
    outdoor_check  = {'test_loss':[],'test_f1':[],'test_acc':[]}
    dark_check = {'test_loss':[],'test_f1':[],'test_acc':[]}

    if use_saveinit == True:
        model_save(model,0,optimizer,0,0,0,0,osp.join(args.model_path, f"{args.message}.pth")) 

    for epoch in range(1, epochs) :
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
            
            optimizer.zero_grad() 
            if args.model == "rgb":
                logits = model(rgb)
            elif args.model == "pointcloud":
                logits = model(cloud)  
            elif args.model == "depth":
                logits = model(depth)   
            elif args.model in ["rgbp_v1", "rgbp_v2"]:
                logits = model(rgb, cloud)
            elif args.model in ["rgbd_v1", "rgbd_v2"]:
                logits = model(rgb, depth)
            elif args.model in ["rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
                logits = model(rgb, depth, cloud)
            
            logits = logits[:,0]  # logtit (4.2) 텐서이던데 첫째는 batch 이겠고. 근데 왜 0번째 값만 쓸까?
            loss = loss_fn(logits, label.float())
            loss.backward()
            optimizer.step()
            
            probs = sigmoid(logits)
            train_loss.append(loss.item())
            train_probs += probs.cpu().detach().tolist()
            train_labels += label.cpu().detach().tolist()
            
            train_bar.set_description("[Train] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(train_loader),
                                                        round(loss.item(),5),round(np.array(train_loss).mean(),10)
                                                                ))
        # if args.scheduler:
        #     scheduler.step()        
        train_loss_mean = round(np.array(train_loss).mean(), 10)        
        logger.Print("[Train] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(train_loader),
                                                        round(loss.item(),5),train_loss_mean))    
        total_train_loss.append(train_loss_mean)
        train_acc = accuracy_score(np.array(train_labels), np.round(train_probs))
        train_f1 = f1_score(np.array(train_labels), np.round(train_probs), average='macro')
        
        writer.add_scalar("(Train) Train Loss ", train_loss_mean, epoch)
        writer.add_scalar("(Train) Accuracy ", train_acc, epoch)
        writer.add_scalar("(Train) F1 Score ", train_f1, epoch)
        
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
        
        indoor_check['train_loss'].append(train_loss_mean)
        indoor_check['train_f1'].append(train_f1)
        indoor_check['train_acc'].append(train_acc)
            

        # # Comment Start
        # # Test Start     
        # model.eval()
    
        # # Indoor test
        # test_loss = []
        # test_probs, test_labels = [],[]
        # test_bar = tqdm(enumerate(test_loader))
        # for step, data in test_bar :  
        #     rgb, cloud, depth, label = data
        #     rgb = rgb.float().to(args.device)
        #     cloud = cloud.float().to(args.device)
        #     depth = depth.float().to(args.device)
        #     label = label.float().to(args.device)
            
        #     if args.model == "rgb":
        #         logits = model(rgb)
        #     elif args.model == "pointcloud":
        #         logits = model(cloud)  
        #     elif args.model == "depth":
        #         logits = model(depth)   
        #     elif args.model in ["rgbp_v1", "rgbp_v2"]:
        #         logits = model(rgb, cloud)
        #     elif args.model in ["rgbd_v1", "rgbd_v2"]:
        #         logits = model(rgb, depth)
        #     elif args.model in ["rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
        #         logits = model(rgb, depth, cloud)            
   
        #     logits = logits[:,0]
        #     loss = loss_fn(logits, label.float())
            
        #     probs = sigmoid(logits)
        #     test_loss.append(loss.item())
        #     test_probs += probs.cpu().detach().tolist()
        #     test_labels += label.cpu().detach().tolist()        
        #     test_bar.set_description("[Test] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(test_loader),
        #                                                 round(loss.item(),5),round(np.array(test_loss).mean(),5)
        #                                                         ))
            
        # indoor_check['test_loss'].append(np.array(test_loss).mean())
        # indoor_check['test_f1'].append(f1_score(np.array(test_labels), np.round(test_probs), average='macro'))
        # indoor_check['test_acc'].append(accuracy_score(np.array(test_labels), np.round(test_probs)))
                     
        # # Outdoor test
        # outdoor_loss = []
        # outdoor_probs, outdoor_labels = [], []
        # outdoor_bar = tqdm(enumerate(outdoor_loader))
        # for step, data in outdoor_bar :  
        #     rgb, cloud, depth, label = data
        #     rgb = rgb.float().to(args.device)
        #     cloud = cloud.float().to(args.device)
        #     depth = depth.float().to(args.device)
        #     label = label.float().to(args.device)

        #     if args.model == "rgb":
        #         logits = model(rgb)
        #     elif args.model == "pointcloud":
        #         logits = model(cloud)  
        #     elif args.model == "depth":
        #         logits = model(depth)   
        #     elif args.model in ["rgbp_v1", "rgbp_v2"]:
        #         logits = model(rgb, cloud)
        #     elif args.model in ["rgbd_v1", "rgbd_v2"]:
        #         logits = model(rgb, depth)
        #     elif args.model in ["rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
        #         logits = model(rgb, depth, cloud)   
                
        #     logits = logits[:,0]
        #     loss = loss_fn(logits, label.float())
            
        #     probs = sigmoid(logits)
        #     outdoor_loss.append(loss.item())
        #     outdoor_probs += probs.cpu().detach().tolist()
        #     outdoor_labels += label.cpu().detach().tolist()        
        #     outdoor_bar.set_description("[Outdoor] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(test_loader),
        #                                                 round(loss.item(),5),round(np.array(outdoor_loss).mean(),5)
        #                                                         ))
        
        # outdoor_check['test_loss'].append(np.array(outdoor_loss).mean())
        # outdoor_check['test_f1'].append(f1_score(np.array(outdoor_labels), np.round(outdoor_probs), average='macro'))
        # outdoor_check['test_acc'].append(accuracy_score(np.array(outdoor_labels), np.round(outdoor_probs)))
          
        # # Dark test
        # dark_loss = []
        # dark_probs, dark_labels = [], []  
        # dark_bar = tqdm(enumerate(dark_loader))
        # for step, data in dark_bar :  
        #     rgb, cloud, depth, label = data
        #     rgb = rgb.float().to(args.device)
        #     cloud = cloud.float().to(args.device)
        #     depth = depth.float().to(args.device)
        #     label = label.float().to(args.device)

        #     if args.model == "rgb":
        #         logits = model(rgb)
        #     elif args.model == "pointcloud":
        #         logits = model(cloud)  
        #     elif args.model == "depth":
        #         logits = model(depth)   
        #     elif args.model in ["rgbp_v1", "rgbp_v2"]:
        #         logits = model(rgb, cloud)
        #     elif args.model in ["rgbd_v1", "rgbd_v2"]:
        #         logits = model(rgb, depth)
        #     elif args.model in ["rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
        #         logits = model(rgb, depth, cloud)   
                
        #     logits = logits[:,0]
        #     loss = loss_fn(logits, label.float())
            
        #     probs = sigmoid(logits)
        #     dark_loss.append(loss.item())
        #     dark_probs += probs.cpu().detach().tolist()
        #     dark_labels += label.cpu().detach().tolist()        
        #     dark_bar.set_description("[Dark] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(test_loader),
        #                                                 round(loss.item(),5),round(np.array(dark_loss).mean(),5)
        #                                                         ))
                
        # dark_check['test_loss'].append(np.array(dark_loss).mean())
        # dark_check['test_f1'].append(f1_score(np.array(dark_labels), np.round(dark_probs), average='macro'))
        # dark_check['test_acc'].append(accuracy_score(np.array(dark_labels), np.round(dark_probs)))          
            
        # logger.Print("[Dark] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch,epochs-1,step+1,len(test_loader),
        #                                                 round(loss.item(),5),round(np.array(test_loss).mean(),5)))
        
        # test_loss_mean = round(np.array(test_loss).mean(),10)
        # outdoor_loss_mean = round(np.array(outdoor_loss).mean(),10)
        # dark_loss_mean = round(np.array(dark_loss).mean(),10)
        
        # total_test_loss.append(test_loss_mean)
        # total_outdoor_loss.append(outdoor_loss_mean)
        # total_dark_loss.append(dark_loss_mean)

        # test_acc = accuracy_score(np.array(test_labels), np.round(test_probs))
        # test_f1 = f1_score(np.array(test_labels), np.round(test_probs), average='macro')
        # outdoor_acc = accuracy_score(np.array(outdoor_labels), np.round(outdoor_probs))
        # outdoor_f1 = f1_score(np.array(outdoor_labels), np.round(outdoor_probs), average='macro')
        # dark_acc = accuracy_score(np.array(dark_labels), np.round(dark_probs))
        # dark_f1 = f1_score(np.array(dark_labels), np.round(dark_probs), average='macro')        
                
        # writer.add_scalar("(Test) Loss ", test_loss_mean, epoch)
        # writer.add_scalar("(Test) Accuracy ", test_acc, epoch)
        # writer.add_scalar("(Test) F1 Score ", test_f1, epoch)
        # writer.add_scalar("(Outdoor) Loss ", outdoor_loss_mean, epoch)
        # writer.add_scalar("(Outdoor) Accuracy ", outdoor_acc, epoch)
        # writer.add_scalar("(Outdoor) F1 Score ", outdoor_f1, epoch)
        # writer.add_scalar("(Dark) Loss ", dark_loss_mean, epoch)
        # writer.add_scalar("(Dark) Accuracy ", dark_acc, epoch)
        # writer.add_scalar("(Dark) F1 Score ", dark_f1, epoch)       
        
        # logger.Print(f'Test Accuracy : {test_acc:.4f}')
        # logger.Print(f'Test F1-score : {test_f1:.4f}')
        
        # test_performs['ACC'].append(test_acc)
        # test_performs['F1'].append(test_f1)
        # outdoor_performs['ACC'].append(outdoor_acc)
        # outdoor_performs['F1'].append(outdoor_f1)
        # dark_performs['ACC'].append(dark_acc)
        # dark_performs['F1'].append(dark_f1)
        
        # # test_cf = confusion_matrix(np.array(test_labels), np.round(test_probs))
        # # test_cf = pd.DataFrame(test_cf)
        # # test_cf.columns = ['Predicted:0','Predicted:1']
        # # test_cf.index = ['Label:0','Label:1']    
        # # logger.Print(' --- [Test] Confustion_Matrix & Classification_Report --- ')
        # # # display(test_cf)
        # # logger.Print(test_cf.to_string())
        # # test_performs['Info'].append(test_cf.to_string())
        # # test_report = classification_report(np.array(test_labels), np.round(test_probs))
        # # logger.Print(test_report)   

        # # 모두 다 저장 
        # logger.Print(' @@ New Save Situation !! @@ ')  
        # logger.Print(f' @@ New Current Epoch : {epoch}')    
        # logger.Print(f' ')
        # logger.Print(f' @@ New Train Accuracy : {train_acc:.4f}')
        # logger.Print(f' @@ New Train F1-Score : {train_f1:.4f}')
        # logger.Print(f' ')
        # logger.Print(f' @@ New Test Accuracy : {test_acc:.4f}')
        # logger.Print(f' @@ New Test F1-Score : {test_f1:.4f}')
        # logger.Print(f' ')
        # logger.Print(f' @@ New Outdoor Accuracy : {outdoor_acc:.4f}')
        # logger.Print(f' @@ New Outdoor F1-Score : {outdoor_f1:.4f}')
        # logger.Print(f' ')
        # logger.Print(f' @@ New Dark Accuracy : {dark_acc:.4f}')
        # logger.Print(f' @@ New Dark F1-Score : {dark_f1:.4f}')
        # logger.Print(f' ')
        # logger.Print(f' @@ New Train Loss : {train_loss_mean:}') 
        # logger.Print(f' @@ New Test Loss : {test_loss_mean}') 
        # logger.Print(f' @@ New Outdoor Loss : {outdoor_loss_mean}')
        # logger.Print(f' @@ New Dark Loss : {dark_loss_mean}')
        # logger.Print(f' ')

        # Comment End 
          
        logger.Print(f'Saving Model ..... ')
        
        model_save(model,epoch,optimizer,np.array(train_loss).mean(), 0,
                train_f1, 0, osp.join(args.model_path, f"epoch_{epoch}_model"+'.pth'))                             
        # model_save(model,epoch,optimizer,np.array(train_loss).mean(),np.array(test_loss).mean(),
        #         train_f1,test_f1, osp.join(args.model_path, f"epoch_{epoch}_model"+'.pth'))    

                
    logger.Print(' @@ THE END @@ ')
    logger.Print(f'  > Train Best Accuracy : {np.array(indoor_check["train_acc"]).max()}')
    logger.Print(f'  > Train Best F1-Score : {np.array(indoor_check["train_f1"]).max()}') 
    # logger.Print(f'  > Test Best Accuracy : {np.array(indoor_check["test_acc"]).max()}')
    # logger.Print(f'  > Test Best F1-Score : {np.array(indoor_check["test_f1"]).max()}')
    # logger.Print(f'  > Outdoor Best Accuracy : {np.array(outdoor_check["test_acc"]).max()}')
    # logger.Print(f'  > Outdoor Best F1-Score : {np.array(outdoor_check["test_f1"]).max()}')    
    # logger.Print(f'  > Dark Best Accuracy : {np.array(dark_check["test_acc"]).max()}')
    # logger.Print(f'  > Dark Best F1-Score : {np.array(dark_check["test_f1"]).max()}')
    logger.Print(f'')
    
    indoor_check_file = f"{args.save_path}/{args.message}_indoor.pkl"
    outdoor_check_file = f"{args.save_path}/{args.message}_outdoor.pkl"
    dark_check_file = f"{args.save_path}/{args.message}_dark.pkl"    
    
    with open(indoor_check_file,'wb') as f:
        pickle.dump(indoor_check,f)
    with open(outdoor_check_file,'wb') as f:
        pickle.dump(outdoor_check,f)
    with open(dark_check_file,'wb') as f:
        pickle.dump(dark_check,f)       
        
    writer.close()
    
    
if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of workers, 4')
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')        
    parser.add_argument('--trainratio', default=1.0, type=float, help='ratio to divide train dataset')                               
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
    parser.add_argument('--attacktype', default='prm', type=str, help='Kinds of Presentation Attacks: r, p, m, prm')
    parser.add_argument('--dataset', default=12, type=int, help='dataset type: 12 or 15')
    parser.add_argument('--model', default='', type=str, help='rgb, rgbd, rgbp, rgbdp')   
    parser.add_argument('--inputchannel', default=3, type=int, help='inputchannel')
    parser.add_argument('--crop', default=False, type=booltype, help='use crop (default: False)')
    parser.add_argument('--scheduler', default=False, type=booltype, help='use scheduelr (default: False)')
    parser.add_argument('--ae-path', default='', type=str, help='Pretrained AutoEncoder path')
    parser.add_argument('--save-path', default='../bc_output/logs/', type=str, help='train logs path')
    parser.add_argument('--model-path', default='', type=str, help='model parameter path')
    parser.add_argument('--init-path', default='', type=str, help='model init parameter path')
    parser.add_argument('--message', default='', type=str, help='parameter file name')                     
    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                         
    parser.add_argument('--device', default='cuda', type=str, help='device when cuda is available')                       
    args = parser.parse_args()
    
    if args.model not in ["rgb", "pointcloud", "depth", "rgbd_v1", "rgbd_v2", "rgbp_v1", "rgbp_v2", "rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
        print("You need to checkout option 'model' [rgb, rgbd, rgbp, rgbpc]")
        sys.exit(0)
        
    # 결과 파일 path 설정 
    args.save_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/logs'
    args.save_path = osp.join(args.save_path, args.message)
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)    
    
    # logger 
    global logger 
    logger = Logger(f'{args.save_path}/Train_logs.logs')
    
    # weight 파일 path
    args.model_path = f'/mnt/nas4/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    if not os.path.exists(args.model_path): 
        os.makedirs(args.model_path)
        
    # (pointcloud, depth, rgbp_v1, rgbp_v2) uses this.
    # cuda 관련 코드
    # use_cuda = True if torch.cuda.is_available() else False
    # if use_cuda : 
    #     if args.cuda == 0:
    #         args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #     elif args.cuda == 1:
    #         args.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    #     print('device :', args.device)

    # random 요소 없애기 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    def seed_worker(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    train_dataset, test_dataset = load_dataset(args)
    outdoor_testset = load_test_dataset(args, "2. Outdoor")
    indoor_dark_testset = load_test_dataset(args, "3. Indoor_dark")

    train_loader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    outdoor_loader = DataLoader(outdoor_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    dark_loader = DataLoader(indoor_dark_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    
    # train 코드
    train_start = time.time()
    train(args, train_loader, test_loader, outdoor_loader, dark_loader)

    train_time = str(timedelta(seconds=time.time()-train_start)).split(".")
    logger.Print(f"Train Execution Time: {train_time}")  