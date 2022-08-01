import os
import os.path as osp
import sys
import random
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
from collections import Counter
import argparse

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch
import torch.nn as nn

from models.Network import Face_Detection_Model
from dataloader.dataloader import load_dataset, load_test_dataset
from utility import draw_accuracy_during_test, draw_f1_during_test
from loger import Logger

def test(args, dataloader):
    
    # args 출력
    logger.Print(args)
  
    # Test 시작 
    test_performs = {'ACC':[], 'F1':[], 'Info':[]}

    start_epoch = 0
    epochs = args.epochs
    sigmoid = nn.Sigmoid()
    loss_fn = nn.BCEWithLogitsLoss()  
    
    for epoch in range(start_epoch+1, epochs+1) :
        if (epoch%5) != 0:
            continue
        
        model_path = osp.join(args.model_path, f"epoch_{epoch}_model.pth")

        # 모델 생성 
        ckpt = torch.load(model_path)
        model = Face_Detection_Model(args.inputchannel)
        model.load_state_dict(ckpt['model_state_dict'])
        model.to(args.device)
        model.eval()
        
        test_loss = []
        test_probs, test_labels = [],[]
        test_bar = tqdm(enumerate(dataloader))
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
            test_bar.set_description("[Test] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch, epochs, step+1,len(dataloader),
                                                        round(loss.item(),5),round(np.array(test_loss).mean(),5)
                                                                ))
 
        test_cf = confusion_matrix(np.array(test_labels), np.round(test_probs))
        test_cf = pd.DataFrame(test_cf)
        test_cf.columns = ['Predicted:0','Predicted:1']
        test_cf.index = ['Label:0','Label:1']    
        logger.Print(f' --- [Test] Confustion_Matrix & Classification_Report --- Epoch {epoch}')
        # display(test_cf)
        logger.Print(test_cf.to_string())
        test_performs['Info'].append(test_cf.to_string())

        test_acc = accuracy_score(np.array(test_labels), np.round(test_probs))
        test_f1 = f1_score(np.array(test_labels), np.round(test_probs), average='macro')  
        logger.Print(f'Test Accuracy : {test_acc:.4f}')
        logger.Print(f'Test F1-score : {test_f1:.4f}')   
        
        test_performs['ACC'].append(test_acc)  
        test_performs['F1'].append(test_f1)  
        print(f"!!!!!!!!!!! Test Accuracy : {test_acc}")

    accu_max = np.array(test_performs["ACC"]).max()
    accu_index = test_performs["ACC"].index(accu_max)
    f1_max = np.array(test_performs["F1"]).max()
    f1_index = test_performs["F1"].index(f1_max)

    logger.Print(f'  > Test Best Accuracy : {np.array(test_performs["ACC"]).max():.4f}')
    logger.Print(f'  > Test Epoch (Best Accuracy): {accu_index*5}')   
    logger.Print(f'  > Test Best F1-Score : {np.array(test_performs["F1"]).max():.4f}')
    logger.Print(f'  > Test Epoch (Best F1-Score): {f1_index*5}')
    logger.Print(f'  > Test Best CF (std: Accuracy)') 
    logger.Print(f'  > {np.array(test_performs["Info"][accu_index])}')
    logger.Print(f'  > Test Best CF (std: F1)') 
    logger.Print(f'  > {np.array(test_performs["Info"][f1_index])}')
    logger.Print(f'')
    
    return test_performs['ACC'], test_performs['F1']
           
    

if __name__ == "__main__":
    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    
    parser.add_argument('--batchsize', default=4, type=int, help='batch size')
    parser.add_argument('--workers', default=4, type=int, help='number of workers')
    parser.add_argument('--epochs', default=100, type=int, help='train epochs')        
    parser.add_argument('--trainratio', default=1.0, type=float, help='ratio to divide train dataset')                               
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    
    parser.add_argument('--attacktype', default='prm', type=str, help='Kinds of Presentation Attacks: r, p, m, prm')
    parser.add_argument('--model', default='', type=str, help='rgb, rgbd, rgbp, rgbdp')   
    parser.add_argument('--inputchannel', default=3, type=int, help='inputchannel')
    # parser.add_argument('--crop', default=False, type=booltype, help='use crop (default: False)')
    
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
        print("You need to checkout correct 'model path' (Modify --message)")
        sys.exit(0)   
    
    # weight 파일 path
    args.model_path = f'/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    if not os.path.exists(args.model_path): 
        print(f"No Such Directory : {args.model_path}")
        sys.exit(0)  

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
    logger = Logger(f'{args.save_path}/Test_logs.logs')

    # Dataset 불러오기     
    _, indoor_testset = load_dataset(args)
    outdoor_testset = load_test_dataset(args, "2. Outdoor")
    indoor_dark_testset = load_test_dataset(args, "3. Indoor_dark")
    
    # 세 가지 경우의 Test Data Loader 
    type1_testset = indoor_testset
    type2_testset = outdoor_testset
    type3_testset = indoor_dark_testset
    type4_testset = torch.utils.data.ConcatDataset([indoor_testset, outdoor_testset])
    type5_testset = torch.utils.data.ConcatDataset([indoor_testset, indoor_dark_testset])
    
    print(f"type1_testset : {len(type1_testset)}")
    print(f"type2_testset : {len(type2_testset)}")
    print(f"type3_testset : {len(type3_testset)}")
    print(f"type4_testset : {len(type4_testset)}")
    print(f"type5_testset : {len(type5_testset)}")

    type1_loader = DataLoader(type1_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    type2_loader = DataLoader(type2_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    type3_loader = DataLoader(type3_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    type4_loader = DataLoader(type4_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    type5_loader = DataLoader(type5_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True)
    
    # 세 가지 경우의 Acc, F1 정리 
    type1_acc, type1_f1 = test(args, type1_loader)
    type2_acc, type2_f1 = test(args, type2_loader)
    type3_acc, type3_f1 = test(args, type3_loader)
    type4_acc, type4_f1 = test(args, type4_loader)
    type5_acc, type5_f1 = test(args, type5_loader)
        
    # plt 로 위의 세 경우 그리기 
    draw_accuracy_during_test(args, type1_acc, type2_acc, type3_acc, type4_acc, type5_acc)
    draw_f1_during_test(args, type1_f1, type2_f1, type3_f1, type4_f1, type5_f1)
