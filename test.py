import os
import os.path as osp
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from tqdm import tqdm
from collections import Counter
import argparse

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from models.Network import Face_Detection_Model, rgbp_twostep_model, rgbd_twostep_model, rgbdp_v1_twostep_model, rgbdp_v2_twostep_model, rgbdp_v3_twostep_model
# from models.hm_Network import Face_Detection_Model
from dataloader.dataloader import load_dataset, load_test_dataset
from utility import plot_roc_curve
from loger import Logger

def deleteBatchnorm(model):
    del model.bn1
    del model.layer1[0].bn1
    del model.layer1[0].bn2
    del model.layer1[1].bn1
    del model.layer1[1].bn2
    del model.layer1[2].bn1
    del model.layer1[2].bn2

    del model.layer2[0].bn1
    del model.layer2[0].bn2
    del model.layer2[0].downsample[1]
    del model.layer2[1].bn1
    del model.layer2[1].bn2
    del model.layer2[2].bn1
    del model.layer2[2].bn2
    del model.layer2[3].bn1
    del model.layer2[3].bn2

    del model.layer3[0].bn1
    del model.layer3[0].bn2
    del model.layer3[0].downsample[1]
    del model.layer3[1].bn1
    del model.layer3[1].bn2
    del model.layer3[2].bn1
    del model.layer3[2].bn2
    del model.layer3[3].bn1
    del model.layer3[3].bn2
    del model.layer3[4].bn1
    del model.layer3[4].bn2
    del model.layer3[5].bn1
    del model.layer3[5].bn2

    del model.layer4[0].bn1
    del model.layer4[0].bn2
    del model.layer4[0].downsample[1]
    del model.layer4[1].bn1
    del model.layer4[1].bn2
    del model.layer4[2].bn1
    del model.layer4[2].bn2

def test(args, dataloader):
    
    # Tensorboard 
    global writer   
    
    # args 출력
    logger.Print(args)
  
    # Test 시작 
    # test_performs = {'ACC':[], 'F1':[], 'Info':[]}
    test_performs = {'apcer':[], 'bpcer':[], 'acer':[], 'auc':[], 'roc_auc':[], 'fpr':[], 'tpr':[], 'thresholds':[]}
    
    epochs = args.epochs
    sigmoid = nn.Sigmoid()
    loss_fn = nn.BCEWithLogitsLoss()  
    
    # # 1
    # fileset = {}    
    # fileset["135"] = "epoch_135_model.pth"

    # 2 
    fileset = {}
    numbers = [i for i in range(1, epochs+1)]
    for number in numbers:
        fileset[str(number)]=f"epoch_{number}_model.pth"
    
    for epoch, filename in fileset.items():
            
        model_path = osp.join(args.modelpath, filename)

        # 모델 생성 
        ckpt = torch.load(model_path)
        model=""
        if args.model == "rgb":
            model = Face_Detection_Model(3).to(args.device)         
        elif args.model == "rgbp":
            model = rgbp_twostep_model(device=args.device)
        elif args.model == "rgbd":
            model = rgbd_twostep_model(device=args.device)
        elif args.model == "rgbdp_v1":
            model = rgbdp_v1_twostep_model(device=args.device)
        elif args.model == "rgbdp_v2":
            model = rgbdp_v2_twostep_model(device=args.device)
        elif args.model == "rgbdp_v3":
            model = rgbdp_v3_twostep_model(device=args.device) 
        # summary(model)
        
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
 
            if args.model == "rgb":
                logits = model(rgb)
            elif args.model == "rgbp":
                 logits = model(rgb, cloud)
            elif args.model == "rgbd":
                logits = model(rgb, depth)
            elif args.model in ["rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
                logits = model(rgb, depth, cloud)
                
            logits = logits[:,0]  # logtit (4.2) 텐서이던데 첫째는 batch 이겠고. 근데 왜 0번째 값만 쓸까?
            loss = loss_fn(logits, label.float())
            
            probs = sigmoid(logits)
            test_loss.append(loss.item())
            test_probs += probs.cpu().detach().tolist()
            test_labels += label.cpu().detach().tolist()        
            test_bar.set_description("[Test] Epoch[{}/{}][{}/{}] Loss:{} Loss(mean):{}".format(epoch, epochs, step+1,len(dataloader),
                                                        round(loss.item(),5),round(np.array(test_loss).mean(),5)
                                                                ))
 
        tn, fp, fn, tp = confusion_matrix(np.array(test_labels), np.round(test_probs)).ravel()
        apcer = fp / (fp + tn)
        bpcer = fn / (fn + tp)
        acer = (apcer + bpcer) / 2
        
        fpr, tpr, thresholds = roc_curve(test_labels, test_probs, pos_label=1)
        auc_value = auc(fpr, tpr)
        roc_auc = roc_auc_score(test_labels, test_probs)
        plot_roc_curve(args.save_path, epoch, test_labels, test_probs) 
        
        acc = accuracy_score(np.array(test_labels), np.round(test_probs))
        f1 = f1_score(np.array(test_labels), np.round(test_probs), average='macro') 
        
        test_performs['apcer'].append(apcer)
        test_performs['bpcer'].append(bpcer)
        test_performs['acer'].append(acer)
        test_performs['fpr'].append(fpr)
        test_performs['tpr'].append(tpr)
        test_performs['thresholds'].append(thresholds)
        test_performs['auc'].append(auc_value)
        test_performs['roc_auc'].append(roc_auc)
        
        writer.add_scalar("1. APCER ", apcer, epoch)
        writer.add_scalar("2. BPCER ", bpcer, epoch)
        writer.add_scalar("3. ACER ", acer, epoch)
        writer.add_scalar("4. AUC ", auc_value, epoch)
        writer.add_scalar("4. ROC_AUC ", roc_auc, epoch)
        writer.add_scalar("5. ACC ", acc, epoch)
        writer.add_scalar("6. F1 ", f1, epoch)
        
        
        logger.Print(f'APCER: {apcer:.4f}')
        logger.Print(f'BPCER: {bpcer:.4f}') 
        logger.Print(f'ACER: {acer:.4f}')
        logger.Print(f'AUC: {auc_value:.4f}')    
        logger.Print(f'ROC_AUC: {roc_auc:.4f}')    
        logger.Print(f'Accuracy: {acc:.4f}')        
        logger.Print(f'F1 Score: {f1:.4f}') 
        
        # test_cf = confusion_matrix(np.array(test_labels), np.round(test_probs))
        # test_cf = pd.DataFrame(test_cf)
        # test_cf.columns = ['Predicted:0','Predicted:1']
        # test_cf.index = ['Label:0','Label:1']    
        # logger.Print(f' --- [Test] Confustion_Matrix & Classification_Report --- Epoch {epoch}')
        # # display(test_cf)
        # logger.Print(test_cf.to_string())
        # test_performs['Info'].append(test_cf.to_string())

        # test_acc = accuracy_score(np.array(test_labels), np.round(test_probs))
        # test_f1 = f1_score(np.array(test_labels), np.round(test_probs), average='macro')  
        # logger.Print(f'Test Accuracy : {test_acc:.4f}')
        # logger.Print(f'Test F1-score : {test_f1:.4f}')   
        
        # test_performs['ACC'].append(test_acc)  
        # test_performs['F1'].append(test_f1)  
        # # writer.add_scalar("Accuracy/Epoch (Test)", test_acc, epoch)
        # # writer.add_scalar("F1 Score/Epoch (Test)", test_f1, epoch)
        # print(f"!!!!!!!!!!! Test Accuracy : {test_acc}")

    # accu_max = np.array(test_performs["ACC"]).max()
    # accu_index = test_performs["ACC"].index(accu_max)
    # f1_max = np.array(test_performs["F1"]).max()
    # f1_index = test_performs["F1"].index(f1_max)

    # logger.Print(f'  > Test Best F1 Score : {np.array(test_performs["F1"]).max():.4f}')
    # logger.Print(f'  > Test Epoch (Best F1 Score): {f1_index+1}')   
    # logger.Print(f'  > Test Best CF (std: F1)') 
    # logger.Print(f'  > {np.array(test_performs["Info"][f1_index])}')
    # logger.Print(f'')
    
    bpcer_min = np.array(test_performs['bpcer']).min()
    bpcer_index = test_performs["bpcer"].index(bpcer_min)
    apcer_min = np.array(test_performs['apcer']).min()
    apcer_index = test_performs["apcer"].index(apcer_min)
    acer_min = np.array(test_performs['acer']).min()
    acer_index = test_performs["acer"].index(acer_min)
    auc_max = np.array(test_performs['auc']).max()
    auc_index = test_performs["auc"].index(auc_max)
    roc_auc_max = np.array(test_performs['roc_auc']).max()
    roc_auc_index = test_performs["roc_auc"].index(roc_auc_max)
    
    logger.Print(f'  > Test Best BPCER : {bpcer_min:.4f} / {bpcer_index}')
    logger.Print(f'  > Test Best APCER : {apcer_min:.4f} / {apcer_index}')
    logger.Print(f'  > Test Best ACER : {acer_min:.4f} / {acer_index}')
    logger.Print(f'  > Test Best AUC : {auc_max:.4f} / {auc_index}')
    logger.Print(f'  > Test Best AUC : {roc_auc_max:.4f} / {roc_auc_index}')
    logger.Print(f'')
    
    return test_performs
           
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
    parser.add_argument('--modelpath', default='', type=str, help='model parameter path')
    parser.add_argument('--message', default='', type=str, help='parameter file name')                     
    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                         
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')                                                  
    args = parser.parse_args()
    
    if args.model not in ["rgb", "rgbd", "rgbp", "rgbdp_v1", "rgbdp_v2", "rgbdp_v3"]:
        print("You need to checkout option 'model' [rgb, rgbd, rgbp, rgbpc]")
        sys.exit(0)
        
    # 결과 파일 path 설정 
    args.save_path = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/logs'
    args.save_path = osp.join(args.save_path, args.message)
    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)  
    
    # weight 파일 path
    # args.modelpath = f'/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    # args.modelpath = f'/mnt/nas4/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.modelpath}'
    args.modelpath = f'/mnt/nas4/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    
    if not os.path.exists(args.modelpath): 
        print(f"No Such Directory : {args.modelpath}")
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
    torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    def seed_worker(worker_id):
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Tensorboard 
    global writer

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

    print(f"type1_testset : {len(type1_testset)}")
    print(f"type2_testset : {len(type2_testset)}")
    print(f"type3_testset : {len(type3_testset)}")

    type1_loader = DataLoader(type1_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    type2_loader = DataLoader(type2_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)
    type3_loader = DataLoader(type3_testset, batch_size=args.batchsize, shuffle=False, num_workers=args.workers, pin_memory=True, worker_init_fn=seed_worker, generator=g)

    #다섯 가지 경우의 Acc, F1 정리 
    loader_types =  ["Indoor(dark)", "Outdoor", "Indoor"]
    test_performs = [[] for i in range(len(loader_types))]
      
    dataloaders = [type3_loader, type2_loader, type1_loader]
    for i, loader in enumerate(dataloaders):
        writer = SummaryWriter(f"runs/result_0914/{args.message}_{args.attacktype}_{loader_types[i]}")
        test_performs[i] = test(args, loader)
        writer.close()
