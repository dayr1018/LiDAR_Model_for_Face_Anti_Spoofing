# python train.py --depth true --model layer4_4to4 --cuda 0 --gr 50 --dr 0.5 --message 0202_layer4_4to4_g50 

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as f
from torch.optim import lr_scheduler
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter

from models.AutoEncoder import AutoEncoder_RGB, AutoEncoder_Depth
from models.AutoEncoder import AutoEncoder_Intergrated_Basic, AutoEncoder_Intergrated_Proposed
from models.Network import Face_Detection_Model
from dataloader.dataloader import Facedata_Loader
from centerloss import CenterLoss

import numpy as np
import random
import time
from datetime import datetime
import argparse
from loger import Logger
import os
import sys

from utility import plot_roc_curve, cal_metrics, cal_metrics2
from skimage.util import random_noise

def booltype(str):
    if isinstance(str, bool):
        return str
    if str.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif str.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentError("Boolean value expected")

def train(args, train_loader, valid_loader):

    # Tensorboard 
    global writer
    writer = SummaryWriter(f"runs/{args.message}")

    # args 출력
    logger.Print(args)

    # Model 생성 및 아키텍쳐 출력 
    model = Face_Detection_Model(args.inputdata_channel).to(args.device)
    summary(model)

    # Loss, 옵티마이저, 스케줄러 생성 
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    ct_loss = CenterLoss(num_classes=2, feat_dim=512, use_gpu=True, device=args.device)
    optimizer2 = torch.optim.Adam(ct_loss.parameters(), lr=args.lr)
    scheduler2 = lr_scheduler.ExponentialLR(optimizer2, gamma=0.95)

    # train 
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    apcer_list = []
    npcer_list = []
    acer_list = []
    epoch_list = []
    
    auc_list = []

    for epoch in range(args.epochs):
        model.train()

        logger.Print(f"***** << Training epoch:{epoch} >>")  
        
        y_true = []
        y_pred = []
        y_prob = []

        for batch, data in enumerate(train_loader, 1):

            rgb_image, depth_image, labels = data            
            rgb_image = rgb_image.to(args.device)
            depth_image = depth_image.to(args.device)
            labels = labels.to(args.device)

            # 모델에 태울 데이터 
            inputdata = rgb_image     
            if args.model == "rgb":
                inputdata = rgb_image
            elif args.model == "depth":
                inputdata = torch.cat((rgb_image, depth_image), dim=1)
            elif args.model == "ae":
                origin_sum = torch.cat((rgb_image, depth_image), dim=1)
                recons_image = autoencoder(rgb_image, depth_image)
                inputdata = torch.cat((origin_sum, recons_image), dim=1)

            # 예측 오류 계산 
            outputs, features = model(inputdata)
            _, pred_outputs = torch.max(outputs, 1)
            prob_outputs = f.softmax(outputs,1)[:,1]

            loss_ce = ce_loss(outputs, labels)
            loss_ct = ct_loss(features, labels)
            loss = loss_ce + args.lamda * loss_ct

            writer.add_scalar("Loss/Epoch(Total)", loss, epoch)
            writer.add_scalar("Loss/Epoch(CrossEntropy)", loss_ce, epoch)
            writer.add_scalar("Loss/Epoch(Center)", loss_ct, epoch)

            # 역전파 
            optimizer.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer2.step()  

            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                print(f"***** loss: {loss:>7f}  [{current:>5d}/{len(train_loader.dataset):>5d}] [{batch}:{len(data[0])}]")

        scheduler.step()
        scheduler2.step()

        if (epoch%5) == 0 or epoch == (args.epochs-1):
            accuracy, precision, recall, f1, apcer, npcer, acer = valid(args, valid_loader, model, epoch)
            logger.Print(f"***** Current Epoch:{epoch}, accuracy:{accuracy:3f}, f1:{f1:3f}, apcer:{apcer:3f}, npcer:{npcer:3f}, acer:{acer:3f}")  
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            apcer_list.append(apcer)
            npcer_list.append(npcer)
            acer_list.append(acer)
            epoch_list.append(epoch)

            writer.add_scalar("Accuracy/Epoch (Valid)", accuracy, epoch)
            writer.add_scalar("F1/Epoch (Valid)", f1, epoch)
            writer.add_scalar("Other Evaluation - APCER/Epoch (Valid)", apcer, epoch)
            writer.add_scalar("Other Evaluation - NPCER/Epoch (Valid)", npcer, epoch)
            writer.add_scalar("Other Evaluation - ACER/Epoch (Valid)", acer, epoch)

            # 0, 5, 10, ... 순서대로 weight들 저장   
            checkpoint = f'{args.checkpoint_path}/epoch_{epoch}_model.pth'
            torch.save(model.state_dict(), checkpoint)

            # 결과 나타내기 (ROC 커브)
            auc_value = plot_roc_curve(args.save_path, f"epoch{epoch}_", y_true, y_prob)
            writer.add_scalar("Other Evaluation - AUC_Value/Epoch (Valid)", auc_value, epoch)

            auc_list.append(auc_value)

    max_accuracy = max(accuracy_list)
    index = accuracy_list.index(max_accuracy)
    ma_precision = precision_list[index]
    ma_recall = recall_list[index]
    ma_f1 = f1_list[index]
    ma_apcer = apcer_list[index]
    ma_npcer = npcer_list[index]
    ma_acer = acer_list[index]
    ma_auc = auc_list[index]
    ma_epoch = epoch_list[index]

    logger.Print(f"***** Total Accuracy per epoch")
    logger.Print(accuracy_list)
    logger.Print(f"***** Total Precision per epoch")
    logger.Print(precision_list)
    logger.Print(f"***** Total Recall per epoch")
    logger.Print(recall_list)
    logger.Print(f"***** Total F1 per epoch")
    logger.Print(f1_list)
    logger.Print(f"***** Total AUC Value")
    logger.Print(auc_list)
    logger.Print(f"***** Total Epoch")
    logger.Print(epoch_list)

    logger.Print(f"\n***** Result (Valid)")
    logger.Print(f"Accuracy: {max_accuracy:3f}")
    logger.Print(f"Precision: {ma_precision:3f}")
    logger.Print(f"Recall: {ma_recall:3f}")
    logger.Print(f"F1: {ma_f1:3f}")
    logger.Print(f"APCER: {ma_apcer:3f}")
    logger.Print(f"NPCER: {ma_npcer:3f}")
    logger.Print(f"ACER: {ma_acer:3f}")
    logger.Print(f"AUC Value: {ma_auc:3f}")
    logger.Print(f"Epoch: {ma_epoch}")

    writer.close()   

    return ma_epoch

def valid(args, valid_loader, model, epoch):

    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    ce_loss = nn.CrossEntropyLoss()
    ct_loss = CenterLoss(num_classes=2, feat_dim=512, use_gpu=True, device=args.device)

    with torch.no_grad():
        for _, data in enumerate(valid_loader):
            rgb_image, depth_image, labels = data

            # 가우시안 노이즈 추가 
            if args.gr != 0 :
                rgb_image =torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                depth_image =torch.FloatTensor(random_noise(depth_image, mode='gaussian', mean=0, var=args.gr, clip=True))

            # 텐서화
            rgb_image = rgb_image.to(args.device) 
            depth_image = depth_image.to(args.device)
            labels = labels.to(args.device)

            # 모델에 태울 데이터 
            inputdata = rgb_image     
            if args.model == "rgb":
                inputdata = rgb_image
            elif args.model == "depth":
                inputdata = torch.cat((rgb_image, depth_image), dim=1)
            elif args.model == "ae":
                origin_sum = torch.cat((rgb_image, depth_image), dim=1)
                recons_image = autoencoder(rgb_image, depth_image)
                inputdata = torch.cat((origin_sum, recons_image), dim=1)

            # 예측 오류 계산 
            outputs, features = model(inputdata)
            _, pred_outputs = torch.max(outputs, 1)
            prob_outputs = f.softmax(outputs,1)[:,1]

            vloss_ce = ce_loss(outputs, labels)
            vloss_ct = ct_loss(features, labels)
            vloss = vloss_ce + args.lamda * vloss_ct

            writer.add_scalar("Total Loss/Epoch (Valid)", vloss, epoch)
            writer.add_scalar("CrossEntropy Loss/Epoch (Valid)", vloss_ce, epoch)
            writer.add_scalar("Center Loss/Epoch (Valid)", vloss_ct, epoch)
        
            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

    # 성능 평가 
    accuracy, precision, recall, f1 = cal_metrics(y_true, y_pred)
    apcer, npcer, acer = cal_metrics2(y_true, y_pred)

    return accuracy, precision, recall, f1, apcer, npcer, acer

def test(args, test_loader, weight_path):
    model = Face_Detection_Model(args.inputdata_channel).to(args.device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for batch, data in enumerate(test_loader, 1):

        rgb_image, depth_image, labels = data            
        rgb_image = rgb_image.to(args.device)
        depth_image = depth_image.to(args.device)
        labels = labels.to(args.device)

    with torch.no_grad():
        for _, data in enumerate(test_loader):
            rgb_image, depth_image, labels = data

            # 가우시안 노이즈 추가 
            if args.gr != 0 :
                rgb_image =torch.FloatTensor(random_noise(rgb_image, mode='gaussian', mean=0, var=args.gr, clip=True))
                depth_image =torch.FloatTensor(random_noise(depth_image, mode='gaussian', mean=0, var=args.gr, clip=True))

            # 텐서화
            rgb_image = rgb_image.to(args.device) 
            depth_image = depth_image.to(args.device)
            labels = labels.to(args.device)

            # 모델에 태울 데이터 
            inputdata = rgb_image     
            if args.model == "rgb":
                inputdata = rgb_image
            elif args.model == "depth":
                inputdata = torch.cat((rgb_image, depth_image), dim=1)
            elif args.model == "ae":
                origin_sum = torch.cat((rgb_image, depth_image), dim=1)
                recons_image = autoencoder(rgb_image, depth_image)
                inputdata = torch.cat((origin_sum, recons_image), dim=1)

            # 예측 오류 계산 
            outputs, features = model(inputdata)
            _, pred_outputs = torch.max(outputs, 1)
            prob_outputs = f.softmax(outputs,1)[:,1]
        
            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

    accuracy, precision, recall, f1 = cal_metrics(y_true, y_pred)
    apcer, npcer, acer = cal_metrics2(y_true, y_pred)
    auc_value = plot_roc_curve(args.save_path_test, f"epoch{epoch}_", y_true, y_prob)

    logger.Print(f"\n***** Result (Test)")
    logger.Print(f"Accuracy: {accuracy:3f}")
    logger.Print(f"Precision: {precision:3f}")
    logger.Print(f"Recall: {recall:3f}")
    logger.Print(f"F1: {f1:3f}")
    logger.Print(f"APCER: {apcer:3f}")
    logger.Print(f"NPCER: {npcer:3f}")
    logger.Print(f"ACER: {acer:3f}")
    logger.Print(f"AUC Value: {auc_value:3f}")


if __name__ == "__main__":

    # args option
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    parser.add_argument('--ae-path', default='', type=str, help='Pretrained AutoEncoder path')
    parser.add_argument('--save-path', default='../bc_output/logs/Train/', type=str, help='train logs path')
    parser.add_argument('--save-path-valid', default='', type=str, help='valid logs path')
    parser.add_argument('--save-path-test', default='../bc_output/logs/Test/', type=str, help='test logs path')
    parser.add_argument('--model', default='', type=str, help='rgb, depth, ae')                                         
    parser.add_argument('--checkpoint-path', default='', type=str, help='checkpoint path')
    parser.add_argument('--message', default='', type=str, help='pretrained model checkpoint')                     
    parser.add_argument('--epochs', default=300, type=int, help='train epochs')                                    
    parser.add_argument('--lowdata', default=True, type=booltype, help='whether low data is included')
    parser.add_argument('--dataset', default=0, type=int, help='data set type')
    parser.add_argument('--loss', default=0, type=int, help='0: mse, 1:rapp')
    parser.add_argument('--gr', default=0.0, type=float, help='gaussian rate(default: 0.01)')
    parser.add_argument('--dr', default=0.5, type=float, help='dropout rate(default: 0.1)')
    parser.add_argument('--batchnorm', default=False, type=booltype, help='batch normalization(default: False)')
    parser.add_argument('--lamda', default=0, type=float, help='rate of center loss (default: 0)')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate(default: 0.001)')
    parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')
    parser.add_argument('--seed', default=1, type=int, help='Seed for random number generator')
    parser.add_argument('--cuda', default=0, type=int, help='gpu number')                                         
    parser.add_argument('--device', default='', type=str, help='device when cuda is available')
    parser.add_argument('--inputdata-channel', default=3, type=int, help='rgb=3, depth=4, ae=8')
    args = parser.parse_args()

    # 중요 옵션 체크 및 model type 배정 
    if args.model == "rgb":
        args.inputdata_channel = 3
    elif args.model == "depth":
        args.inputdata_channel = 4
    elif args.model == "ae":
        args.inputdata_channel = 8
    else:
        print("You need to checkout option 'model' [rgb, depth, ae]")
        sys.exit(0)

    # 결과 파일 path 설정 
    time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time.localtime(time.time()))
    args.save_path = args.save_path + f'{args.message}' + '_' + f'{time_string}'
    args.save_path_valid = args.save_path + '/valid'
    args.save_path_test = args.save_path + '/test'

    if not os.path.exists(args.save_path): 
        os.makedirs(args.save_path)    
    if not os.path.exists(args.save_path_valid): 
        os.makedirs(args.save_path_valid)
    if not os.path.exists(args.save_path_test): 
        os.makedirs(args.save_path_test)
    
    # weight 파일 path
    args.checkpoint_path = f'/mnt/nas3/yrkim/liveness_lidar_project/GC_project/bc_output/checkpoint/{args.message}'
    if not os.path.exists(args.checkpoint_path): 
        os.makedirs(args.checkpoint_path)

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
    logger = Logger(f'{args.save_path}/logs.logs')

    # Autoencoder's path
    args.ae_path = "/mnt/nas3/yrkim/liveness_lidar_project/GC_project/ad_output/checkpoint/0421_Both_3_dr01_gr0/epoch_90_model.pth"

   # Pretrained 된 AutoEncoder 생성 (layer3)
    global autoencoder
    autoencoder = AutoEncoder_Intergrated_Basic(3, False, 0.1).to(args.device)
    autoencoder.load_state_dict(torch.load(args.ae_path))
    autoencoder.eval()

    # data loader
    train_loader, test_loader = Facedata_Loader(train_size=64, test_size=64, use_lowdata=args.lowdata, dataset=args.dataset)

    # train 코드
    epoch = train(args, train_loader, test_loader)
    weight_path = f"{args.checkpoint_path}/epoch_{epoch}_model.pth"

    # test 코드
    test(args, test_loader, weight_path)
    



