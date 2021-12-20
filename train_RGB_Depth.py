import time
import argparse
import os
import matplotlib
from torch.utils.data.dataset import ConcatDataset

matplotlib.use('Agg')
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms 
from torch.utils.data.sampler import SubsetRandomSampler 
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# from data.dataloader_RGB import load_cisia_surf, CISIA_SURF
from dataloader.dataloader_RGB_Depth import load_cisia_surf, CISIA_SURF
# from dataloader.dataloader_RGB_Depth_IR import load_cisia_surf, CISIA_SURF

# from models.model_RGB import Model
from models.model_RGB_Depth import Model
# from models.model_RGB_Depth_IR import Model

from evalution import eval_model
from centerloss import CenterLoss
from utils import plot_roc_curve, plot_eval_metric
from loger import Logger

time_object = time.localtime(time.time())
time_string = time.strftime('%Y-%m-%d_%I:%M_%p', time_object)
use_cuda = True if torch.cuda.is_available() else False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='face anto-spoofing')
parser.add_argument('--batch-size', default='64', type=int, help='train batch size')
parser.add_argument('--test-size', default='64', type=int, help='test batch size')
parser.add_argument('--save-path', default='../output/RGB_Depth/logs/Train/', type=str, help='logs save path')
parser.add_argument('--checkpoint', default='model.pth', type=str, help='pretrained model checkpoint')
parser.add_argument('--message', default='message', type=str, help='pretrained model checkpoint')
parser.add_argument('--epochs', default=50, type=int, help='train epochs')
parser.add_argument('--train', default=True, type=bool, help='train')
parser.add_argument('--mode', default=1, type=int, help='dataset protocol_mode')
parser.add_argument('--tryout', default=0, type=int, help='dataset protocol_tryout')
parser.add_argument('--skf', default=0, type=int, help='stratified k-fold')

args = parser.parse_args()

save_path = args.save_path + f'{time_string}' + '_' + f'{args.message}'
if not os.path.exists(save_path):
    os.makedirs(save_path)

logger = Logger(f'{save_path}/logs.logs')
logger.Print(time_string + " - " + args.message + "\n")

train_data, test_data= load_cisia_surf(train_size=args.batch_size,test_size=args.test_size, mode=args.mode)

model = Model(pretrained=False, num_classes=2)
criterion = nn.CrossEntropyLoss() 
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=5e-4)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

ct_loss = CenterLoss(num_classes=2, feat_dim=512, use_gpu=use_cuda)
optimzer4ct = optim.SGD(ct_loss.parameters(), lr =0.01, momentum=0.9,weight_decay=5e-4)
scheduler4ct = lr_scheduler.ExponentialLR(optimzer4ct, gamma=0.95)

if use_cuda:
    model = torch.nn.DataParallel(model,device_ids=list(range(torch.cuda.device_count()))) #  device_ids=[0, 1, 2]
    model = model.cuda()
    criterion = criterion.cuda()
    ct_loss = ct_loss.cuda()

train_loss = []
eval_loss = []
train_score = []
eval_score = []
test_score = []

def train(fold, epoch, data_loader):

    train_history = []
    auc_history = []

    out_dir = '../output/RGB_Depth/checkpoint'

    if args.mode == 1:
        if not os.path.exists(out_dir +'/checkpoint_v1_'+str(args.tryout)):
            os.makedirs(out_dir +'/checkpoint_v1_'+str(args.tryout))
    elif args.mode == 2:
        if not os.path.exists(out_dir +'/checkpoint_v2_'+str(args.tryout)):
            os.makedirs(out_dir +'/checkpoint_v2_'+str(args.tryout))
    else:
        print("wrong mode!")

    logger.Print(f"***** <<<<<  Train  >>>>>")
    logger.Print(f"***** Dataset info: mode v{args.mode}_{args.tryout}")
    logger.Print(f"***** Max epochs : {args.epochs}")
    logger.Print(f"***** Batch_size: {args.batch_size}, Test_size: {args.test_size}")
    logger.Print(f"***** <<< when fold:{fold}, start >>>")  

    model.train()  

    for epoch in range(epoch):

        logger.Print(f"***** << Training epoch:{epoch} >>")   

        y_true = []
        y_pred = []
        y_prob = []

        for batch, data in enumerate(data_loader, 1):

            size = len(data_loader.dataset)

            rgb_img = data[0]
            depth_img = data[1]
            labels = data[2]

            if use_cuda:
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                labels = labels.cuda()

            # 예측 오류 계산 
            outputs, features = model(rgb_img, depth_img)
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

            if batch % 10 == 0:
                loss, current = loss.item(), batch * len(data[0])
                print(f"***** loss: {loss:>7f}  [{current:>5d}/{size:>5d}] [{batch}:{len(data[0])}]")

        scheduler.step()
        scheduler4ct.step()

        eval_result, score, acer = eval_model(y_true, y_pred, y_prob)
        train_history.append(eval_result)
        train_score.append(score)        

        if epoch == 0:
            min_acer = acer
            ckpt_name = out_dir + '/checkpoint_v'+str(args.mode)+'_'+str(args.tryout)+'/global_min_acer_model.pth'
            torch.save(model.state_dict(), ckpt_name)
            logger.Print(f'***** Saved global min acer model, current epoch: {epoch}')
        elif epoch > 0 and acer < min_acer:
            min_acer = acer
            ckpt_name = out_dir + '/checkpoint_v'+str(args.mode)+'_'+str(args.tryout)+'/global_min_acer_model.pth'
            torch.save(model.state_dict(), ckpt_name)
            logger.Print(f'***** Saved global min acer model, current epoch: {epoch}')
    
        ckpt_name = out_dir + '/checkpoint_v'+str(args.mode)+'_'+str(args.tryout)+'/Cycle_' + str(epoch) + '_min_acer_model.pth'
        torch.save(model.state_dict(), ckpt_name)

        message = f'***** fold:{fold}, epoch:{epoch}, loss:{loss:.6f}, loss_anti:{loss_anti:.6f}, loss_ct:{loss_ct:.6f}'
        logger.Print(message)

        if not os.path.exists(f'{save_path}/train'):
            os.makedirs(f'{save_path}/train')
        title_info = f'fold-{fold}_epoch-{epoch}'
        auc_value = plot_roc_curve(f'{save_path}/train', title_info, y_true, y_prob)
        auc_history.append(auc_value)
        plot_eval_metric(f'{save_path}/train', title_info, y_true, y_pred)

    logger.Print(f"***** <<< Train history (fold={fold}) >>>")
    for i, message in enumerate(train_history):
        logger.Print(f"***** epoch:{i}, {message}")
    
    logger.Print(f"***** <<< AUC history (fold={fold}) >>>")
    for i, message in enumerate(auc_history):
        logger.Print(f"***** epoch:{i}, auc value: {message}")
    
    logger.Print("\n")

def test(fold, epoch, data_loader, weight_dir):

    model = Model(pretrained=False, num_classes=2)

    logger.Print(f"###### <<<<<  Test  >>>>>")
    logger.Print(f"###### <<< fold:{fold}, epoch: {epoch} >>>")

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
        for batch, data in enumerate(data_loader, 1):

            rgb_img = data[0]
            depth_img = data[1]
            labels = data[2]

            if use_cuda:
                rgb_img = rgb_img.cuda()
                depth_img = depth_img.cuda()
                labels = labels.cuda()

            # 예측 오류 계산
            outputs, features = model(rgb_img, depth_img)
            _, pred_outputs = torch.max(outputs, 1)
            prob_outputs = F.softmax(outputs,1)[:,1]
            
            y_true.extend(labels.data.cpu().numpy())
            y_pred.extend(pred_outputs.data.cpu().numpy())
            y_prob.extend(prob_outputs.data.cpu().numpy())

    eval_result, score, acer = eval_model(y_true, y_pred, y_prob)
    logger.Print(f"###### {eval_result}")

    if not os.path.exists(f'{save_path}/test'):
        os.makedirs(f'{save_path}/test')
    title_info = f'fold-{fold}_epoch-{epoch}'
    auc_value = plot_roc_curve(f'{save_path}/test', title_info, y_true, y_prob)
    plot_eval_metric(f'{save_path}/test', title_info, y_true, y_pred)

    logger.Print(f"###### auc_value : {auc_value}")
    logger.Print("\n")

if __name__ == '__main__':
    torch.manual_seed(999)

    if args.skf == 0:
        train_loader, _ = load_cisia_surf(train_size=args.batch_size, test_size=args.test_size, mode=args.mode)
        train(args.epochs, None, train_loader)

    elif args.skf > 0 and args.skf < 10:
       
        transform = transforms.Compose([
            transforms.Resize((124,124)),
            transforms.ToTensor(),
            transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
        ])

        train_data=CISIA_SURF(datatxt='MakeTextFileCode_RGB_Depth/train_data_list.txt', transform=transform)
        test_data=CISIA_SURF(datatxt='MakeTextFileCode_RGB_Depth/test_data_list.txt', transform=transform)
        dataset = ConcatDataset([train_data, test_data])

        nparr_label = []
        for i in range(dataset.__len__()):
            _, _, label = dataset.__getitem__(i)
            np_label = label.numpy()
            nparr_label = np.append(nparr_label, np_label)

        skf = StratifiedKFold(n_splits=args.skf)

        for fold, (train_idx, test_idx) in enumerate(skf.split(dataset, nparr_label)):
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            train_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=32)
            test_loader = DataLoader(dataset=dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=32)

            train(fold=fold, epoch=args.epochs, data_loader=train_loader)

            weight_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/output/RGB_Depth/checkpoint/checkpoint_v' + str(args.mode) + '_0/global_min_acer_model.pth'
            test(fold=fold, epoch="global", data_loader=test_loader, weight_dir=weight_dir)            
            weight_dir = '/mnt/nas3/yrkim/liveness_lidar_project/GC_project/output/RGB_Depth/checkpoint/checkpoint_v' + str(args.mode) + '_0/Cycle_49_min_acer_model.pth'
            test(fold=fold, epoch="49", data_loader=test_loader, weight_dir=weight_dir)          

    else :
        print("Fold value is too high")
