import pdb  
import sys
import time
import argparse
import os
import logging
from scipy import interp
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')
class Logger():
    def __init__(self,logPath):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(logPath)
        handler.setLevel(logging.INFO)
        # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(console)
        self.logger = logger

    def Print(self,message):
        self.logger.info(message)

def add_args():
    parser = argparse.ArgumentParser(description='face anto-spoofing')
    parser.add_argument('--file_path', default='./prob.txt', type=str, help='y prob txt path')
    parser.add_argument('--label_path', default='./val_label.txt', type=str, help='y label txt path')
    args = parser.parse_args()
    return args

def eval_model(y_true, y_pred, y_prob, logger=None):  
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()

    APCER = float(fp)/(tn+fp+0.001)
    NPCER = float(fn)/(fn+tp+0.001)
    ACER = (APCER+NPCER)/2
#    FPR = float(fp)/(fp+tn) # 진짜라고 예측했는데, 틀렸어 -> 가짜가 들어왔는데 진짜라고 한 확률 
#    TPR = float(tp)/(tp+fn) # 진짜라고 예측했는데, 맞췄어 -> 진짜가 들어왔는데 진짜라고 할 확률
    a = tp/(tp+fn+0.001) + tn/(tn+fp+0.001)
    HTER = 1 - a*0.5

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)

    score_1 = interp(0.01,fpr,tpr)
    score_2 = interp(0.001,fpr,tpr)
    score_3 = interp(0.0001,fpr,tpr)

    message = f'|TP:{tp} |TN:{tn} |FP:{fp} |FN:{fn} /HTER:{HTER:.4F} |APCER:{APCER:.6F} |NPCER:{NPCER:.6F} '\
                f'|ACER:{ACER:.6F} |FPR=e2:{score_1:.6f} |FPR=e3:{score_2:.6f} |FPR=e4:{score_3:.6f}|'

    return message, score_3, ACER

if __name__ == '__main__':
    args = add_args()
    logger = Logger('./eval.logs')
    y_prob_list = []
    y_pLabel_list = []
    y_label_list = []
    lines_in_yProb = open(args.file_path,'r')
    lines_in_yLabel = open(args.label_path,'r')

    for line in lines_in_yProb:
        line = line.rstrip()
        split_str = line.split()
        y_prob = float(split_str[3])
        y_prob_list.append(y_prob)
        if y_prob > 0.5:
            y_pLabel_list.append(1)
        else:
            y_pLabel_list.append(0)
    
    for line in lines_in_yLabel:
        line = line.rstrip()
        split_str = line.split()
        y_label = float(split_str[3])
        y_label_list.append(y_label)
    
    eval_model(y_label_list, y_pLabel_list, y_prob_list, logger)