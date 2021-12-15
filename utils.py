import matplotlib
import matplotlib.pyplot as plt
from torch.serialization import location_tag  
matplotlib.use('Agg')
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

def image_crop_f(image):
    image_crops = []
    for i in range(3):
        for j in range(3):
            x_ = i*24
            y_ = j*24
            w_ = x_+64
            h_ = y_+64
            # img_crop = image[:,x_:w_,y_:h_]
            img_crop = image[:,:,x_:w_,y_:h_]
            image_crops.append(img_crop)
    return image_crops

def plot_figure(path, train_loss, eval_loss):

    train_x = np.array([i for i in range(len(train_loss))])
    train_y = np.array(train_loss)
    eval_x = np.array([i for i in range(len(eval_loss))])
    eval_y = np.array(eval_loss)

    fig = plt.figure()
    plt.title("Loss Graph")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(train_x, train_y, lw=1, label='train')
    plt.plot(eval_x, eval_y, lw=1, label='evaluation')    
    plt.legend(loc='upper left')
    plt.savefig(path+f'/loss.png')
    plt.close(fig)

def plot_roc_curve(path, title_info, y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_value = auc(fpr, tpr)

    fig = plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")    
    plt.plot(fpr, tpr, color='red', label=title_info)
    plt.plot([0,1], [0,1], color='green', linestyle='--') 
    plt.legend(loc='upper left')
    plt.text(0.55, 0.1, f"Area Under the Curve:{ auc_value:4f}")
    plt.savefig(path+f'/roc_curve_epoch {title_info}.png')
    plt.close(fig)

    return auc_value
    
def plot_eval_metric(path, title_info, y_true, y_prob):
    APCER_list = []
    NPCER_list = []
    ACER_list = []
    Index_list = []

    for index in range(len(y_true)):
        APCER, NPCER, ACER = cal_metrics(y_true, y_prob, index)
        APCER_list.append(APCER)
        NPCER_list.append(NPCER)
        ACER_list.append(ACER)
        Index_list.append(index)

    ACPER_nparray = np.array(APCER_list)
    NPCER_nparray = np.array(NPCER_list)
    ACER_nparray = np.array(ACER_list)
    Index_nparray = np.array(Index_list)

    fig = plt.figure()
    plt.title(f"Evaluation Metrics - {title_info}")
    plt.xlabel("Number of data")
    plt.ylabel("Performance(%)")
    plt.plot(Index_nparray, ACPER_nparray, color="red", label="APCER")
    plt.plot(Index_nparray, NPCER_nparray, color="blue", label="NPCER")
    plt.plot(Index_nparray, ACER_nparray, color="green", label="ACER")
    plt.legend(loc='upper left')
    plt.savefig(path+f'/eval_metrics_{title_info}.png')
    plt.close(fig)

def cal_metrics(y_true, y_prob, length):
    
    if len(y_true) != len(y_prob):
        print("Evalution Metircs Calculate Error - APCER, NPCER, ACER")
        return 0,0,0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    
    APCER = float(fp)/(tn+fp+0.001)
    NPCER = float(fn)/(fn+tp+0.001)
    ACER = (APCER+NPCER)/2
    
    return APCER, NPCER, ACER

