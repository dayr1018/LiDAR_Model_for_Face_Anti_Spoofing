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

def plot_roc_curve(path, title_info, y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = auc(fpr, tpr)

    fig = plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")    
    plt.plot(fpr, tpr, color='red', label=title_info)
    plt.plot([0,1], [0,1], color='green', linestyle='--') 
    plt.legend(loc='upper left')
    plt.text(0.55, 0.1, f"Area Under the Curve:{ auc_value:4f}")
    plt.savefig(path+f'/{title_info}_roc_curve.png')
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

def cal_metrics(y_true, y_pred):
    
    if len(y_true) != len(y_pred):
        print("Evalution length Error - y_true, y_pred")
        return 0,0,0,0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + fn + fp + tn + 1e-12)
    precision = tp / (tp + fp + 1e-12)  
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-12)

    return accuracy, precision, recall, f1

def cal_metrics2(y_true, y_prob):
    
    if len(y_true) != len(y_prob):
        print("Evalution Metircs Calculate Error - APCER, NPCER, ACER")
        return 0,0,0
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_prob).ravel()
    
    apcer = fp / (tn + fp + 1e-12)
    npcer = fn / (fn + tp + 1e-12) 
    acer = (apcer + npcer)/2
    
    return apcer, npcer, acer
