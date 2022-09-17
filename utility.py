import matplotlib
import matplotlib.pyplot as plt
from torch.serialization import location_tag  
matplotlib.use('Agg')
plt.switch_backend('agg')
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F

# def plot_roc_curve(path, title_info, y_true, y_prob):
#     fpr, tpr, thresholds = roc_curve(y_true, y_prob)
#     optimal_idx = np.argmax(tpr-fpr)
#     optimal_threshold = thresholds[optimal_idx]
#     auc_value = auc(fpr, tpr)

#     fig = plt.figure()
#     plt.title("ROC Curve")
#     plt.xlabel("APCER")
#     plt.ylabel("1-BPCER")    
#     plt.plot(fpr, tpr, color='red', label=title_info)
#     plt.plot([0,1], [0,1], color='green', linestyle='--') 
#     plt.legend(loc='upper left')
#     plt.text(0.55, 0.1, f"Area Under the Curve:{auc_value:4f}")
#     plt.text(0.55, 0.3, f"Best threshold:{optimal_threshold:4f}")
#     plt.text(0.55, 0.3, f"Best 1-BPCER:{tpr[optimal_idx]:4f}")
#     plt.text(0.55, 0.3, f"Best APCER:{fpr[optimal_idx]:4f}")
#     plt.savefig(path+f'/{title_info}_roc_curve.png')
#     plt.close(fig)

#     return auc_value
    
def plot_roc_curve(path, title_info, y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    optimal_idx = np.argmax(tpr-fpr)
    optimal_threshold = thresholds[optimal_idx]
    auc_value = roc_auc_score(y_true, y_prob)

    fig = plt.figure()
    plt.title("ROC Curve")
    plt.xlabel("APCER")
    plt.ylabel("1-BPCER")    
    plt.plot(fpr, tpr, color='red', label=title_info)
    plt.plot([0,1], [0,1], color='green', linestyle='--') 
    plt.legend(loc='upper left')
    plt.text(0.55, 0.4, f"Best threshold:{optimal_threshold:4f}")
    plt.text(0.55, 0.3, f"Best 1-BPCER:{tpr[optimal_idx]:4f}")
    plt.text(0.55, 0.2, f"Best APCER:{fpr[optimal_idx]:4f}")
    plt.text(0.55, 0.1, f"Area Under the Curve:{auc_value:4f}")
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

def draw_train_and_test_loss(args, train_losses, test_losses):
    epoch_list = [i for i in range(1, args.epochs+1)]

    fig = plt.figure()
    title = f"Loss Graph ({args.model}-{args.attacktype})"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epoch_list, train_losses, color="red", label="train_loss")
    plt.plot(epoch_list, test_losses, color="blue", label="test_loss")
    plt.legend(loc='upper right')
    plt.savefig(args.save_path+"/Train_LossGraph.png")
    plt.close(fig)
    
def draw_accuracy_and_f1_during_training(args, accuracy, f1):
    epoch_list = [i for i in range(1, args.epochs+1)]   
    
    fig = plt.figure()
    title = f"Accuracy and F1_score ({args.model}-{args.attacktype})"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Evaluation Protocol")
    plt.plot(epoch_list, accuracy, color="red", label="Accuracy")
    plt.plot(epoch_list, f1, color="blue", label="F1_score")
    plt.legend(loc='lower right')
    plt.savefig(args.save_path+"/Train_EvaluationProtocol.png")
    plt.close(fig)

def draw_train_and_test_loss1(args, train_losses, test_losses, outdoor_losses, dark_losses):
    epoch_list = [i for i in range(1, args.epochs+1)]

    fig = plt.figure()
    title = f"Loss Graph ({args.model}-{args.attacktype})"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epoch_list, train_losses, color="red", label="train_loss")
    plt.plot(epoch_list, test_losses, color="blue", label="test_loss")
    plt.plot(epoch_list, outdoor_losses, color="green", label="outdoor_loss")
    plt.plot(epoch_list, dark_losses, color="purple", label="dark_loss")
    plt.legend(loc='upper right')
    plt.savefig(args.save_path+"/Train_LossGraph.png")
    plt.close(fig)
    
def draw_accuracy_and_f1_during_training1(args, train_accuracy, train_f1, test_accuracy, test_f1, outdoor_accuracy, outdoor_f1, dark_accuracy, dark_f1):
    epoch_list = [i for i in range(1, args.epochs+1)]   
    
    fig = plt.figure()
    title = f"Accuracy and F1_score ({args.model}-{args.attacktype})"
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Evaluation Protocol")
    plt.plot(epoch_list, train_accuracy, color="red", label="Accuracy(train)")
    plt.plot(epoch_list, train_f1, color="blue", label="F1_score(train)")
    plt.plot(epoch_list, test_accuracy, color="green", label="Accuracy(indoor)")
    plt.plot(epoch_list, test_f1, color="purple", label="F1_score(indoor)")
    plt.plot(epoch_list, outdoor_accuracy, color="green", label="Accuracy(outdoor)")
    plt.plot(epoch_list, outdoor_f1, color="purple", label="F1_score(outdoor)")
    plt.plot(epoch_list, dark_accuracy, color="green", label="Accuracy(dark)")
    plt.plot(epoch_list, dark_f1, color="purple", label="F1_score(dark)")
    plt.legend(loc='lower right')
    plt.savefig(args.save_path+"/Train_EvaluationProtocol1(total).png")
    plt.close(fig)
    
    
    
    
def draw_accuracy_during_test(args, acc_type):
    epoch_list = [i for i in range(1, args.epochs+1) if i%5 == 0]   
    
    fig = plt.figure()
    title = f"Accuracy ({args.model}-{args.attacktype})"
    plt.title(title) 
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(epoch_list, acc_type[0], color="red", label="Indoor")
    plt.plot(epoch_list, acc_type[1], color="Green", label="Outdoor")
    plt.plot(epoch_list, acc_type[2], color="blue", label="Dark")
    plt.plot(epoch_list, acc_type[3], color="purple", label="Indoor + Outdoor")
    plt.plot(epoch_list, acc_type[4], color="brown", label="Indoor + Dark")
    plt.legend(loc='upper right')
    plt.savefig(args.save_path+"/Test_Accuracy.png")
    plt.close(fig)


def draw_f1_during_test(args, f1_type):
    epoch_list = [i for i in range(1, args.epochs+1) if i%5 == 0]   
    
    fig = plt.figure()
    title = f"F1 score ({args.model}-{args.attacktype})"
    plt.title(title) 
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.plot(epoch_list, f1_type[0], color="red", label="Indoor")
    plt.plot(epoch_list, f1_type[1], color="Green", label="Outdoor")
    plt.plot(epoch_list, f1_type[2], color="blue", label="Dark")
    plt.plot(epoch_list, f1_type[3], color="purple", label="Indoor + Outdoor")
    plt.plot(epoch_list, f1_type[4], color="brown", label="Indoor + Dark")  
    
    plt.legend(loc='upper right')
    plt.savefig(args.save_path+"/Test_F1.png")
    plt.close(fig)