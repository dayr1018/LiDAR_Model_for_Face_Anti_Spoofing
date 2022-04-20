import argparse
import logging
from scipy import interp
from sklearn import metrics

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
    a = tp/(tp+fn+0.000001) + tn/(tn+fp+0.000001)
    HTER = 1 - a*0.5

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob)

    score_1 = interp(0.01,fpr,tpr)
    score_2 = interp(0.001,fpr,tpr)
    score_3 = interp(0.0001,fpr,tpr)

    # line1 = f'TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}\n'
    # line2 = f'APCER:{APCER:.6F}, NPCER:{NPCER:.6F}, ACER:{ACER:.6F}\n' 
    # line3 = f'FPR=e2:{score_1:.6f}, FPR=e3:{score_2:.6f}, FPR=e4:{score_3:.6f}'
    # message = line1 + line2 + line3
    message = f'TP:{tp}, TN:{tn}, FP:{fp}, FN:{fn}, ACER:{ACER:.6F}'

    return message, score_3, ACER