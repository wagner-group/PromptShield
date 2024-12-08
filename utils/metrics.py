import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def computeROC(scores, labels):
    # ROC curve
    fpr, tpr, thre = roc_curve(labels, scores)

    # Area-under-curve (AUC) associated with ROC curve
    auc_score = roc_auc_score(labels, scores)

    return fpr, tpr, thre, auc_score

def interpolateROC(fpr_val, fpr_list, tpr_list):
    return np.interp(fpr_val, fpr_list, tpr_list)

def interpolateThre(fpr_val, fpr_list, thre):
    return np.interp(fpr_val, fpr_list, thre)