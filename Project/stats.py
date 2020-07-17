import json
import keras
import math
import numpy as np
import os
import sklearn
import sklearn.metrics as skm
import sys

import load
import utils
import scipy.stats as sst
from sklearn.metrics import f1_score, confusion_matrix

model_path = "data/model/2020711659/017-0.230-0.923-0.126-0.956.hdf5"
data_json = "data/validation.json"

val = load.load_dataset("data/validation.json")
preproc = load.preproc(*val)
#preproc = utils.load(os.path.dirname(model_path))
#dataset = load.load_dataset(data_json)

'''
ecgs = []
committee_labels = []
for x, y in zip(*val) :
    ecgs, committee_labels = preproc.process([x], [y])   
'''
ecgs, committee_labels = preproc.process(*val)

model = keras.models.load_model(model_path)
probs = model.predict(ecgs, verbose=1)

def c_statistic_with_95p_confidence_interval(cstat, num_positives, num_negatives, z_alpha_2=1.96):
    """
    Calculates the confidence interval of an ROC curve (c-statistic), using the method described
    under "Confidence Interval for AUC" here:
      https://ncss-wpengine.netdna-ssl.com/wp-content/themes/ncss/pdf/Procedures/PASS/Confidence_Intervals_for_the_Area_Under_an_ROC_Curve.pdf
    Args:
        cstat: the c-statistic (equivalent to area under the ROC curve)
        num_positives: number of positive examples in the set.
        num_negatives: number of negative examples in the set.
        z_alpha_2 (optional): the critical value for an N% confidence interval, e.g., 1.96 for 95%,
            2.326 for 98%, 2.576 for 99%, etc.
    Returns:
        The 95% confidence interval half-width, e.g., the Y in X ± Y.
    """
    q1 = cstat / (2 - cstat)
    q2 = 2 * cstat**2 / (1 + cstat)
    numerator = cstat * (1 - cstat) \
        + (num_positives - 1) * (q1 - cstat**2) \
        + (num_negatives - 1) * (q2 - cstat**2)
    standard_error_auc = math.sqrt(numerator / (num_positives * num_negatives))
    return z_alpha_2 * standard_error_auc

def roc_auc(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=2)
    n_gts = np.zeros_like(gts)
    n_gts[gts==index] = 1
    n_pos = np.sum(n_gts == 1)
    n_neg = n_gts.size - n_pos
    n_ps = probs[..., index].squeeze()
    n_gts, n_ps = n_gts.ravel(), n_ps.ravel()
    #print("type : ", type(n_gts), type(n_ps))
    #print("shape : ", n_gts.shape, n_ps.shape)
    #print(n_gts)
    #print(n_ps)

    return n_pos, n_neg, skm.roc_auc_score(n_gts, n_ps)

def roc_auc_set(ground_truth, probs, index):
    gts = np.argmax(ground_truth, axis=2)
    max_ps = np.max(probs[...,index], axis=1)   #   이상
    max_gts = np.any(gts==index, axis=1)    #   이상
    pos = np.sum(max_gts)
    neg = max_gts.size - pos

    if index == 3:
        #print(max_gts)
        #print("=====")
        #print(max_ps)
        print("111 : \n", probs[..., 3] )
        print("=====")
        print(max_ps)
    auc = skm.roc_auc_score(max_gts, max_ps)
    return pos, neg, auc

def print_aucs(auc_fn, ground_truth, probs):
    macro_average = 0.0; total = 0.0
    #print(type(ground_truth), type(probs))
    #print(ground_truth.shape, probs.shape)
    for idx, cname in preproc.int_to_class.items():
        pos, neg, auc = auc_fn(ground_truth, probs, idx)
        total += pos
        macro_average += pos * auc
        conf = c_statistic_with_95p_confidence_interval(auc, pos, neg)
        print("{: <8}\t{:.3f} ({:.3f}-{:.3f})".format(cname, auc, auc-conf,auc+conf))
    print("Average\t\t{:.3f}".format(macro_average / total))

def stats(ground_truth, preds):
    labels = range(ground_truth.shape[2])
    g = np.argmax(ground_truth, axis=2).ravel()
    p = np.argmax(preds, axis=2).ravel()
    stat_dict = {}
    for i in labels:
        # compute all the stats for each label
        tp = np.sum(g[g==i] == p[g==i])
        fp = np.sum(g[p==i] != p[p==i])
        fn = np.sum(g==i) - tp
        tn = np.sum(g!=i) - fp
        stat_dict[i] = (tp, fp, fn, tn)
        print(i)
        print(tp, fp, fn, tn)
    return stat_dict

def print_results(sd):
    print("\t\tPrecision  Recall     Specificity     NPV        F1")
    tf1 = 0; tot = 0
    for k, v in sd.items():
        tp, fp, fn, tn = v
        precision = tp / float(tp + fp)
        recall = tp / float(tp + fn)
        specificity = tn / float(tn + fp)
        npv = tn / float(tn + fn)
        f1 = 2 * precision * recall / (precision + recall)
        tf1 += (tp + fn) * f1
        tot += (tp + fn)
        print("{:>10} {:10.3f} {:10.3f} {:10.3f} {:15.3f} {:10.3f}".format(preproc.classes[k], precision, recall, specificity, npv, f1))
    print("Average F1 {:.3f}".format(tf1 / float(tot)))

print("Sequence level AUC")
print_aucs(roc_auc, committee_labels, probs)

print("Set level AUC")
print_aucs(roc_auc_set, committee_labels, probs)

print("Model")
print_results(stats(committee_labels, probs))

#print(confusion_matrix(np.argmax(committee_labels, axis=2).ravel(), np.argmax(probs, axis=2).ravel()))