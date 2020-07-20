import argparse
import collections
import json

import load

from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, precision_recall_fscore_support
from tensorflow import keras
import scipy.stats as sst
import numpy as np
import sklearn.metrics as skm

def predict(parser):
    val = load.load_dataset("data/validation.json")
    preproc = load.preproc(*val)

    args = parser.parse_args()
    model = keras.models.load_model(args.model)

    with open("data/train.json", "rb") as fid:
        val_labels = [json.loads(l)['labels'] for l in fid]

    counts = collections.Counter(preproc.class_to_int[l[0]] for l in val_labels)
    counts = sorted(counts.most_common(), key=lambda x: x[0])
    counts = list(zip(*counts))[1]

    print("counts : " , counts)

    smooth = 500
    counts = np.array(counts)[None, None, :]
    total = np.sum(counts) + counts.shape[1]
    print("total : ", total)
    prior = (counts + smooth) / float(total)    # ???
    print("prior : ", prior)

    probs = []
    labels = []

    ecgs, committee_labels = preproc.process(*val)
    m_probs = model.predict(ecgs)

    #print(type(ecgs), ecgs.shape)
    #print(type(committee_labels), committee_labels.shape)

    '''
    for x, y in zip(*val):
        x, y = preproc.process([x], [y])
        probs.append(model.predict(x))
        labels.append(y)

    preds = []
    ground_truth = []

    for p, g in zip(probs, labels):        
        preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
        ground_truth.append(sst.mode(np.argmax(g, axis=2).squeeze())[0][0])

    print("==============")

    '''
    committee_labels = np.argmax(committee_labels, axis=2)
    committee_labels = committee_labels[:, 0]
    #print(committee_labels)
    #print(committee_labels.shape)

    print("===================")
    temp = []
    preds = np.argmax(m_probs / prior, axis = 2)
    for i, j in zip(preds, val_labels):
        t = sst.mode(i[:len(j)-1])[0][0]
        temp.append(t)
        print(i[:len(j)-1])

    preds = temp

    print("preds : \n", preds)

    report = skm.classification_report(committee_labels, preds, target_names=preproc.classes, digits=3)
    scores = skm.precision_recall_fscore_support(committee_labels, preds, average=None)
    print("report : \n", report)
    #print("scores : ", scores)

    cm = confusion_matrix(committee_labels, preds)
    print("confusion matrix : \n", cm)

    f1 = f1_score(committee_labels, preds, average='micro')
    print("f1_score : ", f1)

    
    # ***roc_auc_score - m_probs***

    m_probs = np.sum(m_probs, axis=1)
    m_probs = m_probs / 71  # one data set max size (element count) -> normalization

    #print(ground_truth.shape, m_probs.shape)

    ovo_auroc = roc_auc_score(committee_labels, m_probs, multi_class='ovo')
    ovr_auroc = roc_auc_score(committee_labels, m_probs, multi_class='ovr')

    print("ovr_auroc : ", ovr_auroc)
    print("ovo_auroc : ", ovo_auroc)
    '''

    print(ground_truth)
    print("=====================")
    print(preds)

    report = skm.classification_report(ground_truth, preds, target_names=preproc.classes, digits=3 )
    scores = skm.precision_recall_fscore_support(ground_truth, preds, average=None)
    print("report : ", report)
    print("scores : ", scores)


    cm = confusion_matrix(ground_truth, preds)
    print(cm)

    f1 = f1_score(ground_truth, preds, average='micro')
    print("f1_score : ", f1)
    '''

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/model/base.hmd5")

    predict(parser)
