import argparse
import collections
import json

import load
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from tensorflow import keras
import scipy.stats as sst
import numpy as np
import sklearn.metrics as skm

def predict(parser):
    val = load.load_dataset("data/validation.json")
    preproc = load.preproc(*val)

    args = parser.parse_args()
    model = keras.models.load_model(args.model)

    with open("data/validation.json", "rb") as fid:
        val_labels = [json.loads(l)['labels'] for l in fid]

    counts = collections.Counter(preproc.class_to_int[l[0]] for l in val_labels)
    counts = sorted(counts.most_common(), key=lambda x: x[0])
    counts = list(zip(*counts))[1]

    smooth = 500
    counts = np.array(counts)[None, None, :]
    total = np.sum(counts) + counts.shape[1]
    prior = (counts + smooth) / float(total)

    probs = []
    labels = []

    for x, y in zip(*val):
        x, y = preproc.process([x], [y])
        probs.append(model.predict(x))
        labels.append(y)

    preds = []
    ground_truth = []
    for p, g in zip(probs, labels):
        preds.append(sst.mode(np.argmax(p / prior, axis=2).squeeze())[0][0])
        ground_truth.append(sst.mode(np.argmax(g, axis=2).squeeze())[0][0])


    report = skm.classification_report(ground_truth, preds, target_names=preproc.classes, digits=3)
    scores = skm.precision_recall_fscore_support(ground_truth, preds, average=None)
    print(report)
    print("Average {}".format(np.mean(scores[2][:3])))

    cm = metrics.multilabel_confusion_matrix(ground_truth, preds)
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="data/model/base.hmd5")

    predict(parser)
