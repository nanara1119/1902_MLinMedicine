import json
import os

import numpy as np
import scipy.io as sio

import tqdm
from tensorflow import keras

STEP = 256

class preproc:
    def __init__(self, ecg, labels):
        self.mean, self.std = self.compute_mean_std(ecg)
        self.classes = sorted(set(l for label in labels for l in label))
        self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
        self.class_to_int = {c: i for i, c in self.int_to_class.items()}

    def process(self, x, y):
        return self.process_x(x), self.process_y(y)

    def process_x(self, x):
        x = self.pad(x)
        x = (x - self.mean) / self.std
        x = x[:, :, None]
        return x

    def process_y(self, y):
        y = self.pad([[self.class_to_int[c] for c in s] for s in y], val=3, dtype=np.int32)
        y = keras.utils.to_categorical(y, num_classes=len(self.classes))
        return y

    def compute_mean_std(self, x):
        x = np.hstack(x)
        return (np.mean(x).astype(np.float32), np.std(x).astype(np.float32))

    def pad(self, x, val=0, dtype=np.float32):
        max_len = max(len(i) for i in x)
        padded = np.full((len(x), max_len), val, dtype=dtype)
        for e, i in enumerate(x):
            padded[e, :len(i)] = i
        return padded


def load_dataset(data_json):
    with open(data_json, 'r') as fid:
        data = [json.loads(l) for l in fid]
    labels = [];
    ecgs = []
    for d in tqdm.tqdm(data):
        labels.append(d['labels'])
        ecgs.append(load_ecg(d['ecg']))
    return ecgs, labels


def load_ecg(record):
    if os.path.splitext(record)[1] == ".npy":
        ecg = np.load(record)
    elif os.path.splitext(record)[1] == ".mat":
        ecg = sio.loadmat(record)['val'].squeeze()
    else:
        with open(record, 'r') as fid:
            ecg = np.fromfile(fid, dtype=np.int16)
    trunc_samp = STEP * int(len(ecg) / STEP)
    return ecg[:trunc_samp]

