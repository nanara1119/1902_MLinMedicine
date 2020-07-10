import argparse
import collections
import datetime
import json
import os
import time

import architecture
import load
import numpy as np
import scipy.stats as sst
from tensorflow import keras

default_lr = 0.001

def load_train_val_data(parser):
    print("load_train_val_data ... ")

    train = load.load_dataset("data/train.json")
    val = load.load_dataset("data/validation.json")
    preproc = load.preproc(*train)

    train_x, train_y = preproc.process(*train)
    val_x, val_y = preproc.process(*val)

    print("train size : {}, {}".format(len(train_x), len(train_y)))
    print("val size : {}, {}".format(len(val_x), len(val_y)))
    args = parser.parse_args()

    model = architecture.build_model()
    #print(model.summary())

    save_dir = make_save_dir("data/", "model")
    file_name = get_filename_for_saving(save_dir)
    check_pointer = keras.callbacks.ModelCheckpoint(
        filepath=file_name,
        save_best_only=False)
    stopping = keras.callbacks.EarlyStopping(patience=10)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=2, min_lr=default_lr*0.001)

    model.fit(train_x, train_y,batch_size = int(args.batchsize), epochs = int(args.epochs),
              validation_data=(val_x, val_y), callbacks = [check_pointer, reduce_lr])


def get_filename_for_saving(save_dir):
    return os.path.join(save_dir,
            "{epoch:03d}-{val_loss:.3f}-{val_accuracy:.3f}-{loss:.3f}-{accuracy:.3f}.hdf5")

def make_save_dir(dirname, experiment_name):
    c_time = datetime.datetime.now()
    start_time = "{}{}{}{}{}".format(c_time.year, c_time.month, c_time.day, c_time.hour, c_time.minute)
    #start_time = str(int(time.time())) + '-' + str(random.randrange(1000))
    save_dir = os.path.join(dirname, experiment_name, start_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return save_dir

if __name__ == '__main__':
    print("start train")

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=100)
    parser.add_argument("--batchsize", default=32)

    start_time = time.time()
    load_train_val_data(parser)
    print("total time : ", (time.time() - start_time))


