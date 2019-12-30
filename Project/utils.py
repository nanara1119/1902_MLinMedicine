import os
import pickle as pickle

def load(dirname):
    preporc = os.path.join(dirname, 'preproc.bin')
    with open(preporc, 'rb') as fid :
        preporc = pickle.load(fid)
    return preporc

def save(preproc, dirname):
    t_preproc = os.path.join(dirname, 'preproc.bin')
    with open(t_preproc, 'wb') as fid:
        pickle.dump(preproc.fid)

