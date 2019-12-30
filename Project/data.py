import json
import os
import random
import zipfile
from pathlib import Path

import scipy.io as sio
import tqdm

#   논문에서 매 256 샘플마다 추출(?) 함
STEP = 256


def load_data(folder_path):
    """
    1. train과 validation에 필요한 training2017 파일의 존재 확인
    2. training2017.zip 파일 압축 해제
    3. *.mat 파일의 데이터를 읽어 list로 반환

    :param folder_path: 파일 경로
    :return: 파일 정보를 변환한 list
    """

    #   1. training2017 zip file check
    zip_file = 'data/training2017.zip'
    zip_file_path = Path(zip_file)
    if not zip_file_path.exists():
        print("training2017.zip not exist ... downloading ... ")
        cmd = "mkdir data && cd data && curl -O https://archive.physionet.org/challenge/2017/training2017.zip"
        os.system(cmd)

    #   2. training2017 folder check
    zip_folder_path = Path(folder_path)
    if not zip_folder_path.exists():
        print("training2017 folder not exist ... unzip training2017.zip ... ")
        zip = zipfile.ZipFile('data/training2017.zip')
        zip.extractall('data/')
        zip.close()

    #   3. 파일 내용과 label을 합쳐 리스트 형식으로 반환
    label_file = os.path.join(folder_path, "REFERENCE.csv")

    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]

    dataset = []

    for record, label in tqdm.tqdm(records):
        ecg_file = os.path.join(folder_path, record + ".mat")
        ecg_file = os.path.abspath(ecg_file)
        ecg = load_ecg_mat(ecg_file)
        num_labels = ecg.shape[0] / STEP
        dataset.append((ecg_file, [label] * int(num_labels)))
    return dataset


def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()


def split(dataset, ratio):
    split_size = int(ratio * len(dataset))
    random.shuffle(dataset)
    val = dataset[:split_size]
    train = dataset[split_size:]

    return train, val


def make_json(save_path, dataset):
    with open(save_path, 'w') as fid:
        for d in dataset:
            datum = {'ecg': d[0], 'labels': d[1]}
            json.dump(datum, fid)
            fid.write('\n')


if __name__ == "__main__":
    print("data check")
    folder_path = 'data/training2017/'
    dataset = load_data(folder_path)
    train, val = split(dataset, 0.1)
    make_json("data/train.json", train)
    make_json("data/validation.json", val)
