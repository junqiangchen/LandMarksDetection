import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from detection.model_detection import Vnet2dlandmarkdetectionModule
import numpy as np
import pandas as pd


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvdata = pd.read_csv('dataprocess\\data/landmarktraining.csv')
    maskdata = csvdata.iloc[:, 1].values
    imagedata = csvdata.iloc[:, 0].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(imagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    detection = Vnet2dlandmarkdetectionModule(512, 512, channels=1, numclass=37, costname=('L2-loss',))
    detection.train(imagedata, maskdata, "resnet.pd", "log\\L2-loss\\", 0.001, 0.5, 300, 1, [6, 7])



train()
