from __future__ import print_function, division
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

from detection.model_detection import Vnet2dlandmarkdetectionModule


def inference():
    detection = Vnet2dlandmarkdetectionModule(512, 512, channels=1, numclass=37, costname=('L2-loss',), inference=True,
                                              model_path='log\L2-loss\model/resnet.pd')
    test_path = "dataprocess\data/3167.mha"
    coords_pos, coords_val = detection.inference(test_path)
    print(coords_pos)

    

inference()
