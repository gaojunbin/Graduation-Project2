# -*- coding: utf-8 -*-
"""
Created on 2019.12.4
latest modify 2020.5.13
@author: Junbin
@note: Train
"""

import tensorflow as tf
import numpy as np
import os
import argparse
import random
import yaml
from Network import *
from TrainModel import *
from DatasetReload import *


seed = 123
random.seed(seed)
np.random.seed(seed)


parser = argparse.ArgumentParser(description='param about 2d point cloud training')
parser.add_argument('--config', default='config/config.yaml', type=str)

def main():
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f)
    print("\n**************************")
    for k, v in config['common'].items():
        setattr(args, k, v)
        print('\n[%s]:'%(k), v)
    print("\n**************************\n")
    
    save_dir = dir_apply(checkpoint_path=args.checkpoint_path,logs_path=args.logs_path)
    save_dir()
    pointcloud = PointCloud(
        IsTrain=args.IsTrain,CrossValidation_pro=args.CrossValidation_pro,num_classes=args.num_classes,
        dataset_dir=args.dataset_dir)
    trainmodel = TrainModel(checkpoint_path=args.checkpoint_path,logs_path=args.logs_path,BATCH_SIZE=args.BATCH_SIZE)
    TrainDataset,TrainLable,ValDataset,ValLable = pointcloud()
    trainmodel(TrainDataset,TrainLable,ValDataset,ValLable)
    


if __name__ == "__main__":
    main()