#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 22:11:11 2019

@author: tunan
"""
import numpy as np
import pandas as pd
from tqdm import tqdm 
import os,pickle
from sklearn.utils import shuffle

if __name__ == "__main__":
    train_data = pd.read_csv("inputs/zhengqi_train.txt",sep = '\t')
    
    #查看数据是否有缺失值
    print("缺失值个数为:{}".format(np.sum(np.isnan(train_data).values)))
    
