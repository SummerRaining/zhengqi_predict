#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 21:51:49 2019

@author: tunan
"""

from sklearn.metrics import mean_squared_error
import os
import numpy as np


def print_analyse(ytrue,ypred,name):
    '''
    使用真实值和预测值，画出ROC曲线,auc值，一二类错误率等评估信息。
    inputs:
        ytrue,yproba(array)
    '''
    #计算auc的值
    mse = mean_squared_error(ytrue,ypred)
    print("*"*10+" {} ".format(name)+"*"*10)
    print("{}验证集上的均方误差为{:.4f}".format(name,mse))

    