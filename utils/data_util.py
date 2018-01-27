#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 1/27/17
"""
import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

# import cPickle
import pickle
import pandas as pd
from conf.configure import Configure


def load_train():
    if not os.path.exists(Configure.cleaned_x_train_path):
        with open(Configure.x_train_path, "rb") as f:
            x_train = pickle.load(f)
#             x_train = cPickle.load(f)
        with open(Configure.y_train_path, "rb") as f:
            y_train = pickle.load(f)
#             y_train = cPickle.load(f)
    else:
        with open(Configure.cleaned_x_train_path, "rb") as f:
            x_train = pickle.load(f)
        with open(Configure.cleaned_y_train_path, "rb") as f:
            y_train = pickle.load(f)
    
    print('x_train:', x_train.shape, ', y_train:', y_train.shape)
    return x_train, y_train

def load_test(xTest = True):
    with open(Configure.test_fname_path, "rb") as f:
        test_fname = pickle.load(f)
    
    if xTest:
        if not os.path.exists(Configure.processed_test_path):
            with open(Configure.x_test_path, "rb") as f:
                x_test = pickle.load(f)

        print('x_test:', x_test.shape, ', test_fname:', len(test_fname))
        return x_test, test_fname 
    
    print('test_fname:', len(test_fname))
    return test_fname 

def save_processed_dataset(x_train, y_train, x_test=None):
    if x_train is not None:
        with open(Configure.processed_x_train_path, "wb") as f:
            pickle.dump(x_train, f, -1)
#             cPickle.dump(x_train, f, -1)
            
    if y_train is not None:
        with open(Configure.processed_y_train_path, "wb") as f:
            pickle.dump(y_train, f, -1)
#             cPickle.dump(y_train, f, -1)

    if x_test is not None:
        with open(Configure.processed_x_test_path, "wb") as f:
            pickle.dump(x_test, f, -1)
#             cPickle.dump(x_test, f, -1)
            
def save_cleaned_dataset(x_train, y_train):
    if x_train is not None:
        with open(Configure.cleaned_x_train_path, "wb") as f:
            pickle.dump(x_train, f, -1)
#             cPickle.dump(x_train, f, -1)
            
    if y_train is not None:
        with open(Configure.cleaned_y_train_path, "wb") as f:
            pickle.dump(y_train, f, -1)
#             cPickle.dump(y_train, f, -1)