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

import time


class Configure(object):

    model_path = '../models/savedModels'
    output_path = '../output'
    train_data_path = '../input/train.csv'
    x_test_data_path = '../input/test.csv'
    
    embedding_path = '../input/crawl-300d-2M.vec'
    embedding_matrix = '../input/embedding_matrix.pkl'
    
    glove_path = '../input/glove.42B.300d.txt'
    glove_matrix = '../input/glove_matrix.pkl'
    
    cleaned_x_train_path = '../input/cleaned_x_train.pkl'
    cleaned_y_train_path = '../input/cleaned_y_train.pkl'
    
    processed_x_train_path = '../input/processed_x_train.pkl'
    processed_y_train_path = '../input/processed_y_train.pkl'
    processed_x_test_path = '../input/processed_x_test.pkl'
    
    glove_x_train_path = '../input/glove_x_train.pkl'
    glove_y_train_path = '../input/processed_y_train.pkl'
    glove_x_test_path = '../input/glove_x_test.pkl'
    
    wordChar_x_train_path = '../input/wordChar_x_train.pkl'
    wordChar_y_train_path = '../input/processed_y_train.pkl'
    wordChar_x_test_path = '../input/wordChar_x_test.pkl'
    
    clean_wordChar_x_train_path = '../input/clean_wordChar_x_train.pkl'
    clean_wordChar_y_train_path = '../input/processed_y_train.pkl'
    clean_wordChar_x_test_path = '../input/clean_wordChar_x_test.pkl'
    
    outlierNameList = '../input/outlierNameList.pkl'

    submission_path = '../output/submission_{}.csv'.format(time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time())))
