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

def max_len():
    return 500

def load_dataset():
    if not os.path.exists(Configure.processed_x_train_path):
        maxlen = max_len()
        with open(Configure.train_data_path, "rb") as f:
            train = pickle.load(f)
            list_sentences_train = train["comment_text"].fillna("CVxTz").values
            list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
            y_train = train[list_classes].values
            list_sentences_test = test["comment_text"].fillna("CVxTz").values
            x_train = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

        with open(Configure.x_test_data_path, "rb") as f:
            test = pickle.load(f)
            x_test = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)
    else:
        with open(Configure.processed_x_train_path, "rb") as f:
            x_train = pickle.load(f)
        with open(Configure.processed_y_train_path, "rb") as f:
            y_train = pickle.load(f)
        with open(Configure.processed_x_test_path, "rb") as f:
            x_test = pickle.load(f)
    
    print('x_train:', x_train.shape, ', y_train:', y_train.shape, ', x_test:', x_test.shape)
    return x_train, y_train, x_test

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

def save_embedding_matrix(embedding_matrix):
    if embedding_matrix is not None:
        with open(Configure.embedding_matrix, "wb") as f:
            pickle.dump(embedding_matrix, f, -1)
            
def load_embedding_matrix():
    if os.path.exists(Configure.embedding_matrix):
        with open(Configure.embedding_matrix, "rb") as f:
            embedding_matrix = pickle.load(f)
        return embedding_matrix
    return None
            
def save_cleaned_dataset(x_train, y_train):
    if x_train is not None:
        with open(Configure.cleaned_x_train_path, "wb") as f:
            pickle.dump(x_train, f, -1)
#             cPickle.dump(x_train, f, -1)
            
    if y_train is not None:
        with open(Configure.cleaned_y_train_path, "wb") as f:
            pickle.dump(y_train, f, -1)
#             cPickle.dump(y_train, f, -1)