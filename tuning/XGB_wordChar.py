# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import xgboost as xgb
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import log_loss, roc_auc_score
import os, sys

import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# from configure import Configure as conf
module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf
from utils.clean_util import TextCleaner

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import time

from utils.data_util import cleanWordChar

# CLEAN = cleanWordChar()
# print('##############')
# print('clean: ', CLEAN)
# print('##############')


x_train, y_train, x_test = data_util.load_dataset(processed_x_train_path=conf.clean_wordChar_x_train_path,
                       processed_y_train_path=conf.clean_wordChar_y_train_path,
                       processed_x_test_path=conf.clean_wordChar_x_test_path)

comments_train, comments_valid, train_l, valid_l = train_test_split(x_train, y_train, test_size=0.2, random_state=2018)


col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def score(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    score = 0
    for i, j in enumerate(col):
        dtrain = xgb.DMatrix(comments_train, label=train_l[j])
        dvalid = xgb.DMatrix(comments_valid, label=valid_l[j])
        watchlist = [(dvalid, 'eval'), (dtrain, 'train')]
        gbm_model = xgboost.train(params, 
                                  dtrain, 
                                  num_round,
                                  evals=watchlist,
                                  verbose_eval=False)
        predictions = gbm_model.predict(dvalid, ntree_limit=gbm_model.best_iteration)
        score += roc_auc_score(valid_l[j], np.array(predictions))
    score /= 6.
    print( params )
    print('score: ', score)
    loss = - score
    return loss #{'loss': loss, 'status': STATUS_OK}
 
def opt(evals, cores, trials, optimizer=tpe.suggest, random_state=2018):
    space = {
        'n_estimators': hp.quniform('n_estimators', 200, 600, 1),
        'eta': hp.quniform('eta', 0.05, 0.15, 0.025), # A problem with max_depth casted to float instead of int with the hp.quniform method.
        'max_depth':  hp.choice('max_depth', np.arange(3, 9, dtype=int)),
        'min_child_weight': hp.quniform('min_child_weight', 1, 3, 1),
        'subsample': hp.quniform('subsample', 0.7, .9, 0.05),
#         'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, .9, 0.05),
#         'alpha' :  hp.quniform('alpha', 0, 10, 1),
#         'lambda': hp.quniform('lambda', 1, 2, 0.1),
        'nthread': cores,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'eval_metric' : 'auc',
        'silent' : 1,
        'seed': random_state
    }
    best = fmin(score, space, algo=tpe.suggest, max_evals=evals, trials = trials)
    return best
    
trials = Trials()
cores = 4
n= 100
start = time.time()
best_param = opt(evals = n,
                      optimizer=tpe.suggest,
                      cores = cores,
                      trials = trials)

print("------------------------------------")
print("The best hyperparameters are: ", "\n")
print(best_param)
end = time.time()
print('Time elapsed to optimize {0} executions: {1}'.format(n,end - start))


