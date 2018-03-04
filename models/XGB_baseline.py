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

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.fillna("unknown")
test = test.fillna("unknown")

CLEAN = True
print('##############')
print('clean: ', CLEAN)
print('##############')

if CLEAN:
    train['comment_text'] = train['comment_text'].apply(TextCleaner.clean_text2)
    test['comment_text'] = test['comment_text'].apply(TextCleaner.clean_text2)


train_mes, valid_mes, train_l, valid_l = train_test_split(train['comment_text'],train[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.2, random_state=2)

'''
def text_process(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
'''
#Couldnt remove the stop words using the above function since it is taking too long
#Can try it on a local machine, I feel it improves the score-Not sure though


'''
transform_com = CountVectorizer().fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
comments_train = transform_com.transform(train['comment_text'])
comments_test = transform_com.transform(test['comment_text'])
gc.collect()'''

#Using the tokenize function from Jeremy's kernel
import re, string
re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

transform_com = TfidfVectorizer(ngram_range=(1,2), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1).fit(pd.concat([train['comment_text'],test['comment_text']],axis=0))
'''comments_train = transform_com.transform(train['comment_text'])'''
comments_train = transform_com.transform(train_mes)
comments_valid = transform_com.transform(valid_mes)
comments_test = transform_com.transform(test['comment_text'])
gc.collect()

"""
def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=2017, num_rounds=400):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.1
    param['max_depth'] = 6
    param['silent'] = 1
    param['eval_metric'] = 'auc'
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param['seed'] = seed_val
    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [ (xgtrain,'train'), (xgtest, 'test') ]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=20)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    return model   

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test.shape[0], len(col)))

for i, j in enumerate(col):
    print('fit '+j)
    model = runXGB(comments_train, train_l[j], comments_valid,valid_l[j])
    preds[:,i] = model.predict(xgb.DMatrix(comments_test), ntree_limit = model.best_ntree_limit)
    gc.collect()
    
subm = pd.read_csv('../input/sample_submission.csv')    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv(conf.submission_path, index=False) 
"""

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
 
def opt(evals, cores, trials, optimizer=tpe.suggest, random_state=2017):
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
cores = 32
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


