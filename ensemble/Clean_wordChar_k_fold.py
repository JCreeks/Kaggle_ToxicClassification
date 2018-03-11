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
from utils import data_util
from utils.k_fold_utils import train_folds
from utils.data_util import max_len
from utils.model_wrapper import XgbWrapper, SklearnWrapper, LGBWrapper, GridCVWrapper

SEED = 2018
fold_count = 3

x_train, y_train, x_test = data_util.load_dataset(processed_x_train_path=conf.clean_wordChar_x_train_path,
                       processed_y_train_path=conf.clean_wordChar_y_train_path,
                       processed_x_test_path=conf.clean_wordChar_x_test_path)

# xgb_params = {'n_trees': 520, 
#                'eta': 0.3, 
#                'max_depth': 5, 
#                'subsample': 0.8, 
#                'objective': 'binary:logistic', 
#                'eval_metric': 'auc', 
#                'silent': 1, 
# #                'base_score': y_mean
#              }

# model = XgbWrapper(seed=SEED, params=xgb_params, cv_fold=4)

lgb_params = {
        "objective": "binary",
        'metric': {'auc'},
        "boosting_type": "gbdt",
        "verbosity": -1,
        "num_threads": 4,
        "bagging_fraction": 0.8,
        "feature_fraction": 0.8,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1,
        "rounds": 500
    }

model = LGBWrapper(seed=SEED, params=lgb_params)

def train_meta_prob_val(X, y, fold_count, model):
    trained_y = y
    fold_size = len(X) // fold_count
    models = []
    for fold_id in range(0, fold_count):
        print("training fold: ", fold_id)
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        test_x = X[fold_start:fold_end]

        model.train(train_x, train_y)
        
        trained_y[fold_start:fold_end] = model.predict(test_x)

    return trained_y

# class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_pred = train_y
for j in range(6):
    y = train_y[:,j]
    y_pred[:,j] = train_meta_prob_val(train_x, y, fold_count, model)

sub = pd.read_csv('../input/sample_submission.csv')
INPUT_COLUMN = "comment_text"
LABELS = train.columns[1:]
test_predicts_path = os.path.join(conf.output_path, "LGB_train_predicts_{}fold.csv".format(fold_count))
sub[LABELS] = y_pred 
sub.to_csv(test_predicts_path, index=False)