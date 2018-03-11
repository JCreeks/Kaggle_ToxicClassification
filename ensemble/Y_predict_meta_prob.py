import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf
from utils import data_util
from utils.k_fold_utils import train_folds
from utils.data_util import max_len
from utils.model_utils import GRU_get_model, GRU_params
from utils.model_utils import BiGRU_CNN_get_model, BiGRU_CNN_params
from utils.model_utils import LSTM_get_model, LSTM_params

import warnings
warnings.filterwarnings('ignore')

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.metrics import roc_auc_score
def AUC(y_true, y_pred):
    return roc_auc_score(y_pred)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU, GRU
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import RMSprop

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X, y, x_test = data_util.load_dataset()
# embedding_matrix = data_util.load_embedding_matrix()
del x_test

# maxlen = max_len()
# batch_size = 256
# dropout_rate = .3
# recurrent_units = 64
# dense_size = 32
# fold_count = 3

get_model_func = lambda: GRU_get_model(
        **GRU_params())

trained_y = y
fold_size = len(X) // fold_count
models = []
for fold_id in range(0, fold_count):
    print("training fold: ", fold_id)
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size

    if fold_id == fold_size - 1:
        fold_end = len(X)

    test_x = X[fold_start:fold_end]

    model = get_model_func()
    
    model_path = os.path.join(conf.model_path, "model{0}_weights.npy".format(fold_id))
    model.load_weights(model_path)

    trained_y[fold_start:fold_end] = model.predict(test_x)
    
sub = pd.read_csv('../input/sample_submission.csv')
INPUT_COLUMN = "comment_text"
LABELS = sub.columns[1:]
test_predicts_path = os.path.join(conf.output_path, "GRU_train_predicts_{}fold.csv".format(fold_count))
sub[LABELS] = trained_y 
sub.to_csv(test_predicts_path, index=False)