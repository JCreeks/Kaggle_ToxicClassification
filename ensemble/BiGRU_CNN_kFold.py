import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf
from utils import data_util
from utils.k_fold_utils import train_folds
from utils.data_util import max_len

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

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras.models import Model, load_model
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras import backend as K
from keras.engine import InputSpec, Layer
from keras.optimizers import Adam, RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import GRU, BatchNormalization, Conv1D, MaxPooling1D

maxlen = max_len()
batch_size = 256
dropout_rate = .2
recurrent_units = 128
dense_size = 6
fold_count = 3

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
x_train, y_train, x_test = data_util.load_dataset()

def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size, lr = 1e-3, lr_d = 0):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    
    x = SpatialDropout1D(dropout_rate)(embedding_layer)

    x = Bidirectional(GRU(recurrent_units, return_sequences = True))(x)
    x = Conv1D(64, kernel_size = 2, padding = "valid", kernel_initializer = "he_uniform")(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    x = concatenate([avg_pool, max_pool])

    x = Dense(dense_size, activation = "sigmoid")(x)
    model = Model(inputs = input_layer, outputs = x)
    model.compile(loss = "binary_crossentropy", optimizer = Adam(lr = lr, decay = lr_d), metrics = ["accuracy"])
    return model

embedding_matrix = data_util.load_embedding_matrix()
get_model_func = lambda: get_model(
        embedding_matrix,
        maxlen,
        dropout_rate,
        recurrent_units,
        dense_size)

# modelName = 'dropRate'+str(dropout_rate)+'_bSize'+str(batch_size)+'_epoch'+str(epochs)+'_GRU'+'.model'
# modelFile = os.path.join(conf.model_path, modelName)
# file_path="weights_base.best.hdf5"
# file_path="checkpoints/"+modelName+".hdf5"

print("Starting to train models...")
models = train_folds(x_train, y_train, fold_count, batch_size, get_model_func)

'''
print("Predicting results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    model_path = os.path.join(conf.model_path, "BiGRU_CNN_model{0}_weights.npy".format(fold_id))
    np.save(model_path, model.get_weights())

    test_predicts_path = os.path.join(conf.output_path, "BiGRU_CNN_test_predicts{0}.npy".format(fold_id))
    test_predicts = model.predict(x_test, batch_size=batch_size)
    test_predicts_list.append(test_predicts)
    np.save(test_predicts_path, test_predicts)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))
# test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_classes] = test_predicts

sample_submission.to_csv(conf.submission_path, index=False)
'''

print("Predicting results...")
test_predicts_list = []
trained_y = y_train.astype(np.float32)
fold_size = len(x_train) // fold_count
for fold_id, model in enumerate(models):
#     model_path = os.path.join(conf.model_path, "model{0}_weights.h5".format(fold_id))
#     model.get_weights(model_path)

    test_predicts_path = os.path.join(conf.output_path, "test_predicts{0}.npy".format(fold_id))
    test_predicts = model.predict(x_test, batch_size=batch_size)
    test_predicts_list.append(test_predicts)
    np.save(test_predicts_path, test_predicts)
    
    print("training fold: ", fold_id)
    fold_start = fold_size * fold_id
    fold_end = fold_start + fold_size
    if fold_id == fold_size - 1:
        fold_end = len(X)

    test_x = x_train[fold_start:fold_end]
    trained_y[fold_start:fold_end] = model.predict(test_x, batch_size=batch_size)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = test_predicts
sample_submission.to_csv("../output/trained_models/sub3.csv", index=False)

sample_submission = pd.read_csv("../input/train.csv")
sample_submission[list_classes] = trained_y
sample_submission.to_csv("../output/trained_models/oof3.csv", index=False)
