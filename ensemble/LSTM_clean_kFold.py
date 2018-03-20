import os
import sys
import re

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf
from utils import data_util
from utils.k_fold_utils import train_folds
from utils.data_util import max_len

import numpy as np
np.random.seed(42)
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from keras.models import Model
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import LSTM, Embedding, Dropout, Activation
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalMaxPool1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras import initializers, regularizers, constraints, optimizers, layers

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

maxlen = max_len()
dropout_rate = .1
recurrent_units = 64
batch_size = 256
dense_size1 = 50
dense_size2 = 6
epochs = 2
fold_count = 3

# X_train, y_train, X_test = data_util.load_dataset(processed_x_train_path=conf.glove_x_train_path, processed_y_train_path=conf.glove_y_train_path, processed_x_test_path=conf.glove_x_test_path)

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
x_train, y_train, x_test = data_util.load_dataset()

# embedding_matrix = data_util.load_embedding_matrix(file_name=conf.glove_matrix)
embedding_matrix = data_util.load_embedding_matrix()

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def get_model(embedding_matrix, sequence_length, dropout_rate, recurrent_units, dense_size1, dense_size2):
    input_layer = Input(shape=(sequence_length,))
    embedding_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    x = Bidirectional(LSTM(recurrent_units, return_sequences=True, dropout=0.2, recurrent_dropout=0.1))(embedding_layer)
    x = GlobalMaxPool1D()(x)
    x = Dense(dense_size1, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_size2, activation="sigmoid")(x)
    model = Model(inputs=input_layer, outputs=x)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

get_model_func = lambda: get_model(
        embedding_matrix,
        maxlen,
        dropout_rate,
        recurrent_units,
        dense_size1, 
        dense_size2)


print("Starting to train models...")
print('num of folds: {}'.format(fold_count))
models = train_folds(x_train, y_train, fold_count, batch_size, get_model_func)

'''
print("Predicting results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    model_path = os.path.join(conf.model_path, "LSTM_clean_{0}_weights.npy".format(fold_id))
    np.save(model_path, model.get_weights())

    test_predicts_path = os.path.join(conf.output_path, "LSTM_clean_predicts{0}.npy".format(fold_id))
    test_predicts = model.predict(x_test, batch_size=batch_size)
    test_predicts_list.append(test_predicts)
    np.save(test_predicts_path, test_predicts)
'''
'''
#####################
test_predicts_list = []
for fold_id in range(fold_count):
    test_predicts_path = os.path.join(conf.output_path, "LSTM_clean_predicts{0}.npy".format(fold_id))
    if os.path.exists(test_predicts_path):
        test_predicts = np.load(test_predicts_path)
        test_predicts_list.append(test_predicts)
        continue
#####################
'''

'''
test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))

submission = pd.read_csv("../input/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = test_predicts
submission.to_csv(conf.submission_path, index=False)
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
sample_submission.to_csv("../output/trained_models/sub4.csv", index=False)

sample_submission = pd.read_csv("../input/train.csv")
sample_submission[list_classes] = trained_y
sample_submission.to_csv("../output/trained_models/oof4.csv", index=False)
