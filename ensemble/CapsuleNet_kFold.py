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
from keras.layers import GRU, CuDNNGRU, BatchNormalization, Conv1D, MaxPooling1D

maxlen = max_len()
batch_size = 256
fold_count = 3

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
x_train, y_train, x_test = data_util.load_dataset()
embedding_matrix = data_util.load_embedding_matrix()

from keras.layers import K, Activation
from keras.engine import Layer
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D
gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_model():
    input_layer = Input(shape=(maxlen, ))
    embed_layer = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(input_layer)
    
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)

    #x = Bidirectional(CuDNNGRU(gru_len, return_sequences=True))(embed_layer)
    x = Bidirectional(GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(6, activation='sigmoid')(capsule)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model

get_model_func = lambda: get_model()

print("Starting to train models...")
models = train_folds(x_train, y_train, fold_count, batch_size, get_model_func)

'''
print("Predicting results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    model_path = os.path.join(conf.model_path, "CapsuleNet_{0}_weights.npy".format(fold_id))
    np.save(model_path, model.get_weights())

    test_predicts_path = os.path.join(conf.output_path, "CapsuleNet_predicts{0}.npy".format(fold_id))
    test_predicts = model.predict(x_test, batch_size=batch_size)
    test_predicts_list.append(test_predicts)
    np.save(test_predicts_path, test_predicts)

'''

'''
#####################
test_predicts_list = []
for fold_id in range(fold_count):
    test_predicts_path = os.path.join(conf.output_path, "CapsuleNet_predicts{0}.npy".format(fold_id))
    if os.path.exists(test_predicts_path):
        test_predicts = np.load(test_predicts_path)
        test_predicts_list.append(test_predicts)
        continue
#####################

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
trained_y = y_train
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
    trained_y[fold_start:fold_end] = model.predict(test_x)

test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

test_predicts **= (1. / len(test_predicts_list))

sample_submission = pd.read_csv("../input/sample_submission.csv")
sample_submission[list_classes] = test_predicts
sample_submission.to_csv("../output/trained_models/sub5.csv", index=False)

sample_submission = pd.read_csv("../input/train.csv")
sample_submission[list_classes] = trained_y
sample_submission.to_csv("../output/trained_models/oof5.csv", index=False)