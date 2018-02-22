import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

from conf.configure import Configure as conf
from utils import data_util
from utils.GRUtrain_utils import train_folds
from utils.data_util import max_len

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
def AUC(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.models import load_model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout, CuDNNGRU, GRU
from keras.layers import SpatialDropout1D, concatenate
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K

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


maxlen = max_len()
batch_size = 128
recurrent_units = 64
dense_size = 32
epochs = 3

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
x_train, y_train, x_test = data_util.load_dataset()


embedding_matrix = data_util.load_embedding_matrix()
print('embedding size: ', embedding_matrix.shape[0], ', ',embedding_matrix.shape[1])
def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1], weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(0.2)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(6, activation="sigmoid")(conc)
    
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

model = get_model()

modelName = 'bSize'+str(batch_size)+'_epoch'+str(epochs)+'_PooledGRU'
modelFile = os.path.join(conf.model_path, modelName)
# file_path="weights_base.best.hdf5"
file_path="checkpoints/"+modelName+'.hdf5'

print("Starting to train models...")
[X_tra, X_val, y_tra, y_val] = train_test_split(x_train, y_train, train_size=0.75, random_state=233)
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)

exp_decay = lambda init, fin, steps: (init/fin)**(1/(steps-1)) - 1
steps = int(len(X_tra)/batch_size) * epochs
lr_init, lr_fin = 0.001, 0.0005
lr_decay = exp_decay(lr_init, lr_fin, steps)
K.set_value(model.optimizer.lr, lr_init)
K.set_value(model.optimizer.decay, lr_decay)

if os.path.exists(file_path):
    model.load_weights(file_path)
else:
    print("Start training")
    model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=2)
    model.save_weights(file_path)

print("Predicting results...")
y_pred = model.predict(x_test, batch_size=1024)

sample_submission = pd.read_csv("../input/sample_submission.csv")

sample_submission[list_classes] = y_pred

sample_submission.to_csv(conf.submission_path, index=False)
