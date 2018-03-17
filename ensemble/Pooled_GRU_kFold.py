import os
import sys

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
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

max_features = 30000
maxlen = max_len()
embed_size = 300
batch_size = 128
epochs = 2
fold_count = 3

print('loading data')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
x_train, y_train, x_test = data_util.load_dataset()
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


def get_model():
    inp = Input(shape=(maxlen, ))
    x = Embedding(embedding_matrix.shape[0], embedding_matrix.shape[1],
                                weights=[embedding_matrix], trainable=False)(inp)
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

get_model_func = lambda: get_model(
                            )


print("Starting to train models...")
models = train_folds(x_train, y_train, fold_count, batch_size, get_model_func)

'''
print("Predicting results...")
test_predicts_list = []
for fold_id, model in enumerate(models):
    model_path = os.path.join(conf.model_path, "Pooled_GRU_clean{0}_weights.npy".format(fold_id))
    np.save(model_path, model.get_weights())

    test_predicts_path = os.path.join(conf.output_path, "Pooled_GRU_clean_predicts{0}.npy".format(fold_id))
    test_predicts = model.predict(x_test, batch_size=batch_size)
    test_predicts_list.append(test_predicts)
    np.save(test_predicts_path, test_predicts)
'''
'''
#####################
test_predicts_list = []
for fold_id in range(fold_count):
    test_predicts_path = os.path.join(conf.output_path, "Pooled_GRU_clean_predicts{0}.npy".format(fold_id))
    if os.path.exists(test_predicts_path):
        test_predicts = np.load(test_predicts_path)
        test_predicts_list.append(test_predicts)
        continue
#####################
    
test_predicts = np.ones(test_predicts_list[0].shape)
for fold_predict in test_predicts_list:
    test_predicts *= fold_predict

# test_predicts **= (1. / len(test_predicts_list))

submission = pd.read_csv("../input/sample_submission.csv")
submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = test_predicts
submission.to_csv(conf.submission_path, index=False)
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
sample_submission.to_csv("../output/trained_models/sub0.csv", index=False)

sample_submission = pd.read_csv("../input/train.csv")
sample_submission[list_classes] = trained_y
sample_submission.to_csv("../output/trained_models/oof0.csv", index=False)

