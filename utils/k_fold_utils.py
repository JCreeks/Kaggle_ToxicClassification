from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import numpy as np
import pdb

import pickle


def _train_model(model, batch_size, train_x, train_y, val_x, val_y, metric = roc_auc_score):
    best_score = 2
    best_weights = None
    best_epoch = 0
    best_y_pred = val_y

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=0, verbose=2)#current_epoch
        y_pred = model.predict(val_x, batch_size=batch_size)

        total_score = 0
        count = 0.
        for j in range(6):
#             loss = log_loss(val_y[:, j], y_pred[:, j], eps=1e-7)
            score = metric(val_y[:, j], y_pred[:, j])
            if not np.isnan(score):
                total_score += score
                count +=1.

        total_score /= count
        
        print('count: ', count)
        print("Epoch {0} score {1} best_score {2}".format(current_epoch, total_score, best_score))

        current_epoch += 1
#         if total_loss < best_loss or best_loss == -1:
        if total_score > best_score or best_score == 2:
            best_score = total_score
            best_weights = model.get_weights()
            best_epoch = current_epoch
#             best_y_pred = y_pred
            break
        else:
            if current_epoch - best_epoch == 3:
                break

    model.set_weights(best_weights)
#     total_score = 0
#     for j in range(6):
#         score = metric(val_y[:, j], best_y_pred[:, j])
#         total_score += score
#     total_score /= 6.
    print('###################')
    print("best score: {}".format(best_score))
    print('###################')
    
    return model


def train_folds(X, y, fold_count, batch_size, get_model_func):
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

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        model = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        models.append(model)

    return models

def train_meta_prob_val(X, y, fold_count, batch_size, get_model_func):
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
        
        trn_x, val_x, trn_y, val_y = train_test_split(train_x, train_y, test_size=0.1, random_state=2018)

        model = _train_model(get_model_func(), batch_size, trn_x, trn_y, val_x, val_y)
        
        trained_y[fold_start:fold_end] = model.predict(test_x)

    return trained_y

def train_meta_prob(X, y, fold_count, model):
    trained_y = y
    fold_size = len(X) // fold_count
    for fold_id in range(0, fold_count):
        print("training fold: ", fold_id)
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_size - 1:
            fold_end = len(X)

        train_x = np.concatenate([X[:fold_start], X[fold_end:]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:]])

        test_x = X[fold_start:fold_end]

        model.fit(train_x, train_y)
        
        trained_y[fold_start:fold_end] = model.predict_proba(test_x)

    return trained_y
