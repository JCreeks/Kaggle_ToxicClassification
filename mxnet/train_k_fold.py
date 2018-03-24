import sys
import os
import argparse
import mxnet as mx
import numpy as np
from preprocess import fetch_data, get_word_embedding, get_embed_matrix
from mxnet.gluon import Trainer
from mxnet.gluon.data import DataLoader,Dataset
from mxnet.io import NDArrayIter
from mxnet.ndarray import array
from mxnet import nd
from net import net_define, net_define_eu
from sklearn.model_selection import KFold, StratifiedKFold
import utils
import config
import pickle

def CapLoss(y_pred, y_true):
    L = y_true * nd.square(nd.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * nd.square(nd.maximum(0., y_pred - 0.1))
    return nd.mean(nd.sum(L, 1))

def EntropyLoss(y_pred, y_true):
    L = - y_true*nd.log2(y_pred) - (1-y_true) * nd.log2(1-y_pred)
    return nd.mean(L)

def EntropyLoss1(y_pred, y_true):
    train_pos_ratio = array([ 0.09584448, 0.00999555, 0.05294822, 0.00299553, 0.04936361, 0.00880486], ctx=y_pred.context, dtype=np.float32)*10
    train_neg_ratio = (1.0-train_pos_ratio)*10
    L = - y_true*nd.log2(y_pred) * train_neg_ratio - (1-y_true) * nd.log2(1-y_pred)
    return nd.mean(L)

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--epochs', default=3, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--kfold', default=10, type=int)
    parser.add_argument('--print_batches', default=100, type=int)
    args = parser.parse_args()

#     train_data, train_label, word_index = fetch_data(False)
#     embedding_dict = get_word_embedding()
#     em = get_embed_matrix(embedding_dict, word_index)
    print("fetching train....")
    train_data, train_label, word_index = fetch_data()
    print(len(word_index))
    
    print("starting embedding...")
    if not os.path.exists('../input/GD_EM.plk'):
        embedding_dict = get_word_embedding()
        em = get_embed_matrix(embedding_dict, word_index)
        with open('../input/GD_EM.plk', "wb") as f:
            pickle.dump(em, f, -1)
    else:
         with open('../input/GD_EM.plk', "rb") as f:
            em = pickle.load(f)
    
    print('updating net...')
    em = array(em, ctx=mx.cpu())
    kf_label = np.ones(train_label.shape)
    for i in range(train_label.shape[1]):
        kf_label[:,i] = 2**i
    kf_label = np.sum(kf_label, axis=1)

    kf = StratifiedKFold(n_splits=args.kfold, shuffle=True)
    for i, (inTr, inTe) in enumerate(kf.split(train_data, kf_label)):
        print('training fold: ', i)
#         ctx = [mx.gpu(0), mx.gpu(1)]# , mx.gpu(4), mx.gpu(5)]
        ctx = [utils.try_gpu(),utils.try_gpu()]
        net = net_define_eu()

        xtr = train_data[inTr]
        xte = train_data[inTe]
        ytr = train_label[inTr]
        yte = train_label[inTe]
        data_iter =     NDArrayIter(data= xtr, label=ytr, batch_size=args.batch_size, shuffle=True)
        val_data_iter = NDArrayIter(data= xte, label=yte, batch_size=args.batch_size, shuffle=False)

        # print net.collect_params()
        net.collect_params().reset_ctx(ctx)
        net.collect_params()['sequential'+str(i)+ '_embedding0_weight'].set_data(em)
        net.collect_params()['sequential'+str(i)+ '_embedding0_weight'].grad_req = 'null'
        trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.001})
        # trainer = Trainer(net.collect_params(),'RMSProp', {'learning_rate': 0.001})
        utils.train_multi(data_iter, val_data_iter, i, net, EntropyLoss,
                    trainer, ctx, num_epochs=args.epochs, print_batches=args.print_batches)
