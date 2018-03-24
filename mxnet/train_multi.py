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
from net import net_define,  net_define_eu
import utils
import config

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
    L = - y_true*nd.log2(y_pred) * train_neg_ratio - (1-y_true) * nd.log2(1-y_pred) * train_pos_ratio
    return nd.mean(L)

if __name__ == "__main__":
    # setting the hyper parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--kfold', default=5, type=int)
    args = parser.parse_args()
    '''
    train_data = np.random.randint(0, high=config.MAX_WORDS, size=(10000, config.MAX_LENGTH))
    train_label = np.random.randint(0, high=2, size=(10000, 6)) 
    '''
#     ctx = [mx.gpu(2), mx.gpu(3), mx.gpu(4), mx.gpu(5)]
    ctx = utils.try_gpu()
    net = net_define_eu()
    # net.initialize(mx.init.Xavier(),ctx=ctx)

    print("fetching train....")
    train_data, train_label, word_index = fetch_data()
    
    print("starting embedding...")
    if not os.path.exists('../input/GD_EM.plk'):
        embedding_dict = get_word_embedding()
        print(len(word_index))
        em = get_embed_matrix(embedding_dict, word_index)
        with open('../input/GD_EM.plk', "wb") as f:
            pickle.dump(em, f, -1)
    else:
         with open('../input/GD_EM.plk', "rb") as f:
            em = pickle.load(f)
    
    print('updating net...')
    # print 'copy array'
    em = array(em, ctx=mx.cpu())
    # print 'copy array done'
    net.collect_params()['sequential0_embedding0_weight'].set_data(em)
    net.collect_params().reset_ctx(ctx)
    print net.collect_params()

    print_batches = 1000
    shuffle_idx = np.random.permutation(train_data.shape[0])
    train_data = train_data[shuffle_idx]
    train_label = train_label[shuffle_idx]

    # print em.shape
    data_iter = NDArrayIter(data= train_data[:-10000], label=train_label[:-10000], batch_size=args.batch_size, shuffle=True)
    val_data_iter = NDArrayIter(data= train_data[-10000:], label=train_label[-10000:], batch_size=args.batch_size, shuffle=False)
    trainer = Trainer(net.collect_params(),'adam', {'learning_rate': 0.001})
    # trainer = Trainer(net.collect_params(),'RMSProp', {'learning_rate': 0.001})
    # utils.train(data_iter, val_data_iter, net, EntropyLoss,
    #             trainer, ctx, num_epochs=args.epochs, print_batches=print_batches)
    utils.train_multi(data_iter, val_data_iter, args.kfold, net, EntropyLoss,
                trainer, ctx, num_epochs=args.epochs, print_batches=print_batches)
    net.save_params('../params/net.params')
