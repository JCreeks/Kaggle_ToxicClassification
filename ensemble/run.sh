#!/usr/bin/env bash
nohup python Pooled_GRU_kFold.py > pooled_gru.out
nohup python BiLSTM_CNN_kFold.py > bilstm_cnn.out
nohup python BiGRU_CNN_kFold.py > bigru_cnn.out
nohup python LSTM_clean_kFold.py > lstm.out
nohup python GRU_kFold.py > gru.out
