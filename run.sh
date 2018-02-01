#!/usr/bin/env bash
cd features/
python run.py
echo '============ training model ==========='
cd ../models/
python LSTM_baseline.py