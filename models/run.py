#!/usr/local/miniconda2/bin/python
# _*_ coding: utf-8 _*_

"""
@author: Jing Guo
@time  : 12/29/17
"""
from __future__ import absolute_import, division, print_function

import os
import sys

module_path = os.path.abspath(os.path.join('..'))
sys.path.append(module_path)

cmd = 'rm ../output/subm*.csv'
os.system(cmd)

# cmd = 'python GRU_baseline.py'
# os.system(cmd)

cmd = 'python Bidirectional_LSTM_baseline.py'
os.system(cmd)
