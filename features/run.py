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

cmd = 'rm ../input/*train.pkl'
os.system(cmd)

cmd = 'rm ../input/*test.pkl'
os.system(cmd)

cmd = 'python initialTransform.py'
os.system(cmd)

cmd = 'python initialTransform_cleaned.py'
os.system(cmd)
