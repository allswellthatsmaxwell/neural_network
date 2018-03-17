#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:11:59 2018

@author: mson
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score
from prediction_utils import trn_val_tst
import pandas as pd
import neural_network.neural_network as nn
import neural_network.activations as avs
import neural_network.loss_functions as losses

DATA_DIR = "../../data/china"
trn_path = os.path.join(DATA_DIR, "train.csv")
tst_path = os.path.join(DATA_DIR, "test.csv")

dat = pd.read_csv(trn_path, nrows = 10000)
## tst = pd.read_csv(tst_path, nrows = 1000000)

attributed_rate = dat['is_attributed'].sum() / dat.shape[0]

X = dat.drop('is_attributed', axis = 1)
y = dat['is_attributed']

X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(X, y, 
                                                       8/10, 1/10, 1/10)   
