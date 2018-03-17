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

trn = pd.read_csv(trn_path, nrows = 1000000)

attributed_rate = trn['is_attributed'].sum() / trn.shape[0]
