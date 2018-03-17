#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:11:59 2018

@author: mson
"""

import os
import numpy as np
import pandas as pd
import time
from datetime import datetime, date
from sklearn.metrics import roc_auc_score
from prediction_utils import trn_val_tst, standard_binary_classification_layers
import neural_network.neural_network as nn
import neural_network.activations as avs
import neural_network.loss_functions as losses

def sort_and_get_prior_clicks(dat):
    dat.sort_values(['ip', 'click_time'], inplace = True)
    dat['clicks_so_far'] = dat.groupby('ip').cumcount() + 1
    return dat

def string_to_timestamp(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def timestamp_to_float(ts):
    return time.mktime(ts.timetuple())
    
def seconds_since_midnight(timestamp):
    midnight = datetime.strptime("00:00:00", "%H:%M:%S").time()
    base_day = datetime(1,1,1,0,0,0)
    return (datetime.combine(base_day, timestamp.time()) - 
            datetime.combine(base_day, midnight)).seconds

def prepare_data(dataset):
    dat = sort_and_get_prior_clicks(dataset)
    dat['click_timestamp'] = dat['click_time'].apply(string_to_timestamp)
    dat['click_timefloat'] = dat['click_timestamp'].apply(timestamp_to_float)    
    dat['time_since_last_click'] = dat.groupby('ip').click_timefloat.diff()
    dat['time_of_day'] = dat['click_timestamp'].apply(seconds_since_midnight)
    unique_counts = dat.groupby('ip').agg({'os'    : 'nunique', 
                                           'device': 'nunique', 
                                           'app'   : 'nunique'})
    dat = dat.join(unique_counts, on = 'ip', rsuffix = '_n_distinct')   
    return dat

DATA_DIR = "../../data/china"
trn_path = os.path.join(DATA_DIR, "train.csv")
tst_path = os.path.join(DATA_DIR, "test.csv")

dataset = pd.read_csv(trn_path, nrows = 100000)
## submission_dat = pd.read_csv(tst_path, nrows = 100000)

dat = prepare_data(dataset)

attributed_rate = dat['is_attributed'].sum() / dat.shape[0]

X = dat.drop('is_attributed', axis = 1)
y = dat['is_attributed']

X_trn, y_trn, X_val, y_val, X_tst, y_tst = trn_val_tst(X, y, 
                                                       8/10, 1/10, 1/10)

net_shape = [X.shape[1], 30, 20, 20, 20, 20, 20, 1]
activations = standard_binary_classification_layers(len(net_shape))

net = nn.Net(net_shape, activations, use_adam = True)

