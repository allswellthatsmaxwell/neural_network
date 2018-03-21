#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 19:11:59 2018

@author: mson
"""

import os
os.chdir('/home/mson/home/neural_network/clients')
import numpy as np
import pandas as pd
import time
import csv
from datetime import datetime, date
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from prediction_utils import (trn_val_tst, 
                              standard_binary_classification_layers,
                              Standardizer,
                              bind_and_sort)
import neural_network.neural_network as nn
import neural_network.activations as avs
import neural_network.loss_functions as losses

def sort_and_get_prior_clicks(dat):
    """ for each ip address in dat, for each click that ip address has,
    count the number of clicks that occurred prior to the one in the record
    (and add one to include the current record)"""
    dat.sort_values(['ip', 'click_time'], inplace = True)
    dat['clicks_so_far'] = dat.groupby('ip').cumcount() + 1

def string_to_timestamp(s):
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")

def timestamp_to_float(ts):
    return time.mktime(ts.timetuple())
    
def seconds_since_midnight(timestamp):
    """ 
    return the number of seconds between midnight and the time in timestamp 
    """
    midnight = datetime.strptime("00:00:00", "%H:%M:%S").time()
    base_day = datetime(1,1,1,0,0,0) ## just some default day
    return (datetime.combine(base_day, timestamp.time()) - 
            datetime.combine(base_day, midnight)).seconds

def engineer_features(dataset):
    dat = dataset
    sort_and_get_prior_clicks(dat)
    dat['click_timestamp'] = dat['click_time'].apply(string_to_timestamp)
    dat['click_timefloat'] = dat['click_timestamp'].apply(timestamp_to_float)
    dat['time_since_last_click'] = dat.groupby('ip').click_timefloat.diff()
    dat['time_of_day'] = dat['click_timestamp'].apply(seconds_since_midnight)
    unique_counts = dat.groupby('ip').agg({'os'    : 'nunique', 
                                           'device': 'nunique', 
                                           'app'   : 'nunique',
                                           'channel': 'nunique'})
    group_counts = dat.groupby('ip')['ip'].count()    
    count_dat = (unique_counts.join(group_counts).
                 rename(columns = {'ip': 'total_clicks'}))
    dat = dat.join(count_dat, on = 'ip', rsuffix = '_n_distinct')
    return dat

def prepare_predictors(dataset, continuous, categorical):
    predictors = continuous + categorical
    dat = engineer_features(dataset)
    click_id = dat['click_id']
    dat = dat[predictors]
    if categorical != []:
        dat = pd.get_dummies(dat, columns = categorical)
    return dat.as_matrix(), click_id

#def prepare_submission_file_for_streaming(orig_submit_path):
#    sorted_path = os.path.join(DATA_DIR, "submission_sorted.csv")
#    if not os.path.exists(sorted_path):
#        submission_dat = pd.read_csv(orig_submit_path)
#        submission_dat.sort_values(by = 'ip', inplace = True)
#        submission_dat.to_csv(sorted_path, index = False)
#    return sorted_path

#def stream_predict(sorted_submit_path, net, ips_at_a_time = 100):
#    assert(net.is_trained)
#    ranges = []
#    with open(sorted_submit_path) as f:
#        reader = csv.DictReader(f, delimiter = ',')
#        for row in reader:     
        
def get_X_submission(submit_path,
                     data_dir,
                     continuous_predictors, 
                     categorical_predictors):
    X_pkl_path = os.path.join(data_dir, "X_submit.pkl")
    id_pkl_path = os.path.join(data_dir, "click_id_submit.pkl")
    if not (os.path.exists(X_pkl_path) and os.path.exists(id_pkl_path)):
        submission_dat = pd.read_csv(submit_path)    
        X_submit, click_id = prepare_predictors(submission_dat, 
                                                continuous_predictors, 
                                                categorical_predictors)
        del(submission_dat)
        X_submit.dump(X_pkl_path)
        click_id.to_pickle(id_pkl_path)
        return X_submit, click_id
    else:
        return np.load(X_pkl_path), np.load(id_pkl_path)
    
def chunk_predict(X_t, net, chunk_size = 10000, verbose = True):
    """ 
    Use net to predict the outcome of each training example (column) in X_t,
    sending chunk_size records through the net at a time to fit in memory
    X_t: a matrix where each column is a training example
    net: a trained instance of the neural_network class
    returns: a list with predictions corresponding to the columns of X_t 
    """
    assert(net.is_trained)
    yhat_submit = []
    m = X_t.shape[1]
    for i in range(0, m // chunk_size):
        small_mat = X_t[:, (chunk_size * i):(chunk_size * (i + 1))]
        yhat_chunk = net.predict(small_mat)
        yhat_submit.extend(yhat_chunk)
        if verbose: print(int(i * chunk_size / m * 100), 
                          "% of records processed")
    left_over = m % chunk_size
    small_mat = X_t[:, (m - left_over):]
    yhat_chunk = net.predict(small_mat)
    yhat_submit.extend(yhat_chunk)
    if verbose: print("100 % of records processed")
    assert(len(yhat_submit) == m)
    return yhat_submit

CATEGORICAL_PREDICTORS = []#['os', 'device', 'app', 'channel']
CONTINUOUS_PREDICTORS= ['os_n_distinct', 'device_n_distinct', 
                        'app_n_distinct', 'channel_n_distinct',
                        'time_of_day', 'clicks_so_far']
PREDICTORS = CATEGORICAL_PREDICTORS + CONTINUOUS_PREDICTORS
DATA_DIR = "../../data/china"
trn_path = os.path.join(DATA_DIR, "train.csv")
tst_path = os.path.join(DATA_DIR, "test.csv")

dataset = pd.read_csv(trn_path, nrows = 10000)
dataset['click_id'] = range(dataset.shape[0])
##sorted_submission_path = prepare_submission_file_for_streaming(tst_path)

attributed_rate = dataset['is_attributed'].sum() / dataset.shape[0]



X, _ = prepare_predictors(dataset, CONTINUOUS_PREDICTORS, CATEGORICAL_PREDICTORS)
y = np.array(dataset['is_attributed'])    

stn = Standardizer(X)
X = stn.standardize(X)

(X_trn, y_trn, 
 X_val, y_val, 
 X_tst, y_tst) = trn_val_tst(X, y, 8/10, 1/10, 1/10)

net_shape = [X.shape[1], 30, 20, 20, 20, 20, 20, 1]
activations = standard_binary_classification_layers(len(net_shape))

net = nn.Net(net_shape, activations, use_adam = True)
costs = net.train(X = X_trn.T, y = y_trn, 
                  iterations = 400, 
                  learning_rate = 0.001,
                  minibatch_size = 128,
                  lambd = 0.25,
                  debug = True)
yhat_val = 1 - net.predict(X_val.T)
yyhat_val = bind_and_sort(y_val, yhat_val)
auc_val = roc_auc_score(y_val, yhat_val)
print("auc =", auc_val)

X_submit, click_id_submit = get_X_submission(tst_path, 
                                             DATA_DIR, 
                                             CONTINUOUS_PREDICTORS, 
                                             CATEGORICAL_PREDICTORS)
X_submit_transpose = stn.standardize(X_submit).T
yhat_submit = chunk_predict(X_submit_transpose, net, chunk_size = 1000000,
                            verbose = True)
submission_output = pd.DataFrame({'click_id': list(click_id_submit),
                                  'is_attributed': yhat_submit})
submission_output.to_csv(os.path.join(DATA_DIR, 'submission_output.csv'), index = False)

    
