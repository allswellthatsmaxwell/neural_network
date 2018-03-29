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
from neural_network.evaluator import Evaluator
import neural_network.neural_network as nn
import neural_network.activations as avs
import neural_network.loss_functions as losses
from sklearn.model_selection import train_test_split
#%%

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
    y = click_id = dat['is_attributed']
    dat = dat[predictors]
    if categorical != []:
        dat = pd.get_dummies(dat, columns = categorical)
    return dat, click_id, y

def prepare_submission_file_for_streaming(orig_submit_path, output_dir, 
                                          nrows = None):
    """ 
    Reads the first nrows from the csv at orig_submit_path,
    sorts it by ip, and writes it back into data directory; returns
    the path written to.
    """
    if nrows is not None:
        sorted_fname = f"submission_sorted_{nrows}.csv"
    else:
        sorted_fname = "submission_sorted.csv"
    sorted_path = os.path.join(output_dir, sorted_fname)
    if not os.path.exists(sorted_path):
        submission_dat = pd.read_csv(orig_submit_path, nrows = nrows)
        submission_dat.sort_values(by = 'ip', inplace = True)
        submission_dat.to_csv(sorted_path, index = False)
    return sorted_path

def stream_predict(sorted_submit_path, net, 
                   continuous, categorical, train_columns,
                   standardizer,
                   lines_at_a_time = 10000,
                   verbose = True):
    """
    Reads, prepares as training data was prepared, and predicts upon 
    lines_at_a_time lines at a time (as a dataframe) from the csv file 
    at sorted_submit_path.
    lines_at_a_time is approximate, because ip addresses are read in groups.
    precondition: the file at sorted_submit_path is sorted by ip
    net: a trained net
    continuous, categorical: the same predictors used to train the net
    standardizer: the same standardizer used to standardize the input matrix
    when the net was trained
    returns: a vector of predictions, one per line in sorted_submit_path,
             and corresponding click_ids
    """
    assert(net.is_trained)
    predictions = []
    click_ids = []
    i = 0
    with open(sorted_submit_path) as f:
        reader = csv.DictReader(f, delimiter = ',')
        header = reader.fieldnames
        row = next(reader)
        rows_this_batch = []
        while True:
            rows_this_batch.append(row)
            try:                                
                if len(rows_this_batch) % lines_at_a_time == 0:
                    ## If wrapping up the collection of a batch,
                    ## collect the rest of the ip addresses so that no ip
                    ## is split across batches.
                    current_ip = row['ip']
                    row = next(reader)
                    while row['ip'] == current_ip:
                        rows_this_batch.append(row)
                        row = next(reader)
                    predictions_batch, click_id_batch = do_batch_prediction(
                            rows_this_batch, header, net, standardizer,                                                           
                            continuous, categorical, train_columns)
                    predictions.extend(predictions_batch)
                    click_ids.extend(click_id_batch)
                    i += len(rows_this_batch)
                    if verbose: print(i, "rows processed")
                    rows_this_batch = []
                else:
                    row = next(reader)
            except StopIteration:
                break
        predictions_batch, click_id_batch = do_batch_prediction(
                rows_this_batch, header, net, standardizer,
                continuous, categorical, train_columns)
        predictions.extend(predictions_batch)
        click_ids.extend(click_id_batch)        
    return predictions, click_ids

def do_batch_prediction(rows, header, net, standardizer, 
                        continuous, categorical, train_columns):
    """
    combines rows into a dataframe with column names header, 
    prepares predictors specified in continuous and categorical, 
    standardizes with standardizer, and predicts using net on the resulting 
    matrix.
    returns: a vector of predictions and corresponding click ids
    """
    assert(net.is_trained)
    df = pd.DataFrame(rows, columns = header)
    dat, click_id_batch = prepare_predictors(df, continuous, categorical)
    conform_columns(dat, train_columns)
    X = dat.as_matrix()
    X = standardizer.standardize(X)
    predictions_batch = net.predict_proba(X.T)
    return predictions_batch, click_id_batch

def conform_columns(dat, train_columns):
    """
    remove and add columns from dat inplace so that it has the same columns 
    as train_columns
    """
    new_levels = [colname for colname in dat.columns 
                  if colname not in train_columns]
    dat.drop(labels = new_levels, axis = 1, inplace = True)
    ## Add all-zero columns for dummy levels that are in training, but aren't
    ## in dat
    for colname in train_columns:
        if colname not in dat.columns:
            dat[colname] = 0

def get_X_submission(submit_path,
                     data_dir,
                     continuous_predictors, 
                     categorical_predictors):
    """ 
    Read submit_path contents into dataframe, prepare using prepare_predictors,
    and pickle out the resulting predictor matrix (as a numpy matrix) 
    and the corresponding click_id (as a pandas Series). If the pickled
    files already exist, they are read; otherwise, they are created. In either
    case, the predictor matrix and click_id Series are returned.
    """
    X_pkl_path = os.path.join(data_dir, "X_submit.pkl")
    id_pkl_path = os.path.join(data_dir, "click_id_submit.pkl")
    if not (os.path.exists(X_pkl_path) and os.path.exists(id_pkl_path)):
        submission_dat = pd.read_csv(submit_path)    
        dat_submit, click_id = prepare_predictors(submission_dat,
                                                  continuous_predictors,
                                                  categorical_predictors)
        del(submission_dat)
        X_submit = dat_submit.as_matrix()
        del(dat_submit)
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
        yhat_chunk = net.predict_proba(small_mat)
        yhat_submit.extend(yhat_chunk)
        if verbose: print(int(i * chunk_size / m * 100), 
                          "% of records processed")
    left_over = m % chunk_size
    small_mat = X_t[:, (m - left_over):]
    yhat_chunk = net.predict_proba(small_mat)
    yhat_submit.extend(yhat_chunk)
    if verbose: print("100 % of records processed")
    assert(len(yhat_submit) == m)
    return yhat_submit

def drop_single_occurence_columns(dat):
    """ drop columns with two levels in which one level appears only once """
    cols_to_drop = []
    for colname in dat.columns:
        values_counts = dat[colname].value_counts()
        if len(values_counts) == 2 and values_counts[1] == 1:
            cols_to_drop.append(colname)
    return dat.drop(cols_to_drop, axis = 1)

def ip_aware_split(dat, second_set_prop):
    """
    split dat into two sets by ips (not by record). 
    second_set_prop: probability an ip is assigned to the second set
    returns: two dataframes with the same columns as the input,
             and where no ip address is in both frames.
    """
    assert(0 <= second_set_prop <= 1)
    ips = dat.ip.unique()
    second_set_probas = np.random.rand(len(ips))
    is_second = [True if p < second_set_prop else False 
                 for p in second_set_probas]
    second_ips = ips[is_second]
    first_ips = ips[[not b for b in is_second]]
    first = dat[dat['ip'].isin(first_ips)]
    second = dat[dat['ip'].isin(second_ips)]
    return first.copy(), second.copy()

#%%
    
class Param:
    def __init__(self, alpha, lambd, aucs):
        self.alpha = alpha
        self.lambd = lambd
        self.aucs = aucs
    
def hyperopt(net_shape, activations, X_trn, y_trn, evaluator, n_combos):
    """
    searches over reasonable values of different parameters to a Net and
    returns a list of the aucs achieved with each parameter set.
    
    returns: a list of Param
    Arguments: 
        n_combos: the number of parameter combinations to try
    """
    learning_rates = 10**(-4 * np.random.rand(n_combos))
    lambdas = 10**(-2 * np.random.rand(n_combos))
    results = []
    for alpha, lambd in zip(learning_rates, lambdas):
        print("alpha:", alpha, "| lambda:", lambd)
        
        net = nn.Net(net_shape, activations, use_adam = True)
        try:
            _, aucs = net.fit(X_trn.T, y_trn, 
                              iterations = 50,
                              learning_rate = alpha,
                              minibatch_size = 128 * 24 * 2,
                              lambd = lambd,
                              evaluator = evaluator)
            results.append(Param(alpha, lambd, aucs))
        except:
            aucs = None
            results.append(Param(alpha, lambd, aucs))            
    return results

def get_maximizing_params(param_list):
    """
    param_list: a list of class Param
    returns: the index of and value of the highest auc observed over all
    the parameters
    """
    max_auc = 0
    maximizing_index = -1
    for (i, param) in enumerate(param_list):
        highest_auc_for_param = np.max(param.aucs)
        if (highest_auc_for_param is not None 
            and highest_auc_for_param > max_auc):
            max_auc = highest_auc_for_param
            maximizing_index = i
    return maximizing_index, max_auc

#%%
    
CATEGORICAL_PREDICTORS = []#['os', 'device', 'app', 'channel']
CONTINUOUS_PREDICTORS= ['os_n_distinct', 'device_n_distinct', 
                        'app_n_distinct', 'channel_n_distinct',
                        'time_of_day', 'clicks_so_far']
PREDICTORS = CATEGORICAL_PREDICTORS + CONTINUOUS_PREDICTORS
DATA_DIR = "../../data/china"
OUTPUT_DIR = "../../out/china"
trn_path = os.path.join(DATA_DIR, "train.csv")
tst_path = os.path.join(DATA_DIR, "test.csv")

NROW_TRAIN = 200000
#%%
dataset = pd.read_csv(trn_path, nrows = NROW_TRAIN)
dataset['click_id'] = range(dataset.shape[0])
##sorted_submission_path = prepare_submission_file_for_streaming(tst_path)

#%%
attributed_rate = dataset['is_attributed'].sum() / dataset.shape[0]

train, devtest = ip_aware_split(dataset, 2/10)
dev, test = ip_aware_split(devtest, 5/10)
del(devtest)
train, _, y_trn = prepare_predictors(train, CONTINUOUS_PREDICTORS, 
                                     CATEGORICAL_PREDICTORS)
dev, _, y_dev = prepare_predictors(dev, CONTINUOUS_PREDICTORS, 
                                   CATEGORICAL_PREDICTORS)
test, _, y_tst = prepare_predictors(test, CONTINUOUS_PREDICTORS, 
                                    CATEGORICAL_PREDICTORS)
train = drop_single_occurence_columns(train)
conform_columns(dev, train.columns)
conform_columns(test, train.columns)
X_trn, X_dev, X_tst = train.as_matrix(), dev.as_matrix(), test.as_matrix()
y_trn, y_dev, y_tst = np.array(y_trn), np.array(y_dev), np.array(y_tst)

stn = Standardizer(X_trn)
X_trn = stn.standardize(X_trn)
X_dev = stn.standardize(X_dev)
X_tst = stn.standardize(X_tst)
evaluator = Evaluator(X_dev.T, y_dev)

net_shape = [X_trn.shape[1], 30, 20, 20, 20, 20, 20, 1]
activations = standard_binary_classification_layers(len(net_shape))

#%%
params_aucs = hyperopt(net_shape, activations, X_trn, y_trn, evaluator, 
                       n_combos = 100)
maxer_ind, max_auc = get_maximizing_params(params_aucs)
param = params_aucs[maxer_ind]
alpha, lambd, epochs = param.alpha, param.lambd, np.argmax(param.aucs)
#%%

net = nn.Net(net_shape, activations, use_adam = True)
costs, aucs = net.fit(X = X_trn.T, y = y_trn, 
                      iterations = epochs, ##25, 
                      learning_rate = alpha, ##0.005,
                      minibatch_size = 128 * 24 * 2,
                      lambd = lambd, ##0.02,
                      evaluator = evaluator,
                      debug = True)
yhat_trn = net.predict_proba(X_trn.T)
yhat_dev = net.predict_proba(X_dev.T)
yhat_tst = net.predict_proba(X_tst.T)
auc_trn = roc_auc_score(y_trn, yhat_trn)
auc_dev = roc_auc_score(y_dev, yhat_dev)
auc_tst = roc_auc_score(y_tst, yhat_tst)
yyhat_tst = bind_and_sort(y_tst, yhat_tst)
print("trn auc =", auc_trn)
print("dev auc =", auc_dev)
#print("tst auc =", auc_tst)

#%%

sorted_path = prepare_submission_file_for_streaming(tst_path, OUTPUT_DIR, 
                                                    nrows = None)
predictions, ids = stream_predict(sorted_path, net, 
                                  CONTINUOUS_PREDICTORS, 
                                  CATEGORICAL_PREDICTORS,
                                  train.columns,
                                  stn,
                                  lines_at_a_time = 100000)

submission_output = pd.DataFrame({'click_id': list(ids),
                                  'is_attributed': predictions})
submission_output.to_csv(os.path.join(OUTPUT_DIR, 'submission_output.csv'), 
                         index = False)

